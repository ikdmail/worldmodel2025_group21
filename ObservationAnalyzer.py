import json
import pandas as pd
from pathlib import Path

class ObservationAnalyzer:
    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        self.step_dir = self.metadata_path.parent
        self.unified_map = {}
        self.node_id_map = {}
        self.page_title = "Unknown Page"
        self.focused_bid = ""
        self.stats = {"Total_BIDs": 0, "AX_Match_Rate": "0%"} # デフォルト値で初期化
        
        # 1. metadata.json を読み込む
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # 2. データのロードと結合を実行
        self._fuse_all_data()

    def _read_json_by_key(self, key):
        saved_files = self.metadata.get("saved_files", {})
        filename = saved_files.get(key)
        if not filename: return None
        target_path = self.step_dir / filename
        if not target_path.exists(): return None
        with open(target_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_recursive_text(self, node_id):
        node = self.node_id_map.get(str(node_id))
        if not node: return ""
        if node.get("role", {}).get("value") == "StaticText":
            return node.get("name", {}).get("value", "")
        texts = [self._get_recursive_text(cid) for cid in node.get("childIds", [])]
        return " ".join([t for t in texts if t]).strip()

    def _fuse_all_data(self):
        """metadataに基づいて全てのファイルを読み込み、unified_mapを作成する"""
        dom = self._read_json_by_key("dom_object")
        ax = self._read_json_by_key("axtree_object")
        props = self._read_json_by_key("extra_element_properties")
        obs_meta = self.metadata.get("observation_metadata", {})

        if not (dom and ax and props): return

        # 基本情報のセット
        titles = obs_meta.get("open_pages_titles", [])
        if titles: self.page_title = titles[0]
        self.focused_bid = str(obs_meta.get("focused_element_bid", ""))

        # 結合用マップの作成
        self.node_id_map = {str(n["nodeId"]): n for n in ax['nodes']}
        ax_map = {str(n.get("backendDOMNodeId")): n for n in ax['nodes'] if n.get("backendDOMNodeId")}
        
        strings = dom['strings']
        bid_idx = strings.index('bid') if 'bid' in strings else -1
        
        dom_bid_count, match_count = 0, 0

        for doc in dom['documents']:
            n_data = doc['nodes']
            attr_lists = n_data['attributes']
            for i, backend_id in enumerate(n_data['backendNodeId']):
                current_bid = None
                attrs = {}
                if i < len(attr_lists):
                    al = attr_lists[i]
                    for j in range(0, len(al), 2):
                        k, v = strings[al[j]], strings[al[j+1]]
                        attrs[k] = v
                        if al[j] == bid_idx: current_bid = v
                
                if current_bid is None: continue
                dom_bid_count += 1
                ax_n = ax_map.get(str(backend_id), {})
                if ax_n: match_count += 1
                
                p = props.get(str(current_bid), {})
                ax_props = {item['name']: item['value'].get('value') for item in ax_n.get('properties', [])} if ax_n else {}
                
                self.unified_map[current_bid] = {
                    "BID": current_bid, "Internal_ID": backend_id,
                    "Tag": strings[n_data['nodeName'][i]],
                    "Role": ax_n.get("role", {}).get("value", "generic") if ax_n else "generic",
                    "AX_Name": ax_n.get("name", {}).get("value", "") if ax_n else "",
                    "InnerT": self._get_recursive_text(ax_n.get("nodeId")) if ax_n else "",
                    "Value": str(ax_n.get("value", {}).get("value", "")) if ax_n else "",
                    "BBox": p.get("bbox"), "Vis": p.get("visibility", 0.0),
                    "Z-Index": p.get("z_index", 0), "Clickable": p.get("clickable", False),
                    "Attributes": attrs, "AX_Props": ax_props,
                    "Parent_NodeID": ax_n.get("parentId") if ax_n else None
                }
        
        self.stats = {
            "Total_BIDs": dom_bid_count,
            "AX_Match_Rate": f"{(match_count/dom_bid_count)*100:.1f}%" if dom_bid_count > 0 else "0%"
        }

    def _get_hierarchical_area(self, bid):
        path, curr_bid, visited = [], bid, set()
        while curr_bid and curr_bid not in visited:
            visited.add(curr_bid); node = self.unified_map.get(curr_bid)
            if not node: break
            name = node['AX_Name'] or node['Attributes'].get('aria-label')
            if node['Role'] in ["form", "region", "navigation", "main", "banner"] and name:
                if name not in path: path.append(name)
            p_id = node.get('Parent_NodeID')
            curr_bid = next((b for b, d in self.unified_map.items() if str(d.get('Internal_ID')) == str(p_id)), None)
        return " > ".join(reversed(path)) if path else "Main Content"

    def _get_spatial_labels(self, bid, data):
        if not data['BBox'] or not any(data['BBox']): return "", ""
        tx, ty, tw, th = data['BBox']
        l_label, a_label = "", ""
        for s_bid, s_data in self.unified_map.items():
            if s_bid == bid or not s_data['BBox']: continue
            sx, sy, sw, sh = s_data['BBox']
            txt = s_data['InnerT']
            if not txt or len(txt) > 40: continue
            if abs(ty - sy) < 15 and 0 < (tx - (sx + sw)) < 160: l_label = txt
            if abs((tx + tw/2) - (sx + sw/2)) < 50 and 0 < (ty - (sy + sh)) < 60: a_label = txt
        return l_label, a_label

    def analyze2(self):
        records = []
        for bid, data in self.unified_map.items():
            if data['Vis'] == 0 and bid != self.focused_bid: continue
            l_lab, a_lab = self._get_spatial_labels(bid, data)
            area_path = self._get_hierarchical_area(bid)
            blob = (str(data['Attributes']) + " " + str(data['AX_Props'])).lower()
            records.append({
                "BID": bid, "Area": area_path, "Page": self.page_title, "Internal_ID": data['Internal_ID'],
                "Role": data['Role'], "Tag": data['Tag'],
                "Label_L": l_lab, "Label_A": a_lab, "Label_AX": data['AX_Name'], "Label_P": data['AX_Props'].get('placeholder', ''), "InnerT": data['InnerT'][:500],
                "Value": data['Value'], "Focused": "YES" if bid == self.focused_bid else "no",
                "Status": "Required" if 'required' in blob else "Active", 
                "HasPop": "YES" if 'haspopup' in blob else "no",
                "Expanded": "Open" if 'expanded="true"' in blob else "no",
                "JS": "YES" if data['Clickable'] else "no", "Hover_Strat": "Investigate" if 'haspopup' in blob else "None",
                "Vis": data['Vis'], "Z-Index": data['Z-Index'], "BBox": str(data['BBox']),
                "Scroll": "In-View" if data['BBox'] and 0 <= data['BBox'][1] <= 1080 else "Need",
                "BG": data['Attributes'].get('background_color', ''), "BD": data['Attributes'].get('border_color', ''),
                "TX": data['Attributes'].get('text_color', ''), "Class": data['Attributes'].get('class', '')[:100]
            })
        df = pd.DataFrame(records)
        if not df.empty and "Internal_ID" in df.columns:
            df = df.drop(columns=["Internal_ID"])
        return df

    def analyze3(self):
        records = []
        for bid, data in self.unified_map.items():
            # インタラクティブ要素は Vis=0 でも残す
            is_interactive = data['Tag'] in ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'] or data['Clickable']
            if data['Vis'] == 0 and bid != self.focused_bid and not is_interactive: continue

            l_lab, a_lab = self._get_spatial_labels(bid, data)
            area_path = self._get_hierarchical_area(bid)
            
            # --- 状態判定の強化 ---
            blob = (str(data['Attributes']) + " " + str(data['AX_Props'])).lower()
            checked_val = data['AX_Props'].get('checked')
            is_checked = (checked_val is True or str(checked_val).lower() == 'true')
            
            status_parts = []
            if 'required' in blob: status_parts.append("Required")
            if is_checked: status_parts.append("Checked")
            
            records.append({
                "BID": bid, "Area": area_path, "Page": self.page_title,
                "Role": data['Role'], "Tag": data['Tag'],
                "Label_L": l_lab, "Label_A": a_lab, "Label_AX": data['AX_Name'], "Label_P": data['AX_Props'].get('placeholder', ''), "InnerT": data['InnerT'][:500],
                "Value": data['Value'], "Focused": "YES" if bid == self.focused_bid else "no",
                "Status": "|".join(status_parts) if status_parts else "Active", 
                "HasPop": "YES" if 'haspopup' in blob else "no",
                "Expanded": "Open" if 'expanded="true"' in blob else "no",
                "JS": "YES" if data['Clickable'] else "no", "Hover_Strat": "Investigate" if 'haspopup' in blob else "None",
                "Vis": data['Vis'], "Z-Index": data['Z-Index'], "BBox": str(data['BBox']),
                "Scroll": "In-View" if data['BBox'] and 0 <= data['BBox'][1] <= 1080 else "Need",
                "BG": data['Attributes'].get('background_color', ''), "BD": data['Attributes'].get('border_color', ''),
                "TX": data['Attributes'].get('text_color', ''), "Class": data['Attributes'].get('class', '')[:100]
            })
        df = pd.DataFrame(records)
        if not df.empty and "Internal_ID" in df.columns:
            df = df.drop(columns=["Internal_ID"])
        return df


    def analyze(self):
        records = []
        for bid, data in self.unified_map.items():
            # 1. BBox（物理的な位置・サイズ）の厳格なチェック
            # BBoxが None, または [x, y, 0, 0] のようにサイズがない要素は「幻覚」の原因になるため除外
            bbox = data.get('BBox')
            has_physical_size = bbox and len(bbox) >= 4 and bbox[2] >= 1 and bbox[3] >= 1
            
            # Focused（現在操作中の要素）でない限り、サイズのない要素はノイズとして捨てる
            if not has_physical_size and bid != self.focused_bid:
                continue

            # 2. 可視性フィルター（BBoxがあるものの中で、さらにVisで絞り込む）
            # ただし、操作可能なタグ（INPUT, SELECT等）は Vis=0 でも残す（ServiceNowの実体保護）
            is_interactive = data['Tag'] in ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'] or data['Clickable']
            if data['Vis'] == 0 and bid != self.focused_bid and not is_interactive:
                continue

            # 3. 周辺ラベルとエリアパスの取得
            l_lab, a_lab = self._get_spatial_labels(bid, data)
            area_path = self._get_hierarchical_area(bid)
            
            # 4. AXTreeからの「Checked」状態の抽出
            # AXTreeのプロパティから直接 checked 状態（true/false/mixed）を取得
            checked_val = data['AX_Props'].get('checked')
            is_checked = (checked_val is True or str(checked_val).lower() == 'true')
            
            # 5. Status列の構築
            blob = (str(data['Attributes']) + " " + str(data['AX_Props'])).lower()
            status_parts = []
            if 'required' in blob: 
                status_parts.append("Required")
            if is_checked: 
                status_parts.append("Checked")
            
            # 最終的なステータス文字列（例: "Required|Checked", "Checked", "Active"）
            current_status = "|".join(status_parts) if status_parts else "Active"

            records.append({
                "BID": bid, 
                "Area": area_path, 
                "Page": self.page_title, 
                "Role": data['Role'], 
                "Tag": data['Tag'],
                "Label_L": l_lab, 
                "Label_A": a_lab, 
                "Label_AX": data['AX_Name'], 
                "Label_P": data['AX_Props'].get('placeholder', ''), 
                "InnerT": data['InnerT'][:500],
                "Value": data['Value'], 
                "Focused": "YES" if bid == self.focused_bid else "no",
                "Status": current_status,
                "HasPop": "YES" if 'haspopup' in blob else "no",
                "Expanded": "Open" if 'expanded="true"' in blob else "no",
                "JS": "YES" if data['Clickable'] else "no", 
                "Hover_Strat": "Investigate" if 'haspopup' in blob else "None",
                "Vis": data['Vis'], 
                "Z-Index": data['Z-Index'], 
                "BBox": str(bbox),
                "Scroll": "In-View" if bbox and 0 <= bbox[1] <= 1080 else "Need",
                "BG": data['Attributes'].get('background_color', ''), 
                "BD": data['Attributes'].get('border_color', ''),
                "TX": data['Attributes'].get('text_color', ''), 
                "Class": data['Attributes'].get('class', '')[:100]
            })
            
        df = pd.DataFrame(records)
        if not df.empty and "Internal_ID" in df.columns:
            df = df.drop(columns=["Internal_ID"])
        return df

if __name__ == "__main__":
    meta_file = Path(r"C:\Users\user\Desktop\dev\agent\task_execution_data_full_logs\create-change-request\step_1_state_metadata.json")
    obs = ObservationAnalyzer(meta_file)
    df = obs.analyze()
    if not df.empty:
        df.to_csv(f"create-change-request_step1.csv", index=False)
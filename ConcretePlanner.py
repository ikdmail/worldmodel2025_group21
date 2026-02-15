import json
import traceback
import logging
import pandas as pd
import os
import ast
import time
import datetime
from jinja2 import Environment, FileSystemLoader
from ObservationFilter import ObservationFilter
#import ObservationAnalyzer
from ObservationAnalyzer import ObservationAnalyzer

from functools import lru_cache
import yaml

class ConcretePlanner:
    def __init__(self, connector, template_dir="prompts", output_base_dir=None,sop_enabled=False):
        self.connector = connector
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        self.logger = logging.getLogger(__name__)
        self.output_base_dir = output_base_dir
        self.sop_enabled = sop_enabled
        if self.output_base_dir:
            self.planner_log_dir = os.path.join(self.output_base_dir, "planner_logs")
            os.makedirs(self.planner_log_dir, exist_ok=True)

    @staticmethod # インスタンスに依存しないため staticmethod にし、外部でキャッシュ
    @lru_cache(maxsize=1)
    def _load_yaml_once(file_path):
        """ファイルを一度だけ読み込んで保持する（効率化）"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"❌ SOPファイル読み込みエラー: {e}")
        return {}
    
    def get_sop_content_for_planning(self,manifest):
        """
        マニフェストからSOPを抽出する。
        task_id のドット記法を考慮してマッチングを行う。
        """
        if not self.sop_enabled:
            return None

        sop_file = "dataset_for_sop_generation_updated.yaml"
        sop_database = self._load_yaml_once(sop_file)
        if not sop_database:
            return None

        # 1. マニフェストから task_id を取得
        full_task_id = manifest.get("task_metadata", {}).get("task_id", "")
        if not full_task_id:
            return None

        # 2. task_id の末尾部分だけを抽出 (例: 'workarena.servicenow.all-menu' -> 'all-menu')
        # ドットが含まれていない場合はそのままの名前を使用
        short_task_id = full_task_id.split('.')[-1] if '.' in full_task_id else full_task_id

        # 3. データベース（YAML）から短縮IDで検索
        task_entry = sop_database.get(short_task_id)
        
        # 見つからない場合の予備：全件ループして target_task_id フィールドをチェック
        if not task_entry:
            for entry in sop_database.values():
                if entry.get('target_task_id') == short_task_id or entry.get('target_task_id') == full_task_id:
                    task_entry = entry
                    break

        if not task_entry:
            print(f"⚠️ SOP matching failed for: {full_task_id} (searched as {short_task_id})")
            return None

        # 4. 階層構造を考慮して取得
        sop_content = task_entry.get("template_output_format", {}).get("sop_content") or task_entry.get("sop_content")

        # 有効なリストなら返す
        if isinstance(sop_content, list) and len(sop_content) > 0:
            return sop_content
            
        return None

    def _save_llm_trace(self, prompt, raw_response, mode="generate"):
        """
        日付_時分秒_マイクロ秒 をファイル名にしてプロンプトとレスポンスを対で保存する。
        """
        if not self.output_base_dir:
            return

        # ファイル名の衝突を避け、時系列順に並ぶようにフォーマット
        # 例: 20240520_142030_123456
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 入力(IN)の保存
        in_file = os.path.join(self.planner_log_dir, f"{timestamp}_{mode}_IN.txt")
        with open(in_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # 出力(OUT)の保存
        out_file = os.path.join(self.planner_log_dir, f"{timestamp}_{mode}_OUT.json")
        with open(out_file, "w", encoding="utf-8") as f:
            if isinstance(raw_response, (dict, list)):
                json.dump(raw_response, f, indent=4, ensure_ascii=False)
            else:
                f.write(str(raw_response))

    async def run_repair_loop(self, manifest, raw_df, page_title, focused_bid, 
                               refine_instruction=None, max_retries=3, lang="ja"):
        """
        自己修復機能を備えた具象プラン生成のメインエントリーポイント。
        """
        # 1. 物理Indexの作成 (全量データからロジックで検索)
        print("📍 Building Physical BID Index...")
        bid_index = self.build_bid_index(manifest, raw_df)

        # 2. LLMによるIndexの補完 (救済のため mode="Light" でコンテキストを維持)
        print("📍 Refining Physical BID Index with LLM...")
        bid_index = await self.refine_bid_index_with_llm(manifest, raw_df, bid_index)
        
        # 3. 初回プラン生成 (プランニング時は mode="Aggressive" でトークン節約)
        print("🚀 Generating Initial Concrete Plan...")
        current_steps, current_thought, error = await self.generate_concrete_plan(
            manifest=manifest,
            raw_df=raw_df,
            page_title=page_title,
            focused_bid=focused_bid,
            bid_index=bid_index,
            refine_instruction=refine_instruction,
            lang=lang
        )

        if error:
            return {"concrete_steps": [], "error": error}

        # 4. 自己修復（Repair）ループ
        for i in range(max_retries):
            print(f"⚖️ Validating Plan (Attempt {i+1})...")
            validation_errors = self.validate_plan({"concrete_steps": current_steps}, manifest, bid_index,raw_df)
            
            if not validation_errors:
                print("✅ Plan passed all logical validations!")
                return {
                    "concrete_steps": current_steps, 
                    "thought": current_thought, 
                    "bid_index": bid_index,
                    "attempts": i + 1
                }
            
            print(f"⚠️ Validation errors found: {len(validation_errors)}")
            if i == max_retries - 1:
                break

            print(f"🔄 Starting Repair Loop {i+1}...")
            # 修正時も refine_instruction (履歴や消込情報) を引き継ぐ
            current_steps, current_thought, error = await self.refine_concrete_plan(
                original_plan=current_steps,
                errors=validation_errors,
                manifest=manifest,
                raw_df=raw_df,
                page_title=page_title,
                focused_bid=focused_bid,
                bid_index=bid_index,
                refine_instruction=refine_instruction,
                lang=lang
            )
            if error:
                return {"concrete_steps": [], "error": error}

        return {
            "concrete_steps": current_steps, 
            "thought": current_thought, 
            "bid_index": bid_index,
            "validation_errors": validation_errors
        }


    def build_bid_index(self, manifest, raw_df):
            """
            全量データからスコアリングにより実体BIDを特定。
            BBoxから座標を抽出し、ミッションの意図(Intent)に応じて重要オブジェクトを動的に追加。
            """
            index_map = {}
            spec = manifest.get("specification", {})
            fields = spec.get("FIELDS", {}) if spec else {}
            
            # 1. ミッション・インテントの取得と正規化
            raw_intent = manifest.get("mission_intent") or \
                        manifest.get("task_metadata", {}).get("mission_intent", "")
            raw_intent = str(raw_intent).upper().strip()

            intent_mapping = {
                "FIND": "SEARCH_ANSWER", "QUERY": "SEARCH_ANSWER", "LOOK FOR": "SEARCH_ANSWER",
                "SEARCH": "SEARCH_ANSWER", "GET": "SEARCH_ANSWER",
                "MAKE": "CREATE", "NEW": "CREATE", "INSERT": "CREATE", "ADD": "CREATE",
                "EDIT": "UPDATE", "MODIFY": "UPDATE", "CHANGE": "UPDATE", "FIX": "UPDATE",
                "VIEW": "SHOW", "READ": "SHOW", "DISPLAY": "SHOW", "CHECK": "SHOW"
            }
            
            normalized_intent = intent_mapping.get(raw_intent, raw_intent)
            
            # 2. BBoxから座標情報を数値として抽出するヘルパー
            def get_bbox_coords(bbox_val):
                try:
                    if pd.isna(bbox_val) or bbox_val == "" or bbox_val == "None":
                        return 0, 0, 0, 0
                    import ast
                    coords = ast.literal_eval(bbox_val) if isinstance(bbox_val, str) else bbox_val
                    return coords[0], coords[1], coords[2], coords[3]
                except:
                    return 0, 0, 0, 0

            search_cols = ['Label_L', 'Label_A', 'Label_AX', 'InnerT', 'Label_P']

            # --- A. 指示にあるフィールドの特定 ---
            for field_key in fields.keys():
                norm_key = field_key.lower().replace("_", " ")
                mask = pd.Series(False, index=raw_df.index)
                for col in search_cols:
                    if col in raw_df.columns:
                        mask |= raw_df[col].astype(str).str.contains(norm_key, case=False, na=False)
                
                matches = raw_df[mask].copy()
                if not matches.empty:
                    def score_row(row):
                        score = 0
                        role = str(row.get('Role', '')).lower().strip()
                        tag = str(row.get('Tag', '')).lower().strip()
                        
                        INPUT_ROLES = {'input', 'textarea', 'select', 'combobox', 'checkbox', 'radio', 'searchbox', 'textbox'}
                        INPUT_TAGS = {'input', 'select', 'textarea', 'button'}
                        
                        if role in INPUT_ROLES or tag in INPUT_TAGS:
                            score += 150
                        if role in ['none', 'label', 'text'] or tag == 'label':
                            score -= 20
                        if float(row.get('Vis', 0)) > 0: score += 50
                        if norm_key == str(row.get('Label_L', '')).lower(): score += 20
                        return score

                    matches['score'] = matches.apply(score_row, axis=1)
                    best = matches.sort_values('score', ascending=False).iloc[0]
                    
                    if best['score'] >= 50:
                        index_map[field_key] = {
                            "bid": str(best['BID']),
                            "role": str(best.get('Role', '')),
                            "label": best['Label_L'] if pd.notna(best['Label_L']) and best['Label_L'] != "" else best['InnerT'],
                            "is_visible": float(best.get('Vis', 0)) > 0
                        }
                    else:
                        index_map[field_key] = "NOT_FOUND"
                else:
                    index_map[field_key] = "NOT_FOUND"

            # --- B. インテントに基づく「未指示オブジェクト」の動的追加 ---

            # 1. 検索系インテント (SEARCH_ANSWER, SHOW, ANALYZE)
            if normalized_intent in ["SEARCH_ANSWER", "SHOW", "ANALYZE"]:
                search_masks = (
                    raw_df['Label_P'].astype(str).str.contains('Search|minimum', case=False, na=False) |
                    raw_df['InnerT'].astype(str).str.contains('Search', case=False, na=False) |
                    raw_df['Role'].astype(str).str.contains('search', case=False, na=False)
                )
                searches = raw_df[search_masks & (raw_df['Vis'] > 0)].copy()
                
                if not searches.empty:
                    def score_search_priority(row):
                        s = 100
                        x, y, w, h = get_bbox_coords(row.get('BBox'))
                        if y > 150: s += 400 # ヘッダーより中央を優先
                        if w > 250: s += 150 # 広い入力欄を優先
                        if "minimum" in str(row['Label_P']).lower(): s += 300
                        return s
                    
                    searches['p_score'] = searches.apply(score_search_priority, axis=1)
                    best_search = searches.sort_values('p_score', ascending=False).iloc[0]
                    index_map["PRIMARY_SEARCH_INPUT"] = {
                        "bid": str(best_search['BID']),
                        "role": "searchbox",
                        "label": "Main Search Bar",
                        "is_visible": True,
                        "usage_hint": "Use this input for searching articles or records."
                    }

            # 2. アクション実行系インテント (CREATE, UPDATE, ORDER, ADMIN)
            if any(x in normalized_intent for x in ["CREATE", "UPDATE", "ORDER", "ADMIN", "INSERT"]):
                # ポジティブな実行キーワード
                exec_keywords = 'Submit|Save|Create|Insert|Order|Send|Confirm|Update'
                # ネガティブな参照キーワード (汎用的な除外ワード)
                info_keywords = 'View|Show|Detail|List|Linked|Account|Subscription|History|Log|Info|Help'
                
                btn_masks = (
                    raw_df['InnerT'].astype(str).str.contains(exec_keywords, case=False, na=False) |
                    raw_df['Label_A'].astype(str).str.contains(exec_keywords, case=False, na=False)
                )
                btns = raw_df[btn_masks & (raw_df['Vis'] > 0) & (raw_df['Tag'].isin(['BUTTON', 'A']))].copy()
                
                if not btns.empty:
                    def score_button_priority(row):
                        s = 100
                        x, y, w, h = get_bbox_coords(row.get('BBox'))
                        inner_t = str(row.get('InnerT', '')).lower()
                        cls = str(row.get('Class', '')).lower()
                        
                        # 🚀 汎用除外：参照系単語が含まれていたら実行ボタンではないと判断して大幅減点
                        if any(word.lower() in inner_t for word in info_keywords.split('|')):
                            s -= 1500
                        
                        # 🚀 構造的優先：右上エリア (y < 150 かつ x > 800) は「確定」の標準位置
                        if y < 150 and x > 800:
                            s += 1000
                        
                        # 🚀 クラスによる強調：多くのWebフレームワークでの「主要ボタン」
                        if any(c in cls for c in ['primary', 'success', 'action', 'submit', 'main']):
                            s += 500
                        
                        # 文字列の純度：ボタンテキストが実行ワードそのものなら加点
                        if any(word.lower() == inner_t.strip() for word in exec_keywords.split('|')):
                            s += 300
                            
                        return s
                        
                    btns['b_score'] = btns.apply(score_button_priority, axis=1)
                    best_btn = btns.sort_values('b_score', ascending=False).iloc[0]
                    
                    # スコアがプラス（除外ロジックを突破）した場合のみ採用
                    if best_btn['b_score'] > 0:
                        index_map["SUBMIT_BUTTON"] = {
                            "bid": str(best_btn['BID']),
                            "role": "button",
                            "label": str(best_btn['InnerT']).strip() or "Submit",
                            "is_visible": True,
                            "usage_hint": "Finalize your action by clicking this button."
                        }

            # デバッグ：動的に追加された後のindex_mapを確認
            #print(f"DEBUG: Normalized Intent: {normalized_intent}")
            #print(f"DEBUG: Final Index Map Keys: {list(index_map.keys())}")
            
            return index_map


    async def refine_bid_index_with_llm(self, manifest, raw_df, incomplete_index):
        """NOT_FOUND項目の救済。contextを残すため mode='Light' を使用。"""
        if "NOT_FOUND" not in json.dumps(incomplete_index): return incomplete_index
        filtered_df = ObservationFilter.apply(raw_df, mode="Light")
        
        prompt = f"""
あなたは高度なUI解析スペシャリストです。
以下のIndexで 'NOT_FOUND' となっている項目の正しいBIDをCSVから特定し、JSON形式で補完してください。
## 補完対象のIndex
{json.dumps(incomplete_index, indent=2, ensure_ascii=False)}
## UI Observation (CSV)
{filtered_df.to_csv(index=False)}
"""
        data, _, _, err = await self.connector.fetch_from_api(prompt, None)
        
        self._save_llm_trace(prompt, {"parsed_data": data, "error": err}, mode="refine_bid_index")
        
        if data and isinstance(data, dict):
            updated = incomplete_index.copy()
            for k, v in data.items():
                if k in updated: updated[k] = v
            return updated
        return incomplete_index

    async def generate_concrete_plan(self, manifest, raw_df, page_title, focused_bid, bid_index, refine_instruction=None, lang="ja"):
        try:
            filtered_df = ObservationFilter.apply(raw_df, mode="Aggressive")
            template = self.jinja_env.get_template(f"concrete_planner_{lang}.j2")
            prompt = template.render(
                manifest=manifest, page_title=page_title, focused_bid=focused_bid,
                observation_data_csv=filtered_df.to_csv(index=False),
                bid_index=json.dumps(bid_index, indent=2, ensure_ascii=False),
                refine_instruction=refine_instruction,sop_content=self.get_sop_content_for_planning(manifest)
            )

            data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
            self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="initial")
            
            if err:
                time.sleep(2)
                data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
                self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="initial")

            if err:
                time.sleep(2)
                data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
                self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="initial")
           
            if err: return None, None, err
            return data.get("concrete_steps", []), data.get("thought", reasoning), None
        except Exception: return None, None, traceback.format_exc()

    async def refine_concrete_plan(self, original_plan, errors, manifest, raw_df, page_title, focused_bid, bid_index, refine_instruction=None, lang="ja"):
        """エラー報告と外部指示（消込情報など）を統合して再送する。"""
        try:
            error_report = "\n".join([f"- {err}" for err in errors])
            base_instr = refine_instruction if refine_instruction else ""
            combined_instr = f"{base_instr}\n\n### 【最優先：修正指示】\n{error_report}\n\n※BID捏造禁止、タブ展開後は完遂すること。"
            
            filtered_df = ObservationFilter.apply(raw_df, mode="Aggressive")
            template = self.jinja_env.get_template(f"concrete_planner_{lang}.j2")
            prompt = template.render(
                manifest=manifest, page_title=page_title, focused_bid=focused_bid,
                observation_data_csv=filtered_df.to_csv(index=False),
                bid_index=json.dumps(bid_index, indent=2, ensure_ascii=False),
                refine_instruction=combined_instr,
                original_plan=json.dumps(original_plan, indent=2, ensure_ascii=False),
                sop_content=self.get_sop_content_for_planning(manifest)
            )
            data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
            self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="refine")

            if err:
                time.sleep(2)
                data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
                self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="refine")

            if err:
                time.sleep(2)
                data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
                self._save_llm_trace(prompt, {"parsed_data": data, "reasoning": reasoning, "error": err}, mode="refine")

            if err: return None, None, err
            return data.get("concrete_steps", []), data.get("thought", reasoning), None
        except Exception: return None, None, traceback.format_exc()

    def validate_plan(self, plan_data, manifest, bid_index, raw_df):
        """
        プランを検閲し、実体BIDの状態に基づき既入力項目の消込を行う。
        初期状態で正解のものは「入力済」として扱い、不要なアクションの削除を促す。
        """
        errors = []
        concrete_steps = plan_data.get("concrete_steps", [])
        VALID_ACTIONS = {"fill", "click","focus", "select_option", "scroll", "hover","extractLLM", "send_msg_to_user"}

        # 1. 【物理監査】初期状態で目標と一致しているものを「完了」と見なす
        completed_logic_refs = set()
        fields_spec = manifest.get("specification", {}).get("FIELDS", {})
        #POSITIVE_INDICATORS = {"YES", "TRUE", "ON", "CHECKED", "1", "SELECTED"}
        # 既存の物理的な属性値に加え、動的なUI状態を示すキーワードを追加
        POSITIVE_INDICATORS = {
            "YES", "TRUE", "ON", "CHECKED", "1", "SELECTED",  # 属性値（Value/Status用）
            "ACTIVE", "IS-CHECKED", "CHECKBOX-ACTIVE",        # ServiceNow/Angular等のクラス名用
            "RADIO-ACTIVE", "CHECKED-TRUE"                   # その他SPAで頻出する状態名
        }
        
        for field_label, target_val in fields_spec.items():
            target_val_orig = str(target_val).strip()
            # 🚨 修正：ここで continue せず、空文字指示の場合の判定を行う

            # bid_index から「実体BID」を直接参照
            field_info = bid_index.get(field_label)
            if not field_info or not isinstance(field_info, dict):
                continue 
            
            target_bid = str(field_info.get('bid'))
            target_row = raw_df[raw_df['BID'].astype(str) == target_bid]
            if target_row.empty: continue
            
            cand = target_row.iloc[0]
            actual_val = str(cand.get('Value', '')).strip()
            actual_inner = str(cand.get('InnerT', '')).strip()
            
            # 🚨 修正のキモ：空文字指示パターンの判定
            if target_val_orig == "":
                # 画面側も空、あるいは "-- None --" などの初期状態なら「完了」とみなす
                # これにより 'Service' 等が missing_fields に残る矛盾を防ぐ
                if actual_val == "" or actual_val.lower() in ["none", "-- none --", "null"]:
                    completed_logic_refs.add(field_label)
                continue

            # 真偽値（チェックボックス）判定
            is_target_bool = target_val_orig.lower() in ["true", "false"]
            if is_target_bool:
                expected_bool = target_val_orig.lower() == "true"
                
                val_u = str(cand.get('Value', '')).upper().strip()
                stat_u = str(cand.get('Status', '')).upper().strip()
                cls_u = str(cand.get('Class', '')).upper().strip()
                
                # 実体側の属性から現在のチェック状態を判定
                current_checked = any(ind in val_u for ind in POSITIVE_INDICATORS) or \
                                any(ind in stat_u for ind in POSITIVE_INDICATORS) or \
                                ("CHECKED" in cls_u)
                
                if current_checked == expected_bool:
                    completed_logic_refs.add(field_label)
            else:
                # テキスト/選択判定
                if target_val_orig == actual_val or target_val_orig == actual_inner:
                    completed_logic_refs.add(field_label)

        # 2. 進捗状況の整理
        planned_logic_refs = {a.get("logic_ref") for s in concrete_steps for a in s.get("actions", []) if a.get("logic_ref")}
        required_fields = set(fields_spec.keys())
        
        # 🚨 ここで completed_logic_refs に Service が含まれるため、引き算の結果
        # missing_fields から Service が消え、矛盾が解消される
        missing_fields = (required_fields - completed_logic_refs) - planned_logic_refs
        
        # 3. エラー報告（未入力の指摘）
        if missing_fields:
            truly_missing = [f for f in missing_fields if bid_index.get(f) and bid_index.get(f) != "NOT_FOUND"]
            #if truly_missing and not any(a.get("action_type") == "send_msg_to_user" for s in concrete_steps for a in s.get("actions", [])):
            #    errors.append(f"【未完了】以下の項目は設定が必要です（プランに含まれていません）: {truly_missing}")
            if truly_missing and not any(a.get("action_type") in ["send_msg_to_user", "extractLLM"] for s in concrete_steps for a in s.get("actions", [])):
                #errors.append(f"【未完了】以下の項目は設定が必要です: {truly_missing}") 
                # 複数の場合は「いずれか、または関連する親」を logic_ref に入れるよう促す
                errors.append(
                    f"【未完了】以下の項目がプランに含まれていません: {truly_missing}。 "
                    f"もし現在の操作がこれらの項目を表示させるための準備（カテゴリ展開、タブのクリック等）である場合は、"
                    f"そのアクションの 'logic_ref' に、最も関連の深い項目名（例: '{truly_missing}'）を設定してください。 "
                    f"これにより、その項目への到達意思がシステムに承認されます。"
                )

        # 4. アクション詳細検閲（不要操作の排除）
        for step in concrete_steps:
            step_id = step.get('step_id', 'Unknown')
            for action in step.get("actions", []):
                a_type = action.get("action_type")
                bid = str(action.get("bid", ""))
                logic_ref = action.get("logic_ref")


                # 🚨 未知のアクションチェック
                if a_type not in VALID_ACTIONS:
                    errors.append(f"【アクション不正】{step_id}: 未定義のアクション '{a_type}' です。")
                    continue

                # 🚨 extractLLM 専用のバリデーション
                if a_type == "extractLLM":
                    if not action.get("instruction"):
                        errors.append(f"【引数不足】{step_id}: extractLLM には 'instruction' が必須です。")
                    continue # BIDチェック等は不要なのでスキップ

                # 🚨 send_msg_to_user のスキップ
                if a_type == "send_msg_to_user":
                    continue

                # 🚨 scroll 専用のバリデーション（追加）
                if a_type == "scroll":
                    # dx, dy が数値で存在するか、または direction があるかを確認
                    # 数値指定の scroll(0, 500) 形式を許容する
                    if 'dx' not in action and 'dy' not in action and 'direction' not in action:
                        errors.append(f"【引数不足】{step_id}: scroll には 'dx/dy' または 'direction' が必要です。")
                    continue # BIDチェック（target_row等）は不要なのでスキップ

                target_row = raw_df[raw_df['BID'].astype(str) == bid]
                if not target_row.empty:
                    role = str(target_row.iloc[0].get('Role', '')).lower()
                    tag = str(target_row.iloc[0].get('Tag', '')).upper()

                    # ラベル操作の禁止
                    #if role in ["none", "label"] or tag == "LABEL":
                    #    correct_bid = bid_index.get(logic_ref, {}).get('bid')
                    #    errors.append(f"【ロール不正】{step_id}: ラベル '{bid}' ではなく実体 '{correct_bid}' を操作してください。")


                # --- 修正：実体タグ(INPUT/TEXTAREA/SELECT)であればラベル判定をスキップ ---
                    is_interactive_tag = tag in ["INPUT", "TEXTAREA", "SELECT", "A", "BUTTON"]
                    
                    if (role in ["none", "label"] or tag == "LABEL") and not is_interactive_tag:
                                    
                    # 修正：断定的なメッセージをやめ、再調査を促す
                    #if role in ["none", "label"] or tag == "LABEL":
                        correct_bid_suggestion = bid_index.get(logic_ref, {}).get('bid')
                        errors.append(
                            f"【操作要素の再確認】{step_id}: BID '{bid}' はラベル(LABEL)です。 "
                            f"CSVから、この近傍にある Role: radio や 'input-group-radio' クラスを持つ要素（例: SPAN等）を探して操作してください。 "
                            f"物理Indexの参照値 '{correct_bid_suggestion}' は見出しの可能性があります。"
                        )

                # 完了済み項目へのクリックは「プランからの削除」を命じる
                if a_type == "click" and logic_ref in completed_logic_refs:
                    if str(fields_spec.get(logic_ref, "")).lower() in ["true", "false"]:
                        errors.append(f"【不要操作：削除指示】{step_id}: {logic_ref} は既に正しい状態です。プランからこのアクションを削除してください。")

        return errors

import asyncio
import json
import pandas as pd
import yaml
from ConcretePlanner import ConcretePlanner
from GeminiConnector import GeminiConnector

async def setup_planner():
    """プランナーとコネクタの初期設定"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    connector = GeminiConnector(api_key=config["gemini_api_key"])
    base_path = r"C:\Users\user\Desktop\dev\agent\task_execution_data_full_logs\create-change-request"

    planner = ConcretePlanner(connector,output_base_dir=base_path)
    return planner

def load_test_data():
    """マニフェストと観測データのロード"""
    with open("logs/manifest_create-change-request_ja.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    path = r"C:\Users\user\Desktop\dev\agent\task_execution_data_full_logs\create-change-request\step_1_state_metadata.json"

    #raw_df = pd.read_csv("create-change-request_step1.csv")
    print(f"🔍 Analyzing Observation: {path}")
    obs_analyzer = ObservationAnalyzer(path)
    raw_df = obs_analyzer.analyze()

    page_title = obs_analyzer.page_title
    focused_bid = obs_analyzer.focused_bid
    return manifest, raw_df,page_title,focused_bid

# ==========================================================
# テストケース 1: 初回プランニング (新規フォーム入力)
# ==========================================================
async def test_initial_planning(planner, manifest, raw_df,page_title,focused_bid):
    print("\n" + "="*50)
    print("TEST CASE 1: Initial Planning")
    print("="*50)
    
    # 初回なので指示（履歴）はなし
    result = await planner.run_repair_loop(
        manifest=manifest,
        raw_df=raw_df,
        page_title=page_title,
        focused_bid=focused_bid
    )
    
    print("\n" + "="*50)
    print("📝 FINAL CONCRETE STEPS")
    print("="*50)
    print(f"💡 Thought: {result['thought']}")
    print(f"✅ Steps generated: {len(result['concrete_steps'])}")
    print(json.dumps(result['concrete_steps'], indent=2, ensure_ascii=False))
    
    # 出力確認（一部）
    #for step in result['concrete_steps']:
    #    print(f" - {step['step_id']}: {step['logical_intent']}")

# ==========================================================
# テストケース 2: 途中からのリプラン (消込ロジックの検証)
# ==========================================================
async def test_mid_task_replanning(planner, manifest, raw_df,page_title,focused_bid):
    print("\n" + "="*50)
    print("TEST CASE 2: Mid-task Replanning (with Progress)")
    print("="*50)
    
    # 1. 物理監査のシミュレーション
    # 'Number' と 'Short description' は既に入力済み（Valueが埋まっている）と仮定
    completed_fields = ["Number", "Short description"]
    
    # 物理的な整合性を保つため、ダミーの書き換え（実際の運用では最新のCSVを読み込むだけ）
    # build_bid_indexを使って対象BIDを特定し、値をセット
    bid_index = planner.build_bid_index(manifest, raw_df)
    for field in completed_fields:
        info = bid_index.get(field)
        if isinstance(info, dict):
            raw_df.loc[raw_df['BID'] == info['bid'], 'Value'] = manifest["specification"]["FIELDS"][field]

    # 2. 消込情報を指示文に反映
    refine_instr = f"""
### 現在の進捗状況（消込済み）
以下の項目はシステム上で入力済みであることを確認しました。これらの再入力は不要です：
{completed_fields}

### 今回のミッション
まだ入力されていない残りの項目を処理し、必要であれば Closure Information タブを展開してください。
"""

    # 3. リプランの実行
    result = await planner.run_repair_loop(
        manifest=manifest,
        raw_df=raw_df,
        page_title=page_title,
        focused_bid=focused_bid, # タブ付近にいると仮定
        refine_instruction=refine_instr
    )
    
    print("\n" + "="*50)
    print("📝 FINAL CONCRETE STEPS")
    print("="*50)
    print(f"💡 Thought: {result['thought']}")
    print(f"✅ Steps generated: {len(result['concrete_steps'])}")
    print(json.dumps(result['concrete_steps'], indent=2, ensure_ascii=False))


    # 完了済みの項目がアクションに含まれていないか検証
    planned_refs = [action['logic_ref'] for step in result['concrete_steps'] for action in step['actions']]
    print(f"Planned fields in this run: {planned_refs}")
    
    if "Number" not in planned_refs and "Short description" not in planned_refs:
        print("✅ Success: Completed fields were correctly excluded from the plan.")
    else:
        print("⚠️ Warning: Some completed fields are still in the plan.")

# ==========================================================
# メイン実行部
# ==========================================================
async def main():
    try:
        planner = await setup_planner()
        manifest, raw_df,page_title,focused_bid = load_test_data()

        # 各テストケースの実行
        await test_initial_planning(planner, manifest, raw_df,page_title,focused_bid)
        #await test_mid_task_replanning(planner, manifest, raw_df,page_title,focused_bid)

    except Exception:
        import traceback
        print(f"❌ Test Script Error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
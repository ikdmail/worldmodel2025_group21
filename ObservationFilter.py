import pandas as pd

class ObservationFilter:
    """
    ObservationAnalyzerが生成した25列のDataFrameを受け取り、
    プランニングとアクションに最適な形に要素を削減する独立モジュール。
    ServiceNow等のSPAにおける「非表示だが重要な入力要素」を保護する。
    """
    
    @staticmethod
    def apply(df, mode="Light"):
        if df.empty or mode == "OFF":
            return df

        # --- 1. 必須維持条件 (Critical Elements) ---
        critical_mask = (
            (df['Focused'] == 'YES') | 
            (df['Value'].fillna('').astype(str).str.len() > 0) |
            (df['Hover_Strat'] == 'Investigate') |
            (df['HasPop'] == 'YES')
        )

        # --- 2. インタラクティブ要素の判定強化 ---
        # Roleだけでなく、Tagレベルで入力・選択・操作系を定義
        interactive_roles = [
            'button', 'link', 'textbox', 'combobox', 'checkbox', 
            'radio', 'menuitem', 'searchbox', 'listbox', 'tab', 
            'treeitem', 'option', 'LabelText'
        ]
        
        # Tagによる直接判定（ServiceNowの隠れ要素対策）
        # これによりLabel_Lが空のSELECTやTEXTAREAも保護される
        input_tags = ['input', 'select', 'textarea', 'button', 'a', 'label']
        is_input_tag = df['Tag'].fillna('').astype(str).str.lower().isin(input_tags)

        is_interactive = (
            df['Role'].isin(interactive_roles) | 
            (df['JS'] == 'YES') | 
            is_input_tag
        )
        
        # --- 3. テキスト・ラベル判定 ---
        inner_t_clean = df['InnerT'].fillna('').astype(str).str.strip()
        has_text = inner_t_clean.str.len() > 0
        
        # ラベルとしての価値 (RoleがStaticTextでなくても、一定の長さがあれば保持候補)
        # ServiceNowのセクション名やタブ名などを拾うため条件を調整
        is_label = (
            ((df['Role'] == 'StaticText') | (df['Tag'] == 'SPAN')) & 
            (inner_t_clean.str.len() > 1) & 
            (inner_t_clean.str.len() < 100)
        )

        # --- 4. フィルタリング実行 ---
        if mode == "Light":
            # 基本的に何か意味のある要素（テキストあり、または操作可能）をすべて残す
            filtered_df = df[critical_mask | is_interactive | has_text]
        elif mode == "Aggressive":
            # 操作可能要素と、識別に必要なラベルのみに絞り込む（トークン節約重視）
            filtered_df = df[critical_mask | is_interactive | is_label]
        else:
            filtered_df = df

        return filtered_df.reset_index(drop=True)

    @staticmethod
    def get_stats(before_df, after_df):
        before_cnt, after_cnt = len(before_df), len(after_df)
        reduction = ((before_cnt - after_cnt) / before_cnt * 100) if before_cnt > 0 else 0
        return {
            "before": before_cnt, 
            "after": after_cnt, 
            "reduction_rate": f"{reduction:.1f}%",
            "areas_preserved": f"{after_df['Area'].nunique()}/{before_df['Area'].nunique()}"
        }
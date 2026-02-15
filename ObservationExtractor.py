import json
import logging
import os
import datetime
import pandas as pd
from ObservationFilter import ObservationFilter

class ObservationExtractor:
    def __init__(self, connector, output_base_dir=None):
        """
        抽出特化型コンポーネント。
        Plannerとは独立させ、UIからの情報取得（Read）のみを担当する。
        """
        self.connector = connector
        self.logger = logging.getLogger(__name__)
        self.output_base_dir = output_base_dir
        if self.output_base_dir:
            # 知覚専用のログディレクトリを生成
            self.extractor_log_dir = os.path.join(self.output_base_dir, "extractor_logs")
            os.makedirs(self.extractor_log_dir, exist_ok=True)

    def _save_llm_trace(self, prompt, raw_response, mode="extract"):
        """
        プロンプト(IN)とレスポンス(OUT)を対で保存する。
        タイムスタンプにより実行順序を保持する。
        """
        if not self.output_base_dir:
            return

        # ファイル名用のタイムスタンプ (例: 20260121_123456_789012)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 入力(IN)の保存
        in_file = os.path.join(self.extractor_log_dir, f"{timestamp}_{mode}_IN.txt")
        with open(in_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # 出力(OUT)の保存
        out_file = os.path.join(self.extractor_log_dir, f"{timestamp}_{mode}_OUT.json")
        with open(out_file, "w", encoding="utf-8") as f:
            if isinstance(raw_response, (dict, list)):
                json.dump(raw_response, f, indent=4, ensure_ascii=False)
            else:
                f.write(str(raw_response))

    async def extract(self, instruction, raw_df, page_title):
        """
        UI Observationから情報を「無加工」で抽出するメインメソッド。
        """
        # 知覚精度を上げるため、フィルタリングは"Light"（少し多めに情報を残す）を推奨
        filtered_df = ObservationFilter.apply(raw_df, mode="Light")
        
        # プロンプトの構築：無加工返却を厳格に指示
        prompt = f"""
### UI Perception Mission
あなたはUIから正確な情報を読み取るスペシャリストです。
現在のページ: {page_title}

【抽出指示】
{instruction}

以下のUI Observation (CSV) を解析し、指示に合致する「値」を特定してください。

## UI Observation (CSV)
{filtered_df.to_csv(index=False)}

## ⚠️厳守ルール
1. 抽出した値は、一切の加工（要約、丁寧語への変換、解釈）をせず「元のまま」返却してください。
2. 回答には抽出した値のみを含めてください。
3. 「抽出結果は〜」「以下が回答です」といった前置きや、解説は一切禁止します。
4. UI上に該当する値が存在しない場合は "NOT_FOUND" とのみ回答してください。
"""
        # LLMへの問い合わせ
        data, reasoning, _, err = await self.connector.fetch_from_api(prompt, None)
        
        # 実行トレースの保存
        trace_data = {
            "instruction": instruction,
            "page_title": page_title,
            "extracted_data": data,
            "reasoning": reasoning,
            "error": err
        }
        self._save_llm_trace(prompt, trace_data, mode="extractLLM")

        if err:
            self.logger.error(f"Perception Error: {err}")
            return f"ERROR: {err}"
            
        # --- 戻り値の正規化ロジックの修正 ---
        # 1. 辞書で返ってきた場合（JSONパース済み）
        if isinstance(data, dict):
            # もしLLMが {"result": "val"} のように返してきた場合、
            # キーが何であれ、中身が一つならその値を取り出す。
            # 複数ある場合は、加工を避けるためJSON文字列として丸ごと返す。
            if len(data) == 1:
                raw_val = str(list(data.values())[0])
            else:
                raw_val = json.dumps(data, ensure_ascii=False)
        
        # 2. リストで返ってきた場合
        elif isinstance(data, list):
            raw_val = json.dumps(data, ensure_ascii=False)
            
        # 3. 文字列や数値の場合
        else:
            raw_val = str(data)

        # 前後の引用符や空白だけを最小限に掃除し、中身は「元のまま」返却
        return raw_val.strip().strip('"').strip("'")
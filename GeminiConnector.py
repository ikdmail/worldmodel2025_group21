import re
import json
import traceback
from google import genai
from pydantic import ValidationError

class GeminiConnector:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        # 最新 SDK: Client インスタンスを生成
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def fetch_from_api(self, prompt, schema_class):
        """
        最新の google-genai SDK を使用して、高速にレスポンスを取得する。
        """
        raw_content = None
        try:
            # 最新の generate_content 呼び出し
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json", # JSONモード
                    temperature=0.1,
                ),
            )
            
            # レスポンスの取得
            raw_content = response.text

            # 1. リーズニングの抽出ロジック
            # <think>タグがある場合、それを抜き出して reasoning に格納し、本体から除去する
            reasoning, clean_text = self._extract_reasoning(raw_content)
            
            # JSON 抽出と Pydantic バリデーション
            json_str = self._extract_json(clean_text)
            #validated_data = schema_class.parse_raw(json_str)

            # --- ここを修正 ---
            if schema_class:
                # schema_class (Pydanticモデル) が指定されている場合
                validated_data = schema_class.parse_raw(json_str)
            else:
                # schema_class が None の場合、辞書として返す
                validated_data = json.loads(json_str)
            
            # reasoning は必要に応じて output から分離（現状は text に集約）
            return validated_data, reasoning, raw_content, None

        except Exception:
            # 詳細なトレースバックをキャッチ
            error_detail = traceback.format_exc()
            return None, None, raw_content, error_detail

    def _extract_json(self, text):
        if not text:
            return ""
        # Markdown の装飾をガードして JSON のみを抜き出す
        #match = re.search(r'\{.*\}', text, re.DOTALL)
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        return match.group(0) if match else text
    
    def _extract_reasoning(self, text):
        """<think>タグ等から思考プロセスを分離する"""
        if not text:
            return None, ""
        
        # <think>...</think> の中身を探す
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            # 思考部分を除去したテキストを返す
            clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return reasoning, clean_text
        
        return None, text
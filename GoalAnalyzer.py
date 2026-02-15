import json
import os
import traceback
from jinja2 import Environment, FileSystemLoader



import asyncio
import yaml
import json
import os
import traceback
# 外部ファイルとして保存している前提
from GeminiConnector import GeminiConnector
#from GoalAnalyzer import GoalAnalyzer

class GoalAnalyzer:
    def __init__(self, connector, template_dir="prompts", rules_dir="rules"):
        self.connector = connector
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        self.rules_dir = rules_dir

    def _load_intent_rules(self, intent: str, lang: str) -> dict:
        rule_file = os.path.join(self.rules_dir, f"intent_rules_{lang}.json")
        if not os.path.exists(rule_file):
            return {}
        try:
            with open(rule_file, "r", encoding="utf-8") as f:
                all_rules = json.load(f)
            return all_rules.get(intent, {})
        except Exception:
            # ルールロード失敗時もトレースを出しておくと安全
            print(f"--- Rule Loading Error ---\n{traceback.format_exc()}")
            return {}

    async def analyze(self, raw_goal: str, task_id: str, lang: str = "ja"):
        try:
            # 1. プロンプト生成
            template_name = f"goal_analyzer_{lang}.j2"
            template = self.jinja_env.get_template(template_name)
            prompt = template.render(goal=raw_goal, task_id=task_id)

            # 2. LLM実行
            validated_data, reasoning, raw_content, error = await self.connector.fetch_from_api(prompt, None)

            if error:
                return None, error

            # 3. ルール注入
            intent = validated_data.get("MISSION_INTENT", "GENERAL")
            runtime_hints = self._load_intent_rules(intent, lang)

            # 4. マニフェスト構成
            manifest = {
                "task_metadata": {
                    "task_id": task_id,
                    "mission_intent": intent,
                    "language": lang,
                    "raw_goal": raw_goal
                },
                "abstract_plan": validated_data.get("ABSTRACT_PLAN", []),
                "specification": validated_data.get("SPECIFICATION", {}),
                "success_criteria": validated_data.get("SUCCESS_CRITERIA", {}),
                "runtime_hints": runtime_hints,
                "reasoning": reasoning
            }
            return manifest, None

        except Exception:
            return None, traceback.format_exc()
        



async def main():
    try:
        # 1. 初期設定
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        log_dir = config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        lang = config.get("language", "ja")

        connector = GeminiConnector(api_key=config["gemini_api_key"])
        analyzer = GoalAnalyzer(connector)

        # 2. 入力
        task_id = "create-change-request"
        #raw_goal = "ServiceNowでWindowsサーバーの再起動申請をRisk:Moderateで作成してください。"
        # トリプルクォート（"""）で囲む
        raw_goal = """
        Create a new change request with a value of "Moderate" for field "Risk", a value of "Successful" for field "Close
        code", a value of "3 - Low" for field "Impact", a value of "camptonite cimbia inoffensiveness outthruster
        impossibilist" for field "Risk and impact analysis", a value of "Service Desk" for field "Assignment group", a value
        of "Reboot Windows Server" for field "Short description", a value of "Ensure that the monitoring console is all
        green for this server before closing change request" for field "Test plan", a value of "--In the server monitoring
        console set the server to "Maintenance mode" to prevent alerts being sent--If the server has network attached
        storage (NFS or iSCSI) stop any file server applications and unmount the storage--Reboot the server4. Check
        the Windows system log for any error messages--In the server monitoring console set the server back to
        "Operational" mode" for field "Implementation plan", a value of "" for field "Service", a value of "" for field
        "Configuration item", a value of "Server rebooted" for field "Close notes", a value of "CHG0000028" for field
        "Number", a value of "There is no backout plan for this change template" for field "Backout plan", and a value of
        "subhexagonal ghoul twinned phallic chariotry" for field "Justification".
        """

        print(f"--- Starting Analysis (Lang: {lang}) ---")

        # 3. 解析
        manifest, error = await analyzer.analyze(raw_goal, task_id, lang=lang)

        # 4. 結果出力
        if error:
            print(f"--- Analysis Error Occurred ---\n{error}")
            with open(os.path.join(log_dir, "error_log.txt"), "w") as f:
                f.write(error)
            return

        output_path = os.path.join(log_dir, f"manifest_{task_id}_{lang}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"Success! Manifest: {output_path}")

    except Exception:
        print(f"--- Unexpected Main Error ---\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
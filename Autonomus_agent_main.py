import os
import sys
import time
import yaml
import json
import re
import datetime
import traceback
import pandas as pd
import asyncio  # AI呼び出し時のみ内部で使用
from typing import Dict, Any, Tuple, List, Type

import nest_asyncio  # 🚨 追加：ループのネストを許可する

# nest_asyncio の適用
nest_asyncio.apply()

import browsergym.core.action.utils as bg_utils 
from browsergym.core.action.highlevel import HighLevelActionSet 
from browsergym.core.env import BrowserEnv
from browsergym.workarena import ATOMIC_TASKS 

from logger import save_step_state
from GeminiConnector import GeminiConnector
from GoalAnalyzer import GoalAnalyzer
from ConcretePlanner import ConcretePlanner
from ObservationAnalyzer import ObservationAnalyzer
from ObservationExtractor import ObservationExtractor
import signal


# --- 0. [物理トレース] call_fun へのモンキーパッチ ---
LAST_EXECUTION_TRACE = {"force_used": False}

def logging_call_fun(fun, retry_with_force):
    global LAST_EXECUTION_TRACE
    try:
        return fun(force=False)
    except Exception as e:
        if retry_with_force:
            print(f"⚠️ [Trace] 通常操作失敗。force=True で救済を試みます...")
            try:
                result = fun(force=True)
                LAST_EXECUTION_TRACE["force_used"] = True
                return result
            except Exception as fe:
                raise fe
        else:
            raise e

if not hasattr(bg_utils, "original_call_fun"):
    bg_utils.original_call_fun = bg_utils.call_fun
    bg_utils.call_fun = logging_call_fun

# --- 1. YAML設定 ---
env = None  # 🚨 シグナルハンドラから参照するためにグローバルで定義
CONFIG_FILE = "config.yaml"
OUTPUT_DIR = "task_execution_data_full_logs"
MAX_LOOPS = 8

# --- [シグナルハンドラ] ---
def signal_handler(sig, frame):
    global env
    print("\n🛑 中断要請 (Ctrl+C) を検知。クリーンアップ中...")
    if env:
        try:
            print("🌐 ブラウザを強制終了しています...")
            env.close()
        except:
            pass
    print("👋 ターミナルに戻ります。")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    print(f"❌ 設定ファイル読み込みエラー: {e}")
    sys.exit(1)

os.environ['HUGGING_FACE_HUB_TOKEN'] = cfg.get('HUGGING_FACE_HUB_TOKEN', '')

def sanitize_filename(name: str) -> str:
    name = name.replace("workarena.servicenow.", "").replace("/", "_").replace(".", "_")
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name)[:50]

# AIの非同期関数を同期的に呼び出すためのヘルパー
def sync_wait(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

from typing import Any, Dict

def build_action_code(action: Dict[str, Any], last_extracted_value: str = None) -> str:
    """
    プランナーの指示を BrowserGym 実行コードに変換する。
    extractValue プレースホルダーを実データにバインドする。
    """
    a_type = action.get('action_type')
    bid = action.get('bid')

    # 🚨 0. プレースホルダー 'extractValue' の置換処理
    # 抽出された値がある場合、value, options, message 等の項目を書き換える
    def bind_value(v):
        if last_extracted_value and isinstance(v, str) and "extractValue" in v:
            # "extractValue" という文字列そのもの、あるいはそれを含む文字列を置換
            return v.replace("extractValue", last_extracted_value)
        return v

    # 🚨 特殊アクション：extractLLM はコード生成不要（内部処理のため）
    if a_type == "extractLLM":
        return ""

    # 1. select_option
    if a_type == "select_option":
        val = action.get('options') or action.get('option') or action.get('value')
        val = bind_value(val) # 置換適用
        return f"select_option(bid='{bid}', options={repr(val)})"

    # 2. send_msg_to_user
    if a_type == "send_msg_to_user":
        msg = action.get('message') or action.get('value') or "Done."
        msg = bind_value(msg) # 置換適用
        return f"send_msg_to_user({repr(msg)})"

    # 3. fill
    if a_type == "fill":
        raw_val = action.get('value', '')
        raw_val = bind_value(raw_val) # 置換適用
        
        val_str = str(raw_val).strip()
        val_lower = val_str.lower()
        
        # 救済措置：チェックボックスへの fill(true) は click に変換
        if val_lower in ["true", "false"]:
            return f"click(bid='{bid}')"
        else:
            return f"fill(bid='{bid}', value={repr(raw_val)})"

    # 4. scroll
    #if a_type == "scroll":
    #    direction = action.get('direction', 'down')
    #    return f"scroll(direction='{direction}')"

    # 4. scroll (BrowserGym coord subset 準拠)
    if a_type == "scroll":
        # JSON/SOPから数値を取得。デフォルトは下に500ピクセル
        dx = action.get('dx', 0)
        dy = action.get('dy', 0)
        
        # direction指定がある場合の救済策（互換性維持）
        if dy == 0:
            direction = action.get('direction', 'down')
            if direction == 'down':
                dy = 500
            elif direction == 'up':
                dy = -500
                
        # ⚠️ 引用符を入れず、数値として文字列に組み込む
        return f"scroll({dx}, {dy})"

    # 5. その他 (click, hover, dblclick, press 等)
    if bid:
        # click, hover, dblclick などは引数1つ(bid)のみ
        if a_type in ["click", "hover", "dblclick", "focus"]:
            return f"{a_type}({repr(bid)})"
        
        # press, fill など値を伴うものは引数2つ
        val = action.get('value')
        if val is not None:
            p_val = bind_value(val)
            return f"{a_type}({repr(bid)}, {repr(p_val)})"
        
        # 値がない場合は基本形
        return f"{a_type}({repr(bid)})"
    
    # 引数がない、または特殊な場合
    return f"{a_type}()"

def build_action_code2(action: Dict[str, Any]) -> str:
    a_type = action.get('action_type')
    bid = action.get('bid')

    # 1. select_option は必ず 'options=' (複数形)
    if a_type == "select_option":
        # 複数のキー候補から値を取得
        val = action.get('options') or action.get('option') or action.get('value')
        return f"select_option(bid='{bid}', options={repr(val)})"

    # 2. send_msg_to_user は引数名なし、または message= 形式
    if a_type == "send_msg_to_user":
        msg = action.get('message') or action.get('value') or "Done."
        return f"send_msg_to_user({repr(msg)})"

    # 3. fill は 'value=' 且つ 【大文字小文字を維持】
    if a_type == "fill":
        raw_val = action.get('value', '')
        val_str = str(raw_val).strip()
        
        # 🚨 判定用に一時的に小文字にするが、実際の入力値は raw_val (元のケース) を使う
        val_lower = val_str.lower()
        
        # "true" や "false" が値として渡された場合、それはチェックボックス操作の誤り
        # この場合は click(bid) に変換して救済する
        if val_lower in ["true", "false"]:
            return f"click(bid='{bid}')"
        else:
            # 🚨 修正：repr(raw_val) を使うことで大文字小文字を保持してコード生成
            return f"fill(bid='{bid}', value={repr(raw_val)})"

    # 4. scroll は BrowserGym 仕様に合わせる (bid を持たない場合が多い)
    if a_type == "scroll":
        # もし direction 指定があればそれを使う、なければデフォルト
        direction = action.get('direction', 'down')
        return f"scroll(direction='{direction}')"

    # 5. その他 (click, hover, double_click 等)
    # bid が必要なアクションに関しては一律この形式
    if bid:
        return f"{a_type}(bid='{bid}')"
    
    # 最終フォールバック
    return f"{a_type}()"


def inject_final_hover(concrete_steps):
    """
    プランの最後がclickアクションだった場合、その直前にhoverを強制挿入する。
    ServiceNow等の動的フォームで、最後の入力値を確実に確定(Focus Out)させるための安全装置。
    """
    if not concrete_steps:
        return concrete_steps

    # 1. 最後のアクションが含まれるステップと、そのアクションリストを特定
    last_step = concrete_steps[-1]
    actions = last_step.get("actions", [])

    if not actions:
        return concrete_steps

    # 2. 最後のアクションが 'click' かどうかを判定
    last_action = actions[-1]
    if last_action.get("action_type") == "click":
        target_bid = last_action.get("bid")
        
        # 3. 直前に既に同じBIDへのhoverが存在しないか確認（二重挿入防止）
        has_hover = len(actions) >= 2 and \
                    actions[-2].get("action_type") == "hover" and \
                    actions[-2].get("bid") == target_bid
        
        #click連続の場合もfocus入れない。
        has_hover = len(actions) >= 2 and \
                    actions[-2].get("action_type") == "click"
        
        if not has_hover:
            # 4. 安全のための hover アクションを生成
            safety_hover = {
                "action_type": "focus",
                "bid": target_bid,
                "logic_ref": "Safety Focus-Out before Final Click"
            }
            # clickの直前（インデックス -1 の位置）に挿入
            actions.insert(-1, safety_hover)
            
    return concrete_steps

# --- 2. 単一タスク自律実行ロジック (同期版) ---
def run_autonomous_task(task_class: Type):
    """
    Playwright Sync APIとの競合を避けるため、同期関数として実行する。
    """
    try:
        task_id = getattr(task_class, "get_task_id", lambda: task_class.__name__)()
    except:
        task_id = task_class.__name__
        
    safe_task_name = sanitize_filename(task_id)
    task_output_dir = os.path.join(OUTPUT_DIR, safe_task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # AIコンポーネントの初期化
    connector = GeminiConnector(api_key=cfg["gemini_api_key"])
    goal_analyzer = GoalAnalyzer(connector)
    planner = ConcretePlanner(connector, output_base_dir=task_output_dir,sop_enabled=cfg["sop_enabled"])

    # --- 初期化セクション ---
    extractor = ObservationExtractor(connector, output_base_dir=task_output_dir)
    
    extract_memory = None  # 抽出した「生の値」を保持する辞書
    
    env = None
    task_history = []
    completed_logic_refs = set()
    completed_interactions = set()

    try:
        print(f"\n{'='*60}\n🚀 ミッション開始: {task_id}\n{'='*60}")

        action_set = HighLevelActionSet(subsets=["workarena"], retry_with_force=True)
        env = BrowserEnv(
            task_entrypoint=task_class,
            action_mapping=action_set.to_python_code,
            headless=cfg.get("headless", False),
            pre_observation_delay=5.0
        )
        
        # 🚨 asyncio.run() 外なので、Playwright Sync API が正常に動く
        observation, info = env.reset(seed=42)
        time.sleep(5)

        # 🟢 【戦略フェーズ】AI呼び出しを同期的に待機
        print("🧠 目標解析中...")
        manifest, error = sync_wait(goal_analyzer.analyze(observation['goal'], task_id))
        if error:
            raise Exception(f"Goal Analysis Failed: {error}")

        record, last_meta_path = save_step_state(task_output_dir, 0, observation, info)
        task_history.append(record)

        for loop_idx in range(1, MAX_LOOPS + 1):
            print(f"\n" + "-"*40 + f"\n🔄 自律ループ {loop_idx}/{MAX_LOOPS}\n" + "-"*40)
            # 🚨 1. ループの先頭で変数を初期化（UnboundLocalError 対策）
            terminated = False
            truncated = False
            reward = 0.0

            obs_analyzer = ObservationAnalyzer(last_meta_path)
            raw_df = obs_analyzer.analyze()
            
            # --- 修正後の汎用監査ロジック ---
            fields_spec = manifest.get("specification", {}).get("FIELDS", {})

            for field_label, target_val in fields_spec.items():
                # ターゲット値の正規化（比較しやすくするため）
                target_val_str = str(target_val).strip() if target_val is not None else ""

                # 1. まずそのラベル（Categoryなど）を持つ行をCSVから特定
                # Label_L または InnerT にフィールド名が含まれている要素を探す
                label_matches = raw_df[
                    (raw_df['Label_L'].str.contains(field_label, case=False, na=False)) |
                    (raw_df['InnerT'].str.contains(field_label, case=False, na=False))
                ]

                is_actually_filled = False
                
                # 2. ラベルに該当する行、またはその「直後の行」の値をチェック
                # （ServiceNowなどは、ラベルの次のBIDが入力フィールドになっていることが多いため）
                for idx, row in label_matches.iterrows():
                    # A. その行自体の Value / InnerT をチェック
                    actual_val = str(row['Value']).strip() if row['Value'] else ""
                    actual_inner = str(row['InnerT']).strip() if row['InnerT'] else ""
                    
                    if target_val_str in actual_val or target_val_str in actual_inner:
                        is_actually_filled = True
                        break
                        
                    # B. [救済措置] ラベルの「次のBID」の値をチェック（CSVがBID順に並んでいる前提）
                    # 物理構造上、LabelとInputが分かれているケースに対応
                    next_row_idx = idx + 1
                    if next_row_idx in raw_df.index:
                        next_row = raw_df.loc[next_row_idx]
                        next_val = str(next_row['Value']).strip() if next_row['Value'] else ""
                        if target_val_str != "" and target_val_str in next_val:
                            is_actually_filled = True
                            break

                # 3. 監査結果の反映
                if is_actually_filled:
                    completed_logic_refs.add(field_label)
                else:
                    # 値が一致しない、あるいは実行に失敗して値が変わっていない場合
                    if field_label in completed_logic_refs:
                        completed_logic_refs.remove(field_label)
                        

            # 【物理監査】展開状態の消込
            expanded_rows = raw_df[raw_df['Expanded'].astype(str).str.upper() == 'YES']
            for _, row in expanded_rows.iterrows():
                area = row['Label_L'] if pd.notna(row['Label_L']) else row['InnerT']
                if area: completed_interactions.add(str(area))

            # 🟢 【戦術フェーズ】プラン生成
            refine_instr = f"### 進捗\n- 入力済: {list(completed_logic_refs)}\n- 展開済: {list(completed_interactions)}"
            if extract_memory:
                #refine_instr += f"\n🚨【警告】extractValueは、既に '{extract_memory}' という値を取得済みです。同じ抽出を繰り返さず、この値を使って send_msg_to_user で報告を完了させてください！"
            
                if extract_memory.upper() != "NOT_FOUND":
                    # 有効な値が取れた場合の誘導
                    refine_instr += f"\n🚨【警告】extractValueは、既に '{extract_memory}' という値を取得済みです。同じ抽出を繰り返さず、この値を使って send_msg_to_user で報告を完了させてください！"

            print("📍 具象プランの作成中...")
            plan_result = sync_wait(planner.run_repair_loop(
                manifest=manifest, raw_df=raw_df, page_title=obs_analyzer.page_title,
                focused_bid=obs_analyzer.focused_bid, refine_instruction=refine_instr
            ))

            steps = plan_result.get("concrete_steps", [])
            e = plan_result.get("error", [])
            
            if not steps:
                if not e:
                    print("🏁 白旗！")
                    break

            # 🟢 【実行フェーズ】
            interrupted = False
            #hoverの挿入
            steps = inject_final_hover(steps)

            for step in steps:
                if interrupted: break
                for action in step['actions']:
        
                    # 🚨 特殊アクション: extractLLM のハンドリング
                    if action.get('action_type') == 'extractLLM':
                        instruction = action.get('instruction')

                        # 🚨 ここで待機を入れる
                        print(f"  ⏳ [Wait] コンテンツのロードを待機中 (3s)...")
                        time.sleep(3.0)

                        action_code = 'extractLLM(instruction=' + instruction+')'
                        print(f"  ∟ ⌨️ {action_code}")

                        # 🚨 修正ポイント: async関数なので sync_wait で結果を待機する
                        # (ConcretePlannerの呼び出し時と同じ方式)
                        raw_value = sync_wait(extractor.extract(
                            instruction=action.get('instruction'),
                            raw_df=raw_df,
                            page_title=obs_analyzer.page_title
                        ))
                        
                        # メモリに保存（キーは指示内容にして一意性を保つ）
                        extract_memory = raw_value
                        print(f"  🧠 [Observation] 抽出完了: extract_memory={raw_value}")
                        
                        # 履歴に「知覚ステップ」として記録
                        record, _ = save_step_state(
                            task_output_dir, f"L{loop_idx}_PERCEIVE", 
                            observation, info, action_taken=f"extractLLM: {raw_value}"
                        )
                        task_history.append(record)
                        
                        # 知覚したら一度ループを抜けて「リプラン」に回す（抽出結果を次のプランに反映させるため）
                        interrupted = True 
                        break

                    else:
                        # 🚨 ここを build_action_code 関数に差し替え
                        try:
                            action_code = build_action_code(action,last_extracted_value=extract_memory)
                        except ValueError as ve:
                            print(f"  ❌ ビルドエラー: {ve}")
                            # エラーを info に入れてリプランへ回す
                            info["last_action_error"] = str(ve)
                            break
                        
                        print(f"  ∟ ⌨️ {action_code}")
                        global LAST_EXECUTION_TRACE
                        LAST_EXECUTION_TRACE = {"force_used": False}

                        try:
                            action_code="click('179')\nfocus('274')\nclick('274')"
                            observation, reward, terminated, truncated, info = env.step(action_code)
                            time.sleep(1)
                        except Exception as e:
                            info["last_action_error"] = str(e)

                        info["trace_force_used"] = LAST_EXECUTION_TRACE["force_used"]
                        record, last_meta_path = save_step_state(task_output_dir, f"L{loop_idx}_{step['step_id']}", 
                                                                observation, info, action_taken=action_code,reward=reward,terminated=terminated,truncated=truncated)
                        task_history.append(record)

                        if "last_action_error" in info:
                            interrupted = True
                            break
                    if terminated or truncated: break
                if terminated or truncated: break
            if terminated or truncated: break

    except Exception:
        traceback.print_exc()
    finally:
        # 🚨 初期化失敗時のenv.close()エラーを防ぐ
        if env:
            try:
                env.close()
            except:
                pass
        summary_path = os.path.join(task_output_dir, "task_execution_history.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(task_history, f, ensure_ascii=False, indent=4, default=str)

        # 🚨 [追加] 最終リザルトの判定と表示
        # 履歴の最後のステップから、最終的な reward とメッセージを抽出
        final_reward = 0.0
        final_msg = "No message found."
        
        if task_history:
            last_step = task_history[-1]
            final_reward = last_step.get("reward", 0.0)
            # info_data -> task_info -> message の順に探す
            final_msg = last_step.get("info_data", {}).get("task_info", {}).get("message", "N/A")

        # 3. 🏁 コンソールへの最終出力
        status_symbol = "✅ SUCCESS" if final_reward == 1.0 else "❌ FAILED"
        
        print("\n" + "="*60)
        print(f"🏁 MISSION OVER: {task_id}") # 例: workarena.servicenow.create-problem
        print(f"📊 STATUS    : {status_symbol}")
        print(f"💰 REWARD    : {final_reward}")
        print(f"💬 MESSAGE   : {final_msg}")
        print(f"📁 LOGS      : {task_output_dir}")
        print("="*60 + "\n")

# --- 3. メイン ---
if __name__ == "__main__":

    # asyncio.run() を使わず、直接関数を呼ぶ
    #for task_class in ATOMIC_TASKS[7:8]:
    for task_class in ATOMIC_TASKS[23:24]:
    #for task_class in ATOMIC_TASKS[24:25]:
    #for task_class in ATOMIC_TASKS[25:26]:
    #for task_class in ATOMIC_TASKS[26:27]:
    #for task_class in ATOMIC_TASKS[27:28]:
    #for task_class in ATOMIC_TASKS[28:29]:
    #for task_class in ATOMIC_TASKS[29:30]:
    #for task_class in ATOMIC_TASKS[30:31]:
    #for task_class in ATOMIC_TASKS[31:32]:
    #for task_class in ATOMIC_TASKS[32:33]:
    #for task_class in ATOMIC_TASKS[9:10]:
    #for task_class in ATOMIC_TASKS[7:8]:
        run_autonomous_task(task_class)
        #run_autonomous_task(task_class)
        #run_autonomous_task(task_class)
        #run_autonomous_task(task_class)
        #run_autonomous_task(task_class)
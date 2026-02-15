import os
import sys

# 【最重要】Playwright読み込み前に環境変数をセット
os.environ['DEBUG'] = 'pw:api'

import gymnasium as gym
import time
import yaml 
from typing import Dict, Any, Tuple, List, Type
import random
import re 
import json 
from io import BytesIO
import traceback
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow (PIL) がインストールされていません。")
    sys.exit(1)

from browsergym.core.env import BrowserEnv
from browsergym.workarena import ATOMIC_TASKS 
from logger import save_step_state

# --- 設定 ---
CONFIG_FILE = "config.yaml"
OUTPUT_DIR = "task_execution_data_full_logs"
HEADLESS_MODE: bool = True
FIXED_SEED = 42 
MAX_STEPS = 2
GLOBAL_LOG_PATH = os.path.join(OUTPUT_DIR, "all_playwright_api.log")

all_task_results: List[Dict[str, Any]] = []

# --- 2. YAML読み込み ---
try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    print(f"❌ 設定ファイル読み込み失敗: {e}")
    sys.exit(1)

os.environ['HUGGING_FACE_HUB_TOKEN'] = cfg.get('HUGGING_FACE_HUB_TOKEN', '')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def sanitize_filename(name: str) -> str:
    name = name.replace("workarena.servicenow.", "").replace("/", "_").replace(".", "_")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    return name[:50] 

# --- 3. タスク実行メイン関数 ---
def run_single_task(task_class: Type, log_f):
    global all_task_results
    
    try:
        task_id = task_class.get_task_id() 
    except AttributeError:
        task_id = task_class.__name__
        
    safe_task_name = sanitize_filename(task_id)
    task_output_dir = os.path.join(OUTPUT_DIR, safe_task_name)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)

    # ログに開始マークを刻印 (物理書き込みを確約)
    sep_start = f"\n\n{'='*60}\n=== Task Start: {task_id} ===\n{'='*60}\n"
    log_f.write(sep_start.encode('utf-8'))
    log_f.flush()
    try:
        os.fsync(log_f.fileno())
    except OSError:
        pass

    env = None
    task_results = {'task_id': task_id, 'task_status': 'FAILURE', 'final_reward': 0.0}
    task_history = []
    
    try:
        print(f"\n{'='*50}\n--- ⚙️ WorkArenaタスク実行開始: {task_id} ---", file=sys.stdout)

        env = BrowserEnv(
            task_entrypoint=task_class,
            headless=HEADLESS_MODE,
            #viewport={'width': 1280, 'height': 1280},
            pre_observation_delay=5.0
        )
        
        observation, info = env.reset(seed=FIXED_SEED)
        task_goal = observation.get('goal', 'Goal not found.')
        record = save_step_state(task_output_dir, 0, observation, info)
        task_history.append(record)

        print(f"Task {task_id}: Step 0 saved. Goal: {task_goal}", file=sys.stdout)
        print(f"--- 💡 Executing Cheat ---", file=sys.stdout)
        
        env.task.cheat(env.page, env.chat.messages)
            
        observation, reward, terminated, truncated, info = env.step("noop()")
        
        task_results['final_reward'] = reward
        task_results['task_status'] = 'SUCCESS' if reward >= 1.0 else 'FAIL'
        print(f"✨ Task Reward: {reward}", file=sys.stdout)
            
        record = save_step_state(task_output_dir, 1, observation, info, "noop()", reward, terminated, truncated)
        task_history.append(record)
            
    except Exception as e:
        task_results['task_status'] = f"FATAL_ERROR: {type(e).__name__}"
        traceback.print_exc() 
        
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass
        
        # Playwrightの終了処理ログが出終わるのを待つ
        time.sleep(1) 
        
        # ログに終了マークを刻印
        sep_end = f"=== Task End: {task_id} ===\n"
        log_f.write(sep_end.encode('utf-8'))
        log_f.flush()
        try:
            os.fsync(log_f.fileno())
        except OSError:
            pass
        
        task_results['step_history'] = task_history
        all_task_results.append(task_results)

        summary_path = os.path.join(task_output_dir, "sample_task_execution_history.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(task_history, f, ensure_ascii=False, indent=4, default=str)
        except:
            pass

# --- 4. メイン ---
if __name__ == "__main__":
    # 1. 統合ログファイルをバイナリ・バッファなしで開く
    f_log = open(GLOBAL_LOG_PATH, "wb", buffering=0)

    # 2. OSレベルのリダイレクトをここで「一回だけ」行う
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    os.dup2(f_log.fileno(), stderr_fd)

    try:
        # ATOMIC_TASKS の全範囲またはテスト範囲
        tasks_to_run = ATOMIC_TASKS
        #tasks_to_run = ATOMIC_TASKS[1:3] 

        print(f"🚀 Running {len(tasks_to_run)} tasks. Global Log: {GLOBAL_LOG_PATH}")

        for i, task_class in enumerate(tasks_to_run):
            print(f"\n### 実行 {i+1}/{len(tasks_to_run)} ###", file=sys.stdout)
            run_single_task(task_class, f_log)
            time.sleep(1)

    finally:
        # 3. リダイレクト解除（OSErrorを徹底排除する順序）
        try:
            sys.stderr.flush()
            # Pythonのストリームを元に戻す（最優先）
            sys.stderr = sys.__stderr__ 
            
            # OSレベルの記述子を戻す（失敗しても続行）
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
        except Exception:
            pass
        
        # 4. ログファイルを閉じる
        if not f_log.closed:
            f_log.close()
            
        print(f"\n✅ All tasks finished. Total log: {GLOBAL_LOG_PATH}", file=sys.stdout)

        # レポート表示
        print("\n" + "*"*60, file=sys.stdout)
        print("⭐ 全タスク実行完了レポート ⭐", file=sys.stdout)
        success_count = sum(1 for res in all_task_results if res['task_status'] == 'SUCCESS')
        print(f"総数: {len(all_task_results)} 成功: {success_count}", file=sys.stdout)
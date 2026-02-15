import os
import json
import traceback
import sys
import numpy as np
from io import BytesIO
from PIL import Image

def save_step_state(task_output_dir: str, step: str, obs: dict, info: dict, 
                    action_taken: dict = None, reward: float = 0.0, 
                    terminated: bool = False, truncated: bool = False):
    """
    observationとinfoを保存し、そのステップのサマリー（step_record）を返す。
    """
    heavy_keys = ['screenshot', 'dom_object', 'axtree_object', 'extra_element_properties']
    saved_file_references = {}

    # --- 1. 個別ファイルの保存処理 ---
    # Screenshot
    if obs.get('screenshot') is not None:
        try:
            ss_data = obs['screenshot']
            fname = f"step_{step}_screenshot.png"
            img = Image.fromarray(ss_data) if isinstance(ss_data, np.ndarray) else Image.open(BytesIO(ss_data))
            img.save(os.path.join(task_output_dir, fname))
            saved_file_references['screenshot'] = fname
        except Exception:
            print(f"❌ [Step {step}] Screenshot保存失敗", file=sys.stderr); traceback.print_exc()

    # DOM / AXTree / Extra Props (共通の保存ロジック)
    for key, ext in zip(['dom_object', 'axtree_object', 'extra_element_properties'], ['html', 'json', 'json']):
        data = obs.get(key)
        if data:
            try:
                is_json = ext == 'json' or isinstance(data, (dict, list))
                actual_ext = 'json' if is_json else 'html'
                fname = f"step_{step}_{key}.{actual_ext}"
                with open(os.path.join(task_output_dir, fname), 'w', encoding='utf-8') as f:
                    if is_json:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(str(data))
                saved_file_references[key] = fname
            except Exception:
                print(f"❌ [Step {step}] {key}保存失敗", file=sys.stderr); traceback.print_exc()

    # --- 2. ステップ履歴用レコードの作成 ---
    try:
        # JSON保存可能なメタデータのみを抽出
        metadata_obs = {k: v for k, v in obs.items() if k not in heavy_keys}
        
        # 終了フラグの正規化 (numpy.bool_ 対策)
        term_bool = bool(terminated)
        trun_bool = bool(truncated)

        step_record = {
            'step': step,
            'action_taken': action_taken,
            'reward': float(reward),
            'terminated': term_bool,
            'truncated': trun_bool,
            'observation_metadata': metadata_obs,
            'info_data': info,  # infoの中身も保存
            'saved_files': saved_file_references
        }
        
        # 個別のメタデータファイルとしても保存しておく
        meta_fname = f"step_{step}_state_metadata.json"
        save_path = os.path.join(task_output_dir, meta_fname)
        with open(save_path, 'w', encoding='utf-8') as f:
            # default=str により、JSON化できない型が含まれていてもエラー落ちを防ぐ
            json.dump(step_record, f, ensure_ascii=False, indent=4, default=str)

        return step_record,save_path

    except Exception:
        print(f"❌ [Step {step}] レコード作成失敗", file=sys.stderr); traceback.print_exc()
        return None,None
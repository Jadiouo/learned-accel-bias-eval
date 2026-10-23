# -*- coding: utf-8 -*-
"""共用工具 / shared helpers：config 載入、路徑解析、provenance（git hash / timestamp）。

GitHub-ready：**無硬編絕對路徑**（SDD §1.3）。資料目錄一律走
  1) CLI `--data-dir`（最高優先，由 run.py 注入 cfg["data_dir"]）
  2) 環境變數 `EUROC_DATA_DIR`
  3) config 的 `data_dir`（可寫 `${EUROC_DATA_DIR}` 由本檔展開）
results 目錄相對「執行時的工作目錄」(cwd) 解析，不寫死任何人的家目錄。
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess

import yaml

# EuRoC 序列名 → on-disk 資料夾名（可被 config 的 seq_folders 覆寫）
DEFAULT_SEQ_FOLDERS = {
    "MH_01": "MH_01_easy", "MH_02": "MH_02_easy", "MH_03": "MH_03_medium",
    "MH_04": "MH_04_difficult", "MH_05": "MH_05_difficult",
}

_ENV_PAT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    """把字串裡的 ${VAR} 換成環境變數（找不到 → 空字串，之後會被驗證攔下）。"""
    if isinstance(value, str):
        return _ENV_PAT.sub(lambda m: os.environ.get(m.group(1), ""), value)
    return value


def load_config(path: str, data_dir_override: str = None) -> dict:
    """載入 YAML config；解析 data_dir / results_dir 絕對路徑；無硬編。"""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)

    # ---- data_dir 解析（優先序：CLI > config 的 ${ENV} 展開 > EUROC_DATA_DIR 後備）----
    # config 用 `data_dir: ${TUMVI_DATA_DIR}` / `${EUROC_DATA_DIR}` 指定自己的環境變數，
    # 故先展開 config（資料集各自的 env），最後才退回 EUROC_DATA_DIR（不混用資料集）。
    cfg_data_dir = _expand_env(cfg.get("data_dir", ""))
    data_dir = data_dir_override or cfg_data_dir or os.environ.get("EUROC_DATA_DIR")
    if not data_dir:
        env_hint = cfg.get("data_dir", "${EUROC_DATA_DIR}")
        raise ValueError(
            f"data_dir 未設定。請用 `--data-dir <路徑>`、或設 config 指定的環境變數"
            f"（{env_hint}）、或直接在 config 填 `data_dir`。")
    cfg["_data_dir_abs"] = os.path.abspath(os.path.expanduser(data_dir))

    # ---- results_dir 解析（相對 cwd，不寫死家目錄）----
    results_dir = cfg.get("results_dir", "results")
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.getcwd(), results_dir)
    cfg["_results_dir_abs"] = results_dir

    cfg.setdefault("seq_folders", dict(DEFAULT_SEQ_FOLDERS))
    return cfg


def seq_csv_paths(seq: str, cfg: dict) -> dict:
    """回傳某序列的 imu / gt CSV 絕對路徑（依 data_dir + seq_folders）。"""
    folders = cfg.get("seq_folders", DEFAULT_SEQ_FOLDERS)
    folder = folders.get(seq, seq)
    base = os.path.join(cfg["_data_dir_abs"], folder, "mav0")
    return {"imu": os.path.join(base, "imu0", "data.csv"),
            "gt": os.path.join(base, "state_groundtruth_estimate0", "data.csv")}


def git_hash() -> str:
    """目前 commit hash；非 git repo 時回傳 'nogit'。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        h = out.stdout.strip()
        return h if h else "nogit"
    except Exception:
        return "nogit"


def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

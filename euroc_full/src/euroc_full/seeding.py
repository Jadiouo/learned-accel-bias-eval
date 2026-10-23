# -*- coding: utf-8 -*-
"""統一 seed 控制（SDD §11）：numpy + torch + cuda + PYTHONHASHSEED。

`torch_deterministic` 為 config flag（GPU 全確定性有性能代價）。torch 不在場時只設 numpy。
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, torch_deterministic: bool = False) -> None:
    """把 numpy / random / PYTHONHASHSEED / torch(+cuda) 全部對齊到 `seed`。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            # 全確定性（較慢）。cuDNN benchmark 關閉、deterministic 開啟。
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass  # torch 不在場（純 baseline / 分析路徑）

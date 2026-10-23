# -*- coding: utf-8 -*-
"""
方法 registry（SDD §4）—— config 用名字選方法。

每個方法登記：family（論文 V-B 家族覆蓋）、kind（決定執行路徑）、approx_params。
**module-load 不 import torch**：DL builder 在被呼叫時才 lazy import models/，
這樣純 baseline / 分析 / metric-test 路徑在無 torch 環境也能跑。

kind:
  statistical : constant_pooled / constant_perseq / linear_stats（無 torch）
  filter      : kf / pf（無 torch；bookend）
  dl          : tcn / transformer_* / mamba / m1_cnn / m2 / m3_lstm / m4 / m5 / m6（torch）
"""
from __future__ import annotations

# name -> metadata（builder 對 DL 才有；baseline/filter 由 run.py 直接呼叫 baselines.py）
METHODS = {
    # Layer 1 — statistical
    "constant_pooled": {"family": "statistical", "kind": "statistical", "approx_params": 3},
    "constant_perseq": {"family": "statistical", "kind": "statistical", "approx_params": 3},
    "linear_stats":    {"family": "statistical", "kind": "statistical", "approx_params": 3000},
    # Layer 2 — classical filters
    "kf":              {"family": "classical_filter", "kind": "filter", "approx_params": 0},
    "pf":              {"family": "classical_filter", "kind": "filter", "approx_params": 0},
    # Layer 3 — modern architectures
    "tcn":             {"family": "convolutional_temporal", "kind": "dl", "approx_params": 227000},
    "transformer_s":   {"family": "attention", "kind": "dl", "approx_params": 227000},
    "transformer_m":   {"family": "attention", "kind": "dl", "approx_params": 427000},
    "transformer_l":   {"family": "attention", "kind": "dl", "approx_params": 870000},
    "mamba":           {"family": "selective_state_space", "kind": "dl", "approx_params": 227000},
    # Layer 4 — CNN-LSTM internal ablation
    "m1_cnn":          {"family": "convolutional", "kind": "dl", "approx_params": 22000},
    "m2":              {"family": "convolutional", "kind": "dl", "approx_params": 120000},
    "m3_lstm":         {"family": "recurrent", "kind": "dl", "approx_params": 150000},
    "m4":              {"family": "recurrent", "kind": "dl", "approx_params": 110000},
    "m5":              {"family": "hybrid_conv_recurrent", "kind": "dl", "approx_params": 350000},
    "m6":              {"family": "hybrid_conv_recurrent", "kind": "dl", "approx_params": 872000},
}


def list_methods():
    return list(METHODS.keys())


def method_info(name: str) -> dict:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}（registry 未登記）")
    return METHODS[name]


def kind_of(name: str) -> str:
    return method_info(name)["kind"]


def is_dl(name: str) -> bool:
    return method_info(name)["kind"] == "dl"


def build_dl_model(name: str, cfg: dict):
    """建一個 DL 模型（lazy import torch / models）。回傳 nn.Module。"""
    if not is_dl(name):
        raise ValueError(f"{name!r} 非 DL 方法，不該呼叫 build_dl_model")
    dropout = cfg.get("model", {}).get("dropout", cfg.get("dropout", 0.3))
    mcfg = cfg.get("model", {})

    if name == "tcn":
        from .models.tcn import TCN
        return TCN(channels=tuple(mcfg.get("tcn_channels", (128, 128, 128, 128))),
                   k=mcfg.get("tcn_kernel", 7), dropout=dropout)
    if name in ("transformer_s", "transformer_m", "transformer_l"):
        from .models.transformer import build_transformer
        return build_transformer(name, dropout=dropout)
    if name == "mamba":
        from .models.mamba import MambaModel
        return MambaModel(d_model=mcfg.get("mamba_d_model", 96),
                          n_layers=mcfg.get("mamba_layers", 4), dropout=dropout)
    if name == "m1_cnn":
        from .models.cnn_lstm import M1_CNN
        return M1_CNN(filters=mcfg.get("cnn_filters", 64), dropout=dropout)
    if name == "m2":
        from .models.cnn_lstm import M2_MultiScaleCNN
        return M2_MultiScaleCNN(filters=mcfg.get("cnn_filters", 64), dropout=dropout)
    if name == "m3_lstm":
        from .models.cnn_lstm import M3_LSTM
        return M3_LSTM(hidden=mcfg.get("hidden_dim", 128), layers=mcfg.get("num_layers", 2), dropout=dropout)
    if name == "m4":
        from .models.cnn_lstm import M4_GRU
        return M4_GRU(hidden=mcfg.get("hidden_dim", 128), layers=mcfg.get("num_layers", 2), dropout=dropout)
    if name == "m5":
        from .models.cnn_lstm import M5_CNN_LSTM
        return M5_CNN_LSTM(filters=mcfg.get("cnn_filters", 64),
                           hidden=mcfg.get("hidden_dim", 128), layers=mcfg.get("num_layers", 2), dropout=dropout)
    if name == "m6":
        from .models.cnn_lstm import M6_DeepCNN_LSTM
        return M6_DeepCNN_LSTM(filters=128, hidden=256, layers=2, dropout=dropout)
    raise ValueError(f"unknown DL method {name!r}")


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

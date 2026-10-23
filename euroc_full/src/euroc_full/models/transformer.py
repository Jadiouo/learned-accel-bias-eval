# -*- coding: utf-8 -*-
"""
Causal Transformer（SDD §4，attention 家族；S/M/L 三容量，comparable param scale）。

  transformer_s ~227K（comparable to TCN/M3）
  transformer_m ~427K（comparable to M5）
  transformer_l ~870K（comparable to M6）

★ 因果保證：causal mask（上三角 −inf），位置 t 只看 ≤ t。本模型輸出 per-position 預測，
`forward()` 取最後一步；`causal_sequence_output()` 給整段 per-position 預測供 §7.1 causal-mask
sanity（擾動後半段 input，前半段預測必不變）。
慣例：input (B, W, 6) → output (B, 3)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                     # x: (B, W, d)
        return x + self.pe[:, :x.size(1)]


class CausalTransformer(nn.Module):
    def __init__(self, in_ch=6, d_model=64, nhead=4, layers=3, dim_ff=128,
                 out_dim=3, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d_model)
        self.pos = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def _encode(self, x):
        W = x.size(1)
        mask = torch.triu(torch.full((W, W), float("-inf"), device=x.device), diagonal=1)
        z = self.pos(self.in_proj(x))
        return self.encoder(z, mask=mask)     # (B, W, d) — 每位置只看過去

    def causal_sequence_output(self, x):
        """per-position 預測 (B, W, out_dim)，供 causal-mask sanity。"""
        return self.head(self._encode(x))

    def forward(self, x):
        return self.head(self._encode(x)[:, -1, :])   # 因果：取最後一步


# 容量設定（comparable param scale；可被 config 覆寫）
_CAPACITY = {
    "transformer_s": dict(d_model=64,  nhead=4, layers=3, dim_ff=128),
    "transformer_m": dict(d_model=96,  nhead=6, layers=4, dim_ff=192),
    "transformer_l": dict(d_model=128, nhead=8, layers=5, dim_ff=256),
}


def build_transformer(name: str, dropout=0.3) -> CausalTransformer:
    if name not in _CAPACITY:
        raise ValueError(f"unknown transformer capacity {name!r}")
    return CausalTransformer(dropout=dropout, **_CAPACITY[name])

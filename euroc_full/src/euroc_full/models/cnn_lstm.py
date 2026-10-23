# -*- coding: utf-8 -*-
"""
CNN / LSTM family（SDD §4，Layer 4：m1_cnn … m6，~22K–872K 參數）。

從 legacy 概念 port、**重寫不 import legacy**。所有方法 causal（只看過去 / uni-directional）。
慣例：input (B, W, 6) → output (B, 3)（window-mean accel bias）。

家族覆蓋（寫進論文 V-B）：
  convolutional        : m1_cnn（single-scale）, m2（multi-scale）
  recurrent            : m3_lstm（LSTM）, m4（GRU）
  hybrid conv-recurrent: m5（CNN→LSTM）, m6（deeper CNN→LSTM，最大容量）
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── convolutional ──────────────────────────────────────────────────────────
class M1_CNN(nn.Module):
    """single-scale 1D CNN（~22K）。"""
    def __init__(self, in_ch=6, filters=64, k=5, out_dim=3, dropout=0.3):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, filters, k, padding=pad), nn.ReLU(),
            nn.Conv1d(filters, filters, k, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(filters, out_dim))

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.net(x))


class M2_MultiScaleCNN(nn.Module):
    """multi-scale 1D CNN（並列 k=3/5/7 分支，~120K）。"""
    def __init__(self, in_ch=6, filters=64, out_dim=3, dropout=0.3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, filters, k, padding=k // 2), nn.ReLU(),
                          nn.Conv1d(filters, filters, k, padding=k // 2), nn.ReLU())
            for k in (3, 5, 7)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters * 3, out_dim))

    def forward(self, x):
        x = x.transpose(1, 2)
        feats = [self.pool(b(x)).flatten(1) for b in self.branches]
        return self.head(torch.cat(feats, dim=1))


# ── recurrent ──────────────────────────────────────────────────────────────
class M3_LSTM(nn.Module):
    """uni-directional LSTM（causal，~150K）。"""
    def __init__(self, in_ch=6, hidden=128, layers=2, out_dim=3, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class M4_GRU(nn.Module):
    """uni-directional GRU（causal，~110K）。"""
    def __init__(self, in_ch=6, hidden=128, layers=2, out_dim=3, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


# ── hybrid conv-recurrent ──────────────────────────────────────────────────
class _CNNEncoder(nn.Module):
    def __init__(self, in_ch, filters, k=5):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, filters, k, padding=pad), nn.ReLU(),
            nn.Conv1d(filters, filters, k, padding=pad), nn.ReLU())

    def forward(self, x):                    # x: (B, W, C) → (B, W, filters)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class M5_CNN_LSTM(nn.Module):
    """CNN encoder → LSTM（hybrid，~350K）。"""
    def __init__(self, in_ch=6, filters=64, hidden=128, layers=2, out_dim=3, dropout=0.3):
        super().__init__()
        self.enc = _CNNEncoder(in_ch, filters)
        self.rnn = nn.LSTM(filters, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        out, _ = self.rnn(self.enc(x))
        return self.head(out[:, -1, :])


class M6_DeepCNN_LSTM(nn.Module):
    """deeper CNN encoder → wider LSTM（最大容量，~872K）。"""
    def __init__(self, in_ch=6, filters=128, hidden=256, layers=2, out_dim=3, dropout=0.3):
        super().__init__()
        pad = 2
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, filters, 5, padding=pad), nn.ReLU(),
            nn.Conv1d(filters, filters, 5, padding=pad), nn.ReLU(),
            nn.Conv1d(filters, filters, 5, padding=pad), nn.ReLU())
        self.rnn = nn.LSTM(filters, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        z = self.enc(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(z)
        return self.head(out[:, -1, :])

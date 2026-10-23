# -*- coding: utf-8 -*-
"""
TCN — causal dilated temporal conv（SDD §4，~227K）。複製自驗證過的 MVP，邏輯不改。
慣例：input (B, W, 6) → output (B, 3)。因果：只看過去（left padding + 取最後一步）。
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class _CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal(self, conv, x):
        return conv(F.pad(x, (self.pad, 0)))     # 只在左側 pad → 因果

    def forward(self, x):
        y = self.drop(self.relu(self._causal(self.conv1, x)))
        y = self.drop(self.relu(self._causal(self.conv2, y)))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCN(nn.Module):
    def __init__(self, in_ch=6, channels=(128, 128, 128, 128), k=7, out_dim=3, dropout=0.3):
        super().__init__()
        blocks = []
        prev = in_ch
        for i, ch in enumerate(channels):
            blocks.append(_CausalBlock(prev, ch, k, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, out_dim))

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return self.head(y[:, :, -1])           # 因果：取最後一步

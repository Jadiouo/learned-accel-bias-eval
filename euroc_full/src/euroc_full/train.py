# -*- coding: utf-8 -*-
"""
訓練（SDD §7）。MSE 回歸、Adam、ReduceLROnPlateau、val 早停、seed 全控（seeding.py）。

train_model 吃任何帶 X_train/y_train/X_val/y_val 屬性的物件（SeqData / PooledData 皆可）。
所有 torch 相依集中在此檔與 models/，純 baseline / 分析路徑不 import 本檔。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .seeding import seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tcfg(cfg, key, default):
    return cfg.get("train", {}).get(key, default)


def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_model(model: nn.Module, data, cfg: dict, seed: int):
    """訓練並回傳 (best_model, history)。history 含 train/val loss 與 best_epoch。

    ★ 絕不 shuffle 跨 window 的時序語意？—— window 已是獨立樣本，train 內 batch shuffle 可；
      序列切分本身嚴格時序（data_euroc 不 shuffle 切 split）。
    """
    seed_everything(seed, torch_deterministic=cfg.get("determinism", {}).get("torch_deterministic", False))
    model = model.to(DEVICE)
    batch = _tcfg(cfg, "batch_size", 64)
    tr = _loader(data.X_train, data.y_train, batch, shuffle=True)
    va = _loader(data.X_val, data.y_val, batch, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=_tcfg(cfg, "lr", 1e-3))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss()
    epochs = _tcfg(cfg, "epochs", 50)
    patience = _tcfg(cfg, "early_stop_patience", 7)
    clip = _tcfg(cfg, "clip_norm", 1.0)

    best_val = float("inf"); best_state = None; best_epoch = -1
    no_improve = 0
    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train(); tl = 0.0; nb = 0
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item(); nb += 1
        train_loss = tl / max(nb, 1)

        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += crit(model(xb), yb).item(); nv += 1
        val_loss = vl / max(nv, 1)
        sched.step(val_loss)
        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-9:
            best_val = val_loss; best_epoch = epoch; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch; hist["best_val_loss"] = best_val
    return model, hist


@torch.no_grad()
def infer(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model = model.to(DEVICE).eval()
    xb = torch.as_tensor(X, dtype=torch.float32).to(DEVICE)
    preds = []
    for i in range(0, len(xb), 1024):
        preds.append(model(xb[i:i + 1024]).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float64) if preds else np.zeros((0, 3))

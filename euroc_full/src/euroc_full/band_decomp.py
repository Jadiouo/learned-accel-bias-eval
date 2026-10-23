# -*- coding: utf-8 -*-
"""
★ Bias-error 頻帶分解（SDD_TUMVI_Room §8）—— 讓 L2 判決「有機制」。

對 per-sample bias error ε(t)=b_ref(t)−b̂(t)（取自 `series.npz`）做 PSD（Welch），切
低 [0,0.5] / 中 [0.5,2] / 高 [2,Nyq] Hz 三段能量,回答:
  (a) 各方法**低頻**殘差能量分不分得開（IABE 能否區分方法的前提）
  (b) corr(iabe_rot, 低頻能量) vs corr(iabe_rot, bias_rmse)（IABE=低頻加權、與 RMSE 脫鉤?）
  (c) 方法在「低頻佔比」排序 vs 在「總能量(∝RMSE)」排序是否不同（L2 inversion 的頻域根源）

**純新增、離線、唯讀 series.npz**,不碰任何 metric 邏輯。對 EuRoC 與 TUM-VI 都可算。
注意:ε 時間序列取樣率 = imu_freq/stride（per-window bias，非 raw imu）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .metrics import _spearman


def _band_energy(eps: np.ndarray, fs: float, bands) -> list:
    """ε:(N,3) → [低, 中, 高] 頻帶能量（三軸 PSD 相加後積分）。N 太短回 None。"""
    from scipy.signal import welch
    n = len(eps)
    if n < 16:
        return None
    nper = int(min(256, n))
    freqs = None; psd = None
    for i in range(3):
        f, P = welch(eps[:, i].astype(np.float64), fs=fs, nperseg=nper)
        if psd is None:
            psd = np.zeros_like(P); freqs = f
        psd += P
    out = []
    for lo, hi in bands:
        hi = freqs[-1] if hi is None else float(hi)
        m = (freqs >= float(lo)) & (freqs < hi)
        out.append(float(np.trapezoid(psd[m], freqs[m])) if m.any() else 0.0)
    return out


def band_table(cfg, df, results_dir, regime, methods) -> tuple:
    """逐 (method, seed, eval_seq) 算頻帶能量 + 取 iabe_rot/bias_rmse。回 (DataFrame, fs, bands)。"""
    fs = float(cfg.get("imu_freq", 200)) / float(cfg.get("stride", 5))
    bands = cfg.get("analysis", {}).get("band_decomp", {}).get("bands_hz", [[0, 0.5], [0.5, 2.0], [2.0, None]])
    sub = df[(df["regime"] == regime) & df["metric"].isin(["iabe_rot", "bias_rmse_avg"])]
    if sub.empty:
        return pd.DataFrame(), fs, bands
    piv = sub.pivot_table(index=["run_id", "method", "seed", "eval_sequence"],
                          columns="metric", values="value", aggfunc="first").reset_index()
    rows = []
    for r in piv.itertuples():
        if r.method not in methods:
            continue
        npz = os.path.join(results_dir, "runs", r.run_id, "series.npz")
        if not os.path.exists(npz):
            continue
        try:
            z = np.load(npz)
        except Exception:
            continue
        kh, kr = f"b_hat_{r.eval_sequence}", f"b_ref_{r.eval_sequence}"
        if kh not in z.files or kr not in z.files:
            continue
        be = _band_energy(np.asarray(z[kr]) - np.asarray(z[kh]), fs, bands)
        if be is None:
            continue
        low, mid, high = be; tot = low + mid + high
        rows.append({"method": r.method, "seed": int(r.seed), "seq": r.eval_sequence,
                     "low": low, "mid": mid, "high": high, "total": tot,
                     "low_frac": low / (tot + 1e-30),
                     "iabe_rot": float(getattr(r, "iabe_rot", np.nan)),
                     "bias_rmse": float(getattr(r, "bias_rmse_avg", np.nan))})
    return pd.DataFrame(rows), fs, bands


def _kendall(order_a, order_b) -> float:
    items = list(order_a); idx_b = {m: i for i, m in enumerate(order_b)}
    n = len(items);
    if n < 2:
        return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (i - j) * (idx_b[items[i]] - idx_b[items[j]])
            c += s > 0; d += s < 0
    return (c - d) / (c + d) if (c + d) else 1.0


def analyze(cfg, df, results_dir, dl_methods, regime="B", label="dataset") -> dict:
    """回傳 §8 三項輸出 + per-seq 低頻可分性。空資料回 None。"""
    tab, fs, bands = band_table(cfg, df, results_dir, regime, dl_methods)
    if tab.empty:
        return None
    # (b) 相關性比較
    corr_iabe_low = abs(_spearman(tab["iabe_rot"].to_numpy(), tab["low"].to_numpy()))
    corr_iabe_rmse = abs(_spearman(tab["iabe_rot"].to_numpy(), tab["bias_rmse"].to_numpy()))
    # per-method 聚合
    g = tab.groupby("method").agg(low=("low", "mean"), mid=("mid", "mean"),
                                  high=("high", "mean"), total=("total", "mean"),
                                  low_frac=("low_frac", "mean")).reset_index()
    # (a) 低頻能量跨方法分不分得開（變異係數越大越分得開）
    low_cv = float(g["low"].std(ddof=1) / (g["low"].mean() + 1e-30)) if len(g) > 1 else 0.0
    # (c) 排序比較：低頻佔比 排序 vs 總能量(∝RMSE) 排序
    order_lowfrac = list(g.sort_values("low_frac", ascending=False)["method"])
    order_total = list(g.sort_values("total")["method"])
    tau = _kendall(order_lowfrac, order_total)
    # per-seq 低頻可分性（各序列跨方法低頻能量 spread）
    per_seq = {}
    for seq, gs in tab.groupby("seq"):
        gm = gs.groupby("method")["low"].mean()
        if len(gm) > 1:
            per_seq[seq] = {"low_cv": float(gm.std(ddof=1) / (gm.mean() + 1e-30)),
                            "low_min": float(gm.min()), "low_max": float(gm.max())}
    return {
        "label": label, "regime": regime, "fs_hz": fs, "bands_hz": bands,
        "n_methods": int(len(g)), "n_rows": int(len(tab)),
        "low_energy_cross_method_cv": low_cv,
        "corr_iabe_lowband": float(corr_iabe_low),
        "corr_iabe_biasrmse": float(corr_iabe_rmse),
        "iabe_tracks_lowband_more_than_rmse": bool(corr_iabe_low > corr_iabe_rmse),
        "order_by_low_frac": order_lowfrac,
        "order_by_total_energy": order_total,
        "ordering_kendall_tau": float(tau),
        "orderings_differ": bool(tau < 0.8),
        "per_method": g.round(8).to_dict(orient="records"),
        "per_seq_low_separability": per_seq,
    }

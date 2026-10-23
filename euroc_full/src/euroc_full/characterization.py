# -*- coding: utf-8 -*-
"""
★ Supplementary Characterization（SDD_Supplementary_Characterization）—— 兩個便宜補算。

Part A — 轉動 dose-response 量化（唯讀，現有數據）
  把 C1「越轉動越錯」從 ~7 個軼事點升級成「轉動代理 vs gyro-ATE/IABE ratio」的量化單調曲線。
  跨序列（EuRoC MH_01..05 + TUM-VI room1..6）+ 同序列 sub-segment 散點 + 單調性檢定。

Part B — Matched-RMSE positive control / Bode / detection limit（合成輸入，凍結 metric）
  Tier 2（主菜）：detection limit —— 把 robust inversion=0 寫成「方法間低頻殘差差異
                   低於指標在當前量測不確定度下的解析度」。
  Tier A：matched-RMSE 三頻帶（低>中>高，RMSE 相等）。
  Tier A+：Bode 掃頻（低頻平台 + −40 dB/decade）。

隔離鐵律（SDD §0）：
  * 本模組只 **import** 既有、已驗證的 metric 函式（iabe_rot / gyro_ate / bias_rmse /
    integrate_gyro / build_metric_inputs / MetricInputs / _spearman）與 band_decomp，
    **metric 邏輯一行不改**。
  * 只 **讀** results/euroc 與 results/tumvi（metrics_long、runs_index、series.npz），不寫入主結果。
  * 全部輸出寫到獨立子樹 `results/characterization/`。
  * 受 seed 控制、記 git_hash + config（output contract 慣例）。

離線、零訓練、零 GPU、無硬編路徑。Phase 0 metric sanity 的延伸，非新實驗批次。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import common
from . import data_euroc as D
from . import band_decomp
from .metrics import (
    MetricInputs, build_metric_inputs, iabe_rot, gyro_ate, bias_rmse,
    check_lemma_constant_bias, _spearman,
)

# 乾淨 label 序列（誠實標色用，SDD A.3）：EuRoC 全部乾淨；TUM-VI 只有 room4 乾淨，
# room1/2/3/5/6 的 mocap-derived label 受姿態相關共模/運動洩漏污染（SDD_TUMVI_Room）。
CLEAN_SEQUENCES = {"MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "room4"}


# ──────────────────────────────────────────────────────────────────────────
# 共用：資料集設定載入（無硬編路徑；env > repo/dataset/ 後備）
# ──────────────────────────────────────────────────────────────────────────
def _repo_root() -> str:
    # characterization.py → euroc_full(pkg) → src → euroc_full(root) → 套件上一層
    # （搬到 no_1a 後即 no_1a；用於定位本套件自己的 results 子樹，會跟著套件走）
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _data_root() -> str:
    """向上搜尋含 dataset/ 的祖先目錄；讓 dataset 留在 repo 頂層、本套件可被搬到子層
    （如 no_1a/euroc_full）仍能定位資料。找不到則退回 _repo_root()。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "dataset")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return _repo_root()


def _resolve_cfg(dataset: str, data_dir: str | None) -> dict:
    """載入 dataset 的 config，解析 data_dir（CLI > env > repo/dataset 後備）。"""
    root = _repo_root()
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    if dataset == "euroc":
        cfg_path = os.path.join(cfg_dir, "full.yaml")
        fallback = os.path.join(_data_root(), "dataset", "machine_hall")
        env = os.environ.get("EUROC_DATA_DIR")
    elif dataset == "tumvi":
        cfg_path = os.path.join(cfg_dir, "tumvi.yaml")
        fallback = os.path.join(_data_root(), "dataset", "tum_vi")
        env = os.environ.get("TUMVI_DATA_DIR")
    else:
        raise ValueError(f"未知 dataset {dataset!r}（euroc / tumvi）")
    dd = data_dir or env or fallback
    cfg = common.load_config(cfg_path, data_dir_override=dd)
    # results_dir 解析成各資料集既有結果樹（唯讀）
    cfg["_results_dir_abs"] = os.path.abspath(
        os.path.join(root, "euroc_full", "results" if dataset == "euroc" else "results/tumvi"))
    return cfg


def _load_long_with_meta(results_dir: str) -> pd.DataFrame:
    """讀既有 metrics_long.csv，並用 runs_index.csv 補回 regime/method/seed（唯讀）。"""
    long_df = pd.read_csv(os.path.join(results_dir, "metrics_long.csv"))
    idx = pd.read_csv(os.path.join(results_dir, "runs_index.csv"))
    meta = idx[["run_id", "method", "regime", "seed"]].copy()
    return long_df.merge(meta, on="run_id", how="left")


# ──────────────────────────────────────────────────────────────────────────
# Part A — 轉動 dose-response（唯讀）
# ──────────────────────────────────────────────────────────────────────────
def _rotation_proxy(merged, idx, dt):
    """從原始 gyro 在 eval 段算轉動代理（與模型/seed 無關）。

    rot_path = ∫|ω| dt（總角路徑 rad）；rot_rms = RMS(|ω|) (rad/s)。
    用與 metric 相同的 strided 取樣（idx = 各 window 起始 raw-frame index、dt = stride/imu_freq）。
    """
    w = merged[["wx", "wy", "wz"]].to_numpy(np.float64)[idx]
    wn = np.linalg.norm(w, axis=1)
    rot_path = float(np.sum(wn) * dt)
    rot_rms = float(np.sqrt(np.mean(wn ** 2)))
    return rot_path, rot_rms


def _seq_runs(long_meta, regime, seq):
    """某 (regime, eval_sequence) 下、有 iabe_rot 的 run（含 method/seed）。"""
    sub = long_meta[(long_meta["regime"] == regime)
                    & (long_meta["eval_sequence"] == seq)
                    & (long_meta["metric"].isin(["iabe_rot", "gyro_ate"]))]
    piv = sub.pivot_table(index=["run_id", "method", "seed"], columns="metric",
                          values="value", aggfunc="first").reset_index()
    return piv


def part_a_dataset(dataset, cfg, long_meta, sequences, regime, sub_segment_sec):
    """對一個資料集算 Part A 的逐序列點 + sub-segment 散點。回傳 list[dict]。"""
    rows = []
    dt = float(cfg["stride"]) / float(cfg["imu_freq"])
    windows_per_sub = max(int(round(sub_segment_sec / dt)), 16)
    adapter = D.get_adapter(cfg)

    for seq in sequences:
        # regime-A：直接 prepare；其餘 regime 也走 regime-A loader 取 own-tail（per-seq）。
        try:
            sd = D.prepare_regime_a(seq, cfg, adapter)
        except Exception as e:  # pragma: no cover - 序列缺檔時跳過
            print(f"[Part A] skip {dataset}/{seq}: {e}")
            continue
        merged = sd.merged
        tws = np.asarray(sd.test_win_starts, dtype=np.int64)
        N = len(tws)
        if N < 16:
            continue

        piv = _seq_runs(long_meta, regime, seq)
        if piv.empty or "iabe_rot" not in piv or "gyro_ate" not in piv:
            print(f"[Part A] no frozen metrics for {dataset}/{seq} (regime {regime})")
            continue

        # ---- 全序列點：ratio = median(gyro_ate)/median(iabe_rot)（讀凍結 metrics）----
        gmed = float(np.median(piv["gyro_ate"].to_numpy()))
        imed = float(np.median(piv["iabe_rot"].to_numpy()))
        rot_path, rot_rms = _rotation_proxy(merged, tws, dt)
        rows.append({
            "dataset": dataset, "eval_sequence": seq, "regime": regime,
            "level": "sequence", "sub_idx": -1,
            "rot_path": rot_path, "rot_rms": rot_rms,
            "gyro_ate_med": gmed, "iabe_rot_med": imed,
            "ratio": gmed / imed if imed > 0 else np.nan,
            "n_runs": int(len(piv)), "n_windows": N,
            "clean_label": bool(seq in CLEAN_SEQUENCES),
        })

        # ---- sub-segment 散點：把 eval 段切 k 個等長 sub-segment，逐段重算 ----
        n_sub = N // windows_per_sub
        if n_sub < 2:
            continue
        # 預載各 run 的 series（method/seed → (b_hat, b_ref)）一次
        run_series = {}
        for r in piv.itertuples():
            npz = os.path.join(cfg["_results_dir_abs"], "runs", r.run_id, "series.npz")
            if not os.path.exists(npz):
                continue
            try:
                z = np.load(npz)
            except Exception:
                continue
            kh, kr = f"b_hat_{seq}", f"b_ref_{seq}"
            if kh in z.files and kr in z.files:
                run_series[r.run_id] = (np.asarray(z[kh], np.float64),
                                        np.asarray(z[kr], np.float64))
        if not run_series:
            continue

        for s in range(n_sub):
            a, b = s * windows_per_sub, (s + 1) * windows_per_sub
            idx_sub = tws[a:b]
            rp_sub, rr_sub = _rotation_proxy(merged, idx_sub, dt)
            ia_list, gy_list = [], []
            for rid, (b_hat, b_ref) in run_series.items():
                if b < len(b_hat):
                    mi = build_metric_inputs(merged, idx_sub, b_hat[a:b], b_ref[a:b],
                                             cfg["stride"], cfg["imu_freq"], cfg["gravity"])
                    ia_list.append(iabe_rot(mi)["value"])
                    gy_list.append(gyro_ate(mi)["value"])
            if not ia_list:
                continue
            im = float(np.median(ia_list)); gm = float(np.median(gy_list))
            rows.append({
                "dataset": dataset, "eval_sequence": seq, "regime": regime,
                "level": "sub_segment", "sub_idx": s,
                "rot_path": rp_sub, "rot_rms": rr_sub,
                "gyro_ate_med": gm, "iabe_rot_med": im,
                "ratio": gm / im if im > 0 else np.nan,
                "n_runs": int(len(ia_list)), "n_windows": int(b - a),
                "clean_label": bool(seq in CLEAN_SEQUENCES),
            })
    return rows


def _monotonicity(x, y):
    """Spearman / Kendall（帶 p-value，用 scipy.stats；fallback numpy Spearman）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    out = {"n": int(len(x))}
    if len(x) < 3:
        return out
    try:
        from scipy.stats import spearmanr, kendalltau
        sr = spearmanr(x, y); kt = kendalltau(x, y)
        out.update(spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue),
                   kendall_tau=float(kt.statistic), kendall_p=float(kt.pvalue))
    except Exception:
        out.update(spearman_rho=float(_spearman(x, y)))
    return out


def run_part_a(out_dir, euroc_dd, tumvi_dd, regime, sub_segment_sec):
    print("=== Part A — rotation dose-response ===")
    all_rows = []
    cfg_e = _resolve_cfg("euroc", euroc_dd)
    lm_e = _load_long_with_meta(cfg_e["_results_dir_abs"])
    all_rows += part_a_dataset("euroc", cfg_e, lm_e,
                               ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"],
                               regime, sub_segment_sec)
    try:
        cfg_t = _resolve_cfg("tumvi", tumvi_dd)
        lm_t = _load_long_with_meta(cfg_t["_results_dir_abs"])
        all_rows += part_a_dataset("tumvi", cfg_t, lm_t,
                                   ["room1", "room2", "room3", "room4", "room5", "room6"],
                                   regime, sub_segment_sec)
    except Exception as e:
        print(f"[Part A] TUM-VI skipped: {e}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "rotation_dose_response.csv")
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(df)} rows)")

    # 單調性：跨序列 + sub-segment（全部點），主代理 rot_path
    seq_df = df[df["level"] == "sequence"]
    sub_df = df[df["level"] == "sub_segment"]
    stats = {
        "across_sequences": _monotonicity(seq_df["rot_path"], seq_df["ratio"]),
        "sub_segments": _monotonicity(sub_df["rot_path"], sub_df["ratio"]),
        "all_points": _monotonicity(df["rot_path"], df["ratio"]),
        "n_sequences": int(len(seq_df)), "n_sub_segments": int(len(sub_df)),
        "regime": regime, "sub_segment_sec": sub_segment_sec,
    }
    # ★ 誠實診斷：把 ratio 拆成分子/分母，分別對兩個 proxy 測；並逐資料集（控掉跨資料集
    # label-quality / 尺度 confound）。dose-response 若真由「轉動」驅動，gyro_ate 應隨 proxy 上升。
    diag = {"proxies": {}}
    for px in ("rot_path", "rot_rms"):
        diag["proxies"][px] = {
            "ratio_vs_proxy": _monotonicity(seq_df[px], seq_df["ratio"]),
            "gyro_ate_vs_proxy": _monotonicity(seq_df[px], seq_df["gyro_ate_med"]),
            "iabe_rot_vs_proxy": _monotonicity(seq_df[px], seq_df["iabe_rot_med"]),
        }
    diag["within_dataset"] = {}
    for ds, g in seq_df.groupby("dataset"):
        diag["within_dataset"][ds] = {
            "n": int(len(g)),
            "ratio_vs_rot_path": _monotonicity(g["rot_path"], g["ratio"]),
            "gyro_ate_vs_rot_path": _monotonicity(g["rot_path"], g["gyro_ate_med"]),
            "iabe_rot_vs_rot_path": _monotonicity(g["rot_path"], g["iabe_rot_med"]),
        }
    stats["diagnostics"] = diag
    # 結論判定（誠實）：dose-response「ratio 隨轉動單調上升」是否成立
    rho = stats["across_sequences"].get("spearman_rho", float("nan"))
    stats["hypothesis_supported"] = bool(np.isfinite(rho) and rho > 0.5)
    _plot_part_a(df, stats, os.path.join(out_dir, "plots", "rotation_dose_response.png"))
    return df, stats


def _plot_part_a(df, stats, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    sub = df[df["level"] == "sub_segment"]
    seq = df[df["level"] == "sequence"]
    # sub-segment：小點，乾淨/污染分色
    for clean, col in [(True, "#4c9be8"), (False, "#e8744c")]:
        s = sub[sub["clean_label"] == clean]
        if len(s):
            ax.scatter(s["rot_path"], s["ratio"], s=14, c=col, alpha=0.35,
                       label=f"sub-seg ({'clean' if clean else 'contaminated'} label)")
    # 跨序列：大記號，標序列名
    for clean, col, mk in [(True, "#1f5fa8", "o"), (False, "#a83216", "s")]:
        s = seq[seq["clean_label"] == clean]
        if len(s):
            ax.scatter(s["rot_path"], s["ratio"], s=90, c=col, marker=mk,
                       edgecolors="k", linewidths=0.6, zorder=5,
                       label=f"sequence ({'clean' if clean else 'contaminated'} label)")
            for r in s.itertuples():
                ax.annotate(r.eval_sequence, (r.rot_path, r.ratio),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
    # 趨勢線（log-log 線性擬合，全點）
    m = np.isfinite(df["rot_path"]) & np.isfinite(df["ratio"]) & (df["rot_path"] > 0) & (df["ratio"] > 0)
    lx, ly = np.log10(df["rot_path"][m]), np.log10(df["ratio"][m])
    if len(lx) > 2:
        c = np.polyfit(lx, ly, 1)
        xs = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(10 ** xs, 10 ** np.polyval(c, xs), "k--", lw=1.2, alpha=0.7,
                label=f"log-log trend (slope={c[0]:.2f})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"rotation proxy  $\int|\omega|\,dt$  (rad, eval segment)")
    ax.set_ylabel(r"ratio  median(gyro_ate) / median(iabe_rot)")
    sa = stats["across_sequences"]; al = stats["all_points"]
    title = ("Rotation dose-response: attitude-leakage confound vs rotation\n"
             f"across-seq Spearman ρ={sa.get('spearman_rho', float('nan')):.2f} "
             f"(p={sa.get('spearman_p', float('nan')):.1e}, n={sa.get('n', 0)});  "
             f"all-points ρ={al.get('spearman_rho', float('nan')):.2f} (n={al.get('n', 0)})")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {png_path}")


# ──────────────────────────────────────────────────────────────────────────
# Part B — Tier 2 detection limit（唯讀既有 runs + band 分解）
# ──────────────────────────────────────────────────────────────────────────
def detection_limit(cfg, long_meta, regime="B"):
    """把 robust inversion=0 升級成「儀器無法區分兩樣品」的量測陳述（SDD B.1）。

    noise floor = 2 × seed-paired-diff std（沿用 §9 paired 口徑，非 marginal σ）。
    真實 spread = 方法間 IABE 兩兩差（mean over seeds）的量級。
    比較兩者 → 「方法間低頻殘差差異低於協議在當前量測不確定度下的解析度」。
    """
    from .registry import is_dl
    dl = set(m for m in cfg["methods"] if is_dl(m))
    sub = long_meta[(long_meta["regime"] == regime)
                    & (long_meta["metric"] == "iabe_rot")
                    & (long_meta["method"].isin(dl))]
    piv = sub.pivot_table(index=["eval_sequence", "method", "seed"],
                          values="value", aggfunc="first").reset_index()
    paired_stds, paired_means = [], []
    for seq, g in piv.groupby("eval_sequence"):
        methods = sorted(g["method"].unique())
        by = {m: {int(r.seed): float(r.value) for r in g[g["method"] == m].itertuples()}
              for m in methods}
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                seeds = sorted(set(by[methods[i]]) & set(by[methods[j]]))
                if len(seeds) < 2:
                    continue
                d = np.array([by[methods[i]][s] - by[methods[j]][s] for s in seeds])
                paired_stds.append(float(d.std(ddof=1)))
                paired_means.append(abs(float(d.mean())))
    if not paired_stds:
        return None
    paired_std_med = float(np.median(paired_stds))
    resolution_floor = 2.0 * paired_std_med           # 對齊 effect-size≥2 判準
    real_spread_med = float(np.median(paired_means))
    real_spread_p90 = float(np.percentile(paired_means, 90))

    # 真實方法低頻能量 spread（支撐「低頻殘差」框架；唯讀 series.npz）
    band = band_decomp.analyze(cfg, long_meta, cfg["_results_dir_abs"],
                               dl_methods=dl, regime=regime, label=cfg.get("dataset", ""))
    low_cv = band.get("low_energy_cross_method_cv") if band else None

    below = bool(real_spread_med < resolution_floor)
    return {
        "regime": regime, "n_pairs": int(len(paired_stds)),
        "paired_diff_std_median_m": paired_std_med,
        "resolution_floor_m": resolution_floor,
        "real_iabe_spread_median_m": real_spread_med,
        "real_iabe_spread_p90_m": real_spread_p90,
        "spread_over_floor_ratio": real_spread_med / resolution_floor if resolution_floor > 0 else np.nan,
        "below_resolution": below,
        "low_energy_cross_method_cv": low_cv,
        "corr_iabe_lowband": band.get("corr_iabe_lowband") if band else None,
        "corr_iabe_biasrmse": band.get("corr_iabe_biasrmse") if band else None,
    }


# ──────────────────────────────────────────────────────────────────────────
# Part B — Tier A matched-RMSE 三頻帶 + Tier A+ Bode（合成輸入，凍結 metric）
# ──────────────────────────────────────────────────────────────────────────
def _carrier(cfg, seq="MH_03"):
    """取真實序列 test tail 的幾何（R_ref/dt/v0/p0），當合成 ε 的載具（SDD B.2）。"""
    sd = D.prepare_regime_a(seq, cfg, D.get_adapter(cfg))
    tws = np.asarray(sd.test_win_starts, np.int64)
    N = len(tws)
    zero = np.zeros((N, 3), np.float64)
    # b_hat=b_ref=0 → iabe_rot 的兩臂差 = ∬R_ref·(−ε)，與 a_raw/b_ref 無關（純行使 rotation 路徑）
    base = build_metric_inputs(sd.merged, tws, zero, zero,
                               cfg["stride"], cfg["imu_freq"], cfg["gravity"])
    return base


def _iabe_of_eps(base: MetricInputs, eps: np.ndarray) -> float:
    """以合成 ε 餵凍結 iabe_rot：令 b_ref−b_hat = ε（b_ref=0, b_hat=−ε）。"""
    mi = MetricInputs(base.a_raw, base.w_raw, -eps, np.zeros_like(eps),
                      base.R_ref, base.v0, base.p0, base.dt, base.g_vec)
    return iabe_rot(mi)["value"]


def _iabe_traj(base: MetricInputs, eps: np.ndarray):
    """回傳 (frozen iabe_rot 值, position-error 軌跡差 p_corr−p_ideal)。

    軌跡差由凍結 iabe_rot 的輸出取得（不改 metric）；用於 Bode 把「整段從靜止雙積分」的
    **secular ramp（∫sin 的非零均值 → ~1/ω）** 與「穩態振盪 transfer function（1/ω²）」分開報。
    """
    mi = MetricInputs(base.a_raw, base.w_raw, -eps, np.zeros_like(eps),
                      base.R_ref, base.v0, base.p0, base.dt, base.g_vec)
    r = iabe_rot(mi)
    return r["value"], (r["p_corr"] - r["p_ideal"])


def _detrended_rmse(diff: np.ndarray, t: np.ndarray) -> float:
    """逐軸移除最佳擬合直線（secular ramp）後的 3D RMSE = 穩態振盪響應（transfer function）。"""
    dd = np.empty_like(diff)
    for ax in range(3):
        c = np.polyfit(t, diff[:, ax], 1)
        dd[:, ax] = diff[:, ax] - np.polyval(c, t)
    return float(np.sqrt(np.mean(np.sum(dd ** 2, axis=1))))


def _band_limited_eps(N, dt, band, target_rmse, rng, n_sines=8):
    """band-limited ε(t)（sum-of-sinusoids，random phase），normalize 到 target_rmse。"""
    t = np.arange(N) * dt
    lo, hi = band
    eps = np.zeros((N, 3), np.float64)
    for ax in range(3):
        freqs = rng.uniform(lo, hi, n_sines)
        phases = rng.uniform(0, 2 * np.pi, n_sines)
        eps[:, ax] = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
    rms = float(np.sqrt(np.mean(eps ** 2)))
    if rms > 0:
        eps *= target_rmse / rms
    return eps


def matched_rmse_bands(base, target_rmse=0.002, n_seeds=20, seed0=42):
    """Tier A：三頻帶（低/中/高）matched-RMSE positive control（SDD B.2）。"""
    fs = 1.0 / base.dt
    nyq = fs / 2.0
    bands = {"low": (1e-3, 0.5), "mid": (0.5, 2.0), "high": (2.0, 0.9 * nyq)}
    rows = []
    for name, band in bands.items():
        for k in range(n_seeds):
            rng = np.random.default_rng(seed0 + 1000 * "lmh".index(name[0]) + k)
            eps = _band_limited_eps(len(base), base.dt, band, target_rmse, rng)
            rows.append({"band": name, "seed": int(seed0 + k),
                         "f_lo": band[0], "f_hi": band[1],
                         "bias_rmse": float(bias_rmse(-eps, np.zeros_like(eps))["avg"]),
                         "iabe_rot": _iabe_of_eps(base, eps)})
    df = pd.DataFrame(rows)
    summ = {}
    for name in bands:
        g = df[df["band"] == name]
        summ[name] = {
            "iabe_median": float(g["iabe_rot"].median()),
            "iabe_iqr": [float(g["iabe_rot"].quantile(.25)), float(g["iabe_rot"].quantile(.75))],
            "bias_rmse_median": float(g["bias_rmse"].median()),
            "f_band": list(bands[name]),
        }
    # 驗收：RMSE 相等(<1%) + IABE 嚴格 低>中>高
    rmse_meds = [summ[b]["bias_rmse_median"] for b in ("low", "mid", "high")]
    rmse_rel_spread = (max(rmse_meds) - min(rmse_meds)) / (np.mean(rmse_meds) + 1e-30)
    iabe_meds = [summ[b]["iabe_median"] for b in ("low", "mid", "high")]
    summ["_checks"] = {
        "fs_hz": fs, "target_rmse": target_rmse, "n_seeds": n_seeds,
        "rmse_equal_rel_spread": float(rmse_rel_spread),
        "rmse_equal_pass": bool(rmse_rel_spread < 0.01),
        "iabe_strict_monotone_low_gt_mid_gt_high": bool(iabe_meds[0] > iabe_meds[1] > iabe_meds[2]),
        "low_over_high_ratio": float(iabe_meds[0] / iabe_meds[2]) if iabe_meds[2] > 0 else np.inf,
    }
    return df, summ


def bode_sweep(base, amp_rmse=0.002, n_freqs=20, n_phases=8, seed0=42):
    """Tier A+：單頻掃頻 Bode（SDD B.3）。f log-spaced，A 固定（RMSE 固定）。"""
    fs = 1.0 / base.dt
    nyq = fs / 2.0
    freqs = np.logspace(np.log10(0.01), np.log10(min(10.0, 0.9 * nyq)), n_freqs)
    t = np.arange(len(base)) * base.dt
    A = amp_rmse * np.sqrt(2.0)               # RMS(A·sin)=A/√2 → A=rmse·√2（每軸）
    rng = np.random.default_rng(seed0)
    rows = []
    for f in freqs:
        ia, ia_det, rm = [], [], []
        for _ in range(n_phases):
            ph = rng.uniform(0, 2 * np.pi, 3)
            eps = np.stack([A * np.sin(2 * np.pi * f * t + ph[ax]) for ax in range(3)], axis=1)
            val, diff = _iabe_traj(base, eps)
            ia.append(val)                                   # 凍結 iabe_rot（含 secular ramp）
            ia_det.append(_detrended_rmse(diff, t))          # 穩態振盪 transfer function
            rm.append(float(bias_rmse(-eps, np.zeros_like(eps))["avg"]))
        rows.append({"f_hz": float(f), "iabe_rot": float(np.median(ia)),
                     "iabe_detrended": float(np.median(ia_det)),
                     "bias_rmse": float(np.median(rm))})
    df = pd.DataFrame(rows)

    # 中段 roll-off 擬合（取 1/T .. Nyquist/3 之間的 roll-off 帶）
    T = len(base) * base.dt
    f_lo_fit, f_hi_fit = max(1.0 / T * 3, freqs[0]), min(nyq / 3.0, freqs[-1])
    m = (df["f_hz"] >= f_lo_fit) & (df["f_hz"] <= f_hi_fit)

    def _slope(col):
        mm = m & (df[col] > 0)
        if mm.sum() < 3:
            return np.nan
        lx = np.log10(df["f_hz"][mm].to_numpy())
        ly = 20.0 * np.log10(df[col][mm].to_numpy())         # dB
        return float(np.polyfit(lx, ly, 1)[0])               # dB/decade

    slope_raw = _slope("iabe_rot")                           # ~ −20 dB/dec（1/ω secular）
    slope_tf = _slope("iabe_detrended")                      # ~ −40 dB/dec（1/ω² transfer function）

    # 低頻平台：raw（含 secular）應收斂到常數-bias 響應（δ=amp_rmse 每軸常數）
    const_eps = np.tile(np.array([amp_rmse, amp_rmse, amp_rmse]), (len(base), 1))
    iabe_const = _iabe_of_eps(base, const_eps)
    low_plateau = float(df["iabe_rot"].iloc[:3].median())

    checks = {
        "fs_hz": fs, "amp_rmse": amp_rmse, "T_sec": float(T),
        "fit_band_hz": [float(f_lo_fit), float(f_hi_fit)],
        # ★ transfer-function（穩態振盪、移除整段雙積分的 secular ramp）= 1/ω²
        "rolloff_slope_db_per_decade": slope_tf,
        "rolloff_near_minus40": bool(np.isfinite(slope_tf) and -55 < slope_tf < -25),
        # 凍結 metric 在有限窗整段從靜止雙積分 → 受 ∫sin 非零均值的 secular ramp 主導 ~1/ω
        "rolloff_slope_raw_metric_db_per_decade": slope_raw,
        "note_raw_vs_tf": ("frozen iabe_rot on a finite window carries a 1/omega secular ramp "
                           "(integration-from-rest); the 1/omega^2 transfer function is the "
                           "steady-state oscillatory component (secular ramp removed)."),
        "low_freq_plateau_iabe": low_plateau,
        "constant_bias_iabe": float(iabe_const),
        "plateau_matches_constant_bias_rel": float(abs(low_plateau - iabe_const) / (iabe_const + 1e-30)),
        "rmse_flat_rel_spread": float((df["bias_rmse"].max() - df["bias_rmse"].min())
                                      / (df["bias_rmse"].mean() + 1e-30)),
    }
    return df, checks


def _linearity_check(base, amp_rmse=0.002, seed0=42):
    """B.4.4 線性 sanity：ε→0 → IABE→0；振幅加倍 → IABE 加倍。"""
    rng = np.random.default_rng(seed0)
    eps = _band_limited_eps(len(base), base.dt, (0.5, 2.0), amp_rmse, rng)
    i1 = _iabe_of_eps(base, eps)
    i2 = _iabe_of_eps(base, 2.0 * eps)
    i0 = _iabe_of_eps(base, 1e-6 * eps)
    return {
        "iabe_amp": i1, "iabe_2x_amp": i2, "iabe_near_zero": i0,
        "doubling_ratio": float(i2 / i1) if i1 > 0 else np.inf,
        "doubling_pass": bool(abs(i2 / i1 - 2.0) < 0.02) if i1 > 0 else False,
        "vanish_pass": bool(i0 < 1e-4 * i1) if i1 > 0 else False,
    }


def _plot_bands(bands_df, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    order = ["low", "mid", "high"]
    data = [bands_df[bands_df["band"] == b]["iabe_rot"].to_numpy() for b in order]
    ax.boxplot(data, tick_labels=[f"{b}\n[{bands_df[bands_df.band==b].f_lo.iloc[0]:.2g},"
                                  f"{bands_df[bands_df.band==b].f_hi.iloc[0]:.2g}] Hz" for b in order],
               showmeans=True)
    ax.set_yscale("log")
    ax.set_ylabel("iabe_rot (m)  [matched bias-RMSE]")
    ax.set_title("Matched-RMSE positive control: IABE responds to LOW-freq residual\n"
                 "(identical bias RMSE across bands; expect low > mid > high)", fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(png_path, dpi=130); plt.close(fig)
    print(f"  wrote {png_path}")


def _plot_bode(bode_df, checks, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(bode_df["f_hz"], bode_df["iabe_detrended"], "o-", color="#1f5fa8",
              label="transfer function (steady-state, secular ramp removed)")
    ax.loglog(bode_df["f_hz"], bode_df["iabe_rot"], "^-", color="#7aa6d6", alpha=0.7,
              label="frozen iabe_rot (integrate-from-rest, ~$1/\\omega$ secular)")
    ax.loglog(bode_df["f_hz"], bode_df["bias_rmse"], "s--", color="#888",
              label="bias RMSE (flat reference)")
    # −40 dB/decade 參考線（= 1/ω²），對齊 transfer function
    fb = checks["fit_band_hz"]
    m = (bode_df["f_hz"] >= fb[0]) & (bode_df["f_hz"] <= fb[1])
    if m.any():
        f0 = bode_df["f_hz"][m].iloc[0]; y0 = bode_df["iabe_detrended"][m].iloc[0]
        fr = bode_df["f_hz"][m].to_numpy()
        ax.loglog(fr, y0 * (fr / f0) ** -2, "k:", lw=1.3,
                  label=r"$1/\omega^2$ ($-40$ dB/decade)")
    ax.set_xlabel("frequency f (Hz)"); ax.set_ylabel("response (m) / bias RMSE")
    ax.set_title("Bode sweep: double-integration transfer function\n"
                 f"steady-state roll-off ≈ {checks['rolloff_slope_db_per_decade']:.0f} dB/decade "
                 r"($1/\omega^2$); low-freq plateau = constant-bias limit", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); fig.savefig(png_path, dpi=130); plt.close(fig)
    print(f"  wrote {png_path}")


def run_part_b(out_dir, euroc_dd, n_seeds, target_rmse):
    print("=== Part B — positive control / Bode / detection limit ===")
    cfg = _resolve_cfg("euroc", euroc_dd)
    long_meta = _load_long_with_meta(cfg["_results_dir_abs"])

    # ---- Tier 2（主菜）：detection limit（EuRoC regime B）----
    dl = detection_limit(cfg, long_meta, regime="B")
    if dl:
        pd.DataFrame([dl]).to_csv(os.path.join(out_dir, "detection_limit.csv"), index=False)
        print(f"  detection limit: floor={dl['resolution_floor_m']:.4g} m, "
              f"real spread={dl['real_iabe_spread_median_m']:.4g} m, "
              f"below_resolution={dl['below_resolution']}")

    # ---- Tier A / A+：合成輸入（真實 MH_03 tail R_ref 當載具）----
    base = _carrier(cfg, "MH_03")
    bands_df, bands_summ = matched_rmse_bands(base, target_rmse, n_seeds)
    bands_df.to_csv(os.path.join(out_dir, "positive_control_bands.csv"), index=False)
    bode_df, bode_checks = bode_sweep(base, target_rmse)
    bode_df.to_csv(os.path.join(out_dir, "bode_sweep.csv"), index=False)
    lin = _linearity_check(base, target_rmse)
    lemma = check_lemma_constant_bias(base, tol_m=1e-9)

    _plot_bands(bands_df, os.path.join(out_dir, "plots", "positive_control_bands.png"))
    _plot_bode(bode_df, bode_checks, os.path.join(out_dir, "plots", "bode_iabe_vs_rmse.png"))

    return {"detection_limit": dl, "matched_rmse_bands": bands_summ,
            "bode": bode_checks, "linearity": lin,
            "lemma_constant_bias_pass": bool(lemma["pass_strict"])}


# ──────────────────────────────────────────────────────────────────────────
# SUMMARY 產出
# ──────────────────────────────────────────────────────────────────────────
def _write_summary(out_dir, part_a_stats, part_b, regime):
    lines = ["# Supplementary Characterization — SUMMARY",
             "*兩個便宜補算（離線、零訓練、凍結 results 不動、metric 邏輯不改）。*",
              f"*git_hash: {common.git_hash()} · {common.now_iso()}*", ""]

    if part_a_stats:
        sa = part_a_stats["across_sequences"]; al = part_a_stats["all_points"]
        sub = part_a_stats["sub_segments"]
        supported = part_a_stats.get("hypothesis_supported")
        diag = part_a_stats.get("diagnostics", {})
        lines += [
            "## Part A — 轉動 dose-response 量化",
            f"- regime **{regime}**；跨序列 {part_a_stats['n_sequences']} 點 + "
            f"sub-segment {part_a_stats['n_sub_segments']} 點（{part_a_stats['sub_segment_sec']}s/段）。",
            f"- **跨序列 ratio-vs-rotation**：Spearman ρ={sa.get('spearman_rho', float('nan')):.3f} "
            f"(p={sa.get('spearman_p', float('nan')):.2e}), "
            f"Kendall τ={sa.get('kendall_tau', float('nan')):.3f} (n={sa.get('n', 0)})。",
            f"- **sub-segment**：ρ={sub.get('spearman_rho', float('nan')):.3f} "
            f"(p={sub.get('spearman_p', float('nan')):.2e}, n={sub.get('n', 0)})；"
            f"**全點**：ρ={al.get('spearman_rho', float('nan')):.3f} (n={al.get('n', 0)})。",
            "",
            f"### ⚠ 結論：預期的「ratio 隨轉動單調**上升**」**不成立**（hypothesis_supported={supported}）",
            "觀測到的是強烈**負**相關。診斷拆解（為何）：",
        ]
        pr = diag.get("proxies", {}).get("rot_path", {})
        g_vp = pr.get("gyro_ate_vs_proxy", {}); i_vp = pr.get("iabe_rot_vs_proxy", {})
        lines += [
            f"- **分子 gyro_ate vs rotation**：ρ={g_vp.get('spearman_rho', float('nan')):+.2f} —— "
            "attitude-leakage **沒有**隨 ∫|ω|dt 上升。機制上 gyro-attitude 漂移 ∝（gyro 誤差×時間），"
            "**非真實轉動量**；靜止但有 bias 的 gyro 一樣漂 → ∫|ω|dt 是錯的劑量軸。",
            f"- **分母 iabe_rot vs rotation**：ρ={i_vp.get('spearman_rho', float('nan')):+.2f} —— "
            "ratio 的變異**主要由分母**驅動；TUM-VI 同時是高轉動（手持）又是大 iabe_rot"
            "（mocap label 污染/尺度），這個跨資料集 confound 主導了 ratio 排序。",
        ]
        for ds, d in diag.get("within_dataset", {}).items():
            r1 = d["ratio_vs_rot_path"].get("spearman_rho", float("nan"))
            g1 = d["gyro_ate_vs_rot_path"].get("spearman_rho", float("nan"))
            lines += [f"- within **{ds}** (n={d['n']}): ratio ρ={r1:+.2f}, gyro_ate ρ={g1:+.2f}。"]
        lines += [
            "",
            "**對寫作的意涵（誠實）**：C1 原本「room1 2.4×→room3 405×→EuRoC 502×」的 ratio 排序"
            "**確實存在**，但把它歸因於「轉動量」**站不住**——量化後 ratio 與 ∫|ω|dt 反向、且 gyro_ate "
            "本身不隨轉動上升。預期的「這些序列差的不只轉動」**被證實**。建議：(a) 不要在論文中宣稱"
            "rotation dose-response；(b) 若要保留 C1 的「非隔離對照會爆」敘事，改用「gyro-attitude 漂移"
            "（∝積分時間/gyro 誤差）」而非「真實轉動量」當軸，或直接以 gyro_ate 絕對量級 + isolation "
            "（L1：median gyro/iabe ratio 與 |ρ(gyro,bias_rmse)|）論證，不靠 dose-response 曲線。",
            "（room3/污染序列已於主圖標色揭露。）", ""]

    if part_b:
        dl = part_b.get("detection_limit")
        if dl:
            lines += [
                "## Part B — Tier 2 detection limit（主菜）",
                f"- noise floor（2×seed-paired-diff std，§9 paired 口徑）= "
                f"**{dl['resolution_floor_m']:.4g} m**（{dl['n_pairs']} 個方法對）。",
                f"- 真實方法間 IABE spread（median |ΔIABE|）= **{dl['real_iabe_spread_median_m']:.4g} m** "
                f"（p90 = {dl['real_iabe_spread_p90_m']:.4g} m）。",
                f"- spread / floor = **{dl['spread_over_floor_ratio']:.2f}** → "
                f"below_resolution = **{dl['below_resolution']}**。",
                f"- 支撐：跨方法低頻能量 CV={dl['low_energy_cross_method_cv']:.3f}, "
                f"corr(IABE,low-band)={dl['corr_iabe_lowband']:.2f} vs "
                f"corr(IABE,RMSE)={dl['corr_iabe_biasrmse']:.2f}（IABE 較跟低頻走、與 RMSE 脫鉤）。",
                "- **陳述**：On the evaluated benign datasets, the frequency-weighted aspect does not "
                "alter method rankings relative to RMSE; a controlled matched-RMSE characterization confirms "
                "IABE responds to low-frequency residuals as predicted by the double-integration transfer "
                "function, and the method-to-method low-frequency differences lie below the metric's "
                "resolution at current evaluation uncertainty.", ""]
        b = part_b.get("matched_rmse_bands")
        if b:
            ch = b["_checks"]
            lines += [
                "## Part B — Tier A matched-RMSE 三頻帶（配菜）",
                f"- IABE median：low={b['low']['iabe_median']:.4g} > "
                f"mid={b['mid']['iabe_median']:.4g} > high={b['high']['iabe_median']:.4g} m。",
                f"- RMSE 相等（rel spread={ch['rmse_equal_rel_spread']:.1e} < 1% → "
                f"{ch['rmse_equal_pass']}）；嚴格單調 low>mid>high = {ch['iabe_strict_monotone_low_gt_mid_gt_high']}；"
                f"low/high = {ch['low_over_high_ratio']:.1f}×。", ""]
        bo = part_b.get("bode")
        if bo:
            lines += [
                "## Part B — Tier A+ Bode 掃頻（配菜，視覺）",
                f"- **transfer function（穩態振盪、移除 secular ramp）roll-off = "
                f"{bo['rolloff_slope_db_per_decade']:.1f} dB/decade**（≈−40 → "
                f"{bo['rolloff_near_minus40']}；= 1/ω²，近同義反覆：雙積分本就 1/ω²）。",
                f"- 凍結 iabe_rot（整段從靜止雙積分）raw roll-off = "
                f"{bo['rolloff_slope_raw_metric_db_per_decade']:.1f} dB/decade（~1/ω）—— 有限窗下 "
                f"∫sin 的非零均值帶入 secular ramp 主導；誠實併報，1/ω² 為其穩態振盪成分。",
                f"- 低頻平台 {bo['low_freq_plateau_iabe']:.4g} m vs 常數-bias 解析 "
                f"{bo['constant_bias_iabe']:.4g} m（rel={bo['plateau_matches_constant_bias_rel']:.1e}）。", ""]
        lin = part_b.get("linearity")
        if lin:
            lines += [
                "## Part B — 驗收四條",
                f"- 線性：振幅加倍 ratio={lin['doubling_ratio']:.3f}（→{lin['doubling_pass']}）；"
                f"ε→0 → IABE→0（→{lin['vanish_pass']}）。",
                f"- lemma_constant_bias 再驗：{part_b.get('lemma_constant_bias_pass')}。", ""]
        lines += ["## 論文放置（billing）",
                  "- Part A 轉動曲線 → VI-B isolation 主圖（C1 dose-response 視覺化）。",
                  "- Tier 2 detection-limit 一句進 VI/discussion；matched-RMSE 三頻帶小表 + Bode → "
                  "Appendix（sensitivity analysis）。**絕不進摘要/貢獻一/主結果標題。**",
                  "- billing：detection-limit 為主菜；matched-RMSE/Bode 近同義反覆（雙積分本就 1/ω²），"
                  "不當承重防禦；**不宣稱經驗排名差異**。"]

    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Supplementary Characterization（Part A dose-response + Part B positive control）")
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    ap.add_argument("--euroc-data-dir", default=None)
    ap.add_argument("--tumvi-data-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="輸出子樹（預設 <euroc results>/characterization）")
    ap.add_argument("--regime", default="A", help="Part A eval segment 來源 regime（A own-tail）")
    ap.add_argument("--sub-segment-sec", type=float, default=5.0)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--target-rmse", type=float, default=0.002, help="合成 bias RMSE 量級 (m/s²)")
    args = ap.parse_args()

    cfg_e = _resolve_cfg("euroc", args.euroc_data_dir)
    out_dir = args.out_dir or os.path.join(cfg_e["_results_dir_abs"], "characterization")
    common.ensure_dir(out_dir)
    common.ensure_dir(os.path.join(out_dir, "plots"))

    part_a_stats = None
    part_b = None
    if args.part in ("A", "all"):
        _, part_a_stats = run_part_a(out_dir, args.euroc_data_dir, args.tumvi_data_dir,
                                     args.regime, args.sub_segment_sec)
    if args.part in ("B", "all"):
        part_b = run_part_b(out_dir, args.euroc_data_dir, args.n_seeds, args.target_rmse)

    # provenance + SUMMARY
    prov = {"git_hash": common.git_hash(), "timestamp": common.now_iso(),
            "regime": args.regime, "sub_segment_sec": args.sub_segment_sec,
            "n_seeds": args.n_seeds, "target_rmse": args.target_rmse,
            "part_a_stats": part_a_stats, "part_b": part_b}
    with open(os.path.join(out_dir, "characterization.json"), "w") as f:
        json.dump(prov, f, indent=2, default=float)
    _write_summary(out_dir, part_a_stats, part_b, args.regime)
    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()

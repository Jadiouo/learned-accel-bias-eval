#!/usr/bin/env python3
"""產生主文 Section VI-D 用的 characterization 圖（IEEE 單欄）。

Panel (a) Bode 掃頻：transfer function、凍結 metric、1/w^2 漸近線，
          並標出 1/T（有限窗解析度）與常數偏差的解析上限 —— 這兩條是
          「為何 naive 1/w^2 預測不適用」的視覺證據。
Panel (b) matched-RMSE 三頻帶：實測 median + bootstrap CI95，並疊上
          有限窗離散算子的預測值。

用法: python -m euroc_full.plot_characterization_paper
輸出: paper/latex/Paper1A_Figure_characterization.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── 常數（與 characterization.json / item0 對帳結果一致） ────────────────
T_SEC = 19.6
F_WIN = 1.0 / T_SEC                      # 0.0510 Hz，有限窗基頻
F_BOXCAR = 1.0                           # 1 s bias 平均窗的第一零點 (W=200 @ 200 Hz)
CONST_ANALYTIC_I = 0.29757               # ||d||T^2/(2 sqrt5), C = I
CONST_ROT = 0.22996                      # rotation-aware 數值
TARGET_RMSE = 0.002
BANDS = [("low", 1e-3, 0.5), ("mid", 0.5, 2.0), ("high", 2.0, 18.0)]
SIM_CEQI = {"low": 0.0703036, "mid": 0.00591203, "high": 0.00100926}

C_TF, C_RAW, C_RMSE, C_PRED = "#1f4e79", "#8ab4d8", "#7a7a7a", "#c1660a"


def _root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    root = _root()
    cdir = os.path.join(root, "euroc_full", "results", "characterization")
    bode = pd.read_csv(os.path.join(cdir, "bode_sweep.csv"))
    pcb = pd.read_csv(os.path.join(cdir, "positive_control_bands.csv"))

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.2,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.linewidth": 0.6, "grid.linewidth": 0.35,
        "lines.linewidth": 1.1, "font.family": "serif",
    })
    # panel (a) 進主文、panel (b) 進 supplementary —— 分成兩個獨立檔案
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.45))
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.30))

    # ───────────────────────── Panel (a) Bode ─────────────────────────
    f = bode["f_hz"].to_numpy()
    ax1.loglog(f, bode["iabe_detrended"], "o-", color=C_TF, ms=2.6,
               label=r"transfer function ($-38.3$ dB/dec)", zorder=5)
    ax1.loglog(f, bode["iabe_rot"], "^-", color=C_RAW, ms=2.6,
               label=r"frozen IABE (with secular ramp)", zorder=4)
    ax1.loglog(f, bode["bias_rmse"], "s--", color=C_RMSE, ms=2.2,
               label="bias RMSE (flat)", zorder=3)

    # 1/w^2 漸近線（錨在中頻）
    anchor = np.argmin(np.abs(f - 0.2))
    ref = bode["iabe_detrended"].iloc[anchor] * (f / f[anchor]) ** -2
    ax1.loglog(f, ref, ":", color="k", lw=0.8, label=r"$1/\omega^2$ ($-40$ dB/dec)")

    # 有限窗解析度 1/T —— 左側為 secular 主導區
    ax1.axvline(F_WIN, color="#b03060", lw=0.8, ls="-.", zorder=2)
    ax1.axvspan(f.min(), F_WIN, color="#b03060", alpha=0.07, zorder=1)
    ax1.annotate(r"$1/T$", xy=(F_WIN, 1.05), xytext=(0.115, 1.05),
                 fontsize=6.2, color="#b03060", va="center", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#b03060", lw=0.7))
    ax1.text(f.min() * 1.06, 1.05, "secular-\ndominated", fontsize=5.6,
             color="#b03060", va="center", ha="left", linespacing=0.95)

    # boxcar 第一零點：1 秒 bias 平均窗 → 受測物殘差在此之上被構造性衰減
    ax1.axvline(F_BOXCAR, color="#2e7d32", lw=0.9, ls=":", zorder=2)
    ax1.axvspan(F_BOXCAR, f.max(), color="#2e7d32", alpha=0.05, zorder=1)
    ax1.annotate("1 Hz\n(boxcar null)", xy=(F_BOXCAR, 1.6e-4),
                 xytext=(2.0, 1.6e-4), fontsize=5.8, color="#2e7d32",
                 va="center", ha="left", linespacing=0.95,
                 arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.7))

    # 常數偏差上限
    ax1.axhline(CONST_ANALYTIC_I, color=C_PRED, lw=0.9, ls="--", zorder=2)
    ax1.text(9.5, CONST_ANALYTIC_I * 1.3, r"constant-bias limit $\|\delta\|T^2/(2\sqrt{5})$",
             fontsize=5.8, color=C_PRED, ha="right", va="bottom")

    ax1.set_xlabel("frequency $f$ (Hz)")
    ax1.set_ylabel("response (m)")
    ax1.set_ylim(1e-5, 2.2)
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.95,
               handlelength=1.9, borderpad=0.35, labelspacing=0.3)

    # ─────────────────── Panel (b) matched-RMSE 三頻帶 ───────────────────
    rng = np.random.default_rng(0)
    xs, meds, los, his, preds, labels = [], [], [], [], [], []
    for i, (name, lo, hi) in enumerate(BANDS):
        g = pcb[pcb["band"] == name]["iabe_rot"].to_numpy()
        bm = np.median(rng.choice(g, size=(20000, len(g)), replace=True), axis=1)
        ci = np.percentile(bm, [2.5, 97.5])
        xs.append(i); meds.append(np.median(g))
        los.append(np.median(g) - ci[0]); his.append(ci[1] - np.median(g))
        preds.append(SIM_CEQI[name])
        labels.append(f"{name}\n[{lo:g}, {hi:g}] Hz")

    # 受測物可達區（boxcar 零點以下）
    ax2.axvspan(-0.46, 0.5, color="#2e7d32", alpha=0.06, zorder=0)
    ax2.text(-0.40, 0.20, "reachable by\nmethods under test", fontsize=5.7,
             color="#2e7d32", va="top", ha="left", linespacing=1.0)

    ax2.errorbar(xs, meds, yerr=[los, his], fmt="o", color=C_TF, ms=5,
                 capsize=3, lw=1.1, label="observed (rotation-aware)", zorder=5)
    ax2.plot(xs, preds, "D", color=C_PRED, ms=4.5, mfc="none", mew=1.2,
             label="finite-window operator ($C\\!=\\!I$)", zorder=6)
    ax2.axhline(TARGET_RMSE, color=C_RMSE, ls="--", lw=0.9, zorder=2)
    ax2.text(0.62, TARGET_RMSE * 1.16, "bias RMSE (identical, 0.0% spread)",
             fontsize=5.9, color=C_RMSE, ha="left", va="bottom")

    # low->high 的分離度：畫成左側括號，避免與資料點重疊
    xb = -0.30
    ax2.plot([xb, xb], [meds[2], meds[0]], color="#333", lw=0.8, zorder=4)
    for yv in (meds[0], meds[2]):
        ax2.plot([xb, xb + 0.055], [yv, yv], color="#333", lw=0.8, zorder=4)
    ax2.text(xb + 0.09, np.sqrt(meds[0] * meds[2]),
             "$70.6\\times$\n[49.3, 100.9]", fontsize=6.2, va="center",
             ha="left", linespacing=1.1)

    ax2.set_yscale("log")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.46, 2.46)
    ax2.set_ylim(6e-4, 0.30)
    ax2.set_ylabel("IABE (m)")
    ax2.grid(True, which="both", axis="y", alpha=0.22)
    h, l = ax2.get_legend_handles_labels()
    order = [l.index("observed (rotation-aware)"),
             l.index("finite-window operator ($C\\!=\\!I$)")]
    ax2.legend([h[i] for i in order], [l[i] for i in order], loc="upper right",
               frameon=True, framealpha=0.95, handlelength=1.6,
               borderpad=0.35, labelspacing=0.3)

    outdir = os.path.join(root, "paper", "latex")
    fig1.tight_layout(pad=0.35)
    out = os.path.join(outdir, "Paper1A_Figure_bode.png")
    fig1.savefig(out, dpi=400, bbox_inches="tight")
    print("saved:", out)
    fig2.tight_layout(pad=0.35)
    out2 = os.path.join(outdir, "Paper1A_Figure_bands.png")
    fig2.savefig(out2, dpi=400, bbox_inches="tight")
    print("saved:", out2)
    for i, (name, lo, hi) in enumerate(BANDS):
        print(f"  {name:>5}: obs={meds[i]:.6f} "
              f"CI95=[{meds[i]-los[i]:.6f}, {meds[i]+his[i]:.6f}] "
              f"pred={preds[i]:.6f}")


if __name__ == "__main__":
    main()

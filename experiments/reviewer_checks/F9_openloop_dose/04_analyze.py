# -*- coding: utf-8 -*-
"""Step 4. Pooled curves, uncertainty, crossings (0.8 / 0.9), leakage table, figure. Reads dose_counts.csv / dose_chunks.npz / ops.pkl."""
import os, sys, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import f9_lib as F
H0 = F.H0
Z = np.load(HERE + "/dose_chunks.npz"); CH = Z["CH"]; MODES = list(Z["modes"]); SRCI = list(Z["sources"]); GRID = Z["grid"]; UN = [tuple(u.split("|")) for u in Z["units"]]
NCH, NPAIR = CH.shape[-1], 100
ncase = len(UN); seqs = [u[1] for u in UN]
D = pickle.load(open(HERE + "/ops.pkl", "rb")); OPS, RUNS = D["OPS"], D["RUNS"]
pool = np.array([r["rmse_avg_ref"] for r in RUNS]); PQ_DL = dict(median=float(np.median(pool)), p95=float(np.quantile(pool, .95)), max=float(pool.max()))   # 990 DL runs only
PQ = dict(median=2.8e-3, p95=1.05e-2, max=5.8e-2)     # F3 / paper pool (1026 runs incl. baselines): the values drawn on the figure
TH_S15 = 1e-2
GRP = {"nine cases": list(range(ncase)), "regime A": [i for i, u in enumerate(UN) if u[0] == "A"], "regime B": [i for i, u in enumerate(UN) if u[0] == "B"]}
rng = np.random.default_rng(20261126)

def counts(mode, src):            # (case, grid) agree counts, n per case
    return CH[MODES.index(mode), :, SRCI.index(src)].sum(-1).astype(float), np.full(ncase, NCH * NPAIR, float)
def curve(mode, src, idx):        # pooled agreement over the cases in idx
    a, n = counts(mode, src); return a[idx].sum(0) / n[idx].sum()
def jk(mode, src, idx):
    """point, jackknife-t lo, hi (leave-one-sequence-out, df = G-1), per grid point (vectorised copy of H0.jackknife)"""
    from scipy import stats
    a, n = counts(mode, src); cl = np.array(seqs); idx = np.array(idx); labs = sorted(set(cl[idx])); Gc = len(labs)
    pt = a[idx].sum(0) / n[idx].sum()
    th = np.array([a[[i for i in idx if cl[i] != g]].sum(0) / n[[i for i in idx if cl[i] != g]].sum() for g in labs])
    se = np.sqrt((Gc - 1) / Gc * ((th - th.mean(0)) ** 2).sum(0)); t = stats.t.ppf(.975, Gc - 1)
    return pt, pt - t * se, pt + t * se, th, Gc
def dir_boot(mode, src, idx, nb=2000):   # resample the 30 direction chunks (all selected cases together)
    x = CH[MODES.index(mode), idx, SRCI.index(src)].sum(0) / (len(idx) * NPAIR)          # (grid, chunk)
    b = np.array([x[:, rng.integers(0, NCH, NCH)].mean(1) for _ in range(nb)])
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
def crossing(y, thr):
    """smallest Dx such that curve >= thr there and for every larger grid point (log-linear interpolation to the previous point); nan = never"""
    ok = y >= thr
    if not ok[-1]: return np.nan
    k = len(y) - 1
    while k > 0 and ok[k - 1]: k -= 1
    if k == 0: return GRID[0]
    x0, x1 = np.log10(GRID[k - 1]), np.log10(GRID[k]); return 10 ** (x0 + (thr - y[k - 1]) / (y[k] - y[k - 1]) * (x1 - x0))
def at(y, dx): return float(np.interp(np.log10(dx), np.log10(GRID), y))

# ---------------- leakage per source
leak = []
for s in F.NAMES:
    th = np.array([OPS[(seq, s)]["theta_deg"] for _, seq in UN])          # 9 cases (A and B share the sequence)
    g = F.G * np.radians(th); rl = np.array([OPS[(seq, s)]["rmsL"] for _, seq in UN])
    leak.append(dict(source=s, label=F.NAMES[s][1], theta_deg_median=np.median(th), g_theta_median=np.median(g), g_theta_min=g.min(), g_theta_max=g.max(),
                     g_theta_over_sqrt3=np.median(g) / F.SQ3, rms_P_leak_m_median=np.median(rl)))
leak = pd.DataFrame(leak); leak.to_csv(HERE + "/leakage_by_source.csv", index=False)

# ---------------- main table
rows = []
for mode in MODES:
    for src in SRCI:
        for grp, idx in GRP.items():
            pt, lo, hi, th, Gc = jk(mode, src, idx); dlo, dhi = dir_boot(mode, src, idx)
            c8 = [crossing(th[j], .8) for j in range(Gc)]; c9 = [crossing(th[j], .9) for j in range(Gc)]
            r = dict(mode=mode, source=src, group=grp, clusters=Gc, cross_0p8=crossing(pt, .8), cross_0p9=crossing(pt, .9),
                     cross_0p8_loo_min=np.nanmin(c8) if not np.all(np.isnan(c8)) else np.nan, cross_0p8_loo_max=np.nanmax(c8) if not np.any(np.isnan(c8)) else np.nan,
                     cross_0p9_loo_min=np.nanmin(c9) if not np.all(np.isnan(c9)) else np.nan, cross_0p9_loo_max=np.nanmax(c9) if not np.any(np.isnan(c9)) else np.nan,
                     n_loo_never_0p8=int(np.isnan(c8).sum()), n_loo_never_0p9=int(np.isnan(c9).sum()))
            for lab, dx in (("1e-4", 1e-4), ("1e-3", 1e-3), ("2.8e-3", PQ["median"]), ("1e-2", 1e-2), ("1.05e-2", PQ["p95"]), ("3e-2", 3e-2), ("1e-1", 1e-1), ("1", 1.0), ("10", 10.0)):
                r["agree@" + lab] = at(pt, dx); r["jk_lo@" + lab] = at(lo, dx); r["jk_hi@" + lab] = at(hi, dx); r["dirboot_lo@" + lab] = at(dlo, dx); r["dirboot_hi@" + lab] = at(dhi, dx)
            rows.append(r)
T = pd.DataFrame(rows); T.to_csv(HERE + "/dose_summary.csv", index=False)
# full pooled curves (for the figure / re-plotting)
curves = []
for mode in MODES:
    for src in SRCI:
        for grp, idx in GRP.items():
            pt, lo, hi, _, _ = jk(mode, src, idx); dlo, dhi = dir_boot(mode, src, idx, 500)
            for g, dx in enumerate(GRID): curves.append(dict(mode=mode, source=src, group=grp, dx=dx, agree=pt[g], jk_lo=lo[g], jk_hi=hi[g], dir_lo=dlo[g], dir_hi=dhi[g]))
pd.DataFrame(curves).to_csv(HERE + "/dose_curves_pooled.csv", index=False)

# ---------------- figure
MAIN = ["raw", "mahony", "gmahony", "vqf", "static", "oracle"]
COL = {"raw": "#c0392b", "static": "#e67e22", "mahony": "#8e44ad", "gmahony": "#2980b9", "vqf": "#16a085", "oracle": "#111111", "IABE": "#7f8c8d",
       "comp90": "#a0a0a0", "comp99": "#808080", "orc10": "#2ecc71", "orc03": "#27ae60", "orc30": "#58d68d", "comp50": "#bdc3c7"}
def panel(ax, mode, srcs, title, band_for=("raw", "oracle")):
    for s in srcs:
        pt, lo, hi, _, _ = jk(mode, s, GRP["nine cases"])
        ax.plot(GRID, pt, color=COL.get(s, "k"), lw=2.2 if s in ("raw", "oracle", "IABE") else 1.4, ls="--" if s == "IABE" else "-",
                label=F.NAMES[s][1] if s in F.NAMES else "IABE (reference attitude)")
        if s in band_for: ax.fill_between(GRID, lo.clip(0, 1), hi.clip(0, 1), color=COL[s], alpha=.12, lw=0)
    for v, ls, lab in ((TH_S15, "-", "0.01 (S15 threshold)"), (PQ["median"], ":", "pool median 2.8e-3"), (PQ["p95"], "-.", "pool 95th pct 1.05e-2")):
        ax.axvline(v, color="#555", ls=ls, lw=1, alpha=.8, label=lab)
    for h in (.8, .9): ax.axhline(h, color="#bbb", lw=.8, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.3, 1.02); ax.set_xlim(GRID[0], GRID[-1]); ax.set_title(title, fontsize=10); ax.set_xlabel(r"offset difference $\Delta$ in bias RMSE (m/s$^2$)")
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
panel(axs[0], "C-pool", MAIN + ["IABE"], "A. constant synthetic estimators (base d from the real pool)")
panel(axs[1], "R-same", MAIN + ["IABE"], "B. real estimator output + offset (same run, jitter kept)")
panel(axs[2], "C-zero", MAIN + ["IABE"], "C. constant estimators, one member exact (d_a = 0)")
axs[0].set_ylabel("P(ATE order = bias-RMSE order)")
axs[0].legend(fontsize=8, loc="lower right", ncol=1)
gth = leak.set_index("source").g_theta_median
fig.suptitle("Open-loop dose response (post hoc, synthetic offsets); bands = leave-one-sequence-out jackknife t 95%% (5 sequence clusters), nine cases pooled. "
             "gθ̄ (m/s², ||δ||=√3Δ scale): raw %.2f, Mahony %.2f, oracle %.3f" % (gth["raw"], gth["mahony"], gth["oracle"]), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(HERE + "/fig_openloop_dose.png", dpi=150)
# supplementary figure: all sources, main mode
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
panel(axs[0], "C-pool", list(F.NAMES) + ["IABE"], "C-pool, all sources", band_for=())
panel(axs[1], "R-diff", MAIN + ["IABE"], "R-diff: real run a vs another real run b + offset")
axs[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(HERE + "/fig_openloop_dose_allsources.png", dpi=130)

# ---- crossing vs leakage: sqrt(3)*Dx is the offset norm |delta| that gθ̄ (an acceleration) should be compared with
cl = leak.set_index("source"); cr = []
for src in F.NAMES:
    for mode in ("C-pool", "R-same"):
        t = T[(T["mode"] == mode) & (T.source == src) & (T.group == "nine cases")].iloc[0]
        cr.append(dict(mode=mode, source=src, g_theta=cl.g_theta_median[src], cross_0p8_bias_rmse=t.cross_0p8, cross_0p9_bias_rmse=t.cross_0p9,
                       ratio_0p8=F.SQ3 * t.cross_0p8 / cl.g_theta_median[src], ratio_0p9=F.SQ3 * t.cross_0p9 / cl.g_theta_median[src],
                       cross_0p8_over_0p01=t.cross_0p8 / 1e-2, cross_0p8_over_pool_p95=t.cross_0p8 / PQ["p95"], cross_0p8_over_pool_max=t.cross_0p8 / PQ["max"]))
cr = pd.DataFrame(cr); cr.to_csv(HERE + "/crossing_vs_leakage.csv", index=False)
from scipy import stats as _st
c = cr[cr["mode"] == "C-pool"]
print("log-log Spearman(g_theta, cross_0p8) over 12 sources: %.3f ; Pearson on logs %.3f ; slope %.3f" % (
    _st.spearmanr(c.g_theta, c.cross_0p8_bias_rmse)[0], np.corrcoef(np.log10(c.g_theta), np.log10(c.cross_0p8_bias_rmse))[0, 1],
    np.polyfit(np.log10(c.g_theta), np.log10(c.cross_0p8_bias_rmse), 1)[0]))
# ---- context: size of the real pairwise bias-RMSE differences (seed-mean per method, 55 pairs per case)
rr = pd.DataFrame([{k: r[k] for k in ("regime", "seq", "method", "seed", "rmse_avg_ref")} for r in RUNS]); dd = []
for (reg, seq), g in rr.groupby(["regime", "seq"]):
    m = g.groupby("method").rmse_avg_ref.mean().values; dd += list(np.abs(m[:, None] - m[None, :])[np.triu_indices(len(m), 1)])
dd = np.array(dd); ctx = dict(n_pairs=len(dd), median=float(np.median(dd)), q25=float(np.quantile(dd, .25)), q75=float(np.quantile(dd, .75)), q95=float(np.quantile(dd, .95)),
                              frac_above_0p01=float((dd > 1e-2).mean()), pool_DL_only=PQ_DL)
json.dump(ctx, open(HERE + "/real_pair_scale.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("pool quantiles used on figure", PQ, "DL-only", PQ_DL, "real pair |dRMSE|", ctx); print(cr[cr["mode"] == "C-pool"].round(3).to_string()); print(leak.round(4).to_string())
cols = ["mode", "source", "group", "cross_0p8", "cross_0p9", "cross_0p8_loo_min", "cross_0p8_loo_max", "agree@1e-3", "agree@1e-2", "agree@1e-1", "agree@1", "agree@10"]
print(T[(T.group == "nine cases") & (T["mode"] == "C-pool")][cols].round(4).to_string())

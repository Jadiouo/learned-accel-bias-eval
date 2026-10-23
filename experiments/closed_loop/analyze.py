# -*- coding: utf-8 -*-
import os, sys, json, itertools
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(HERE + "/consequences_long.csv"); P = pd.read_csv(HERE + "/predictors.csv")
PREDS = ["iabe", "bias_rmse", "mean_eps", "eps_rms", "ate_raw", "ate_oracle", "ate_static", "ate_mahony", "slie_10"]
ATES = ["ate_raw", "ate_oracle", "ate_static", "ate_mahony"]
NB = 2000; RNG = np.random.default_rng(7)

def rho(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean(); d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0

def scenario(cfg, mode, L, cons):
    s = C[(C.cfg == cfg) & (C["mode"] == mode) & (C.L == L)]
    K = s.rep.nunique()
    # RMS over reps, per (run_id, seq); also per-rep matrix
    g = s.groupby(["run_id", "seq"])
    agg = g[cons].apply(lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))).rename("c").reset_index()
    m = P[P.regime.isin(["A", "B"])].merge(agg, on=["run_id", "seq"])
    m = m.merge(s[["run_id", "seq", "rep", cons]].rename(columns={cons: "cr"}), on=["run_id", "seq"]) if False else m
    cells = []
    for (reg, seq), d in m.groupby(["regime", "seq"]):
        meth = d.groupby("method")
        X = meth[PREDS + [f"iabe_out{L}"]].mean(); c = meth.c.apply(lambda x: np.sqrt(np.mean(x ** 2)))
        # per-rep consequence per method (mean over train seeds of per-rep value)
        sr = s[s.seq == seq].merge(d[["run_id", "method", "seed"]], on="run_id")
        R = sr.groupby(["method", "rep"])[cons].mean().unstack()
        cells.append(dict(reg=reg, seq=seq, X=X.loc[c.index], c=c, R=R.loc[c.index], K=K))
    return cells, K

def s1(cells, preds, idx=None):
    out = {p: [] for p in preds}
    for i, ce in enumerate(cells):
        ii = idx[i] if idx is not None else slice(None)
        for p in preds: out[p].append(rho(ce["X"][p].to_numpy()[ii], ce["c"].to_numpy()[ii]))
    return {p: float(np.median(v)) for p, v in out.items()}, out

def pairs(ce, thr=1.25):
    c = ce["c"].to_numpy(); R = ce["R"].to_numpy(); K = R.shape[1]; need = 4 if K >= 5 else 2
    res = []
    n = len(c)
    for i, j in itertools.combinations(range(n), 2):
        hi, lo = max(c[i], c[j]), min(c[i], c[j])
        if lo <= 0 or hi / lo < thr: continue
        sgn = np.sign(c[i] - c[j]); cons = (np.sign(R[i] - R[j]) == sgn).sum()
        if cons >= need: res.append((i, j, sgn))
    return res

def s2(cells, preds, boot_idx=None):
    num = {p: 0.0 for p in preds}; den = 0
    for ce in cells:
        for i, j, sgn in pairs(ce):
            den += 1
            for p in preds:
                x = ce["X"][p].to_numpy(); d = np.sign(x[i] - x[j])
                num[p] += 1.0 if d == sgn else (0.5 if d == 0 else 0.0)
    return {p: (num[p] / den if den else np.nan) for p in preds}, den

def s3(cells, preds):
    xs = {p: [] for p in preds}; ys = []
    for ce in cells:
        y = np.log(ce["c"].to_numpy()); ys.append(y - y.mean())
        for p in preds:
            x = np.log(np.maximum(ce["X"][p].to_numpy(), 1e-12)); xs[p].append(x - x.mean())
    y = np.concatenate(ys)
    return {p: float(np.corrcoef(np.concatenate(xs[p]), y)[0, 1] ** 2) for p in preds}

def boot_diff(cells, preds, ref="iabe"):
    """two-level bootstrap: resample sequences, then methods within cell. returns dict p -> samples of (rho_ref - rho_p) and rho_p."""
    seqs = sorted({ce["seq"] for ce in cells}); by = {s: [i for i, ce in enumerate(cells) if ce["seq"] == s] for s in seqs}
    # precompute per-cell rho under method-resample
    res = {p: [] for p in preds}
    for b in range(NB):
        pick = RNG.choice(seqs, len(seqs)); cl = [i for s in pick for i in by[s]]
        rr = {p: [] for p in preds}
        for i in cl:
            n = len(cells[i]["c"]); ii = RNG.integers(0, n, n)
            for p in preds: rr[p].append(rho(cells[i]["X"][p].to_numpy()[ii], cells[i]["c"].to_numpy()[ii]))
        for p in preds: res[p].append(np.median(rr[p]))
    return {p: np.array(v) for p, v in res.items()}

def ci(x): return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

rows = []
scen = sorted({(r.cfg, r["mode"], r.L) for _, r in C[["cfg", "mode", "L"]].drop_duplicates().iterrows()}) if False else \
    [tuple(x) for x in C[["cfg", "mode", "L"]].drop_duplicates().itertuples(index=False)]
MAIN = ("base", "O", 10)
for (cfg, mode, L) in scen:
    for cons in ("delta", "abs"):
        cells, K = scenario(cfg, mode, L, cons)
        if len(cells) < 4: continue
        preds = PREDS + [f"iabe_out{L}"]
        med, per = s1(cells, preds)
        a2, npairs = s2(cells, preds); r2 = s3(cells, preds)
        # noise ceiling: split reps
        ceil = []
        for ce in cells:
            R = ce["R"].to_numpy(); h = R.shape[1] // 2
            if h >= 1: ceil.append(rho(np.sqrt((R[:, :h] ** 2).mean(1)), np.sqrt((R[:, h:] ** 2).mean(1))))
        bs = boot_diff(cells, ["iabe", "bias_rmse", "mean_eps"] + ATES) if (cfg, mode, L) == MAIN or True else None
        best_ate = max(ATES, key=lambda p: med[p])
        for p in preds:
            row = dict(cfg=cfg, mode=mode, L=L, cons=cons, pred=p, n_cells=len(cells), K=K, n_pairs=npairs, s1_median=med[p],
                       s1_pos_cells=int(sum(v > 0 for v in per[p])), s2=a2[p], s3_r2=r2[p], ceiling=float(np.median(ceil)) if ceil else np.nan)
            if p in bs:
                lo, hi = ci(bs[p]); row.update(s1_lo=lo, s1_hi=hi)
                if p != "iabe":
                    d = bs["iabe"] - bs[p]; lo, hi = ci(d); row.update(diff_iabe_minus=med["iabe"] - med[p], diff_lo=lo, diff_hi=hi,
                                                                       iabe_wins_cells=int(sum(a > b for a, b in zip(per["iabe"], per[p]))))
            rows.append(row)
        # best ATE row
        d = bs["iabe"] - bs[best_ate]; lo, hi = ci(d)
        rows.append(dict(cfg=cfg, mode=mode, L=L, cons=cons, pred="BEST_ATE=" + best_ate, n_cells=len(cells), K=K, n_pairs=npairs,
                         s1_median=med[best_ate], s2=a2[best_ate], s3_r2=r2[best_ate], diff_iabe_minus=med["iabe"] - med[best_ate],
                         diff_lo=lo, diff_hi=hi, iabe_wins_cells=int(sum(a > b for a, b in zip(per["iabe"], per[best_ate])))))
        print(cfg, mode, L, cons, "iabe %.2f rmse %.2f bestATE(%s) %.2f ceil %.2f pairs %d" % (med["iabe"], med["bias_rmse"], best_ate, med[best_ate], np.median(ceil), npairs), flush=True)
S = pd.DataFrame(rows); S.to_csv(HERE + "/scenario_stats.csv", index=False)

# controls & consequence magnitude table
rr = []
for (cfg, mode, L), d in C.groupby(["cfg", "mode", "L"]):
    if cfg != "base": continue
    g = d.groupby(["run_id", "seq"])[["delta", "abs"]].agg(lambda x: np.sqrt(np.mean(x ** 2))).reset_index()
    ctrl = g[g.run_id.str.startswith("CTRL")]; meth = g[~g.run_id.str.startswith("CTRL")]
    for n, x in ctrl.groupby("run_id"): rr.append(dict(mode=mode, L=L, group=n, delta_med=x.delta.median(), abs_med=x["abs"].median(), delta_min=x.delta.min(), delta_max=x.delta.max()))
    rr.append(dict(mode=mode, L=L, group="learned+const methods (all)", delta_med=meth.delta.median(), abs_med=meth["abs"].median(), delta_min=meth.delta.min(), delta_max=meth.delta.max()))
pd.DataFrame(rr).to_csv(HERE + "/table_controls.csv", index=False)

# figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for k, cons in enumerate(("delta", "abs")):
    cells, _ = scenario("base", "O", 10, cons)
    for ce in cells:
        ax[k].scatter(ce["X"]["iabe"], ce["c"], s=14, label=f"{ce['reg']}/{ce['seq']}")
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("IABE (m)"); ax[k].set_ylabel("outage-end " + ("delta vs b_ref" if cons == "delta" else "abs error") + " (m)")
    ax[k].set_title(f"mode O, L=10 s: {cons}"); ax[k].grid(alpha=.3)
ax[0].legend(fontsize=6, ncol=2)
sub = S[(S.cfg == "base") & (S["mode"] == "O") & (S.L == 10) & (S.pred.isin(PREDS))]
w = 0.38
for k, cons in enumerate(("delta", "abs")):
    t = sub[sub.cons == cons].set_index("pred").loc[PREDS]
    ax[2].bar(np.arange(len(PREDS)) + (k - .5) * w, t.s1_median, w, label=cons)
ax[2].set_xticks(range(len(PREDS))); ax[2].set_xticklabels(PREDS, rotation=60, fontsize=7); ax[2].axhline(0, c="k", lw=.5)
ax[2].set_ylabel("median within-cell Spearman"); ax[2].legend(); ax[2].set_title("S1, base config")
fig.tight_layout(); fig.savefig(HERE + "/fig_main.png", dpi=140)

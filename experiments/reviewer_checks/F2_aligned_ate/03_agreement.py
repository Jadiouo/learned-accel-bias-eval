# -*- coding: utf-8 -*-
"""Step 3: direction-agreement of ATE (none/4dof/se3 alignment) vs bias_rmse; split-half reliability; magnitudes.
Agreement definition COPIED from verification/R-F_noncircular/rF_noncircular_stratified.py
(11 DL methods, 10 seeds, 9 units, 1000 random 5/5 seed splits, BH selection on bias_rmse in half 1, sign agreement in half 2,
 sequence-cluster bootstrap). Differences: bootstrap index sets are drawn once per group and shared (paired) across measures;
 per-group rng seeded deterministically. Point estimates use the same split draw (default_rng(0).integers) as the original -> identical."""
import os, itertools, json, numpy as np, pandas as pd
from scipy import stats
OUT = os.path.dirname(os.path.abspath(__file__))
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
d = pd.read_csv(OUT + "/ate_aligned_per_run.csv"); d = d[d.method.isin(DL)]
SRCS = ["a_raw_gyro(paper)", "b_gyro-GT-bias(oracle)", "a2_raw_gyro_200Hz", "c_static-mean-bias",
        "d_mahony(kp=1,ki=0.05)", "e_mahony_gated(kp=1,ki=0.05)"]
ALN = ["none", "4dof", "se3"]
UNITS = [(r, s) for r in "AB" for s in sorted(d[d.regime == r].seq.unique())]
piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
PAIRS = np.array(list(itertools.combinations(range(11), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
names = ["bias_rmse"] + ["iabe_rot"] + [f"{s}|{a}" for s in SRCS for a in ALN]
mats = {}; seeds = None
for reg, seq in UNITS:
    g = d[(d.regime == reg) & (d.seq == seq)]
    g1 = g[g.source == SRCS[0]]
    b = piv(g1, "bias_rmse"); seeds = seeds or list(b.index); assert list(b.index) == seeds and b.shape == (10, 11)
    mats[(reg, seq, "bias_rmse")] = b.values; mats[(reg, seq, "iabe_rot")] = piv(g1, "iabe_rot").values
    for s in SRCS:
        gs = g[g.source == s]
        for a in ALN:
            p = piv(gs, "ate_" + a); assert list(p.index) == seeds; mats[(reg, seq, f"{s}|{a}")] = p.values
def select(M, idx):
    X = M[idx]; D = X[:, PI]-X[:, PJ]
    p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
    return bh(p) < .05, D.mean(0)
ALL = np.arange(10); halves = [np.array(c) for c in itertools.combinations(range(10), 5)]
rng = np.random.default_rng(0); NS = 1000; hs = rng.integers(0, len(halves), NS)
A = np.zeros((NS, len(UNITS), len(names))); N = np.zeros((NS, len(UNITS)))
REL = np.zeros((NS, len(UNITS), len(names)))   # split-half Spearman of method means
for si, h in enumerate(hs):
    h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
    for ui, u in enumerate(UNITS):
        mask, d1 = select(mats[(*u, "bias_rmse")], h1); N[si, ui] = mask.sum()
        for ci, k in enumerate(names):
            M = mats[(*u, k)]; m1 = M[h1].mean(0); m2 = M[h2].mean(0)
            REL[si, ui, ci] = stats.spearmanr(m1, m2)[0]
            if mask.sum():
                A[si, ui, ci] = (np.sign(m2[PI]-m2[PJ]) == np.sign(d1))[mask].sum()
uidx = {u: i for i, u in enumerate(UNITS)}
groups = {"A": [uidx[u] for u in UNITS if u[0] == "A"], "B": [uidx[u] for u in UNITS if u[0] == "B"], "ALL": list(range(len(UNITS)))}
for u in UNITS: groups[f"{u[0]}/{u[1]}"] = [uidx[u]]
def pooled(ix, c): return A[:, ix, c].sum()/max(N[:, ix].sum(), 1)
rows = []
for gname, gi in groups.items():
    gi = np.array(gi); brng = np.random.default_rng(1234 + len(gname) + sum(gi))
    BI = [gi[brng.integers(0, len(gi), len(gi))] for _ in range(4000)] if len(gi) > 1 else []
    base = {nm: [pooled(ix, ci) for ix in BI] for ci, nm in enumerate(names)} if BI else None
    for ci, nm in enumerate(names):
        sr = A[:, gi, ci].sum(1)/np.maximum(N[:, gi].sum(1), 1); ok = N[:, gi].sum(1) > 0
        r0 = pooled(gi, ci)
        rl = REL[:, gi, ci].mean(1)          # mean over units, per split
        rel_med = float(np.median(rl)); rel_sb = 2*rel_med/(1+rel_med) if rel_med > -1 else np.nan
        row = dict(group=gname, n_units=len(gi), measure=nm, pooled_rate=r0, split_p2_5=np.percentile(sr[ok], 2.5),
                   split_p97_5=np.percentile(sr[ok], 97.5), splithalf_spearman_median=rel_med, spearman_brown=rel_sb)
        if BI:
            b = np.array(base[nm]); row.update(seqboot_lo=np.percentile(b, 2.5), seqboot_hi=np.percentile(b, 97.5))
            if "|" in nm and not nm.endswith("|none"):
                b0 = np.array(base[nm.split("|")[0] + "|none"]); dd = b - b0
                row.update(delta_vs_none=r0 - pooled(gi, names.index(nm.split("|")[0] + "|none")),
                           delta_lo=np.percentile(dd, 2.5), delta_hi=np.percentile(dd, 97.5))
        rows.append(row)
P = pd.DataFrame(rows); P.to_csv(OUT + "/agreement_aligned.csv", index=False)
# units-level table
U = P[~P.group.isin(["A", "B", "ALL"])].pivot(index="measure", columns="group", values="pooled_rate"); U.to_csv(OUT + "/agreement_aligned_by_unit.csv")
# magnitudes
wide_i = d[d.source == SRCS[0]].groupby("regime").iabe_rot.median()
mg = []
for s in SRCS:
    for reg in ["A", "B", "ALL"]:
        g = d[(d.source == s) & ((d.regime == reg) | (reg == "ALL"))]
        row = dict(source=s, regime=reg, n=len(g), iabe_rot_median=g.iabe_rot.median(), bias_rmse_median=g.bias_rmse.median())
        for a in ALN:
            row[f"ate_{a}_median"] = g["ate_" + a].median(); row[f"ate_{a}_min"] = g["ate_" + a].min(); row[f"ate_{a}_max"] = g["ate_" + a].max()
        row["ratio_none_over_4dof_median"] = (g.ate_none / g.ate_4dof).median(); row["ratio_none_over_se3_median"] = (g.ate_none / g.ate_se3).median()
        row["ate_se3_over_iabe_median"] = (g.ate_se3 / g.iabe_rot).median(); row["ate_4dof_over_iabe_median"] = (g.ate_4dof / g.iabe_rot).median()
        mg.append(row)
pd.DataFrame(mg).to_csv(OUT + "/ate_magnitudes.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(P[P.group.isin(["A", "B", "ALL"])][["group", "measure", "pooled_rate", "seqboot_lo", "seqboot_hi", "delta_vs_none", "delta_lo", "delta_hi", "splithalf_spearman_median", "spearman_brown"]].round(3).to_string())
print(pd.DataFrame(mg).round(3).to_string())

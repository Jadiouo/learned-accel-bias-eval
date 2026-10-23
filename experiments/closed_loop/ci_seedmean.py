# -*- coding: utf-8 -*-
"""Seed-mean aggregation (same as analyze_gated.py / table_gated_spearman.csv) with the
analyze.py bootstrap (resample sequences, then methods within cell; NB=2000; default_rng(7)).
Writes ci_seedmean.csv. Does not modify any existing file."""
import os, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
G = pd.read_csv(HERE + "/gated_consequences.csv"); P = pd.read_csv(HERE + "/predictors.csv")
NB = 2000; PR = ["iabe", "bias_rmse"]
rms = lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))
def rho(a, b):
    ra, rb = rankdata(a), rankdata(b); ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()); return float((ra * rb).sum() / d) if d > 0 else 0.0
def cells_for(var, L):
    d = G[(G.variant == var) & (G.L == L) & ~G.run_id.str.startswith("CTRL:")]
    a = d.groupby(["run_id", "seq"]).delta.apply(rms).rename("c").reset_index()   # RMS over reps
    m = P[P.regime.isin(["A", "B"])].merge(a, on=["run_id", "seq"])
    out = []
    for (reg, seq), c in m.groupby(["regime", "seq"]):
        g = c.groupby("method")[PR + ["c"]].mean()                               # mean over train seeds
        if len(g) > 3: out.append(dict(seq=seq, X={p: g[p].to_numpy() for p in PR}, c=g.c.to_numpy()))
    return out
def run(var, L=10):
    cells = cells_for(var, L); rng = np.random.default_rng(7)
    pt = {p: float(np.median([rho(ce["X"][p], ce["c"]) for ce in cells])) for p in PR}
    seqs = sorted({ce["seq"] for ce in cells}); by = {s: [i for i, ce in enumerate(cells) if ce["seq"] == s] for s in seqs}
    bs = {p: [] for p in PR}
    for _ in range(NB):
        pick = rng.choice(seqs, len(seqs)); cl = [i for s in pick for i in by[s]]; rr = {p: [] for p in PR}
        for i in cl:
            n = len(cells[i]["c"]); ii = rng.integers(0, n, n)
            for p in PR: rr[p].append(rho(cells[i]["X"][p][ii], cells[i]["c"][ii]))
        for p in PR: bs[p].append(np.median(rr[p]))
    d = np.array(bs["iabe"]) - np.array(bs["bias_rmse"]); lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(variant=var, L=L, n_cells=len(cells), rho_iabe=pt["iabe"], rho_bias_rmse=pt["bias_rmse"],
                diff=pt["iabe"] - pt["bias_rmse"], ci_lo=float(lo), ci_hi=float(hi))
R = pd.DataFrame([run("nogate"), run("gated")]); R.to_csv(HERE + "/ci_seedmean.csv", index=False)
print(R.round(4).to_string())

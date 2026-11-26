"""Reproduce 0.554 [0.444, 0.674] (nine cases pooled, raw-attitude ATE) and the A/B values.
(1) 1000 random splits (rng seed 0) and the original script's bootstrap random-number order give the same CI as the original csv;
(2) point estimate from all 252 splits enumerated (used by the simulation)."""
import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
OUT = HERE + "/"
units, bias, ate = load(); sel = precompute_selection(units, bias); G = groups(units)
rng = np.random.default_rng(0); hs = rng.integers(0, len(HALVES), 1000)
# Group order of the original script: A, B, ALL, then each unit; 8 measures per group (bias, IABE, 6 ATE sources)
gnames = ["A", "B", "ALL"]; Gd = dict(G)
for i, u in enumerate(units): Gd[f"{u[0]}/{u[1]}"] = [i]; gnames.append(f"{u[0]}/{u[1]}")
measures = ["bias", "IABE", "raw", "oracle", "static", "mahony", "gated", "mahony5"]   # same order as the original script
Aall = {"raw": agree_counts(ate["raw"], units, sel)[0][hs], "oracle": agree_counts(ate["oracle"], units, sel)[0][hs]}
Ns = agree_counts(ate["raw"], units, sel)[1][hs]
bs_store = {}
for g in gnames:
    gi = np.array(Gd[g])
    for m in measures:
        if len(gi) > 1:
            ix_all = [gi[rng.integers(0, len(gi), len(gi))] for _ in range(4000)]
            if m in Aall: bs_store[(g, m)] = [Aall[m][:, ix].sum()/Ns[:, ix].sum() for ix in ix_all]
rows = []
for k in ("raw", "oracle"):
    A, N = agree_counts(ate[k], units, sel)
    for g in ["A", "B", "ALL"]:
        gi = np.array(Gd[g]); lo, hi = np.percentile(bs_store[(g, k)], [2.5, 97.5])
        rows.append(dict(source=k, group=g, rate_rand1000=A[hs][:, gi].sum()/Ns[:, gi].sum(), rate_enum252=A[:, gi].sum()/N[:, gi].sum(),
                         boot_lo=lo, boot_hi=hi, pairs_per_split=N[:, gi].sum(1).mean()))
P = pd.DataFrame(rows); P.to_csv(OUT+"reproduce_observed.csv", index=False); print(P.round(4).to_string())

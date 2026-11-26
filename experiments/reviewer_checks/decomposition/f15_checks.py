import os, sys; sys.dont_write_bytecode = True
RB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, RB + "F1_noise_ceiling")
import numpy as np, pandas as pd
from common import *
units, bias, ate = load(); G = groups(units); seqs = sorted({u[1] for u in units})
sel = precompute_selection(units, bias)
A_obs, N_obs = agree_counts(ate["raw"], units, sel); a_obs, n = A_obs.sum(0), N_obs.sum(0)
# ---- F1: paired ceiling - observed with sequence-cluster and case-cluster bootstrap
rng = np.random.default_rng(1)
def seqW(gi, NB):
    ss = [s for s in seqs if any(units[i][1] == s for i in gi)]; M = np.zeros((NB, len(units)))
    for b in range(NB):
        for j in rng.integers(0, len(ss), len(ss)):
            for i in gi:
                if units[i][1] == ss[j]: M[b, i] += 1
    return M
def caseW(gi, NB):
    M = np.zeros((NB, len(units)))
    for b in range(NB):
        for j in rng.integers(0, len(gi), len(gi)): M[b, gi[j]] += 1
    return M
print("F1 paired (ceiling_sim - observed), raw, ALL; fraction <=0 and 2.5/50/97.5 pct")
for var in ["V1", "V2", "V3", "C1"]:
    S = np.load(RB + f"F1_noise_ceiling/sim_agree_raw_{var}_N2000.npy")      # (2000, 9)
    for nm, W in [("seq", seqW(G["ALL"], 2000)), ("case", caseW(G["ALL"], 2000))]:
        ceil = (W * S).sum(1) / (W @ n); obs = (W @ a_obs) / (W @ n); d = ceil - obs
        print(var, nm, "frac<=0 %.4f" % (d <= 0).mean(), np.round(np.percentile(d, [2.5, 50, 97.5]), 3),
              "obs CI", np.round(np.percentile(obs, [2.5, 97.5]), 3), "ceil CI", np.round(np.percentile(ceil, [2.5, 97.5]), 3))
# ---- F5: leave-one-sequence-out and per-sequence delta for RM|r=1 vs W|s=1
d = pd.read_csv(RB + "F5_scaled_attitude/ate_scaled_per_run.csv")
def mats(c):
    x = d[d.config == c]
    return {u: x[(x.regime == u[0]) & (x.seq == u[1])].pivot_table(index="seed", columns="method", values="ate")[DL].values for u in units}
cfgs = ["W|s=1", "RM|r=3", "RM|r=1", "RM|r=0.3", "W|s=0"]
C = {c: [v.sum(0) for v in agree_counts(mats(c), units, sel)] for c in cfgs}
print("\nF5 per-unit agreement")
print(pd.DataFrame({c: C[c][0] / C[c][1] for c in cfgs}, index=[f"{u[0]}/{u[1]}" for u in units]).round(3).to_string())
print("npairs per unit", dict(zip([f"{u[0]}/{u[1]}" for u in units], C["W|s=1"][1].astype(int))))
for c in ["RM|r=1", "RM|r=3", "W|s=0"]:
    out = []
    for s in seqs:
        keep = np.array([u[1] != s for u in units])
        pt = lambda cc: C[cc][0][keep].sum() / C[cc][1][keep].sum()
        out.append(f"-{s}:{pt(c) - pt('W|s=1'):+.3f}")
    per = []
    for s in seqs:
        m = np.array([u[1] == s for u in units])
        per.append(f"{s}:{C[c][0][m].sum()/C[c][1][m].sum() - C['W|s=1'][0][m].sum()/C['W|s=1'][1][m].sum():+.3f}")
    print(c, "minus s=1, leave-one-seq-out:", " ".join(out)); print("   per-sequence:", " ".join(per))
# share of pooled pairs from regime A vs B
print("pairs A/B share", C["W|s=1"][1][G["A"]].sum() / C["W|s=1"][1].sum())

# Red-team: decompose ATE = ||L + Q + I|| (Prop 1) and test which term drives the drop in agreement.
import sys; sys.dont_write_bytecode = True
import os
RB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, RB + "F2_aligned_ate"); sys.path.insert(0, RB + "F1_noise_ceiling")
import numpy as np, pandas as pd
import f2_lib as L
L.mahony = lambda w, f, R0, dt, kp, ki: np.repeat(R0[None], len(w), 0)       # skip slow unused sources
L.mahony_gated = L.mahony
from common import load, precompute_selection, agree_counts, groups, DL
from scipy.spatial.transform import Rotation as Rot
OUT = sys.argv[1]
units, bias, ate = load()
wide = pd.read_csv(L.ROOT + "/results/metrics_wide.csv").merge(
    pd.read_csv(L.ROOT + "/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
dl = wide[wide.method.isin(DL)]
SRC = {"raw": "a_raw_gyro(paper)", "oracle": "b_gyro-GT-bias(oracle)"}
rng = np.random.default_rng(7)
NROT = 30
ROTS = Rot.random(NROT, random_state=11).as_matrix()
recs = []
for seq in L.SEQS:
    mi, S = L.load_seq(seq)
    Ls, Cs = {}, {}
    for k, src in SRC.items():
        C = S[src]; Cs[k] = C
        Ls[k] = L.dint(L.world(C - mi.R_ref, mi.a_raw - mi.b_ref))
    for _, r in dl[dl.eval_sequence == seq].iterrows():
        bh = np.load(f"{L.ROOT}/results/runs/{r.run_id}/series.npz")[f"b_hat_{seq}"]
        eps = mi.b_ref - bh
        I = L.dint(L.world(mi.R_ref, eps))
        for k in SRC:
            C = Cs[k]; Lk = Ls[k]
            Q = L.dint(L.world(C - mi.R_ref, eps))
            row = dict(run_id=r.run_id, regime=r.regime, method=r.method, seed=int(r.seed), seq=seq, src=k,
                       full=L.rms3(Lk + Q + I), noL=L.rms3(Q + I), noQ=L.rms3(Lk + I), I=L.rms3(I), negL=L.rms3(-Lk + Q + I),
                       normL=L.rms3(Lk), normQ=L.rms3(Q))
            for j in range(NROT):
                row[f"rot{j}"] = L.rms3(Lk @ ROTS[j].T + Q + I)
            recs.append(row)
    print(seq, flush=True)
D = pd.DataFrame(recs); D.to_csv(OUT + "/f5_decomp_runs.csv", index=False)
# check vs existing
att = pd.read_csv(RB + "../attitude_controls/ate_per_run_gated.csv")
for k, src in SRC.items():
    x = D[D.src == k].merge(att[att.source == src][["run_id", "seq", "ate", "iabe_rot"]], on=["run_id", "seq"])
    print(k, "n", len(x), "max rel full vs csv", float((abs(x.full - x.ate) / x.ate).max()),
          "max rel I vs iabe_rot", float((abs(x.I - x.iabe_rot) / x.iabe_rot).max()),
          "median |Q|/|I|", float((x.normQ / x.I).median()), "median |L|/|I|", float((x.normL / x.I).median()))
G = groups(units); seqs = sorted({u[1] for u in units})
sel = precompute_selection(units, bias)
def mats(k, col):
    return {u: D[(D.src == k) & (D.regime == u[0]) & (D.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
NB = 4000; brng = np.random.default_rng(20261126); Wt = {}
for g, gi in G.items():
    ss = [s for s in seqs if any(units[i][1] == s for i in gi)]
    M = np.zeros((NB, len(units)))
    for b in range(NB):
        for j in brng.integers(0, len(ss), len(ss)):
            for i in gi:
                if units[i][1] == ss[j]: M[b, i] += 1
    Wt[g] = M
def cnt(m):
    A, N = agree_counts(m, units, sel); return A.sum(0), N.sum(0)
res = []
for k in SRC:
    base = cnt(mats(k, "full"))
    cols = ["full", "noL", "noQ", "I", "negL"] + [f"rot{j}" for j in range(NROT)]
    C = {c: cnt(mats(k, c)) for c in cols}
    for g in ["ALL", "A", "B"]:
        gi = G[g]
        for c in cols:
            a, n = C[c]; pt = a[gi].sum() / n[gi].sum(); bs = (Wt[g] @ a) / (Wt[g] @ n)
            b1 = (Wt[g] @ base[0]) / (Wt[g] @ base[1])
            res.append(dict(src=k, group=g, variant=c, agree=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                            d_vs_full=pt - base[0][gi].sum() / base[1][gi].sum(), dlo=np.percentile(bs - b1, 2.5), dhi=np.percentile(bs - b1, 97.5)))
P = pd.DataFrame(res); P.to_csv(OUT + "/f5_decomp_agree.csv", index=False)
pd.set_option("display.width", 200)
print(P[~P.variant.str.startswith("rot")].round(3).to_string())
rot = P[P.variant.str.startswith("rot")].groupby(["src", "group"]).agree.describe()[["mean", "min", "25%", "50%", "75%", "max"]]
print("random fixed rotation of the shared leakage term (30 rotations):"); print(rot.round(3).to_string())

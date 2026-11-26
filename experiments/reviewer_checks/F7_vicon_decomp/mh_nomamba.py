"""F7 task 2: Machine Hall nine cases, raw (and oracle) attitude, with vs without Mamba.
Loads via F1_noise_ceiling/common.load (read-only import), decomposition/f5_decomp_runs.csv (read-only) for the decomposition variants."""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, RB + "/F1_noise_ceiling")
import numpy as np, pandas as pd
import common as C1
from agree_lib import *
units, bias, ate = C1.load(); DL = C1.DL; seqs_of = [u[1] for u in units]
w = pd.read_csv(C1.R + "euroc_full/results/metrics_wide.csv").merge(pd.read_csv(C1.R + "euroc_full/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
w = w[w.method.isin(DL)]
iabe = {u: w[(w.regime == u[0]) & (w.eval_sequence == u[1])].pivot_table(index="seed", columns="method", values="iabe_rot")[DL].values for u in units}
RP = pd.read_csv(RB + "/decomposition/f5_decomp_runs.csv")
def rpmat(src, col): return {u: RP[(RP.src == src) & (RP.regime == u[0]) & (RP.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
SETS = {"with_mamba(K=11)": list(range(11)), "no_mamba(K=10)": [i for i, m in enumerate(DL) if m != "mamba"]}
SPL = {"252": HALVES252, "rng0_1000": halves_rng0(1000)}
lines = []; rows = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
for setn, cols in SETS.items():
    for spn, hv in SPL.items():
        ag = Agree(units, {u: bias[u][:, cols] for u in units}, len(cols), hv)
        meas = {"bias RMSE self-retest": bias, "IABE (iabe_rot)": iabe, "raw ATE": ate["raw"], "oracle ATE": ate["oracle"],
                "raw noL": rpmat("raw", "noL"), "raw noQ": rpmat("raw", "noQ"), "raw negL": rpmat("raw", "negL")}
        res = {k: ag.counts({u: m[u][:, cols] for u in units}) for k, m in meas.items()}
        a0, n0 = res["raw ATE"]
        for k, (a, n) in res.items():
            pt, lo, hi, se, loo = jackknife(a, n, seqs_of); d = jackknife_diff(a, n, a0, n0, seqs_of)
            rows.append(dict(set=setn, splits=spn, measure=k, agree=pt, jk_lo=lo, jk_hi=hi, d_vs_rawATE=d[0], d_lo=d[1], d_hi=d[2], pairs_per_split=n.sum() / len(hv)))
O = pd.DataFrame(rows); O.to_csv(HERE + "/mh_nomamba_agree.csv", index=False)
pd.set_option("display.width", 200)
for spn in SPL:
    P(f"\n== Machine Hall nine cases, raw attitude, {spn} splits, jackknife t4 over 5 sequences")
    x = O[O.splits == spn].pivot_table(index="measure", columns="set", values=["agree", "jk_lo", "jk_hi", "pairs_per_split"])
    P(x.round(3).to_string())
P("\nreference: RT 252-split raw ATE 0.559, IABE 0.802, noL 0.799; old paper S6 (rng0 1000 splits, bootstrap): with Mamba raw ATE 0.554, IABE 0.804; no Mamba raw ATE 0.454 [0.313,0.668], IABE 0.942")
open(HERE + "/mh_nomamba_stdout.txt", "w").write("\n".join(lines))

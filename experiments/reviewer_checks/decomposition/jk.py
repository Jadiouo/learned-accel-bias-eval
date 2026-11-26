import os, sys; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
import numpy as np, pandas as pd
from common import *
from scipy import stats
units, bias, ate = load(); seqs = sorted({u[1] for u in units})
sel = precompute_selection(units, bias)
D = pd.read_csv(os.path.join(HERE, "f5_decomp_runs.csv"))
def mat(k, col): return {u: D[(D.src == k) & (D.regime == u[0]) & (D.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
items = {"raw ATE": ate["raw"], "oracle ATE": ate["oracle"], "raw noL": mat("raw", "noL")}
for nm, m in items.items():
    a, n = [v.sum(0) for v in agree_counts(m, units, sel)]
    pt = a.sum() / n.sum()
    th = np.array([a[[u[1] != s for u in units]].sum() / n[[u[1] != s for u in units]].sum() for s in seqs])
    se = np.sqrt((len(seqs) - 1) / len(seqs) * ((th - th.mean()) ** 2).sum()); t = stats.t.ppf(.975, len(seqs) - 1)
    print(nm, "pt %.3f  jackknife-t95 [%.3f, %.3f]  LOO range %.3f-%.3f" % (pt, pt - t * se, pt + t * se, th.min(), th.max()))

"""分解：原 11 方法的切分半一致率中，含 mamba 的對 vs 不含 mamba 的對（regime A 與 ALL；同 rng/切分）。"""
import sys, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba_exclusion import *
UNITS, mats = build(DL11); rng = np.random.default_rng(0)
PAIRS = np.array(list(itertools.combinations(range(11), 2))); PI, PJ = PAIRS[:,0], PAIRS[:,1]
mm = (PI == 4) | (PJ == 4)
halves = [np.array(c) for c in itertools.combinations(range(10), 5)]; hs = rng.integers(0, len(halves), 1000)
ms = ["bias_rmse","IABE","ATE:a_raw_gyro(paper)","ATE:b_gyro-GT-bias(oracle)","ATE:c_static-mean-bias"]
acc = {(reg, g, m): [0, 0] for reg in ("A","B") for g in ("mamba","other") for m in ms}
for h in hs:
    h1 = halves[h]; h2 = np.setdiff1d(np.arange(10), h1)
    for u in UNITS:
        X = mats[(*u,"bias_rmse")][h1]; D = X[:,PI]-X[:,PJ]
        p = np.nan_to_num(stats.ttest_1samp(D,0,axis=0).pvalue, nan=1.0); sel = bh(p) < .05; d1 = D.mean(0)
        for m in ms:
            mu = mats[(*u,m)][h2].mean(0); ag = np.sign(mu[PI]-mu[PJ]) == np.sign(d1)
            for g, msk in (("mamba", mm), ("other", ~mm)):
                a = acc[(u[0], g, m)]; a[0] += (ag & sel & msk).sum(); a[1] += (sel & msk).sum()
rows = []
for reg in ("A","B"):
    tot = acc[(reg,"mamba",ms[0])][1] + acc[(reg,"other",ms[0])][1]
    for m in ms:
        a, b = acc[(reg,"mamba",m)], acc[(reg,"other",m)]
        rows.append(dict(regime=reg, measure=m, mamba_pair_share_of_selected=a[1]/tot, rate_mamba_pairs=a[0]/max(a[1],1), rate_other_pairs=b[0]/max(b[1],1)))
df = pd.DataFrame(rows); df.to_csv(os.path.dirname(os.path.abspath(__file__))+"/mamba_pair_share.csv", index=False); print(df.round(3).to_string())

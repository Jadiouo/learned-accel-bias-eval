# 探索性（非 PREREG 預登錄）：成對反轉率依 Truth 比值分箱。混合池所有估測器對（不抽 K=4）。
import numpy as np
from run_all import TRAJ, SEEDS, LEV_I, LEV_M, OUT
bins = [(1.25, 2), (2, 5), (5, 20), (20, 1e9)]
rows = ["| Mode | level | Truth 比值箱 | ATE單次 | IABE | RMSE | 對數 |", "|---|---|---|---|---|---|---|"]
for mode, li in [("I", 1), ("I", 3), ("I", 5), ("M", 0), ("M", 2)]:
    us = [np.load(f"{OUT}/{mode}_{t}_s{s}_L{li}.npz") for t in TRAJ for s in SEEDS]
    acc = {b: np.zeros((3, 2)) for b in bins}
    for u in us:
        m = np.where(u["grp"] == -1)[0]
        tb, ia, rm, at = u["truthB"][m], u["iabe"][m], u["rmse"][m], u["ate"][m][:, 0]
        i, j = np.triu_indices(len(m), 1)
        r = np.maximum(tb[i] / tb[j], tb[j] / tb[i])
        for b in bins:
            s = (r >= b[0]) & (r < b[1])
            for q, v in enumerate([at, ia, rm]):
                wrong = (v[i][s] < v[j][s]) != (tb[i][s] < tb[j][s])
                acc[b][q] += [wrong.sum(), s.sum()]
    lev = (LEV_I if mode == "I" else LEV_M)[li]
    for b in bins:
        a = acc[b]; rows.append(f"| {mode} | {lev:g} | {b[0]}-{b[1] if b[1]<1e8 else 'inf'} | " + " | ".join(f"{a[q,0]/a[q,1]:.3f}" for q in range(3)) + f" | n={int(a[0,1])} |")
open("explore_pairs.md", "w").write("\n".join(rows)); print("\n".join(rows))

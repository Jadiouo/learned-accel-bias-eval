# -*- coding: utf-8 -*-
"""Step 1: per-run ATE under {source} x {none,4dof,se3}; also reproduction check. Writes ate_aligned_per_run.csv, repro_check.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import f2_lib as L
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
wide = pd.read_csv(L.ROOT + "/results/metrics_wide.csv")
idx = pd.read_csv(L.ROOT + "/results/runs_index.csv")[["run_id", "regime", "method", "seed"]]
wide = wide.merge(idx, on="run_id")
LIMIT = int(os.environ.get("F2_LIMIT", "0"))
if LIMIT: wide = wide.sample(LIMIT, random_state=0)
BASE, SRCS, PR, PREF = {}, {}, {}, {}
for seq in L.SEQS:
    if not (wide.eval_sequence == seq).any(): continue
    mi, S = L.load_seq(seq); BASE[seq] = mi; SRCS[seq] = S
    PREF[seq] = L.dint(L.world(mi.R_ref, mi.a_raw - mi.b_ref) + mi.g_vec)
    print(seq, len(mi), {k: v.shape for k, v in S.items()}, "%.0fs" % (time.time() - t0), flush=True)
recs = []
for n, (_, r) in enumerate(wide.iterrows()):
    seq = r.eval_sequence; mi = BASE[seq]
    bh = np.load(f"{L.ROOT}/results/runs/{r.run_id}/series.npz")[f"b_hat_{seq}"]
    assert len(bh) == len(mi)
    Q = PREF[seq]
    for src, Cg in SRCS[seq].items():
        P = L.dint(L.world(Cg, mi.a_raw - bh) + mi.g_vec)
        row = dict(run_id=r.run_id, regime=r.regime, method=r.method, seed=int(r.seed), seq=seq, source=src,
                   bias_rmse=r.bias_rmse_avg, iabe_rot=r.iabe_rot, gyro_ate_paper=r.gyro_ate)
        for k, f in L.ALIGN.items(): row["ate_" + k] = L.rms3(f(P, Q) - Q)
        recs.append(row)
    if n % 100 == 0: print("run", n, "%.0fs" % (time.time() - t0), flush=True)
df = pd.DataFrame(recs)
df.to_csv(OUT + ("/ate_aligned_per_run_SMOKE.csv" if LIMIT else "/ate_aligned_per_run.csv"), index=False)
print("done %.0fs" % (time.time() - t0))

# -*- coding: utf-8 -*-
"""Step 1. Per (seq, attitude source): linear-algebra operators of the open-loop ATE; per real run: its error series reduced to a few moments.
Open loop is exactly quadratic in a constant bias-error offset delta (eps = eps_real + delta):
  position error p_t = L_t + P[C eps]_t,  P[C delta]_t = T_t delta  with T = dint(C) (N,3,3)
  ATE^2(delta) = mean_t |Y_t + T_t delta|^2 = Y2 + 2 delta.vY + delta' H delta, Y = L + P[C eps_real].
IABE uses C=R_ref, L=0. bias RMSE^2(delta) = (m2 + 2 delta.m1 + |delta|^2)/3.
CPU only; writes ops.pkl in this folder."""
import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f9_lib as F
L = F.L
t0 = time.time()
wide = pd.read_csv(L.ROOT + "/results/metrics_wide.csv").merge(
    pd.read_csv(L.ROOT + "/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
dl = wide[wide.method.isin(F.DL)]
OPS = {}      # (seq, src) -> dict(H, L2, vL, theta_deg, rmsL)
HR = {}       # seq -> H of R_ref
RUNS = []     # list of dict per run
for seq in L.SEQS:
    mi, S = F.load_seq_sources(seq); N = len(mi)
    print(seq, "N", N, "%.0fs" % (time.time() - t0), flush=True)
    TR = L.dint(mi.R_ref)                                     # (N,3,3)
    HR[seq] = np.einsum("tki,tkj->ij", TR, TR) / N
    rr = dl[dl.eval_sequence == seq]
    eps_all = []
    for _, r in rr.iterrows():
        bh = np.load(f"{L.ROOT}/results/runs/{r.run_id}/series.npz")[f"b_hat_{seq}"]; assert len(bh) == N
        eps_all.append(mi.b_ref - bh)
    base = [dict(regime=r.regime, seq=seq, run_id=r.run_id, method=r.method, seed=int(r.seed), rmse_avg_ref=r.bias_rmse_avg, iabe_ref=r.iabe_rot) for _, r in rr.iterrows()]
    for b, e in zip(base, eps_all):
        b["m1"] = e.mean(0); b["m2"] = float(np.mean(np.sum(e ** 2, 1)))
        Ia = L.dint(L.world(mi.R_ref, e)); b["I2"] = float(np.mean(np.sum(Ia ** 2, 1))); b["vI"] = np.einsum("tki,tk->i", TR, Ia) / N
        b["Y2"] = {}; b["vY"] = {}
    for src, C in S.items():
        T = L.dint(C); H = np.einsum("tki,tkj->ij", T, T) / N
        Lt = L.dint(L.world(C - mi.R_ref, mi.a_raw - mi.b_ref))
        OPS[(seq, src)] = dict(H=H, L2=float(np.mean(np.sum(Lt ** 2, 1))), vL=np.einsum("tki,tk->i", T, Lt) / N,
                               theta_deg=F.tilt_mean_deg(C, mi.R_ref), rmsL=L.rms3(Lt))
        for b, e in zip(base, eps_all):
            Y = Lt + L.dint(L.world(C, e))
            b["Y2"][src] = float(np.mean(np.sum(Y ** 2, 1))); b["vY"][src] = np.einsum("tki,tk->i", T, Y) / N
    RUNS += base
    print(seq, "done %.0fs" % (time.time() - t0), flush=True)
pickle.dump(dict(OPS=OPS, HR=HR, RUNS=RUNS), open(HERE + "/ops.pkl", "wb"))
print("saved, %d runs, %.0fs" % (len(RUNS), time.time() - t0))

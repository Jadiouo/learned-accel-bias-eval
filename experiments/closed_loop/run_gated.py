# -*- coding: utf-8 -*-
"""POST-HOC (after review). Outage with gravity-gated tilt update (accelerometer gravity direction, low dynamics only).
Nonlinear per-source outage propagation (tilt update depends on the bias used). variant 'nogate' reproduces the original
linear-form result (sanity check). CPU only. Output: gated_consequences.csv"""
import os, sys, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eskf
from eskf import FS, DT, GV, skew, exp_so3
import run_closed_loop as rc

THR_A, THR_W = 0.3, 0.2      # | ||f-b|| - g | [m/s^2], ||w-bg|| [rad/s]; fixed post hoc, not tuned
SIG_A = 0.3                  # tilt measurement sigma [m/s^2], inflated for residual linear accel
TILT_EVERY = 20              # 10 Hz
UP = np.array([0, 0, eskf.G])

def outage_prop(S, f, w, k0, k1, bser, gated):
    p, v, R, bg, P = S["p"].copy(), S["v"].copy(), S["R"].copy(), S["bg"], S["P"].copy()
    I15 = np.eye(15); Q = rc.eskf.make_Q(1.0); Rm = np.eye(3) * SIG_A ** 2
    ng = 0; nt = 0
    for k in range(k0, k1):
        fa = f[k] - bser[k - k0]; wc = w[k] - bg
        aw = R @ fa + GV
        p = p + v * DT + 0.5 * aw * DT * DT; v = v + aw * DT
        F = I15.copy(); F[0:3, 3:6] = np.eye(3) * DT; F[3:6, 6:9] = -R @ skew(fa) * DT; F[3:6, 12:15] = -R * DT
        F[6:9, 6:9] = np.eye(3) - skew(wc) * DT; F[6:9, 9:12] = -np.eye(3) * DT
        R = R @ exp_so3(wc * DT); P = F @ P @ F.T + Q
        nt += 1
        if gated and (k - k0) % TILT_EVERY == 0 and abs(np.linalg.norm(fa) - eskf.G) < THR_A and np.linalg.norm(wc) < THR_W:
            ng += 1
            h = R.T @ UP; H = np.zeros((3, 15)); H[:, 6:9] = skew(h)
            Sm = H @ P @ H.T + Rm; K = np.linalg.solve(Sm, H @ P).T
            K[9:15] = 0.0                                   # bg/ba frozen during outage
            dx = K @ (fa - h)
            IKH = I15 - K @ H; P = IKH @ P @ IKH.T + K @ Rm @ K.T; P = 0.5 * (P + P.T)
            p = p + dx[0:3]; v = v + dx[3:6]; R = R @ exp_so3(dx[6:9])
    return p, ng / max(nt // TILT_EVERY, 1)

def task(args):
    seq, rep0 = args
    cname = "base"; sig, attS, qS = rc.CFGS[cname]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; Q = eskf.make_Q(qS); src = rc.sources(D)
    Ls = [L for L in rc.LS if i1 - L * FS >= i0]; rows = []
    for rep in [rep0]:
        nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
        S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
        kos = {L: i1 - L * FS for L in Ls}
        snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
        for L in Ls:
            Sn = snaps[kos[L]]; ko = kos[L]
            names = list(src) + ["CTRL:eskf_self"]
            for var, gt in (("nogate", False), ("gated", True)):
                pe = {}; fr = {}
                for n in names:
                    b = np.tile(Sn["ba"], (i1 - ko, 1)) if n == "CTRL:eskf_self" else src[n][ko:i1]
                    pe[n], fr[n] = outage_prop(Sn, f, w, ko, i1, b, gt)
                for n in names:
                    rows.append(dict(seq=seq, L=L, rep=rep, run_id=n, variant=var, gate_frac=fr[n],
                                     delta=float(np.linalg.norm(pe[n] - pe["CTRL:bref"])), abs=float(np.linalg.norm(pe[n] - pg[i1]))))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    with Pool(16) as pool: R = pool.map(task, [(s, r) for s in rc.SEQS for r in range(rc.K_O)], chunksize=1)
    df = pd.DataFrame([r for x in R for r in x]); df.to_csv(HERE + "/gated_consequences.csv", index=False)
    print(df.shape, "%.0fs" % (time.time() - t0))

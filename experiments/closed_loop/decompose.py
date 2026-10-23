# -*- coding: utf-8 -*-
"""POST-HOC 2 (outline stage). Decompose b_ref-group end-of-outage position error (C_abs) into sources by replacing ESKF state
components at outage start with GT (pos, vel, tilt, yaw, bg), re-propagating the outage (gated and nogate, nonlinear run_gated.outage_prop),
exact Shapley over the 5 components (32 coalitions) + one-at-a-time. CPU only. Outputs: decomp_units.csv, decomp_summary.csv"""
import os, sys, itertools, math, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eskf, run_closed_loop as rc, run_gated as rg
from eskf import FS
COMP = ["pos", "vel", "tilt", "yaw", "bg"]

def task(args):
    seq, rep = args
    sig, attS, qS = rc.CFGS["base"]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; Q = eskf.make_Q(qS); bref = rc.hold_series(D, D["bref"])
    Ls = [L for L in rc.LS if i1 - L * FS >= i0]; rows = []
    nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    kos = {L: i1 - L * FS for L in Ls}
    snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
    for L in Ls:
        ko = kos[L]; Sn = snaps[ko]; Rg = D["R"][ko]
        ew = Rot.from_matrix(Sn["R"] @ Rg.T).as_rotvec()               # world-frame attitude error (R = exp(ew) Rg)
        R_notilt = Rot.from_rotvec([0, 0, ew[2]]).as_matrix() @ Rg      # tilt removed, yaw error kept
        R_noyaw = Rot.from_rotvec([ew[0], ew[1], 0]).as_matrix() @ Rg   # yaw removed, tilt error kept
        R_none = Rg
        errs = dict(pos=float(np.linalg.norm(Sn["p"] - D["p"][ko])), vel=float(np.linalg.norm(Sn["v"] - D["v"][ko])),
                    tilt_deg=float(np.degrees(np.hypot(ew[0], ew[1]))), yaw_deg=float(np.degrees(abs(ew[2]))),
                    bg=float(np.linalg.norm(Sn["bg"] - D["bw"][ko])))
        b = bref[ko:i1]
        for var, gt in (("nogate", False), ("gated", True)):
            val = {}
            for m in range(32):
                z = {c: bool(m >> j & 1) for j, c in enumerate(COMP)}   # True = replaced by GT
                S1 = dict(Sn); S1["P"] = Sn["P"]
                S1["p"] = D["p"][ko] if z["pos"] else Sn["p"]; S1["v"] = D["v"][ko] if z["vel"] else Sn["v"]
                S1["bg"] = D["bw"][ko] if z["bg"] else Sn["bg"]
                if z["tilt"] and z["yaw"]: S1["R"] = R_none
                elif z["tilt"]: S1["R"] = R_notilt
                elif z["yaw"]: S1["R"] = R_noyaw
                else: S1["R"] = Sn["R"]
                p, _ = rg.outage_prop(S1, f, w, ko, i1, b, gt)
                val[m] = float(np.linalg.norm(p - pg[i1]))
            v0, vall = val[0], val[31]
            row = dict(seq=seq, rep=rep, L=L, variant=var, total=v0, floor=vall, **errs)
            for j, c in enumerate(COMP):
                row["oat_" + c] = v0 - val[1 << j]
                phi = 0.0
                for m in range(32):
                    if m >> j & 1: continue
                    s = bin(m).count("1")
                    phi += math.factorial(s) * math.factorial(4 - s) / math.factorial(5) * ((val[m] ) - val[m | 1 << j])
                row["shap_" + c] = phi
            rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    with Pool(16) as pool: R = pool.map(task, [(s, r) for s in rc.SEQS for r in range(rc.K_O)], chunksize=1)
    df = pd.DataFrame([r for x in R for r in x]); df.to_csv(HERE + "/decomp_units.csv", index=False)
    print(df.shape, "%.0fs" % (time.time() - t0))

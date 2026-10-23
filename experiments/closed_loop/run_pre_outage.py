# -*- coding: utf-8 -*-
"""POST-HOC 3 (PREREG_2.md). Exp1: pre-outage injection (mode P) ; Exp2: filter variants (Q_ba x10 / x0.1), mode O. CPU only.
Outputs: pre_outage_long.csv, variants_long.csv, variants_decomp_units.csv"""
import os, sys, time, math
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eskf, run_closed_loop as rc, run_gated as rg
from eskf import FS

def make_Q_ba(ba_scale=1.0, ba_on=True):
    Q = eskf.make_Q(1.0)
    Q[12:15, 12:15] *= (ba_scale ** 2) if ba_on else 0.0
    return Q

def state_diag(Sn, D, ko, bofs_k):
    Rg = D["R"][ko]; ew = Rot.from_matrix(Sn["R"] @ Rg.T).as_rotvec()
    ev = Sn["v"] - D["v"][ko]; bt = Sn["ba"] + bofs_k
    return dict(tx=np.degrees(ew[0]), ty=np.degrees(ew[1]), yaw=np.degrees(ew[2]), vx=ev[0], vy=ev[1], vz=ev[2],
                batx=bt[0], baty=bt[1], batz=bt[2], dax=Sn["ba"][0], day=Sn["ba"][1], daz=Sn["ba"][2])

def extend_back(b, i0):
    b = b.copy(); b[:i0] = b[i0]; return b

def task_P(args):
    seq, mode, rep = args           # mode in {"P_est","P_off"}
    sig, attS, qS = rc.CFGS["base"]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; src = rc.sources(D)
    Q = make_Q_ba(1.0, mode == "P_est")
    Ls = [L for L in rc.LS if i1 - L * FS >= i0]; kos = {L: i1 - L * FS for L in Ls}
    nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
    rows = []
    for n, b in src.items():
        bofs = extend_back(b, i0) if not n == "CTRL:zero" else b
        S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
        if mode == "P_off": S["P"][12:15, 12:15] = 0.0
        snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, bofs=bofs, snaps=list(kos.values()))
        for L in Ls:
            ko = kos[L]; Sn = snaps[ko]
            Rs = eskf.outage_rotations(Sn, w, ko, i1)
            pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, bofs[ko:i1] + Sn["ba"])[0]
            r = dict(seq=seq, cfg=mode, L=L, rep=rep, run_id=n, pend=pe, pgt=pg[i1])
            r.update(state_diag(Sn, D, ko, bofs[ko])); rows.append(r)
    out = []
    for L in Ls:
        rr = [r for r in rows if r["L"] == L]; ref = [r for r in rr if r["run_id"] == "CTRL:bref"][0]["pend"]
        for r in rr:
            r["delta"] = float(np.linalg.norm(r["pend"] - ref)); r["abs"] = float(np.linalg.norm(r["pend"] - r["pgt"]))
            del r["pend"], r["pgt"]; out.append(r)
    return out

VAR = {"base": 1.0, "q10": 10.0, "q01": 0.1}

def task_V(args):
    seq, vname, rep = args
    sig, attS, qS = rc.CFGS["base"]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; src = rc.sources(D); Q = make_Q_ba(VAR[vname])
    Ls = [L for L in (5, 10, 20) if i1 - L * FS >= i0]; kos = {L: i1 - L * FS for L in Ls}
    nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
    rows = []
    for L in Ls:
        ko = kos[L]; Sn = snaps[ko]; Rs = eskf.outage_rotations(Sn, w, ko, i1)
        names = list(src); B = np.stack([src[n][ko:i1] for n in names])
        pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, B); ib = names.index("CTRL:bref")
        for j, n in enumerate(names):
            rows.append(dict(seq=seq, cfg=vname, L=L, rep=rep, run_id=n, delta=float(np.linalg.norm(pe[j] - pe[ib])),
                             abs=float(np.linalg.norm(pe[j] - pg[i1]))))
    # Shapley decomposition of b_ref C_abs (L=10 and 5), nogate, same as decompose.py
    COMP = ["pos", "vel", "tilt", "yaw", "bg"]; drows = []; bref = src["CTRL:bref"]
    for L in (5, 10):
        if L not in kos: continue
        ko = kos[L]; Sn = snaps[ko]; Rg = D["R"][ko]
        ew = Rot.from_matrix(Sn["R"] @ Rg.T).as_rotvec()
        R_notilt = Rot.from_rotvec([0, 0, ew[2]]).as_matrix() @ Rg; R_noyaw = Rot.from_rotvec([ew[0], ew[1], 0]).as_matrix() @ Rg
        b = bref[ko:i1]; val = {}
        for m in range(32):
            z = {c: bool(m >> j & 1) for j, c in enumerate(COMP)}; S1 = dict(Sn)
            S1["p"] = D["p"][ko] if z["pos"] else Sn["p"]; S1["v"] = D["v"][ko] if z["vel"] else Sn["v"]
            S1["bg"] = D["bw"][ko] if z["bg"] else Sn["bg"]
            S1["R"] = Rg if (z["tilt"] and z["yaw"]) else R_notilt if z["tilt"] else R_noyaw if z["yaw"] else Sn["R"]
            p, _ = rg.outage_prop(S1, f, w, ko, i1, b, False); val[m] = float(np.linalg.norm(p - pg[i1]))
        row = dict(seq=seq, cfg=vname, rep=rep, L=L, total=val[0], floor=val[31])
        for j, c in enumerate(COMP):
            phi = 0.0
            for m in range(32):
                if m >> j & 1: continue
                s = bin(m).count("1")
                phi += math.factorial(s) * math.factorial(4 - s) / math.factorial(5) * (val[m] - val[m | 1 << j])
            row["shap_" + c] = phi
        drows.append(row)
    return rows, drows

if __name__ == "__main__":
    t0 = time.time(); which = sys.argv[1] if len(sys.argv) > 1 else "all"
    with Pool(14) as pool:
        if which in ("all", "P"):
            R = pool.map(task_P, [(s, m, r) for s in rc.SEQS for m in ("P_est", "P_off") for r in range(rc.K_O)], chunksize=1)
            pd.DataFrame([r for x in R for r in x]).to_csv(HERE + "/pre_outage_long.csv", index=False); print("P done %.0fs" % (time.time() - t0), flush=True)
        if which in ("all", "V"):
            R = pool.map(task_V, [(s, v, r) for s in rc.SEQS for v in VAR for r in range(rc.K_O)], chunksize=1)
            pd.DataFrame([r for x in R for r in x[0]]).to_csv(HERE + "/variants_long.csv", index=False)
            pd.DataFrame([r for x in R for r in x[1]]).to_csv(HERE + "/variants_decomp_units.csv", index=False)
            print("V done %.0fs" % (time.time() - t0), flush=True)

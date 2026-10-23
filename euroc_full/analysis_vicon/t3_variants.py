# -*- coding: utf-8 -*-
"""[Vicon port of experiments/closed_loop/run_pre_outage.py (modes P_est/P_off, variants Q_ba x10/x0.1 + their Shapley units),
run_gated.outage_prop, and decompose.py (base Shapley, nogate+gated)]. Copied; only paths/sequences/results come from vcommon/t2.
Outputs (CSV, first line is the [TAG] provenance comment): t3_pre_outage_long.csv, t3_variants_long.csv, t3_variants_decomp_units.csv,
t3_decomp_units.csv.  CPU only."""
import os, sys, time, math
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vcommon as vc
import eskf
import t2_closed_loop as rc          # same data loading / sources / noise as the Table II port (parses the same CLI args)
from eskf import FS, DT, GV, skew, exp_so3
COMP = ["pos", "vel", "tilt", "yaw", "bg"]

# ---- run_gated.outage_prop (verbatim, rc.eskf -> eskf) ----
THR_A, THR_W = 0.3, 0.2
SIG_A = 0.3
TILT_EVERY = 20
UP = np.array([0, 0, eskf.G])
def outage_prop(S, f, w, k0, k1, bser, gated):
    p, v, R, bg, P = S["p"].copy(), S["v"].copy(), S["R"].copy(), S["bg"], S["P"].copy()
    I15 = np.eye(15); Q = eskf.make_Q(1.0); Rm = np.eye(3) * SIG_A ** 2
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
            p, _ = outage_prop(S1, f, w, ko, i1, b, False); val[m] = float(np.linalg.norm(p - pg[i1]))
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

def task_B(args):
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
                p, _ = outage_prop(S1, f, w, ko, i1, b, gt)
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
    t0 = time.time(); A = rc.A; SEQS = rc.SEQS
    with Pool(A.procs) as pool:
        R = pool.map(task_P, [(s, m, r) for s in SEQS for m in ("P_est", "P_off") for r in range(rc.K_O)], chunksize=1)
        vc.write_csv(pd.DataFrame([r for x in R for r in x]), A, "t3_pre_outage_long.csv", index=False); print("P done %.0fs" % (time.time() - t0), flush=True)
        R = pool.map(task_V, [(s, v, r) for s in SEQS for v in VAR for r in range(rc.K_O)], chunksize=1)
        vc.write_csv(pd.DataFrame([r for x in R for r in x[0]]), A, "t3_variants_long.csv", index=False)
        vc.write_csv(pd.DataFrame([r for x in R for r in x[1]]), A, "t3_variants_decomp_units.csv", index=False)
        print("V done %.0fs" % (time.time() - t0), flush=True)
        R = pool.map(task_B, [(s, r) for s in SEQS for r in range(rc.K_O)], chunksize=1)
        vc.write_csv(pd.DataFrame([r for x in R for r in x]), A, "t3_decomp_units.csv", index=False); print("B done %.0fs" % (time.time() - t0), flush=True)

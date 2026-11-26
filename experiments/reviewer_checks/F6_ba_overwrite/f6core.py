# -*- coding: utf-8 -*-
"""F6 core. Imports (does NOT execute the __main__ blocks of) experiments/closed_loop/{run_closed_loop, eskf, run_pre_outage};
structure follows F3_dose_response/core.py. Nothing here writes into experiments/closed_loop."""
import os, sys, math
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.spatial.transform import Rotation as Rot
CL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "closed_loop"))
sys.path.insert(0, CL)
import run_closed_loop as RC
import run_pre_outage as RPO          # only make_Q_ba is used
import eskf
from eskf import FS, G, exp_so3
SEQS = RC.SEQS; K = RC.K_O; LS = (5, 10)
QS = {"q01": 0.1, "base": 1.0, "q10": 10.0}
COMP = ["pos", "vel", "tilt", "yaw", "bg"]

def snapshots(seq, rep, qscale):
    """shared pre-outage filter run per (seq, rep, Q_ba scale), snapshots at outage starts for L=5,10 (as RC.task_O)."""
    D = RC.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]
    sig, attS, _ = RC.CFGS["base"]; Q = RPO.make_Q_ba(qscale, True)
    Ls = [L for L in LS if i1 - L * FS >= i0]
    nz, mn = RC.rep_noise(rep, len(f) // FS + 2)
    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    kos = {L: i1 - L * FS for L in Ls}
    snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
    return D, {L: (kos[L], i1, snaps[kos[L]]) for L in Ls}

def state_with(Sn, **kw):
    S = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in Sn.items()}; S.update(kw); return S

class Outage:
    """linear-form outage end position for a given onset state; caches rotations."""
    def __init__(self, D, ko, i1):
        self.D, self.ko, self.i1 = D, ko, i1; self.f, self.w = D["f"], D["w"]
    def endpos(self, S, B):
        Rs = eskf.outage_rotations(S, self.w, self.ko, self.i1)
        return eskf.outage_end_pos(S, self.f, Rs, self.ko, self.i1, B)   # (M,3)

def tilt_split(Sn, Rg):
    """world-frame attitude error ew (R = exp(ew) Rg) and the tilt-removed / yaw-removed rotations (decompose.py)."""
    ew = Rot.from_matrix(Sn["R"] @ Rg.T).as_rotvec()
    R_notilt = Rot.from_rotvec([0, 0, ew[2]]).as_matrix() @ Rg
    R_noyaw = Rot.from_rotvec([ew[0], ew[1], 0]).as_matrix() @ Rg
    return ew, R_notilt, R_noyaw

def min_rot(u, v):
    """rotation taking direction u to direction v."""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    ax = np.cross(u, v); s = np.linalg.norm(ax); c = float(u @ v)
    if s < 1e-14: return np.eye(3)
    return exp_so3(ax / s * math.atan2(s, c))

def consistent_tilt_R(Sn, b_new, first_order=False, fbar=None):
    """(c): R_c with R_c (fbar - b_new) = R_f (fbar - b_a), fbar = R_f^T [0,0,g] (static gravity) unless fbar is given."""
    Rf = Sn["R"]
    if fbar is None: fbar = Rf.T @ np.array([0, 0, G])
    u = fbar - b_new; v = fbar - Sn["ba"]
    if first_order:
        d = b_new - Sn["ba"]; phi = np.cross(u, d) / (u @ u); return Rf @ exp_so3(phi)
    return Rf @ min_rot(u, v)

def shapley(OG, Sn, D, ko, i1, b, ew):
    """exact Shapley over 5 components (linear form; same replacement rules as decompose.py). b: (n,3) bias series."""
    Rg = D["R"][ko]; _, R_notilt, R_noyaw = tilt_split(Sn, Rg)
    cache = {}; pg = D["p"][i1]
    def val(m):
        z = {c: bool(m >> j & 1) for j, c in enumerate(COMP)}
        key = (z["tilt"], z["yaw"], z["bg"])
        S1 = state_with(Sn)
        S1["p"] = D["p"][ko] if z["pos"] else Sn["p"]; S1["v"] = D["v"][ko] if z["vel"] else Sn["v"]
        S1["bg"] = D["bw"][ko] if z["bg"] else Sn["bg"]
        S1["R"] = Rg if (z["tilt"] and z["yaw"]) else R_notilt if z["tilt"] else R_noyaw if z["yaw"] else Sn["R"]
        if key not in cache:
            Rs = eskf.outage_rotations(S1, OG.w, ko, i1); cache[key] = Rs
        Rs = cache[key]
        pe = eskf.outage_end_pos(S1, OG.f, Rs, ko, i1, b)[0]
        return float(np.linalg.norm(pe - pg))
    v = {m: val(m) for m in range(32)}
    row = dict(total=v[0], floor=v[31])
    for j, c in enumerate(COMP):
        phi = 0.0
        for m in range(32):
            if m >> j & 1: continue
            s = bin(m).count("1")
            phi += math.factorial(s) * math.factorial(4 - s) / math.factorial(5) * (v[m] - v[m | 1 << j])
        row["shap_" + c] = phi; row["oat_" + c] = v[0] - v[1 << j]
    return row

# -*- coding: utf-8 -*-
"""F3 core: reuse (import) the closed-loop machinery of experiments/closed_loop/run_closed_loop.py + eskf.py.
Nothing here writes into experiments/closed_loop. The main program of run_closed_loop.py is __main__-guarded
and is NOT executed; only its functions (load_seq, hold_series, rep_noise, sources) and eskf are imported."""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
CL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "closed_loop"))
sys.path.insert(0, CL)
import run_closed_loop as RC          # import only; reads configs/metrics_wide, writes nothing
import eskf
from eskf import FS
SEQS = RC.SEQS; K = RC.K_O; LS = (5, 10)

def snapshots(seq, rep):
    """Replicates RC.task_O for cfg 'base': one shared filter run per (seq, rep), snapshots at every outage start.
    Returns dict L -> (ko, i1, Sn, Rs) with Rs = outage rotations (gyro-only, frozen bg)."""
    D = RC.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]
    sig, attS, qS = RC.CFGS["base"]; Q = eskf.make_Q(qS)
    Ls = [L for L in LS if i1 - L * FS >= i0]
    nz, mn = RC.rep_noise(rep, len(f) // FS + 2)
    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    kos = {L: i1 - L * FS for L in Ls}
    snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
    out = {}
    for L in Ls:
        Sn = snaps[kos[L]]; ko = kos[L]
        out[L] = (ko, i1, Sn, eskf.outage_rotations(Sn, w, ko, i1))
    return D, out

def end_err(D, L_pack, bseries):
    """bseries (M,n,3) bias series over the outage -> (M,) ||p_end - p_gt_end||, and (M,3) end position."""
    ko, i1, Sn, Rs = L_pack
    pe = eskf.outage_end_pos(Sn, D["f"], Rs, ko, i1, bseries)
    return np.linalg.norm(pe - D["p"][i1], axis=1), pe

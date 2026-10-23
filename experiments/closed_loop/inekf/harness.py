# -*- coding: utf-8 -*-
"""Protocol harness (PREREG.md): same protocol as ../run_closed_loop.py / ../run_pre_outage.py (task_V base, task_P P_off), filter selectable.
Modes:
  O   <eskf|inekf> <outdir>            -> variants_long.csv, variants_decomp_units.csv  (cfg=base; mode O_base)
  P   inekf <outdir> [max_src]         -> pre_outage_long.csv, pre_outage_decomp_units.csv (cfg=P_off; batched over sources)
  Psub eskf <outdir> <src,src,...>     -> same files, ESKF backend on a source subset (validation V2 / ESKF P_off Shapley)
CPU only; Pool<=4. Existing modules are imported read-only."""
import os, sys, math, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); PAR = os.path.dirname(HERE)
sys.path.insert(0, PAR); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
import eskf, inekf, run_closed_loop as rc, run_gated as rg, run_pre_outage as rp
from eskf import FS
COMP = ["pos", "vel", "tilt", "yaw", "bg"]
CHUNK = 64


def shapley(Sn, D, ko, i1, b):
    """Exact Shapley of C_abs over 5 state components (identical to run_pre_outage.task_V)."""
    f, w, pg = D["f"], D["w"], D["p"]; Rg = D["R"][ko]
    ew = Rot.from_matrix(Sn["R"] @ Rg.T).as_rotvec()
    R_notilt = Rot.from_rotvec([0, 0, ew[2]]).as_matrix() @ Rg; R_noyaw = Rot.from_rotvec([ew[0], ew[1], 0]).as_matrix() @ Rg
    val = {}
    for m in range(32):
        z = {c: bool(m >> j & 1) for j, c in enumerate(COMP)}; S1 = dict(Sn)
        S1["p"] = D["p"][ko] if z["pos"] else Sn["p"]; S1["v"] = D["v"][ko] if z["vel"] else Sn["v"]
        S1["bg"] = D["bw"][ko] if z["bg"] else Sn["bg"]
        S1["R"] = Rg if (z["tilt"] and z["yaw"]) else R_notilt if z["tilt"] else R_noyaw if z["yaw"] else Sn["R"]
        p, _ = rg.outage_prop(S1, f, w, ko, i1, b, False); val[m] = float(np.linalg.norm(p - pg[i1]))
    row = dict(total=val[0], floor=val[31])
    for j, c in enumerate(COMP):
        phi = 0.0
        for m in range(32):
            if m >> j & 1: continue
            s = bin(m).count("1")
            phi += math.factorial(s) * math.factorial(4 - s) / math.factorial(5) * (val[m] - val[m | 1 << j])
        row["shap_" + c] = phi
    return row


def task_O(args):
    seq, rep, fname = args; filt = {"eskf": eskf, "inekf": inekf}[fname]
    sig, attS, qS = rc.CFGS["base"]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; src = rc.sources(D); Q = rp.make_Q_ba(1.0)
    Ls = [L for L in (5, 10, 20) if i1 - L * FS >= i0]; kos = {L: i1 - L * FS for L in Ls}
    nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
    S = filt.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    snaps = filt.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
    rows, drows = [], []; bref = src["CTRL:bref"]
    for L in Ls:
        ko = kos[L]; Sn = snaps[ko]; Rs = eskf.outage_rotations(Sn, w, ko, i1)
        names = list(src); B = np.stack([src[n][ko:i1] for n in names])
        pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, B); ib = names.index("CTRL:bref")
        for j, n in enumerate(names):
            rows.append(dict(seq=seq, cfg="base", L=L, rep=rep, run_id=n, delta=float(np.linalg.norm(pe[j] - pe[ib])),
                             abs=float(np.linalg.norm(pe[j] - pg[i1]))))
        if L in (5, 10):
            drows.append(dict(seq=seq, cfg="base", rep=rep, L=L, **shapley(Sn, D, ko, i1, bref[ko:i1])))
    return rows, drows


def _chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i + n]


def task_P(args):
    """P_off arm for (seq, rep). backend 'inekf' (batched over sources) or 'eskf' (loop). names=None -> all sources."""
    seq, rep, backend, names = args
    sig, attS, qS = rc.CFGS["base"]; D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; src = rc.sources(D)
    if names is None: names = list(src)
    if "FIRST2" in names: names = [n for n in names if n != "FIRST2"] + [n for n in src if not n.startswith("CTRL:")][:2]
    assert "CTRL:bref" in names
    Q = rp.make_Q_ba(1.0, False)
    Ls = [L for L in rc.LS if i1 - L * FS >= i0]; kos = {L: i1 - L * FS for L in Ls}
    nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
    rows, drows = [], []
    for ch in _chunks(names, CHUNK if backend == "inekf" else 1):
        bo = np.stack([rp.extend_back(src[n], i0) if n != "CTRL:zero" else src[n] for n in ch])        # (B,N,3)
        if backend == "inekf":
            S0 = inekf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS); S0["P"][12:15, 12:15] = 0.0
            Sb = {k: np.repeat(x[None], len(ch), 0) for k, x in S0.items()}
            snb = inekf.run_aided_batch(Sb, f, w, pg, 0, max(kos.values()), mn, sig, Q, bofs=bo, snaps=list(kos.values()))
            get = lambda ko, j: {k: x[j] for k, x in snb[ko].items()}
            Rsb = {L: inekf.outage_rotations_b(snb[kos[L]], w, kos[L], i1) for L in Ls}
            rots = lambda L, j, Sn: Rsb[L][j]
        else:
            S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS); S["P"][12:15, 12:15] = 0.0
            sn1 = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, bofs=bo[0], snaps=list(kos.values()))
            get = lambda ko, j: sn1[ko]
            rots = lambda L, j, Sn: eskf.outage_rotations(Sn, w, kos[L], i1)
        for j, n in enumerate(ch):
            for L in Ls:
                ko = kos[L]; Sn = get(ko, j); Rs = rots(L, j, Sn)
                b = bo[j][ko:i1] + Sn["ba"]
                pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, b)[0]
                r = dict(seq=seq, cfg="P_off", L=L, rep=rep, run_id=n, pend=pe, pgt=pg[i1]); r.update(rp.state_diag(Sn, D, ko, bo[j][ko]))
                rows.append(r)
                if n == "CTRL:bref" and L in (5, 10):
                    drows.append(dict(seq=seq, cfg="P_off", rep=rep, L=L, **shapley(Sn, D, ko, i1, b)))
    out = []
    for L in Ls:
        rr = [r for r in rows if r["L"] == L]; ref = [r for r in rr if r["run_id"] == "CTRL:bref"][0]["pend"]
        for r in rr:
            r["delta"] = float(np.linalg.norm(r["pend"] - ref)); r["abs"] = float(np.linalg.norm(r["pend"] - r["pgt"]))
            del r["pend"], r["pgt"]; out.append(r)
    return out, drows


if __name__ == "__main__":
    t0 = time.time(); mode, fname, outdir = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(outdir, exist_ok=True)
    nproc = int(os.environ.get("NPROC", "4")); assert nproc <= 4
    K = rc.K_O
    if mode == "O":
        tasks = [(s, r, fname) for s in rc.SEQS for r in range(K)]; fn = task_O
        a, b = "variants_long.csv", "variants_decomp_units.csv"
    else:
        names = None
        if mode == "Psub": names = sys.argv[4].split(",")
        if mode == "P" and len(sys.argv) > 4: names = ["CTRL:bref", "CTRL:zero", "CTRL:const_oracle"]    # smoke test
        seqs = os.environ.get("SEQS", ",".join(rc.SEQS)).split(","); reps = int(os.environ.get("REPS", K))
        tasks = [(s, r, fname, names) for s in seqs for r in range(reps)]; fn = task_P
        a, b = "pre_outage_long.csv", "pre_outage_decomp_units.csv"
    with Pool(nproc) as pool: R = pool.map(fn, tasks, chunksize=1)
    pd.DataFrame([r for x in R for r in x[0]]).to_csv(os.path.join(outdir, a), index=False)
    pd.DataFrame([r for x in R for r in x[1]]).to_csv(os.path.join(outdir, b), index=False)
    print(mode, fname, "done %.0fs" % (time.time() - t0), flush=True)

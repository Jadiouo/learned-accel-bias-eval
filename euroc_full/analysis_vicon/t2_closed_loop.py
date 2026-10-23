# -*- coding: utf-8 -*-
"""[Vicon port of experiments/closed_loop/run_closed_loop.py (mode O, base config only) + unify_table.py -> paper Table II]
Closed-loop ESKF outage experiment on real EuRoC b_hat (PREREG.md). CPU only."""
import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, yaml
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
import argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vcommon as vc
import eskf                                  # verbatim copy of experiments/closed_loop/eskf.py
from eskf import FS, DT
from euroc_full import data_euroc, metrics
ap = vc.add_args(argparse.ArgumentParser())
ap.add_argument("--reps", type=int, default=5, help="noise repetitions K_O (MH: 5)")
ap.add_argument("--procs", type=int, default=4)
A = ap.parse_args()
cfg = vc.load_cfg(A); SEQS = list(cfg["sequences"]); STRIDE = cfg["stride"]
wide, _idx = vc.load_wide(A.results_dir)
ROOT = None

# configs: name -> (sig_meas, att_scale, qscale)
CFGS = {"base": (0.05, 1.0, 1.0), "sig0.02": (0.02, 1.0, 1.0), "sig0.20": (0.20, 1.0, 1.0),
        "att0": (0.05, 0.0, 1.0), "att4": (0.05, 4.0, 1.0), "q3": (0.05, 1.0, 3.0)}
LS = (5, 10, 20)
K_O, K_D = A.reps, 3
D_CFGS = ("base", "sig0.20", "att4")
_cache = {}

def load_seq(seq):
    if seq in _cache: return _cache[seq]
    d = data_euroc.prepare_regime_a(seq, cfg); m = d.merged
    tws = np.asarray(d.test_win_starts, np.int64)
    D = dict(f=m[["ax", "ay", "az"]].to_numpy(), w=m[["wx", "wy", "wz"]].to_numpy(), p=m[["px", "py", "pz"]].to_numpy(),
             v=m[["vx", "vy", "vz"]].to_numpy(), bw=m[["bwx", "bwy", "bwz"]].to_numpy(),
             R=Rot.from_quat(m[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix(), i0=int(tws[0]), i1=int(tws[-1]), n=len(tws))
    runs = wide[wide.eval_sequence == seq]
    bh, bref, rid = [], None, []
    for _, r in runs.iterrows():
        z = np.load(f"{A.results_dir}/runs/{r.run_id}/series.npz")
        bh.append(z[f"b_hat_{seq}"]); rid.append(r.run_id); bref = z[f"b_ref_{seq}"]
    D["run_ids"] = rid; D["bhat"] = bh; D["bref"] = bref
    _cache[seq] = D
    return D

def hold_series(D, b40):
    """40 Hz series -> 200 Hz zero-order hold placed at frames i0.. (length Ntot)."""
    N = len(D["f"]); out = np.zeros((N, 3)); up = np.repeat(b40, STRIDE, axis=0)
    e = min(N, D["i0"] + len(up)); out[D["i0"]:e] = up[:e - D["i0"]]
    return out

def rep_noise(rep, n_upd):
    rng = np.random.default_rng(1000 + rep)
    return dict(att=rng.standard_normal(3), bg=rng.standard_normal(3)), rng.standard_normal((n_upd, 3))

def sources(D):
    """label -> (200Hz bias series incl. controls). Mode O/D bias sources."""
    out = {}
    for rid, b in zip(D["run_ids"], D["bhat"]): out[rid] = hold_series(D, b)
    out["CTRL:bref"] = hold_series(D, D["bref"])
    out["CTRL:zero"] = np.zeros((len(D["f"]), 3))
    out["CTRL:const_oracle"] = hold_series(D, np.tile(D["bref"].mean(0), (len(D["bref"]), 1)))
    return out

def task_O(args):
    seq, cname = args
    sig, attS, qS = CFGS[cname]; D = load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; Q = eskf.make_Q(qS); src = sources(D)
    Ls = [L for L in LS if i1 - L * FS >= i0]
    if cname != "base": Ls = [L for L in Ls if L == 10 or L == 5]
    rows = []
    for rep in range(K_O):
        nz, mn = rep_noise(rep, len(f) // FS + 2)
        S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
        kos = {L: i1 - L * FS for L in Ls}
        snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
        for L in Ls:
            Sn = snaps[kos[L]]; ko = kos[L]
            Rs = eskf.outage_rotations(Sn, w, ko, i1)
            names = list(src); B = np.stack([src[n][ko:i1] for n in names] + [np.tile(Sn["ba"], (i1 - ko, 1))])
            pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, B)
            ib = names.index("CTRL:bref")
            for j, n in enumerate(names + ["CTRL:eskf_self"]):
                rows.append(dict(seq=seq, cfg=cname, mode="O", L=L, rep=rep, run_id=n,
                                 delta=float(np.linalg.norm(pe[j] - pe[ib])), abs=float(np.linalg.norm(pe[j] - pg[i1]))))
    return rows

def task_D(args):
    seq, cname, rep = args
    sig, attS, qS = CFGS[cname]; D = load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]
    i0, i1 = D["i0"], D["i1"]; Q = eskf.make_Q(qS); src = sources(D)
    Ls = [L for L in (5, 10) if i1 - L * FS >= i0]
    kos = {L: i1 - L * FS for L in Ls}
    nz, mn = rep_noise(rep, len(f) // FS + 2)
    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
    SA = eskf.run_aided(S, f, w, pg, 0, i0, mn, sig, Q, snaps=[i0])[i0]
    res = {}
    for n, bofs in src.items():
        S1 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in SA.items()}
        S1["ba"] = SA["ba"] - bofs[i0]
        snaps = eskf.run_aided(S1, f, w, pg, i0, max(kos.values()), mn, sig, Q, bofs=bofs, snaps=list(kos.values()))
        for L in Ls:
            ko = kos[L]; Sn = snaps[ko]
            Rs = eskf.outage_rotations(Sn, w, ko, i1)
            res[(n, L)] = eskf.outage_end_pos(Sn, f, Rs, ko, i1, bofs[ko:i1] + Sn["ba"])[0]
    rows = []
    for (n, L), pe in res.items():
        rows.append(dict(seq=seq, cfg=cname, mode="D", L=L, rep=rep, run_id=n,
                         delta=float(np.linalg.norm(pe - res[("CTRL:bref", L)])), abs=float(np.linalg.norm(pe - pg[i1]))))
    return rows

def table_unified(C):
    """== experiments/closed_loop/unify_table.py (cfg base, mode O): RMS over reps per (run,seq), quantiles over (run,seq) units."""
    C = C[(C.cfg == "base") & (C["mode"] == "O")].copy()
    C["group"] = C.run_id.map(lambda r: r if r.startswith("CTRL:") else "learned+const (all runs)")
    rows = []
    for L, d in C.groupby("L"):
        a = d.groupby(["group", "run_id", "seq"]).agg(abs=("abs", lambda x: np.sqrt(np.mean(x**2))), delta=("delta", lambda x: np.sqrt(np.mean(x**2)))).reset_index()
        for g, e in a.groupby("group"):
            q = lambda c: np.quantile(e[c], [.25, .5, .75])
            rows.append(dict(L=L, group=g, n=len(e), abs_q25=q("abs")[0], abs_med=q("abs")[1], abs_q75=q("abs")[2],
                             delta_q25=q("delta")[0], delta_med=q("delta")[1], delta_q75=q("delta")[2]))
    return pd.DataFrame(rows)

# ---- unit-level shared-error / between-estimator ratio (paper Table III row O_base) ------------------
# copied from experiments/closed_loop/analyze_3.py: units(), fr(), boot() with sequence-cluster bootstrap B=5000
def units(d):
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq", "rep"])
    u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max() - g.delta.min(), iqr=g.delta.quantile(.75) - g.delta.quantile(.25),
                          sd_abs=g["abs"].std())).join(ctl).reset_index()
    u["r_med"] = u.common / u.med; u["r_rng"] = u.common / u.rng; u["r_iqr"] = u.common / u.iqr; u["sd_rel"] = u.sd_abs / u.common
    return u
fr = lambda u: np.array([u.r_med.median(), u.r_iqr.median(), u.r_rng.median(), u.sd_rel.median(), u.common.median(), u.med.median()])
def boot(u, fn, B=5000, seed=20261016):
    rng = np.random.default_rng(seed); seqs = sorted(u.seq.unique()); pt = fn(u); bs = []
    for _ in range(B):
        s = rng.choice(seqs, len(seqs)); bs.append(fn(pd.concat([u[u.seq == x] for x in s])))
    bs = np.array(bs, float); return pt, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
def table_ratio(C):
    rows = []
    for L, d in C[(C.cfg == "base") & (C["mode"] == "O")].groupby("L"):
        u = units(d); pt, lo, hi = boot(u, fr)
        row = dict(cfg="O_base", L=L, n_seq=u.seq.nunique(), n_units=len(u))
        for i, k in enumerate(["ratio_med", "ratio_iqr", "ratio_rng", "sdabs_over_common", "common_m", "median_delta_m"]):
            row[k] = pt[i]; row[k + "_lo"] = lo[i]; row[k + "_hi"] = hi[i]
        ps = u.groupby("seq").apply(lambda x: x.common.median() / x.med.median(), include_groups=False)
        row["per_seq_min"], row["per_seq_max"] = ps.min(), ps.max(); rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    t0 = time.time()
    tasksO = [(s, c) for s in SEQS for c in ("base",)]
    with Pool(A.procs) as pool:
        rO = pool.map(task_O, tasksO, chunksize=1)
    print("O done %.0fs" % (time.time() - t0), flush=True)
    df = pd.DataFrame([r for x in rO for r in x])
    vc.write_csv(df, A, "t2_consequences_long.csv", index=False)
    T = table_unified(df); vc.write_csv(T, A, "t2_table_unified.csv", index=False)
    TR = table_ratio(df); vc.write_csv(TR, A, "t2_table3_ratio_obase.csv", index=False); print(TR.round(3).to_string())
    # per-sequence view (as unify_table.py): eskf_self vs bref, L=10
    d = df[(df.cfg == "base") & (df.L == 10)]
    pv = d.assign(group=d.run_id.map(lambda r: r if r.startswith("CTRL:") else "learned")).pivot_table(index=["seq", "rep"], columns="group", values="abs")
    frac = float((pv["CTRL:eskf_self"] < pv["CTRL:bref"]).mean()) if "CTRL:eskf_self" in pv else np.nan
    vc.write_csv(pv.groupby("seq").median().reset_index(), A, "t2_L10_abs_by_seq.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(3).to_string())
    print("L=10: eskf_self<bref fraction over (seq,rep):", frac, "| total %.0fs" % (time.time() - t0))

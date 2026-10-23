# -*- coding: utf-8 -*-
"""Closed-loop ESKF outage experiment on real EuRoC b_hat (PREREG.md). CPU only."""
import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, yaml
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eskf
from eskf import FS, DT
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ROOT = os.path.join(REPO, "euroc_full"); sys.path.insert(0, ROOT + "/src")
from euroc_full import data_euroc, metrics
cfg = yaml.safe_load(open(ROOT + "/configs/full.yaml"))
cfg["_data_dir_abs"] = os.environ.get("EUROC_DATA_DIR", os.path.join(REPO, "data", "euroc", "machine_hall"))
cfg["seq_folders"] = {"MH_01": "MH_01_easy", "MH_02": "MH_02_easy", "MH_03": "MH_03_medium",
                      "MH_04": "MH_04_difficult", "MH_05": "MH_05_difficult"}
SEQS = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]; STRIDE = cfg["stride"]
wide = pd.read_csv(ROOT + "/results/metrics_wide.csv")
wide["regime"] = wide.run_id.str.split("__").str[0]; wide["method"] = wide.run_id.str.split("__").str[1]
wide["seed"] = wide.run_id.str.extract(r"seed(\d+)").astype(int)

# configs: name -> (sig_meas, att_scale, qscale)
CFGS = {"base": (0.05, 1.0, 1.0), "sig0.02": (0.02, 1.0, 1.0), "sig0.20": (0.20, 1.0, 1.0),
        "att0": (0.05, 0.0, 1.0), "att4": (0.05, 4.0, 1.0), "q3": (0.05, 1.0, 3.0)}
LS = (5, 10, 20)
K_O, K_D = 5, 3
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
        z = np.load(f"{ROOT}/results/runs/{r.run_id}/series.npz")
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

def build_predictors():
    ate = pd.read_csv(HERE + "/../attitude_controls/ate_per_run.csv")
    srcmap = {"a_raw_gyro(paper)": "ate_raw", "b_gyro-GT-bias(oracle)": "ate_oracle", "c_static-mean-bias": "ate_static",
              "d_mahony(kp=1,ki=0.05)": "ate_mahony"}
    A = ate[ate.source.isin(srcmap)].pivot_table(index=["run_id", "seq"], columns="source", values="ate").rename(columns=srcmap)
    rows = []
    for seq in SEQS:
        D = load_seq(seq)
        d = data_euroc.prepare_regime_a(seq, cfg)
        tws = np.asarray(d.test_win_starts, np.int64)
        mi = metrics.build_metric_inputs(d.merged, tws, np.zeros((len(tws), 3)), d.y_test, STRIDE, FS, 9.81)
        def feats(b):
            e = mi.b_ref - b
            o = dict(iabe=metrics.iabe_rot(metrics._mi_with_bhat(mi, b))["value"],
                     bias_rmse=float(np.sqrt(np.mean(e ** 2))), mean_eps=float(np.linalg.norm(e.mean(0))),
                     eps_rms=float(np.sqrt(np.mean(np.sum(e ** 2, 1)))))
            for L in LS:
                n = L * 40
                if n > len(e): o[f"iabe_out{L}"] = np.nan; continue
                a = np.einsum("nij,nj->ni", mi.R_ref[-n:], e[-n:]); dt = mi.dt
                p = np.cumsum(np.cumsum(a * dt, 0) * dt, 0); o[f"iabe_out{L}"] = float(np.sqrt(np.mean(np.sum(p ** 2, 1))))
            return o
        for rid, b in zip(D["run_ids"], D["bhat"]):
            o = feats(b); o.update(run_id=rid, seq=seq)
            r = wide[(wide.run_id == rid) & (wide.eval_sequence == seq)].iloc[0]
            o.update(iabe_paper=r.iabe_rot, bias_rmse_paper=r.bias_rmse_avg, slie_10=r.slie_10, regime=r.regime, method=r.method, seed=r.seed)
            for c in A.columns: o[c] = A.loc[(rid, seq), c]
            rows.append(o)
        for n, b in (("CTRL:bref", D["bref"]), ("CTRL:zero", np.zeros_like(D["bref"])),
                     ("CTRL:const_oracle", np.tile(D["bref"].mean(0), (len(D["bref"]), 1)))):
            o = feats(b); o.update(run_id=n, seq=seq, regime="C", method=n, seed=0); rows.append(o)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    t0 = time.time()
    PRED_ONLY = len(sys.argv) > 1 and sys.argv[1] == "pred"
    P = build_predictors(); P.to_csv(HERE + "/predictors.csv", index=False)
    chk = P[P.regime.isin(["A", "B"])]
    print("pred check iabe vs paper max rel", float((abs(chk.iabe - chk.iabe_paper) / chk.iabe_paper).max()),
          "bias_rmse vs paper", float((abs(chk.bias_rmse - chk.bias_rmse_paper) / chk.bias_rmse_paper).max()), flush=True)
    if PRED_ONLY: sys.exit(0)
    tasksO = [(s, c) for s in SEQS for c in CFGS]
    tasksD = [(s, c, r) for s in SEQS for c in D_CFGS for r in range(K_D)]
    with Pool(16) as pool:
        rO = pool.map(task_O, tasksO, chunksize=1); print("O done %.0fs" % (time.time() - t0), flush=True)
        rD = pool.map(task_D, tasksD, chunksize=1); print("D done %.0fs" % (time.time() - t0), flush=True)
    df = pd.DataFrame([r for x in rO + rD for r in x]); df.to_csv(HERE + "/consequences_long.csv", index=False)
    print(df.shape, "total %.0fs" % (time.time() - t0))

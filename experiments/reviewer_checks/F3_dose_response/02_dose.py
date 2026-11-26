# F3 step 2: dose-response. Constant offset (random + axis directions) and time-varying (alpha * actual estimator error).
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = 10.0 ** (-5 + np.arange(21) / 4.0)           # r = bias-RMSE-equivalent, m/s^2
S3 = np.sqrt(3.0)

def directions():
    """16 uniform random sphere directions per level (independent, fixed seeds) + 6 axis directions."""
    out = []
    ax = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    for k in range(len(LEVELS)):
        g = np.random.default_rng(20261126 + k).standard_normal((16, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        out.append(np.vstack([g, ax]))
    return np.stack(out)                                  # (21, 22, 3)
DIRS = directions()

def pick_runs():
    """per seq: 3 learned runs (regime A; exclude constant baselines): min / median / max bias RMSE."""
    w = RC.wide[(RC.wide.regime == "A") & (~RC.wide.method.str.startswith("constant"))]
    sel = {}
    for seq in SEQS:
        d = w[w.eval_sequence == seq].sort_values("bias_rmse_avg")
        mid = (d.bias_rmse_avg - d.bias_rmse_avg.median()).abs().idxmin()
        sel[seq] = [("min", d.iloc[0].run_id), ("med", d.loc[mid].run_id), ("max", d.iloc[-1].run_id)]
    return sel
SEL = pick_runs()

def task(args):
    seq, rep = args
    D, packs = snapshots(seq, rep); bref_h = RC.hold_series(D, D["bref"])
    rows, units, tvrows = [], [], []
    rid2b = dict(zip(D["run_ids"], D["bhat"]))
    for L, pk in packs.items():
        ko, i1, Sn, Rs = pk; n = i1 - ko
        b0 = bref_h[ko:i1]
        ab0, pe0 = end_err(D, pk, b0[None]); ab0 = float(ab0[0]); pe0 = pe0[0]
        # M from unit-vector perturbations (end position linear in bias)
        E = np.eye(3)[:, None, :] + b0[None]
        _, pe_u = end_err(D, pk, E); M = -(pe_u - pe0).T          # pe(b+d) = pe0 - M d
        units.append(dict(seq=seq, rep=rep, L=L, abs0=ab0, **{f"e0{i}": float((pe0 - D["p"][i1])[i]) for i in range(3)},
                          **{f"M{i}{j}": float(M[i, j]) for i in range(3) for j in range(3)}))
        # constant-offset dose response
        lin_err = 0.0
        for k, r in enumerate(LEVELS):
            dl = DIRS[k] * r * S3                               # (22,3)
            ab, pe = end_err(D, pk, b0[None] + dl[:, None, :])
            lin = pe0[None] - dl @ M.T; lin_err = max(lin_err, float(np.abs(lin - pe).max()))
            for j in range(22):
                rows.append(dict(seq=seq, rep=rep, L=L, k=k, r=r, dir=j, kind="axis" if j >= 16 else "rand",
                                 ux=DIRS[k, j, 0], uy=DIRS[k, j, 1], uz=DIRS[k, j, 2], abs=float(ab[j]),
                                 delta=float(np.linalg.norm(pe[j] - pe0)), abs0=ab0))
        units[-1]["lin_maxerr"] = lin_err
        # time-varying versions
        for strength, rid in SEL[seq]:
            e40 = rid2b[rid] - D["bref"]; rm = float(np.sqrt(np.mean(e40 ** 2))); em = e40.mean(0)
            hv = {"TV": RC.hold_series(D, e40), "TVmean": RC.hold_series(D, np.tile(em, (len(e40), 1))),
                  "TVdm": RC.hold_series(D, e40 - em)}
            for var, h in hv.items():
                a = LEVELS / rm
                Bm = b0[None] + a[:, None, None] * h[ko:i1][None]
                ab, pe = end_err(D, pk, Bm)
                for k, r in enumerate(LEVELS):
                    tvrows.append(dict(seq=seq, rep=rep, L=L, strength=strength, run_id=rid, rmse_run=rm, variant=var, k=k, r=r,
                                       alpha=float(a[k]), abs=float(ab[k]), delta=float(np.linalg.norm(pe[k] - pe0)), abs0=ab0))
    return rows, units, tvrows

if __name__ == "__main__":
    t = time.time()
    with Pool(4) as p: out = p.map(task, [(s, r) for s in SEQS for r in range(K)], chunksize=1)
    pd.DataFrame([r for o in out for r in o[0]]).to_csv(HERE + "/dose_const_long.csv", index=False)
    U = pd.DataFrame([r for o in out for r in o[1]]); U.to_csv(HERE + "/units.csv", index=False)
    pd.DataFrame([r for o in out for r in o[2]]).to_csv(HERE + "/dose_tv_long.csv", index=False)
    print("max linear-vs-direct err", U.lin_maxerr.max(), "time %.0fs" % (time.time() - t))
    print({s: SEL[s] for s in SEQS})

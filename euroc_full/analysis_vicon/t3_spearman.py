# -*- coding: utf-8 -*-
"""[Vicon port of experiments/closed_loop/run_closed_loop.py build_predictors() + analyze.py (S1 = median within-cell Spearman of a bias-error
predictor vs the closed-loop consequence; S2 pair agreement; S3 R2)]. Copied; changes: paths/results from vcommon; ATE-based predictors dropped
(IABE and bias RMSE are what the paper reports); only cfg base / mode O (the only one ported); cells = (regime, sequence), on V regime A only;
the bootstrap RNG is re-seeded per scenario (MH original shares one RNG over all scenarios, so only point estimates are bit-comparable).
Outputs: t3_predictors.csv, t3_spearman_scenario.csv.  CPU only."""
import os, sys, itertools
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vcommon as vc
import t2_closed_loop as rc
from euroc_full import data_euroc, metrics
from eskf import FS
A = rc.A; cfg = rc.cfg; wide = rc.wide; SEQS = rc.SEQS; STRIDE = rc.STRIDE; LS = rc.LS

def build_predictors():
    rows = []
    for seq in SEQS:
        D = rc.load_seq(seq)
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
            o.update(iabe_paper=r.iabe_rot, bias_rmse_paper=r.bias_rmse_avg, regime=r.regime, method=r.method, seed=r.seed)
            rows.append(o)
        for n, b in (("CTRL:bref", D["bref"]), ("CTRL:zero", np.zeros_like(D["bref"])),
                     ("CTRL:const_oracle", np.tile(D["bref"].mean(0), (len(D["bref"]), 1)))):
            o = feats(b); o.update(run_id=n, seq=seq, regime="C", method=n, seed=0); rows.append(o)
    return pd.DataFrame(rows)


C = None; P = None
PREDS = ["iabe", "bias_rmse", "mean_eps", "eps_rms"]
NB = 2000; RNG = np.random.default_rng(7)
def rho(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean(); d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0

def scenario(cfg, mode, L, cons):
    s = C[(C.cfg == cfg) & (C["mode"] == mode) & (C.L == L)]
    K = s.rep.nunique()
    # RMS over reps, per (run_id, seq); also per-rep matrix
    g = s.groupby(["run_id", "seq"])
    agg = g[cons].apply(lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))).rename("c").reset_index()
    m = P[P.regime.isin(["A", "B"])].merge(agg, on=["run_id", "seq"])
    m = m.merge(s[["run_id", "seq", "rep", cons]].rename(columns={cons: "cr"}), on=["run_id", "seq"]) if False else m
    cells = []
    for (reg, seq), d in m.groupby(["regime", "seq"]):
        meth = d.groupby("method")
        X = meth[PREDS + [f"iabe_out{L}"]].mean(); c = meth.c.apply(lambda x: np.sqrt(np.mean(x ** 2)))
        # per-rep consequence per method (mean over train seeds of per-rep value)
        sr = s[s.seq == seq].merge(d[["run_id", "method", "seed"]], on="run_id")
        R = sr.groupby(["method", "rep"])[cons].mean().unstack()
        cells.append(dict(reg=reg, seq=seq, X=X.loc[c.index], c=c, R=R.loc[c.index], K=K))
    return cells, K

def s1(cells, preds, idx=None):
    out = {p: [] for p in preds}
    for i, ce in enumerate(cells):
        ii = idx[i] if idx is not None else slice(None)
        for p in preds: out[p].append(rho(ce["X"][p].to_numpy()[ii], ce["c"].to_numpy()[ii]))
    return {p: float(np.median(v)) for p, v in out.items()}, out

def pairs(ce, thr=1.25):
    c = ce["c"].to_numpy(); R = ce["R"].to_numpy(); K = R.shape[1]; need = 4 if K >= 5 else 2
    res = []
    n = len(c)
    for i, j in itertools.combinations(range(n), 2):
        hi, lo = max(c[i], c[j]), min(c[i], c[j])
        if lo <= 0 or hi / lo < thr: continue
        sgn = np.sign(c[i] - c[j]); cons = (np.sign(R[i] - R[j]) == sgn).sum()
        if cons >= need: res.append((i, j, sgn))
    return res

def s2(cells, preds, boot_idx=None):
    num = {p: 0.0 for p in preds}; den = 0
    for ce in cells:
        for i, j, sgn in pairs(ce):
            den += 1
            for p in preds:
                x = ce["X"][p].to_numpy(); d = np.sign(x[i] - x[j])
                num[p] += 1.0 if d == sgn else (0.5 if d == 0 else 0.0)
    return {p: (num[p] / den if den else np.nan) for p in preds}, den

def s3(cells, preds):
    xs = {p: [] for p in preds}; ys = []
    for ce in cells:
        y = np.log(ce["c"].to_numpy()); ys.append(y - y.mean())
        for p in preds:
            x = np.log(np.maximum(ce["X"][p].to_numpy(), 1e-12)); xs[p].append(x - x.mean())
    y = np.concatenate(ys)
    return {p: float(np.corrcoef(np.concatenate(xs[p]), y)[0, 1] ** 2) for p in preds}

def boot_diff(cells, preds, ref="iabe"):
    """two-level bootstrap: resample sequences, then methods within cell. returns dict p -> samples of (rho_ref - rho_p) and rho_p."""
    seqs = sorted({ce["seq"] for ce in cells}); by = {s: [i for i, ce in enumerate(cells) if ce["seq"] == s] for s in seqs}
    # precompute per-cell rho under method-resample
    res = {p: [] for p in preds}
    for b in range(NB):
        pick = RNG.choice(seqs, len(seqs)); cl = [i for s in pick for i in by[s]]
        rr = {p: [] for p in preds}
        for i in cl:
            n = len(cells[i]["c"]); ii = RNG.integers(0, n, n)
            for p in preds: rr[p].append(rho(cells[i]["X"][p].to_numpy()[ii], cells[i]["c"].to_numpy()[ii]))
        for p in preds: res[p].append(np.median(rr[p]))
    return {p: np.array(v) for p, v in res.items()}

def ci(x): return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


rows = []
C = None
def run_scenarios(C_, P_):
    global C, P
    C, P = C_, P_
    out = []
    global RNG
    for (cfg_, mode, L) in [tuple(x) for x in C[["cfg", "mode", "L"]].drop_duplicates().itertuples(index=False)]:
        for cons in ("delta", "abs"):
            RNG = np.random.default_rng(7)
            cells, K = scenario(cfg_, mode, L, cons)
            if len(cells) < 4: continue
            preds = PREDS + [f"iabe_out{L}"]
            med, per = s1(cells, preds)
            a2, npairs = s2(cells, preds); r2 = s3(cells, preds)
            ceil = []
            for ce in cells:
                R = ce["R"].to_numpy(); h = R.shape[1] // 2
                if h >= 1: ceil.append(rho(np.sqrt((R[:, :h] ** 2).mean(1)), np.sqrt((R[:, h:] ** 2).mean(1))))
            bs = boot_diff(cells, ["iabe", "bias_rmse", "mean_eps"])
            for p in preds:
                row = dict(cfg=cfg_, mode=mode, L=L, cons=cons, pred=p, n_cells=len(cells), K=K, n_pairs=npairs, s1_median=med[p],
                           s1_pos_cells=int(sum(v > 0 for v in per[p])), s2=a2[p], s3_r2=r2[p], ceiling=float(np.median(ceil)) if ceil else np.nan)
                if p in bs:
                    lo, hi = ci(bs[p]); row.update(s1_lo=lo, s1_hi=hi)
                    if p != "iabe":
                        d = bs["iabe"] - bs[p]; lo, hi = ci(d); row.update(diff_iabe_minus=med["iabe"] - med[p], diff_lo=lo, diff_hi=hi,
                                                                           iabe_wins_cells=int(sum(a > b for a, b in zip(per["iabe"], per[p]))))
                out.append(row)
            print(cfg_, mode, L, cons, "iabe %.3f bias_rmse %.3f ceil %.2f pairs %d cells %d" % (med["iabe"], med["bias_rmse"], np.median(ceil), npairs, len(cells)), flush=True)
    return pd.DataFrame(out)

if __name__ == "__main__":
    Pr = build_predictors(); vc.write_csv(Pr, A, "t3_predictors.csv", index=False)
    chk = Pr[Pr.regime.isin(["A", "B"])]
    print("pred check iabe vs wide max rel", float((abs(chk.iabe - chk.iabe_paper) / chk.iabe_paper).max()),
          "bias_rmse vs wide", float((abs(chk.bias_rmse - chk.bias_rmse_paper) / chk.bias_rmse_paper).max()), flush=True)
    Cc = pd.read_csv(os.path.join(A.out_dir, "t2_consequences_long.csv"), comment="#")
    S = run_scenarios(Cc, Pr); vc.write_csv(S, A, "t3_spearman_scenario.csv", index=False)

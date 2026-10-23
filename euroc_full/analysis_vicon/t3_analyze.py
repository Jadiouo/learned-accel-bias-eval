# -*- coding: utf-8 -*-
"""[Vicon port of experiments/closed_loop/analyze_3.py -> Table III rows P_est/P_off/O_q10/O_q01 (+O_base) = Table IV, onset-state table,
within-sequence Spearman, Shapley shares (Fig. 3)]. Copied; changes: inputs/outputs via vcommon (t3_*.csv), O_base taken from the
t2/t3 variants run (cfg 'base'), predictors from t3_spearman.py. Outputs: t3_table3_ratio.csv, t3_table3_state.csv, t3_table3_spearman.csv,
t3_table3_shapley.csv, t3_decomp_summary.csv (base nogate/gated).  CPU only."""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vcommon as vc
ap = vc.add_args(argparse.ArgumentParser()); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--procs", type=int, default=4)
A = ap.parse_args()
B = 5000
rng = np.random.default_rng(20261016)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
rd = lambda n: pd.read_csv(os.path.join(A.out_dir, n), comment="#")
P = rd("t3_pre_outage_long.csv"); V = rd("t3_variants_long.csv")
ALL = pd.concat([P, V]); ALL["cfg"] = ALL.cfg.replace({"base": "O_base", "q10": "O_q10", "q01": "O_q01"})
pred = rd("t3_predictors.csv")

def boot(u, fn):
    seqs = sorted(u.seq.unique()); pt = fn(u); bs = []
    for _ in range(B):
        s = rng.choice(seqs, len(seqs)); bs.append(fn(pd.concat([u[u.seq == x] for x in s])))
    bs = np.array(bs, float); return pt, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)

def units(d):
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq", "rep"])
    u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max() - g.delta.min(), iqr=g.delta.quantile(.75) - g.delta.quantile(.25),
                          sd_abs=g["abs"].std())).join(ctl).reset_index()
    u["r_med"] = u.common / u.med; u["r_rng"] = u.common / u.rng; u["r_iqr"] = u.common / u.iqr; u["sd_rel"] = u.sd_abs / u.common
    return u
fr = lambda u: np.array([u.r_med.median(), u.r_iqr.median(), u.r_rng.median(), u.sd_rel.median(), u.common.median(), u.med.median()])
rows = []
for (cfg, L), d in ALL.groupby(["cfg", "L"]):
    u = units(d); pt, lo, hi = boot(u, fr)
    row = dict(cfg=cfg, L=L, n_seq=u.seq.nunique(), n_units=len(u))
    for i, k in enumerate(["ratio_med", "ratio_iqr", "ratio_rng", "sdabs_over_common", "common_m", "median_delta_m"]):
        row[k] = pt[i]; row[k + "_lo"] = lo[i]; row[k + "_hi"] = hi[i]
    ps = u.groupby("seq").apply(lambda x: x.common.median() / x.med.median(), include_groups=False)
    row["per_seq_min"], row["per_seq_max"] = ps.min(), ps.max(); rows.append(row)
T = pd.DataFrame(rows); vc.write_csv(T, A, "t3_table3_ratio.csv", index=False)
print(T[T.L.isin([5, 10])].round(2).to_string())

# start-state differences (P only)
srows = []
for (cfg, L), d in P.groupby(["cfg", "L"]):
    d = d.copy(); d["bhx"] = d.batx - d.dax; d["bhy"] = d.baty - d.day
    ref = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"]); m = d[~d.run_id.str.startswith("CTRL:")].join(ref, on=["seq", "rep"], rsuffix="_r")
    m["dtilt"] = np.hypot(m.tx - m.tx_r, m.ty - m.ty_r); m["dvel"] = np.sqrt((m.vx - m.vx_r) ** 2 + (m.vy - m.vy_r) ** 2 + (m.vz - m.vz_r) ** 2)
    m["dbat"] = np.hypot(m.batx - m.batx_r, m.baty - m.baty_r); m["dbh"] = np.hypot(m.bhx - m.bhx_r, m.bhy - m.bhy_r)
    g = m.groupby(["seq", "rep"]); u = pd.DataFrame(dict(dtilt=g.dtilt.median(), dvel=g.dvel.median(), dbat=g.dbat.median(), dbh=g.dbh.median())).reset_index()
    rr = ref.reset_index(); u = u.merge(rr[["seq", "rep", "tx", "ty", "vx", "vy", "vz"]], on=["seq", "rep"])
    u["tilt_ref"] = np.hypot(u.tx, u.ty); u["vel_ref"] = np.sqrt(u.vx ** 2 + u.vy ** 2 + u.vz ** 2)
    u["r_tilt"] = u.dtilt / u.tilt_ref; u["r_vel"] = u.dvel / u.vel_ref
    u["abs_ratio"] = u.dbat / u.dbh      # total-ba spread / b_hat spread (absorption)
    fs = lambda x: np.array([x.r_tilt.median(), x.r_vel.median(), x.dtilt.median(), x.tilt_ref.median(), x.dvel.median(), x.vel_ref.median(),
                             x.dbat.median(), x.dbh.median(), x.abs_ratio.median()])
    pt, lo, hi = boot(u, fs); row = dict(cfg=cfg, L=L, n_units=len(u))
    for i, k in enumerate(["r_tilt", "r_vel", "dtilt_deg", "tilt_ref_deg", "dvel", "vel_ref", "d_ba_total_xy", "d_bhat_xy", "ba_total_over_bhat_spread"]):
        row[k] = pt[i]; row[k + "_lo"] = lo[i]; row[k + "_hi"] = hi[i]
    srows.append(row)
S = pd.DataFrame(srows); vc.write_csv(S, A, "t3_table3_state.csv", index=False); print(S.round(3).to_string())

# within-sequence Spearman
srows = []
for (cfg, L), d in ALL.groupby(["cfg", "L"]):
    m = d[~d.run_id.str.startswith("CTRL:")]
    a = m.groupby(["seq", "run_id"]).agg(abs=("abs", lambda x: np.sqrt(np.mean(x ** 2))), delta=("delta", lambda x: np.sqrt(np.mean(x ** 2)))).reset_index()
    a = a.merge(pred[["run_id", "seq", "iabe", "bias_rmse"]], on=["run_id", "seq"])
    for tgt in ("abs", "delta"):
        for pr in ("iabe", "bias_rmse"):
            per = pd.DataFrame([dict(seq=s, rho=spearmanr(x[pr], x[tgt])[0]) for s, x in a.groupby("seq")])
            pt, lo, hi = boot(per, lambda x: np.array([x.rho.median()]))
            srows.append(dict(cfg=cfg, L=L, target=tgt, predictor=pr, n_seq=len(per), rho_med=pt[0], lo=lo[0], hi=hi[0], rho_min=per.rho.min(), rho_max=per.rho.max()))
SP = pd.DataFrame(srows); vc.write_csv(SP, A, "t3_table3_spearman.csv", index=False); print(SP[SP.L == 10].round(2).to_string())

# Shapley for variants
U = rd("t3_variants_decomp_units.csv"); C = ["pos", "vel", "tilt", "yaw", "bg"]; hr = []
def shares(u):
    t = u.total.sum(); return np.array([u["shap_" + c].sum() / t for c in C] + [u.floor.sum() / t, u.total.mean()])
for (cfg, L), u in U.groupby(["cfg", "L"]):
    pt, lo, hi = boot(u, shares); row = dict(cfg=cfg, L=L, n_units=len(u))
    for i, k in enumerate(C + ["floor", "mean_total_m"]): row[k] = pt[i]; row[k + "_lo"] = lo[i]; row[k + "_hi"] = hi[i]
    hr.append(row)
H = pd.DataFrame(hr); vc.write_csv(H, A, "t3_table3_shapley.csv", index=False); print(H.round(3).to_string())

# ---- summarize_decomp.py (base config, nogate + gated Shapley shares) ----
U = rd("t3_decomp_units.csv"); C = ["pos", "vel", "tilt", "yaw", "bg"]
rng = np.random.default_rng(7); B = 5000
def shares(u):   # pooled share = sum phi / sum total ; floor = sum floor / sum total
    t = u.total.sum(); d = {c: u["shap_" + c].sum() / t for c in C}; d["floor"] = u.floor.sum() / t
    d.update({"oat_" + c: u["oat_" + c].sum() / t for c in C}); return d
rows = []
for (var, L), u in U.groupby(["variant", "L"]):
    seqs = sorted(u.seq.unique()); pt = shares(u); bs = {k: [] for k in pt}
    for _ in range(B):
        s = rng.choice(seqs, len(seqs)); st = shares(pd.concat([u[u.seq == x] for x in s]))
        for k in bs: bs[k].append(st[k])
    for k in pt:
        ps = [shares(u[u.seq == x])[k] for x in seqs]
        rows.append(dict(variant=var, L=L, n_seq=len(seqs), n_units=len(u), part=k, share=pt[k], lo=np.percentile(bs[k], 2.5),
                         hi=np.percentile(bs[k], 97.5), per_seq_min=min(ps), per_seq_max=max(ps)))
    rows.append(dict(variant=var, L=L, n_seq=len(seqs), n_units=len(u), part="mean_total_m", share=u.total.mean(), lo=np.nan, hi=np.nan,
                     per_seq_min=np.nan, per_seq_max=np.nan))
    for k in ("pos", "vel", "tilt_deg", "yaw_deg", "bg"):
        rows.append(dict(variant=var, L=L, n_seq=len(seqs), n_units=len(u), part="startErr_median_" + k, share=u[k].median(), lo=u[k].quantile(.1), hi=u[k].quantile(.9), per_seq_min=np.nan, per_seq_max=np.nan))
S = pd.DataFrame(rows); vc.write_csv(S, A, "t3_decomp_summary.csv", index=False)
vc.write_csv(S, A, "t3_decomp_summary.csv", index=False); print(S[S.part.isin(C+["floor","mean_total_m"])].round(3).to_string())

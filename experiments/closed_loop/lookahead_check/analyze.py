# -*- coding: utf-8 -*-
"""Reads out_O.csv/out_P.csv (this dir) + stored pre_outage_long.csv/variants_long.csv/table3_ratio.csv (read-only). Writes only here."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))  # this folder
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HD = HERE
CL = os.path.normpath(os.path.join(HERE, ".."))  # experiments/closed_loop
def units(d):   # verbatim copy of analyze_3.units
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq", "rep"])
    u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max() - g.delta.min(), iqr=g.delta.quantile(.75) - g.delta.quantile(.25),
                          sd_abs=g["abs"].std())).join(ctl).reset_index()
    u["r_med"] = u.common / u.med; u["r_rng"] = u.common / u.rng; u["r_iqr"] = u.common / u.iqr; u["sd_rel"] = u.sd_abs / u.common
    return u
def stats(d):
    u = units(d)
    ps = u.groupby("seq").apply(lambda x: x.common.median() / x.med.median(), include_groups=False)
    return dict(n_units=len(u), common_m=u.common.median(), median_delta_m=u.med.median(), ratio_med=u.r_med.median(),
                per_seq_min=ps.min(), per_seq_max=ps.max(), per_seq=";".join(f"{k}:{v:.2f}" for k, v in ps.items()))

def build(df, align_e, align_b):
    """delta_est = |pe_est[align_e] - pe_bref[align_b]|, common = |pe_bref[align_b]-gt|"""
    key = ["seq", "cfg", "L", "rep"]
    e = df[(df["align"] == align_e) & (df.run_id != "CTRL:bref")]
    b = df[(df["align"] == align_b) & (df.run_id == "CTRL:bref")][key + ["ex", "ey", "ez", "abs"]].rename(columns=dict(ex="bx", ey="by", ez="bz", abs="babs"))
    m = e.merge(b, on=key)
    m["delta"] = np.sqrt((m.ex - m.bx) ** 2 + (m.ey - m.by) ** 2 + (m.ez - m.bz) ** 2)
    out = m[key + ["run_id", "delta", "abs"]].copy()
    c = b[key + ["babs"]].rename(columns=dict(babs="abs")); c["run_id"] = "CTRL:bref"; c["delta"] = 0.0
    return pd.concat([out, c[out.columns]])

O = pd.read_csv(HD + "/out_O.csv"); frames = [O]
if os.path.exists(HD + "/out_P.csv"): frames.append(pd.read_csv(HD + "/out_P.csv"))
R = pd.concat(frames)
Pst = pd.read_csv(CL + "/pre_outage_long.csv"); Vst = pd.read_csv(CL + "/variants_long.csv")
Pst = Pst[["seq", "cfg", "L", "rep", "run_id", "delta", "abs"]]
Vst = Vst[Vst.cfg == "base"].assign(cfg="O_base")[["seq", "cfg", "L", "rep", "run_id", "delta", "abs"]]
ST = pd.concat([Pst, Vst]); T3 = pd.read_csv(CL + "/table3_ratio.csv")

# ---------------- A: reproduction ----------------
lines = []
cur = build(R, "cur", "cur")
key = ["seq", "cfg", "L", "rep", "run_id"]
J = cur.merge(ST, on=key, suffixes=("", "_st"), how="left")
J.to_csv(HD + "/A_rowwise_join.csv", index=False)
for cfg, g in J.groupby("cfg"):
    g = g[g.run_id != "CTRL:bref"]
    lines.append(f"[A rowwise] {cfg}: rows rerun={len(g)}, matched in stored={g.delta_st.notna().sum()}, "
                 f"max|d delta|={np.nanmax(abs(g.delta - g.delta_st)):.3e} m, max rel d delta={np.nanmax(abs(g.delta - g.delta_st) / g.delta_st):.3e}, "
                 f"max|d abs|={np.nanmax(abs(g['abs'] - g.abs_st)):.3e} m, max rel d abs={np.nanmax(abs(g['abs'] - g.abs_st) / g.abs_st):.3e}")
    c = J[(J.cfg == cfg) & (J.run_id == "CTRL:bref")]
    lines.append(f"[A rowwise] {cfg} CTRL:bref abs: rows={len(c)}, matched={c.abs_st.notna().sum()}, max|d|={np.nanmax(abs(c['abs'] - c.abs_st)):.3e}, max rel={np.nanmax(abs(c['abs'] - c.abs_st) / c.abs_st):.3e}")
# ratio recomputation, rerun vs stored (same rows) vs table3_ratio.csv
arows = []
for (cfg, L), g in cur.groupby(["cfg", "L"]):
    names = set(g.run_id)
    s = ST[(ST.cfg == cfg) & (ST.L == L)]
    s_same = s[s.run_id.isin(names)]; s_full = s
    t = T3[(T3.cfg == cfg) & (T3.L == L)].iloc[0]
    r = dict(cfg=cfg, L=L, n_est_rerun=len(names) - 1, n_est_stored=s_full.run_id[~s_full.run_id.str.startswith("CTRL:")].nunique())
    for nm, d in (("rerun", g), ("stored_same_rows", s_same), ("stored_full", s_full)):
        for k, v in stats(d).items():
            if k != "per_seq": r[f"{nm}.{k}"] = v
    r["table3.ratio_med"] = t.ratio_med; r["table3.per_seq_min"] = t.per_seq_min; r["table3.common_m"] = t.common_m; r["table3.median_delta_m"] = t.median_delta_m
    arows.append(r)
A = pd.DataFrame(arows); A.to_csv(HD + "/A_ratio_compare.csv", index=False)

# ---------------- B/C: alignment comparison ----------------
combos = {"cur": ("cur", "cur"), "caus_est": ("caus", "cur"), "caus_both": ("caus", "caus"),
          "holdrule_est": ("caush", "cur"), "holdrule_both": ("caush", "caush")}
crow = []; ranks = {}
for name, (ae, ab) in combos.items():
    d = build(R, ae, ab)
    for (cfg, L), g in d.groupby(["cfg", "L"]):
        st = stats(g); st.update(cfg=cfg, L=L, align=name); crow.append(st); ranks[(name, cfg, L)] = g
C = pd.DataFrame(crow)
# Spearman of per-estimator terminal difference (delta) rank, cur vs each causal variant: per unit, and per (seq) using rms over reps
sp = []
for (name, cfg, L), g in ranks.items():
    if name == "cur": continue
    g0 = ranks[("cur", cfg, L)]
    a = g0[g0.run_id != "CTRL:bref"].merge(g[g.run_id != "CTRL:bref"], on=["seq", "cfg", "L", "rep", "run_id"], suffixes=("_c", "_k"))
    ru = [spearmanr(x.delta_c, x.delta_k)[0] for _, x in a.groupby(["seq", "rep"])]
    rs = []
    for s_, x in a.groupby("seq"):
        y = x.groupby("run_id").agg(c=("delta_c", lambda v: np.sqrt(np.mean(v ** 2))), k=("delta_k", lambda v: np.sqrt(np.mean(v ** 2))))
        rs.append(spearmanr(y.c, y.k)[0])
    ra = [spearmanr(x["abs_c"], x["abs_k"])[0] for _, x in a.groupby(["seq", "rep"])]
    sp.append(dict(cfg=cfg, L=L, align=name, rho_delta_unit_median=np.median(ru), rho_delta_unit_min=np.min(ru),
                   rho_delta_seqrms_median=np.median(rs), rho_delta_seqrms_min=np.min(rs), rho_abs_unit_median=np.median(ra), rho_abs_unit_min=np.min(ra)))
S = pd.DataFrame(sp)
C = C.merge(S, on=["cfg", "L", "align"], how="left"); C.to_csv(HD + "/summary_C.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
with open(HD + "/analyze_stdout.txt", "w") as fh:
    def P(*a):
        print(*a); print(*a, file=fh)
    P("\n".join(lines)); P("\n== A ratio compare =="); P(A.round(4).T.to_string())
    P("\n== C summary (L=5,10) =="); P(C[C.L.isin([5, 10])].drop(columns=["per_seq"]).round(3).to_string())
    P("\n== per-seq ratios =="); 
    for _, r in C[C.L.isin([5, 10])].iterrows(): P(r.cfg, r.L, r["align"], r.per_seq)

# -*- coding: utf-8 -*-
"""Commander check (2026-11-18): full-pool P_off rerun (out_P_full.csv, stride 1) vs stored pre_outage_long.csv,
and current vs causal alignment. Read-only on stored files; prints only. units()/build() copied from analyze.py in this dir."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HD = os.path.dirname(os.path.abspath(__file__))  # this folder
CL = os.path.normpath(os.path.join(HD, ".."))  # experiments/closed_loop
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
def units(d):
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq", "rep"])
    u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max() - g.delta.min(),
                          iqr=g.delta.quantile(.75) - g.delta.quantile(.25))).join(ctl).reset_index()
    u["r_med"] = u.common / u.med; u["r_rng"] = u.common / u.rng; u["r_iqr"] = u.common / u.iqr
    return u
def build(df, ae, ab):
    key = ["seq", "cfg", "L", "rep"]
    e = df[(df["align"] == ae) & (df.run_id != "CTRL:bref")]
    b = df[(df["align"] == ab) & (df.run_id == "CTRL:bref")][key + ["ex", "ey", "ez", "abs"]].rename(columns=dict(ex="bx", ey="by", ez="bz", abs="babs"))
    m = e.merge(b, on=key)
    m["delta"] = np.sqrt((m.ex - m.bx) ** 2 + (m.ey - m.by) ** 2 + (m.ez - m.bz) ** 2)
    out = m[key + ["run_id", "delta", "abs"]].copy()
    c = b[key + ["babs"]].rename(columns=dict(babs="abs")); c["run_id"] = "CTRL:bref"; c["delta"] = 0.0
    return pd.concat([out, c[out.columns]])
R = pd.read_csv(HD + "/out_P_full.csv")
print("rows:", len(R), "| aligns:", sorted(R["align"].unique()), "| cfg:", sorted(R.cfg.unique()), "| L:", sorted(R.L.unique()),
      "| estimators per seq:", R[R.run_id != "CTRL:bref"].groupby("seq").run_id.nunique().to_dict())
ST = pd.read_csv(CL + "/pre_outage_long.csv"); ST = ST[ST.cfg == "P_off"][["seq", "cfg", "L", "rep", "run_id", "delta", "abs"]]
T3 = pd.read_csv(CL + "/table3_ratio.csv"); print("table3_ratio.csv columns:", list(T3.columns))
print(T3[T3.cfg == "P_off"].to_string(index=False))
cur = build(R, "cur", "cur")
J = cur.merge(ST, on=["seq", "cfg", "L", "rep", "run_id"], suffixes=("", "_st"), how="outer", indicator=True)
print("\n[A] row match rerun vs stored:", J["_merge"].value_counts().to_dict())
B = J[J["_merge"] == "both"]; e = B[B.run_id != "CTRL:bref"]
print(f"[A] estimators rows={len(e)}: max|d delta|={abs(e.delta - e.delta_st).max():.3e} m, max|d abs|={abs(e['abs'] - e.abs_st).max():.3e} m")
c = B[B.run_id == "CTRL:bref"]; print(f"[A] bref rows={len(c)}: max|d abs|={abs(c['abs'] - c.abs_st).max():.3e} m")
rows = []
for name, ae, ab in [("cur (as in paper)", "cur", "cur"), ("causal est, bref cur", "caus", "cur"), ("causal est+bref", "caus", "caus"),
                     ("hold-rule est, bref cur", "caush", "cur"), ("hold-rule est+bref", "caush", "caush")]:
    d = build(R, ae, ab)
    for L, g in d.groupby("L"):
        u = units(g)
        ps_a = u.groupby("seq").r_med.median()                                     # median over reps of the unit ratio
        ps_b = u.groupby("seq").apply(lambda x: x.common.median() / x.med.median(), include_groups=False)  # ratio of medians
        rows.append(dict(align=name, L=L, n_units=len(u), shared_m=u.common.median(), diff_m=u.med.median(), ratio_med=u.r_med.median(),
                         ratio_iqr=u.r_iqr.median(), ratio_rng=u.r_rng.median(), perseq_min_medratio=ps_a.min(), perseq_min_ratiomed=ps_b.min(),
                         argmin=ps_a.idxmin(), perseq=";".join(f"{k[-2:]}:{v:.2f}" for k, v in ps_a.items())))
S = pd.DataFrame(rows); S.to_csv(HD + "/summary_P_full.csv", index=False)
print("\n[B/C] full pool, P_off:"); print(S.to_string(index=False))
# rank stability of per-run delta between current and causal alignment (seed-wise, per unit)
a = build(R, "cur", "cur"); b = build(R, "caus", "cur"); k = ["seq", "L", "rep", "run_id"]
m = a[a.run_id != "CTRL:bref"].merge(b[b.run_id != "CTRL:bref"], on=k, suffixes=("_cur", "_caus"))
rho = m.groupby(["seq", "L", "rep"]).apply(lambda x: spearmanr(x.delta_cur, x.delta_caus)[0], include_groups=False)
print("\n[C] Spearman of per-run delta, current vs causal, per unit: median=%.4f min=%.4f" % (rho.median(), rho.min()))
print("[C] relative change of delta (causal/current - 1): median=%.3f, 5%%=%.3f, 95%%=%.3f" % tuple(np.percentile(m.delta_caus / m.delta_cur - 1, [50, 5, 95])))

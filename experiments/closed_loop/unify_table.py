# -*- coding: utf-8 -*-
"""Post-review: unified C_abs / C_delta table from consequences_long.csv (read-only). Writes table_unified.csv"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(HERE + "/consequences_long.csv")
C = C[(C.cfg == "base") & (C["mode"] == "O")]
def grp(r):
    if r.startswith("CTRL:"): return r
    return "learned+const (all runs)"
C["group"] = C.run_id.map(grp)
rows = []
for L, d in C.groupby("L"):
    # per (run_id, seq): RMS over reps (PREREG def) -> then pooled quantiles over (run,seq) units
    a = d.groupby(["group", "run_id", "seq"]).agg(abs=("abs", lambda x: np.sqrt(np.mean(x**2))), delta=("delta", lambda x: np.sqrt(np.mean(x**2)))).reset_index()
    for g, e in a.groupby("group"):
        q = lambda c: np.quantile(e[c], [.25, .5, .75])
        rows.append(dict(L=L, group=g, n=len(e), abs_q25=q("abs")[0], abs_med=q("abs")[1], abs_q75=q("abs")[2],
                         delta_q25=q("delta")[0], delta_med=q("delta")[1], delta_q75=q("delta")[2]))
T = pd.DataFrame(rows); T.to_csv(HERE + "/table_unified.csv", index=False)
pd.set_option("display.width", 200); print(T.round(3).to_string())
# per-sequence view for eskf_self vs bref at L=10 (same seq, same rep): is eskf_self abs < bref abs consistently?
d = C[(C.L == 10)]
pv = d.pivot_table(index=["seq", "rep"], columns="group", values="abs")
print("\nL=10 per seq,rep abs: eskf_self < bref fraction", float((pv["CTRL:eskf_self"] < pv["CTRL:bref"]).mean()))
print(pv[["CTRL:bref", "CTRL:eskf_self", "CTRL:zero"]].groupby("seq").median().round(2))

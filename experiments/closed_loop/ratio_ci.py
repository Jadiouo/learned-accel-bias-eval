# -*- coding: utf-8 -*-
"""POST-HOC 2 (outline stage). Paired-unit ratio with sequence-cluster bootstrap. Input: gated_consequences.csv. Output: table_ratio_ci.csv"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
G = pd.read_csv(HERE + "/gated_consequences.csv")
G["method"] = G.run_id.str.split("__").str[1]
rng = np.random.default_rng(20261015); B = 5000
def units(d, learned_only):
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]
    if learned_only: m = m[~m.method.str.startswith("constant")]
    g = m.groupby(["seq", "rep"]).delta
    u = pd.DataFrame(dict(med=g.median(), rng=g.max() - g.min(), iqr=g.quantile(.75) - g.quantile(.25))).join(ctl).reset_index()
    for k in ("med", "rng", "iqr"): u["r_" + k] = u.common / u[k]            # common / spread
    return u
def stat(u):  # median over units of per-unit ratio; and ratio of medians (old definition)
    return dict(paired_med=u.r_med.median(), paired_rng=u.r_rng.median(), paired_iqr=u.r_iqr.median(),
                old=u.common.median() / u.med.median())
rows = []
for (var, L), d in G.groupby(["variant", "L"]):
    for lo in (False, True):
        u = units(d, lo); seqs = sorted(u.seq.unique()); pt = stat(u)
        bs = {k: [] for k in pt}
        for _ in range(B):
            s = rng.choice(seqs, len(seqs)); uu = pd.concat([u[u.seq == x] for x in s]); st = stat(uu)
            for k in bs: bs[k].append(st[k])
        # per-sequence ratio of rep-RMS-free medians, for range
        ps = u.groupby("seq").apply(lambda x: x.common.median() / x.med.median(), include_groups=False)
        row = dict(variant=var, L=L, set="learned_only" if lo else "learned+const", n_seq=len(seqs), n_units=len(u))
        for k in pt:
            row[k] = pt[k]; row[k + "_lo"], row[k + "_hi"] = np.percentile(bs[k], [2.5, 97.5])
        row["inv_paired_med"] = 1 / pt["paired_med"]; row["inv_paired_rng"] = 1 / pt["paired_rng"]
        row["per_seq_min"], row["per_seq_max"] = ps.min(), ps.max()
        row["unit_ratio_q10"], row["unit_ratio_q90"] = u.r_med.quantile(.1), u.r_med.quantile(.9)
        rows.append(row)
T = pd.DataFrame(rows); T.to_csv(HERE + "/table_ratio_ci.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T.round(1).to_string())

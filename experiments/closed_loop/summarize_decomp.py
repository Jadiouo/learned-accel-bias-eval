# -*- coding: utf-8 -*-
"""POST-HOC 2. Summarize decomp_units.csv: Shapley shares with sequence-cluster bootstrap CI + stacked bar. Outputs decomp_summary.csv, fig_decomp.png"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
U = pd.read_csv(HERE + "/decomp_units.csv"); C = ["pos", "vel", "tilt", "yaw", "bg"]
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
S = pd.DataFrame(rows); S.to_csv(HERE + "/decomp_summary.csv", index=False)
pd.set_option("display.width", 250); print(S.round(3).to_string())
# stacked bar
fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
col = dict(pos="#8c8c8c", vel="#4c72b0", tilt="#dd8452", yaw="#55a868", bg="#c44e52", floor="#d0d0d0")
for ax, L in zip(axs, (5, 10, 20)):
    xs = []; 
    for i, var in enumerate(("nogate", "gated")):
        s = S[(S.variant == var) & (S.L == L) & S.part.isin(C + ["floor"])].set_index("part")
        bot_p = 0; bot_n = 0
        for c in C + ["floor"]:
            v = s.loc[c, "share"]
            if v >= 0: ax.bar(i, v, bottom=bot_p, color=col[c], label=c if (i == 0 and L == 5) else None, edgecolor="w"); bot_p += v
            else: ax.bar(i, v, bottom=bot_n, color=col[c], edgecolor="w", label=c if (i == 0 and L == 5) else None); bot_n += v
        n = int(s.n_seq.iloc[0]) if "n_seq" in s else 0
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no gate", "gated"]); ax.axhline(0, c="k", lw=.5)
    ax.set_title(f"L = {L} s (n_seq = {int(S[(S.L==L)].n_seq.iloc[0])})")
axs[0].set_ylabel("share of b_ref end-of-outage error C_abs\n(Shapley)"); axs[0].legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(HERE + "/fig_decomp.png", dpi=160)

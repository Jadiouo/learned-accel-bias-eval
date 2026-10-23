# -*- coding: utf-8 -*-
"""POST-HOC: summarize gated_consequences.csv. Writes table_gated.csv, table_gated_spearman.csv"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
G = pd.read_csv(HERE + "/gated_consequences.csv"); P = pd.read_csv(HERE + "/predictors.csv")
rms = lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))
rows = []; srows = []
for (var, L), d in G.groupby(["variant", "L"]):
    a = d.groupby(["run_id", "seq"]).agg(abs=("abs", rms), delta=("delta", rms), gf=("gate_frac", "mean")).reset_index()
    lm = a[~a.run_id.str.startswith("CTRL:")]
    def ctrl(n): return a[a.run_id == n]
    bref = ctrl("CTRL:bref")
    row = dict(variant=var, L=L, gate_frac=lm.gf.mean(), bref_abs_med=bref["abs"].median(),
               bref_abs_q25=bref["abs"].quantile(.25), bref_abs_q75=bref["abs"].quantile(.75),
               learn_abs_med=lm["abs"].median(), learn_abs_q25=lm["abs"].quantile(.25), learn_abs_q75=lm["abs"].quantile(.75),
               learn_delta_med=lm.delta.median(), learn_delta_q25=lm.delta.quantile(.25), learn_delta_q75=lm.delta.quantile(.75),
               learn_delta_min=lm.delta.min(), learn_delta_max=lm.delta.max(),
               eskf_abs_med=ctrl("CTRL:eskf_self")["abs"].median(), eskf_delta_med=ctrl("CTRL:eskf_self").delta.median(),
               zero_abs_med=ctrl("CTRL:zero")["abs"].median(), zero_delta_med=ctrl("CTRL:zero").delta.median())
    row["ratio_common_over_method"] = row["bref_abs_med"] / row["learn_delta_med"]
    rows.append(row)
    # within-sequence (cell = regime,seq) Spearman: predictors vs consequence, mean over train seeds per method
    m = P[P.regime.isin(["A", "B"])].merge(lm[["run_id", "seq", "abs", "delta"]], on=["run_id", "seq"])
    for cons in ("delta", "abs"):
        for pr in ("iabe", "bias_rmse", "ate_oracle", "ate_raw"):
            rs = []
            for (reg, seq), c in m.groupby(["regime", "seq"]):
                g = c.groupby("method")[[pr, cons]].mean()
                if len(g) > 3: rs.append(spearmanr(g[pr], g[cons])[0])
            srows.append(dict(variant=var, L=L, cons=cons, pred=pr, n_cells=len(rs), rho_med=np.nanmedian(rs), rho_min=np.nanmin(rs), rho_max=np.nanmax(rs)))
T = pd.DataFrame(rows); T.to_csv(HERE + "/table_gated.csv", index=False)
S = pd.DataFrame(srows); S.to_csv(HERE + "/table_gated_spearman.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).T.to_string()); print(S.round(2).to_string())

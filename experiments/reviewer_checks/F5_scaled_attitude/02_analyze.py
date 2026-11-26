# -*- coding: utf-8 -*-
"""Step 2: reproduction checks + direction agreement vs scale. Pipeline: ../F1_noise_ceiling/common.py (252 splits enumerated),
sequence-cluster bootstrap as in ../F4_seqcluster/f4_ci.py but index draws shared across configs (paired). Seed 20261126."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
from common import *
from scipy.stats import spearmanr
units, bias, ate = load()
w = pd.read_csv(R + "euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R + "euroc_full/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
wdl = w[w.method.isin(DL)]
iabe = {u: wdl[(wdl.regime == u[0]) & (wdl.eval_sequence == u[1])].pivot_table(index="seed", columns="method", values="iabe_rot")[DL].values for u in units}
d = pd.read_csv(HERE + "/ate_scaled_per_run.csv"); dg = pd.read_csv(HERE + "/diag_by_case.csv")
d = d.merge(wdl.rename(columns={"eval_sequence": "seq"})[["run_id", "seq", "iabe_rot", "bias_rmse_avg"]], on=["run_id", "seq"], validate="many_to_one")
configs = list(dict.fromkeys(d.config))
mats = {c: {u: g.pivot_table(index="seed", columns="method", values="ate")[DL].values
            for u in units for g in [d[(d.config == c) & (d.regime == u[0]) & (d.seq == u[1])]]} for c in configs}
chk = {}
# ---- reproduction checks ----
att = pd.read_csv(R + "experiments/attitude_controls/ate_per_run_gated.csv"); att = att[(att.source == "b_gyro-GT-bias(oracle)") & att.method.isin(DL)][["run_id", "seq", "ate"]].rename(columns={"ate": "ate_ref"})
x = d[d.config == "W|s=1"].merge(att, on=["run_id", "seq"], validate="one_to_one"); chk["n_s1_vs_oracle_csv"] = len(x)
chk["max_rel_s1_vs_oracle_csv"] = float((abs(x.ate - x.ate_ref) / x.ate_ref).max())
x = d[d.config == "W|s=0"]; chk["n_s0_vs_iabe"] = len(x); chk["max_rel_s0_vs_iabe_rot"] = float((abs(x.ate - x.iabe_rot) / x.iabe_rot).max())
for v in "BT":
    x = d[d.config == f"{v}|s=1"].merge(att, on=["run_id", "seq"], validate="one_to_one"); chk[f"max_rel_{v}_s1_vs_oracle_csv"] = float((abs(x.ate - x.ate_ref) / x.ate_ref).max())
    x = d[d.config == f"{v}|s=0"]; chk[f"max_rel_{v}_s0_vs_iabe_rot"] = float((abs(x.ate - x.iabe_rot) / x.iabe_rot).max())
# ---- agreement machinery ----
G = groups(units); seqs = sorted({u[1] for u in units})
sel_b = precompute_selection(units, bias); sel_i = precompute_selection(units, iabe)
rng = np.random.default_rng(20261126); NB = 4000
Wt = {}
for g, gi in G.items():
    ss = [s for s in seqs if any(units[i][1] == s for i in gi)]
    M = np.zeros((NB, len(units)))
    for b in range(NB):
        for j in rng.integers(0, len(ss), len(ss)):
            for i in gi:
                if units[i][1] == ss[j]: M[b, i] += 1
    Wt[g] = M
def counts(src, sel): A, N = agree_counts(src, units, sel); return A.sum(0), N.sum(0)
def summ(a, n, g):
    gi = G[g]; pt = a[gi].sum() / n[gi].sum(); bs = (Wt[g] @ a) / (Wt[g] @ n); return pt, bs
cache = {}
refs = {"s1_csv_oracle": counts(ate["oracle"], sel_b), "raw_csv": counts(ate["raw"], sel_b), "iabe_vs_bias": counts(iabe, sel_b)}
for c in configs: cache[c] = (counts(mats[c], sel_b), counts(mats[c], sel_i))
chk["agree_ALL_W_s1"] = summ(*cache["W|s=1"][0], "ALL")[0]; chk["agree_ALL_oracle_csv"] = summ(*refs["s1_csv_oracle"], "ALL")[0]
chk["agree_ALL_W_s0"] = summ(*cache["W|s=0"][0], "ALL")[0]; chk["agree_ALL_iabe_vs_bias"] = summ(*refs["iabe_vs_bias"], "ALL")[0]
for g in "AB": chk[f"agree_{g}_W_s0"] = summ(*cache["W|s=0"][0], g)[0]; chk[f"agree_{g}_iabe_vs_bias"] = summ(*refs["iabe_vs_bias"], g)[0]
chk["agree_ALL_raw_csv"] = summ(*refs["raw_csv"], "ALL")[0]
# leakage diagnostics at the table's definitions
regime_iabe = w.groupby("regime").iabe_rot.median(); all_iabe = w.iabe_rot.median()
dg["regime_iabe"] = dg.regime.map(regime_iabe)
rows = []
for c in configs:
    for g in ["ALL", "A", "B"]:
        sub = dg[(dg.config == c) & (dg.regime.isin(list("AB") if g == "ALL" else [g]))]
        iab = all_iabe if g == "ALL" else regime_iabe[g]
        dd = d[(d.config == c) & (d.regime.isin(list("AB") if g == "ALL" else [g]))]
        (a, n), (ai, ni) = cache[c]; pt, bs = summ(a, n, g); (a1, n1), _ = cache["W|s=1"]; p1, b1 = summ(a1, n1, g)
        pti, bsi = summ(ai, ni, g)
        rows.append(dict(config=c, group=g, theta_bar_med=sub.theta_bar_deg.median(), theta_bar_min=sub.theta_bar_deg.min(), theta_bar_max=sub.theta_bar_deg.max(),
                         ratio_med=sub.leak_ratio.median(), ratio_min=sub.leak_ratio.min(), ratio_max=sub.leak_ratio.max(),
                         leak_multiple=sub.leak_PL_m.median() / iab, ate_med=dd.ate.median(), ate_over_iabe_med=(dd.ate / dd.iabe_rot).median(),
                         s_med=sub.s_used.median(), agree=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         delta_vs_s1=pt - p1, dlo=np.percentile(bs - b1, 2.5), dhi=np.percentile(bs - b1, 97.5),
                         agree_with_iabe=pti, ilo=np.percentile(bsi, 2.5), ihi=np.percentile(bsi, 97.5)))
P = pd.DataFrame(rows); P.to_csv(HERE + "/results_agreement.csv", index=False)
refrows = []
for k, (a, n) in refs.items():
    for g in ["ALL", "A", "B"]:
        pt, bs = summ(a, n, g); refrows.append(dict(ref=k, group=g, agree=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
pd.DataFrame(refrows).to_csv(HERE + "/results_reference.csv", index=False)
# trend statistic on ratio-matched series (targets 30..0.1)
tr = {}
for g in ["ALL", "A", "B"]:
    sub = P[(P.group == g) & P.config.str.startswith("RM|")].copy(); sub["t"] = sub.config.str[5:].astype(float); sub = sub[sub.t >= 0.1]
    tr[g] = float(spearmanr(sub.agree, -np.log(sub.t))[0])
chk["spearman_agree_vs_neg_log_ratio_RM_30_to_0.1"] = tr
json.dump(chk, open(HERE + "/repro_check.json", "w"), indent=1, default=float)
pd.set_option("display.width", 280); pd.set_option("display.max_columns", 30)
print(json.dumps(chk, indent=1, default=float))
cols = ["config", "group", "theta_bar_med", "ratio_med", "leak_multiple", "ate_over_iabe_med", "agree", "lo", "hi", "delta_vs_s1", "dlo", "dhi", "agree_with_iabe"]
print(P[cols].round(3).to_string())
print(pd.DataFrame(refrows).round(3).to_string())

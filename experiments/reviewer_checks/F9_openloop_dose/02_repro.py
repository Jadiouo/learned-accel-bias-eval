# -*- coding: utf-8 -*-
"""Step 2. Reproduction checks (must pass before any new analysis).
(a) operator-based ATE/IABE/bias RMSE at delta=0 vs stored per-run values (ate_per_run_gated.csv, vqf/v2/ate_per_run_v2.csv source v_vqf_clip10(cold), metrics_wide).
(b) scaled-oracle operator ATE vs F5 ate_scaled_per_run.csv.
(c) 9-case direction agreement from the stored ATEs and from the operator-rebuilt ATEs (H0 pipeline): raw 0.5586, oracle 0.5598, IABE 0.8024.
(d) leakage diagnostic: oracle median theta-bar 0.1245 deg (paper 0.125).  Writes repro_check.json."""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f9_lib as F
H0 = F.H0; R = H0.R
D = pickle.load(open(HERE + "/ops.pkl", "rb")); OPS, HR, RUNS = D["OPS"], D["HR"], D["RUNS"]
runs = pd.DataFrame([{k: v for k, v in r.items() if k in ("regime", "seq", "run_id", "method", "seed", "rmse_avg_ref", "iabe_ref")} for r in RUNS])
out = {}
# (a) values at delta = 0
att = pd.read_csv(R + "experiments/attitude_controls/ate_per_run_gated.csv")
vq = pd.read_csv(R + "experiments/attitude_controls/vqf/v2/ate_per_run_v2.csv")
ref = pd.concat([att, vq[vq.source == "v_vqf_clip10(cold)"]])
ref = ref[ref.method.isin(F.DL)]
rows = []
for i, r in enumerate(RUNS):
    for k, (sid, _) in F.NAMES.items():
        if sid is None: continue
        rows.append((r["run_id"], r["seq"], sid, np.sqrt(r["Y2"][k])))
mine = pd.DataFrame(rows, columns=["run_id", "seq", "source", "ate_mine"]).merge(ref[["run_id", "seq", "source", "ate"]], on=["run_id", "seq", "source"])
out["ate_n_compared"] = len(mine); out["ate_max_rel_err"] = float((abs(mine.ate_mine - mine.ate) / mine.ate).max())
out["ate_max_rel_err_by_source"] = {s: float((abs(g.ate_mine - g.ate) / g.ate).max()) for s, g in mine.groupby("source")}
out["iabe_max_rel_err"] = float(max(abs(np.sqrt(r["I2"]) - r["iabe_ref"]) / r["iabe_ref"] for r in RUNS))
out["bias_rmse_max_rel_err"] = float(max(abs(np.sqrt(r["m2"] / 3) - r["rmse_avg_ref"]) / r["rmse_avg_ref"] for r in RUNS))
# (b) scaled oracle vs F5
f5 = pd.read_csv(R + "experiments/reviewer_checks/F5_scaled_attitude/ate_scaled_per_run.csv")
bad = {}
for key, cfg in (("orc30", "W|s=0.3"), ("orc10", "W|s=0.1"), ("orc03", "W|s=0.03")):
    a = pd.DataFrame([(r["run_id"], r["seq"], np.sqrt(r["Y2"][key])) for r in RUNS], columns=["run_id", "seq", "m"])
    b = f5[f5.config == cfg][["run_id", "seq", "ate"]].merge(a, on=["run_id", "seq"]); assert len(b) == 990
    bad[cfg] = float((abs(b.m - b.ate) / b.ate).max())
out["f5_scaled_max_rel_err"] = bad
# (c) agreement
units, bias, iabe, seeds = H0.load_mh(); cl = [u[1] for u in units]; allidx = list(range(len(units)))
ag = H0.Agree(units, bias, 11)
def mats(valfn):
    M = {}
    for u in units:
        g = runs[(runs.regime == u[0]) & (runs.seq == u[1])]
        idx = {(r.method, r.seed): i for i, r in zip(g.index, g.itertuples())}
        M[u] = np.array([[valfn(RUNS[idx[(m, s)]]) for m in F.DL] for s in seeds])
    return M
res = {}
for name, M in (("raw ATE (stored csv)", H0.ate_from_long(att[att.method.isin(F.DL)], units, seeds, "a_raw_gyro(paper)")),
                ("oracle ATE (stored csv)", H0.ate_from_long(att[att.method.isin(F.DL)], units, seeds, "b_gyro-GT-bias(oracle)")),
                ("raw ATE (operators)", mats(lambda r: np.sqrt(r["Y2"]["raw"]))),
                ("oracle ATE (operators)", mats(lambda r: np.sqrt(r["Y2"]["oracle"]))),
                ("IABE (operators)", mats(lambda r: np.sqrt(r["I2"]))),
                ("VQF clip10 cold ATE (operators)", mats(lambda r: np.sqrt(r["Y2"]["vqf"])))):
    a, n = ag.counts(M); pt, lo, hi, g = H0.jackknife(a, n, cl, allidx); res[name] = dict(point=pt, lo=lo, hi=hi, clusters=g)
out["agreement_9case"] = res
out["agreement_expected"] = {"VQF clip10 cold": "0.5686 [0.4647, 0.6726] (H0 canonical_agreement.csv, nine cases pooled)", "raw": "0.5586 [0.4600, 0.6573]", "oracle": "0.5598 [0.4227, 0.6969]", "IABE": "0.8024"}
# (d) leakage
th = {s: [OPS[(seq, s)]["theta_deg"] for seq in F.SEQS] for s in F.NAMES}
out["oracle_theta_bar_median_deg_over_5seq"] = float(np.median(th["oracle"]))
json.dump(out, open(HERE + "/repro_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))

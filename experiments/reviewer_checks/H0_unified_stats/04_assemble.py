"""H0 step 4: add noise-ceiling rows (K=11 included, K=10 excluded) to canonical_core.csv -> canonical_agreement.csv; also ceiling_controls.csv."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
core = pd.read_csv(HERE+"/canonical_core.csv")
c = pd.concat([pd.read_csv(HERE+"/ceiling_K11_N2000.csv"), pd.read_csv(HERE+"/ceiling_K10_N2000.csv")])
c[c.variant.isin(["C0", "C1"])].to_csv(HERE+"/ceiling_controls.csv", index=False)
rows = []
for (att, grp, mam), x in c[c.variant.isin(["V1", "V2", "V3"])].groupby(["attitude", "group", "mamba"]):
    sid = {"raw": "a_raw_gyro(paper)", "oracle": "b_gyro-GT-bias(oracle)"}[att]
    for _, r in x.iterrows():
        rows.append(dict(dataset="Machine Hall", selector="bias RMSE", measure=f"ATE noise ceiling {r.variant} (simulated)", attitude_source=att, source_id=sid, group=grp, mamba=mam,
                         point=r.ceil_median, lo=r.ceil_p2_5, hi=r.ceil_p97_5, lo_raw=r.ceil_p2_5, hi_raw=r.ceil_p97_5, n_pairs=r.n_pairs, n_clusters=np.nan,
                         interval_type="2.5-97.5 percentile of %d simulations (fixed cases); point = median" % r.n_sim,
                         note="seed %s; observed %.3f lies at percentile %.4f of the ceiling distribution" % (r.seed, r.observed_point, r.obs_percentile_in_ceiling)))
    m = x.set_index("variant").ceil_median[["V1", "V2", "V3"]]
    rows.append(dict(dataset="Machine Hall", selector="bias RMSE", measure="ATE noise ceiling band (min-max of V1-V3 medians)", attitude_source=att, source_id=sid, group=grp, mamba=mam,
                     point=float(m.median()), lo=float(m.min()), hi=float(m.max()), lo_raw=float(m.min()), hi_raw=float(m.max()), n_pairs=x.n_pairs.iloc[0], n_clusters=np.nan,
                     interval_type="range of the three variants' medians (Fig. 2 band); point = median of the three", note=""))
P = pd.concat([core, pd.DataFrame(rows)], ignore_index=True)
P["clipped"] = (P.lo_raw < 0) | (P.hi_raw > 1)
key = ["dataset", "selector", "measure", "attitude_source", "group", "mamba"]
assert not P.duplicated(key).any(), P[P.duplicated(key, keep=False)][key]
P.to_csv(HERE+"/canonical_agreement.csv", index=False); print(len(P))

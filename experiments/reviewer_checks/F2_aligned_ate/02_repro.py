import os, json, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(OUT, "..", "..", "..")) + "/"
new = pd.read_csv(OUT + "/ate_aligned_per_run.csv")
old = pd.read_csv(R + "experiments/attitude_controls/ate_per_run_gated.csv")
m = new.merge(old[["run_id", "seq", "source", "ate"]], on=["run_id", "seq", "source"], )
m["rel"] = (m.ate_none - m.ate).abs() / m.ate
rng = np.random.default_rng(20260126)
ids = rng.choice(m.run_id.unique(), 24, replace=False)
samp = m[m.run_id.isin(ids) & m.source.isin(["a_raw_gyro(paper)", "b_gyro-GT-bias(oracle)"])]
out = dict(n_pairs_total=len(m), max_rel_all_sources_all_runs=float(m.rel.max()),
           sample24_runs=dict(n_rows=len(samp), max_rel=float(samp.rel.max()),
                              raw_max=float(samp[samp.source.str.startswith('a_raw')].rel.max()),
                              oracle_max=float(samp[samp.source.str.startswith('b_')].rel.max())),
           per_source_max_rel={k: float(v) for k, v in m.groupby("source").rel.max().items()})
raw = new[new.source == "a_raw_gyro(paper)"]
out["raw_vs_metrics_wide_gyro_ate_max_rel_all_rows"] = float(((raw.ate_none - raw.gyro_ate_paper).abs() / raw.gyro_ate_paper).max())
out["n_raw_run_seq_rows"] = len(raw)
json.dump(out, open(OUT + "/repro_check.json", "w"), indent=1); print(json.dumps(out, indent=1))

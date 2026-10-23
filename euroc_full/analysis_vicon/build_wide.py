# -*- coding: utf-8 -*-
"""Step 0: results_vicon -> metrics_wide.csv + coverage tables (which methods/runs are missing).
MH counterpart: euroc_full/results/metrics_wide.csv (made by euroc_full.analysis.load_runs/write_long_wide)."""
import argparse, os
import pandas as pd
import vcommon as vc

ap = vc.add_args(argparse.ArgumentParser()); a = ap.parse_args()
w, idx = vc.load_wide(a.results_dir)
vc.write_csv(w, a, "metrics_wide.csv", index=False)
# coverage: expected = methods x seqs x seeds from vicon.yaml
cfg = vc.load_cfg(a)
seeds = cfg["seeds"]; methods = cfg["methods"]; seqs = cfg["sequences"]
single = {"constant_perseq", "linear_stats", "kf", "pf"}
idx["seq"] = idx.eval_sequences
tab = idx.assign(ok=(idx.status == "ok")).groupby(["method", "seq"]).agg(ok=("ok", "sum"), failed=("ok", lambda x: (~x).sum())).reset_index()
exp = {m: (1 if m in single else len(seeds)) for m in methods}
cov = pd.DataFrame([(m, s, exp[m], int(tab[(tab.method == m) & (tab.seq == s)].ok.sum()),
                     int(tab[(tab.method == m) & (tab.seq == s)].failed.sum())) for m in methods for s in seqs],
                   columns=["method", "seq", "expected", "ok", "failed"])
cov["not_run"] = cov.expected - cov.ok - cov.failed
vc.write_csv(cov, a, "coverage_by_method_seq.csv", index=False)
keep, cnt = vc.method_set(w, seqs, a.min_seeds)
tot = cov.groupby("method")[["expected", "ok", "failed", "not_run"]].sum().reindex(methods)
L = [f"# Coverage ({a.tag})", "", f"results_dir: `{a.results_dir}`", "",
     f"runs_index rows: {len(idx)}; ok: {(idx.status=='ok').sum()}; failed: {(idx.status=='failed').sum()}; expected total: {int(tot.expected.sum())}", "",
     "| method | expected | ok | failed | not run |", "|---|---|---|---|---|"]
for m, r in tot.iterrows(): L.append(f"| {m} | {r.expected} | {r.ok} | {r.failed} | {r.not_run} |")
L += ["", f"Complete-case deep-method set (>= {a.min_seeds} ok seeds on every sequence): {keep}",
      f"Excluded from cross-method statistics: {[m for m in vc.DL if m not in keep]}", "",
      "ok seeds per deep method x sequence:", "", "```", cnt.to_string(), "```"]
open(os.path.join(a.out_dir, "coverage.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:20]))

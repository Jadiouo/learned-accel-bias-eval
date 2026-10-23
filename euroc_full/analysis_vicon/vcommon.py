# -*- coding: utf-8 -*-
"""Shared helpers for the Vicon (V1_01-V2_03) analysis pipeline. CPU only, read-only on results.

Every script takes the same three path parameters:
  --results-dir  (default ../results_vicon)   runs_index.csv + runs/*/metrics.json + series.npz
  --data-dir     (default ../kaggle_vicon/data_vicon)  raw EuRoC folders (V1_01_easy/mav0/...)
  --out-dir      (default ./partial_2026-10-20)
Switch to the full results by changing --results-dir (and --out-dir); nothing else is hard-coded.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                   # .../euroc_full
sys.path.insert(0, os.path.join(ROOT, "src"))

# Same 11 deep methods and order as MH analysis scripts (rF_noncircular.py, rE_stats.py, tost_backfill.py)
DL = ['tcn', 'transformer_s', 'transformer_m', 'transformer_l', 'mamba', 'm1_cnn', 'm2', 'm3_lstm', 'm4', 'm5', 'm6']
SEQS = ["V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03"]


def add_args(ap):
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results_vicon"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "kaggle_vicon", "data_vicon"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "vicon.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "partial_2026-10-20"))
    ap.add_argument("--tag", default="PARTIAL", help="label stamped on every output file (use FULL for the complete set)")
    ap.add_argument("--min-seeds", type=int, default=10,
                    help="a deep method enters the cross-method statistics only if it has >= this many ok seeds on EVERY sequence")
    return ap


def load_cfg(a):
    from euroc_full import common
    cfg = common.load_config(a.config, data_dir_override=a.data_dir)
    return cfg


def load_wide(results_dir):
    """runs_index.csv + metrics.json -> wide table (one row per ok run x eval_sequence).
    Same logic as euroc_full.analysis.load_runs/write_long_wide (copied; only ok runs, pivot metric -> column)."""
    idx = pd.read_csv(os.path.join(results_dir, "runs_index.csv"))
    rows = []
    for _, r in idx.iterrows():
        if str(r["status"]) != "ok":
            continue
        mp = os.path.join(results_dir, "runs", r["run_id"], "metrics.json")
        if not os.path.exists(mp):
            continue
        blk = json.load(open(mp))
        for seq, mt in blk.items():
            d = {"run_id": r["run_id"], "eval_sequence": seq}
            for k, v in (mt or {}).items():
                if v is not None: d[k] = float(v)
            rows.append(d)
    w = pd.DataFrame(rows)
    w = w.merge(idx[["run_id", "regime", "method", "seed"]], on="run_id")
    return w, idx


def method_set(w, seqs, min_seeds, pool=DL):
    """Deep methods with >= min_seeds ok seeds on every sequence in `seqs` (complete-case set)."""
    cnt = w[w.method.isin(pool)].groupby(["method", "eval_sequence"]).seed.nunique().unstack(fill_value=0)
    cnt = cnt.reindex(index=pool, columns=seqs, fill_value=0)
    keep = [m for m in pool if (cnt.loc[m] >= min_seeds).all()]
    return keep, cnt


def banner(a, extra=""):
    return f"[{a.tag}] results_dir={os.path.abspath(a.results_dir)} {extra}".strip()


def write_csv(df, a, name, **kw):
    """Write CSV whose first line is a '# ' provenance comment (tag + source); read back with comment='#'."""
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, name)
    with open(p, "w") as f:
        f.write(f"# {banner(a)} ; coverage: see coverage.md\n")
        df.to_csv(f, **kw)
    return p

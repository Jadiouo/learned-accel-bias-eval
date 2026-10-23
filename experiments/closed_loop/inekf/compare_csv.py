# -*- coding: utf-8 -*-
"""Compare two CSVs (same columns, same row order after sorting by key columns). usage: compare_csv.py A B [key1,key2,...] -> prints max abs / rel diff."""
import sys, numpy as np, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
keys = sys.argv[3].split(",") if len(sys.argv) > 3 else []
if keys: a = a.sort_values(keys).reset_index(drop=True); b = b.sort_values(keys).reset_index(drop=True)
assert list(a.columns) == list(b.columns) or set(a.columns) == set(b.columns), (a.columns, b.columns)
b = b[a.columns]; assert a.shape == b.shape, (a.shape, b.shape)
num = a.select_dtypes("number").columns; obj = [c for c in a.columns if c not in num]
for c in obj: assert (a[c].astype(str) == b[c].astype(str)).all(), c
d = (a[num] - b[num]).abs().to_numpy(float); sc = np.maximum(a[num].abs().to_numpy(float), 1e-300)
print(f"{sys.argv[1]} vs {sys.argv[2]}: shape {a.shape}, max abs diff {np.nanmax(d):.3e}, max rel diff {np.nanmax(d/sc):.3e}")

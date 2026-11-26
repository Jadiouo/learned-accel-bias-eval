# -*- coding: utf-8 -*-
"""A8 helpers. Imports existing closed-loop code READ-ONLY (never runs their __main__, never writes into the project).
Alignment variants: cur (existing hold_series), caus (estimates delayed 200 frames = window end)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))  # this folder
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
CL = os.path.normpath(os.path.join(HERE, ".."))  # experiments/closed_loop; sys.path.insert(0, CL)
sys.path.insert(0, CL)
import eskf, run_closed_loop as rc
from eskf import FS
WIN = 200   # window length in frames (cfg window=200)

def hold_cur(D, b40):          # identical to rc.hold_series
    return rc.hold_series(D, b40)

def hold_caus(D, b40):
    """Causal alignment: window k (start i0+5k, uses frames [start,start+200)) becomes available at its END, frame i0+5k+200.
    Series value at frame j = estimate of the latest window with end <= j, i.e. window start = j-200 (rounded down to stride).
    Before the first window completes (j < i0+200) the first window's estimate is held (same extrapolation as extend_back; only
    matters for pre-outage aided phase in P mode; never reached by outage frames, checked in assert_bounds)."""
    N = len(D["f"]); out = np.empty((N, 3)); up = np.repeat(b40, rc.STRIDE, axis=0)
    out[:] = b40[0]
    s = D["i0"] + WIN; e = min(N, s + len(up)); out[s:e] = up[:e - s]
    if e < N: out[e:] = up[-1]
    return out

def sources(D, mode):
    """mode: cur | caus_est (estimators causal, bref/const_oracle current) | caus_both"""
    out = {}
    hc = hold_caus if mode in ("caus_est", "caus_both") else hold_cur
    hb = hold_caus if mode == "caus_both" else hold_cur
    for rid, b in zip(D["run_ids"], D["bhat"]): out[rid] = hc(D, b)
    out["CTRL:bref"] = hb(D, D["bref"])
    out["CTRL:zero"] = np.zeros((len(D["f"]), 3))
    out["CTRL:const_oracle"] = hb(D, np.tile(D["bref"].mean(0), (len(D["bref"]), 1)))
    return out

def hold_first_second(b200, ko):
    """Task rule 'B-hold': first 1 s of outage re-uses the value at ko-1 (last window completed before outage)."""
    b = b200[ko:].copy() if False else b200.copy(); b[ko:ko + WIN] = b200[ko - 1]; return b

def units(d):   # verbatim from analyze_3.py
    ctl = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"].rename("common")
    m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq", "rep"])
    u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max() - g.delta.min(), iqr=g.delta.quantile(.75) - g.delta.quantile(.25),
                          sd_abs=g["abs"].std())).join(ctl).reset_index()
    u["r_med"] = u.common / u.med; u["r_rng"] = u.common / u.rng; u["r_iqr"] = u.common / u.iqr; u["sd_rel"] = u.sd_abs / u.common
    return u

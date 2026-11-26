"""Shared loading and agreement-rate functions for the F1 noise-ceiling simulation.
Adapted from verification/R-F_noncircular/rF_noncircular_stratified.py (5/5 seed split halves, BH pair selection,
direction agreement, pooled agreement rate). The original script is neither imported nor modified."""
import itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
R = str(Path(__file__).resolve().parents[3]) + "/"   # repository root
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
SRC = {"raw": "a_raw_gyro(paper)", "oracle": "b_gyro-GT-bias(oracle)"}

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out

PAIRS = np.array(list(itertools.combinations(range(11), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
ALL = np.arange(10)
HALVES = [np.array(c) for c in itertools.combinations(range(10), 5)]   # 252 5/5 splits

def load():
    w = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
        pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
    w = w[w.method.isin(DL)]
    att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
    units = [(r, s) for r in "AB" for s in sorted(w[w.regime == r].eval_sequence.unique())]
    piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
    bias, ate, seeds = {}, {s: {} for s in SRC}, None
    for reg, seq in units:
        g = w[(w.regime == reg) & (w.eval_sequence == seq)]
        b = piv(g, "bias_rmse_avg")
        if seeds is None: seeds = list(b.index)
        assert list(b.index) == seeds and b.shape == (10, 11)
        bias[(reg, seq)] = b.values
        for k, src in SRC.items():
            gs = att[(att.regime == reg) & (att.seq == seq) & (att.source == src)]
            p = piv(gs, "ate"); assert list(p.index) == seeds and p.shape == (10, 11) and not np.isnan(p.values).any()
            ate[k][(reg, seq)] = p.values
    return units, bias, ate

def select(M, idx):
    X = M[idx]; D = X[:, PI]-X[:, PJ]
    p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
    return bh(p) < .05, D.mean(0)

def precompute_selection(units, bias, halves=HALVES):
    """Per (half1, unit): BH-selected pair mask and sign of the half-1 bias difference. Independent of ATE."""
    sel = {}
    for hi, h1 in enumerate(halves):
        for u in units:
            mask, d1 = select(bias[u], h1); sel[(hi, u)] = (mask, np.sign(d1))
    return sel

def agree_counts(ate_by_unit, units, sel, halves=HALVES, split_ids=None):
    """Return A[split, unit], N[split, unit] (as in the original pipeline: half-2 mean-ATE direction vs half-1 bias direction)."""
    split_ids = range(len(halves)) if split_ids is None else split_ids
    A = np.zeros((len(split_ids), len(units))); N = np.zeros_like(A)
    for si, hi in enumerate(split_ids):
        h2 = np.setdiff1d(ALL, halves[hi])
        for ui, u in enumerate(units):
            mask, s1 = sel[(hi, u)]; N[si, ui] = mask.sum()
            if not mask.sum(): continue
            m = ate_by_unit[u][h2].mean(0)
            A[si, ui] = (np.sign(m[PI]-m[PJ]) == s1)[mask].sum()
    return A, N

def groups(units):
    ui = {u: i for i, u in enumerate(units)}
    return {"A": [ui[u] for u in units if u[0]=="A"], "B": [ui[u] for u in units if u[0]=="B"], "ALL": list(range(len(units)))}

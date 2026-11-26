"""F7: direction-agreement pipeline generalised to any number of methods K and any set of units.
Logic copied from F1_noise_ceiling/common.py (itself from R-F_noncircular_stratified.py): 5/5 seed halves, paired t over
pairs + Benjamini-Hochberg q=.05 select the pairs on half 1 by bias RMSE; the other half's method-mean ATE difference
must have the same sign. Pooled rate = sum agree / sum pairs. Nothing here writes files."""
import sys; sys.dont_write_bytecode = True
import itertools, numpy as np
from scipy import stats

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i] * n / (n - k)); out[i] = run
    return out

ALL10 = np.arange(10)
HALVES252 = [np.array(c) for c in itertools.combinations(range(10), 5)]
def halves_rng0(ns=1000):          # paper pipeline: rng(0).integers(0,252,1000) as first draw
    hs = np.random.default_rng(0).integers(0, len(HALVES252), ns); return [HALVES252[h] for h in hs]

class Agree:
    def __init__(self, units, bias, K, halves):
        self.units, self.K, self.halves = units, K, halves
        P = np.array(list(itertools.combinations(range(K), 2))); self.PI, self.PJ = P[:, 0], P[:, 1]
        self.sel = {}
        for hi, h1 in enumerate(halves):
            for u in units:
                X = bias[u][h1]; D = X[:, self.PI] - X[:, self.PJ]
                p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
                self.sel[(hi, u)] = (bh(p) < .05, np.sign(D.mean(0)))
    def counts(self, ate):
        """ate[u]: (10,K). returns A[S,U], N[S,U]"""
        S, U = len(self.halves), len(self.units); A = np.zeros((S, U)); N = np.zeros((S, U))
        for hi, h1 in enumerate(self.halves):
            h2 = np.setdiff1d(ALL10, h1)
            for ui, u in enumerate(self.units):
                mask, s1 = self.sel[(hi, u)];
                N[hi, ui] = mask.sum()
                if mask.sum():
                    m = ate[u][h2].mean(0); A[hi, ui] = (np.sign(m[self.PI] - m[self.PJ]) == s1)[mask].sum()
        return A.sum(0), N.sum(0)

def pooled(a, n, idx): return a[idx].sum() / n[idx].sum()

def jackknife(a, n, clusters, idx=None):
    """a,n per unit; clusters: list of cluster label per unit. leave-one-cluster-out jackknife, t(df=G-1). returns pt, lo, hi, se, LOO array"""
    a, n = np.asarray(a), np.asarray(n); cl = np.asarray(clusters)
    idx = np.arange(len(a)) if idx is None else np.asarray(idx)
    labs = sorted(set(cl[idx])); G = len(labs)
    pt = a[idx].sum() / n[idx].sum()
    th = np.array([a[[i for i in idx if cl[i] != g]].sum() / n[[i for i in idx if cl[i] != g]].sum() for g in labs])
    se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum()); t = stats.t.ppf(.975, G - 1)
    return pt, pt - t * se, pt + t * se, se, th

def jackknife_diff(a1, n1, a2, n2, clusters):
    """jackknife CI of (rate1 - rate2) (paired over clusters)."""
    cl = np.asarray(clusters); labs = sorted(set(cl)); G = len(labs)
    f = lambda m: a1[m].sum() / n1[m].sum() - a2[m].sum() / n2[m].sum()
    pt = f(np.ones(len(cl), bool)); th = np.array([f(cl != g) for g in labs])
    se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum()); t = stats.t.ppf(.975, G - 1)
    return pt, pt - t * se, pt + t * se

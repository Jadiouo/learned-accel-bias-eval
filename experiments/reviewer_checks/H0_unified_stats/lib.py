"""H0 shared library. Logic copied from ../F7_vicon_decomp/agree_lib.py and ../F1_noise_ceiling/common.py
(themselves from verification/R-F_noncircular/rF_noncircular_stratified.py). Nothing here writes files."""
import sys; sys.dont_write_bytecode = True
import itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
R = str(Path(__file__).resolve().parents[3]) + "/"   # repository root
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
NOM = [i for i, m in enumerate(DL) if m != "mamba"]; ALLM = list(range(11))
MAMBA_SETS = {"included": ALLM, "excluded": NOM}
HALVES = [np.array(c) for c in itertools.combinations(range(10), 5)]   # 252 splits
ALL10 = np.arange(10)

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out

class Agree:
    """bias[u]: (10,K) ; selects pairs on half1 by paired t + BH(q=.05), reused for any measure."""
    def __init__(self, units, bias, K, halves=HALVES):
        self.units, self.K, self.halves = units, K, halves
        P = np.array(list(itertools.combinations(range(K), 2))); self.PI, self.PJ = P[:, 0], P[:, 1]
        self.sel = {}
        for hi, h1 in enumerate(halves):
            for u in units:
                X = bias[u][h1]; D = X[:, self.PI]-X[:, self.PJ]
                p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
                self.sel[(hi, u)] = (bh(p) < .05, np.sign(D.mean(0)))
    def counts(self, meas):
        """meas[u]: (10,K). returns a[U], n[U] summed over splits"""
        U = len(self.units); A = np.zeros(U); N = np.zeros(U)
        for hi, h1 in enumerate(self.halves):
            h2 = np.setdiff1d(ALL10, h1)
            for ui, u in enumerate(self.units):
                mask, s1 = self.sel[(hi, u)]; N[ui] += mask.sum()
                if mask.sum():
                    m = meas[u][h2].mean(0); A[ui] += (np.sign(m[self.PI]-m[self.PJ]) == s1)[mask].sum()
        return A, N

def jackknife(a, n, clusters, idx):
    """leave-one-cluster-out jackknife, t(df=G-1). returns pt, lo, hi (unclipped), G"""
    a, n = np.asarray(a), np.asarray(n); cl = np.asarray(clusters); idx = np.asarray(idx)
    labs = sorted(set(cl[idx])); G = len(labs)
    pt = a[idx].sum()/n[idx].sum()
    th = []
    for g in labs:
        k = [i for i in idx if cl[i] != g]; th.append(a[k].sum()/n[k].sum())
    th = np.array(th); se = np.sqrt((G-1)/G*((th-th.mean())**2).sum()); t = stats.t.ppf(.975, G-1)
    return pt, pt-t*se, pt+t*se, G

def jk_diff(a1, n1, a2, n2, clusters, idx):
    cl = np.asarray(clusters); idx = np.asarray(idx); labs = sorted(set(cl[idx])); G = len(labs)
    f = lambda m: a1[m].sum()/n1[m].sum()-a2[m].sum()/n2[m].sum()
    msk = np.zeros(len(cl), bool); msk[idx] = True
    pt = f(msk); th = np.array([f(msk & (cl != g)) for g in labs])
    se = np.sqrt((G-1)/G*((th-th.mean())**2).sum()); t = stats.t.ppf(.975, G-1)
    return pt, pt-t*se, pt+t*se

# ---------------- Machine Hall data
def load_mh():
    w = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
        pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
    w = w[w.method.isin(DL)]
    units = [(r, s) for r in "AB" for s in sorted(w[w.regime == r].eval_sequence.unique())]
    def piv(df, c): return df.pivot_table(index="seed", columns="method", values=c)[DL]
    bias, iabe, seeds = {}, {}, None
    for reg, seq in units:
        g = w[(w.regime == reg) & (w.eval_sequence == seq)]
        b = piv(g, "bias_rmse_avg"); seeds = seeds or list(b.index)
        assert list(b.index) == seeds and b.shape == (10, 11)
        bias[(reg, seq)] = b.values; iabe[(reg, seq)] = piv(g, "iabe_rot").values
    return units, bias, iabe, seeds

def ate_from_long(df, units, seeds, source, col="ate", regcol="regime", seqcol="seq"):
    out = {}
    for reg, seq in units:
        g = df[(df[regcol] == reg) & (df[seqcol] == seq) & (df.source == source)]
        p = g.pivot_table(index="seed", columns="method", values=col)[DL]
        assert list(p.index) == seeds and p.shape == (10, 11) and not np.isnan(p.values).any(), (source, reg, seq)
        out[(reg, seq)] = p.values
    return out

def groups_of(units):
    return {"regime A": [i for i, u in enumerate(units) if u[0] == "A"],
            "regime B": [i for i, u in enumerate(units) if u[0] == "B"],
            "nine cases pooled": list(range(len(units)))}

"""R-F: 非循環方向一致率。CPU only。用法: python3 rF_noncircular.py"""
import itertools, os, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), ""); OUT = os.path.dirname(os.path.abspath(__file__))
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
rng = np.random.default_rng(0)
w = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
w = w[w.method.isin(DL)]
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run.csv"); att = att[att.method.isin(DL)]
UNITS = [(r, s) for r in "AB" for s in sorted(w[w.regime == r].eval_sequence.unique())]
piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
PAIRS = np.array(list(itertools.combinations(range(11), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
mats = {}; seeds = None
for reg, seq in UNITS:
    g = w[(w.regime == reg) & (w.eval_sequence == seq)]
    a, b = piv(g, "iabe_rot"), piv(g, "bias_rmse_avg")
    if seeds is None: seeds = list(a.index)
    assert list(a.index) == seeds and a.shape == (10, 11)
    mats[(reg, seq, "IABE")] = a.values; mats[(reg, seq, "bias_rmse")] = b.values
    for src, gs in att[(att.regime == reg) & (att.seq == seq)].groupby("source"):
        p = piv(gs, "ate"); assert list(p.index) == seeds; mats[(reg, seq, "ATE:"+src)] = p.values
SRC = sorted({k[2] for k in mats if k[2].startswith("ATE:")})
def select(M, idx):  # paired t + BH on pairs, using seeds idx; returns mask, mean diff
    X = M[idx]; D = X[:, PI]-X[:, PJ]
    p = stats.ttest_1samp(D, 0, axis=0).pvalue; p = np.nan_to_num(p, nan=1.0)
    return bh(p) < .05, D.mean(0)
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.96; p = k/n; c = (p+z*z/2/n)/(1+z*z/n); h = z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return (c-h, c+h)
ALL = np.arange(10)
# ---------- Part 1: full-data, 以 ref 篩選 ----------
rows = []
for ref in ["bias_rmse", "IABE"]:
    sel = {}; refd = {}
    for u in UNITS:
        sel[u], refd[u] = select(mats[(*u, ref)], ALL)
    n_sel = sum(sel[u].sum() for u in UNITS)
    others = ["IABE" if ref == "bias_rmse" else "bias_rmse"] + SRC
    for k in others:
        def rate(sidx_by_unit):  # sidx: seed index array per unit (same for all)
            ag = n = 0
            for u in UNITS:
                d = mats[(*u, k)][sidx_by_unit]; dd = d.mean(0)[PI]-d.mean(0)[PJ]
                ag += (np.sign(dd) == np.sign(refd[u]))[sel[u]].sum(); n += sel[u].sum()
            return ag/n, ag, n
        r0, ag, n = rate(ALL)
        # seed-cluster bootstrap: 同一 seed 重抽索引套用到所有序列 (B 共用 seed), 篩選集合固定; ref 方向固定於全資料
        bs = [rate(rng.integers(0, 10, 10))[0] for _ in range(2000)]
        # unit-cluster bootstrap
        per = []
        for u in UNITS:
            dm = mats[(*u, k)].mean(0); dd = dm[PI]-dm[PJ]
            per.append(((np.sign(dd) == np.sign(refd[u]))[sel[u]].sum(), sel[u].sum()))
        per = np.array(per); bu = []
        for _ in range(2000):
            ix = rng.integers(0, len(UNITS), len(UNITS)); bu.append(per[ix, 0].sum()/max(per[ix, 1].sum(), 1))
        rows.append(dict(ref=ref, measure=k, n_pairs=n, agree=ag, rate=r0, wilson_lo=wilson(ag, n)[0], wilson_hi=wilson(ag, n)[1],
                         seedboot_lo=np.percentile(bs, 2.5), seedboot_hi=np.percentile(bs, 97.5),
                         unitboot_lo=np.percentile(bu, 2.5), unitboot_hi=np.percentile(bu, 97.5)))
P1 = pd.DataFrame(rows); P1.to_csv(OUT+"/part1_fulldata_agreement.csv", index=False)
# ---------- Part 2: 獨立 seed 切半 ----------
halves = [np.array(c) for c in itertools.combinations(range(10), 5)]  # 252
NS = 1000; hs = rng.integers(0, len(halves), NS)
res = []  # per (ref, split, unit): n, agree_ceiling, agree for each measure
for ref in ["bias_rmse", "IABE"]:
    others = ["IABE" if ref == "bias_rmse" else "bias_rmse"] + SRC
    A = np.zeros((NS, len(UNITS), 1+len(others))); N = np.zeros((NS, len(UNITS)))
    for si, h in enumerate(hs):
        h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
        for ui, u in enumerate(UNITS):
            mask, d1 = select(mats[(*u, ref)], h1)
            N[si, ui] = mask.sum()
            if not mask.sum(): continue
            def dir2(k):
                m = mats[(*u, k)][h2].mean(0); return np.sign(m[PI]-m[PJ])
            A[si, ui, 0] = (dir2(ref) == np.sign(d1))[mask].sum()
            for oi, k in enumerate(others): A[si, ui, 1+oi] = (dir2(k) == np.sign(d1))[mask].sum()
    names = [ref+" (自身重測上限)"] + others
    # 每切分的彙整率
    sp_rate = A.sum(1)/np.maximum(N.sum(1), 1)[:, None]
    ok = N.sum(1) > 0
    for ci, nm in enumerate(names):
        r = sp_rate[ok, ci]
        # unit-cluster bootstrap on pooled-over-splits rate, and on gap vs ceiling
        def pooled(ix, c): return A[:, ix, c].sum()/max(N[:, ix].sum(), 1)
        bs = []; gap = []; ab = []
        for _ in range(2000):
            ix = rng.integers(0, len(UNITS), len(UNITS))
            bs.append(pooled(ix, ci)); gap.append(pooled(ix, 0)-pooled(ix, ci)); ab.append(pooled(ix, ci)-.5)
        res.append(dict(ref=ref, measure=nm, mean_pairs_per_split=N.sum(1)[ok].mean(), pooled_rate=pooled(np.arange(9), ci),
                        split_p2_5=np.percentile(r, 2.5), split_median=np.median(r), split_p97_5=np.percentile(r, 97.5),
                        unitboot_lo=np.percentile(bs, 2.5), unitboot_hi=np.percentile(bs, 97.5),
                        gap_vs_ceiling=pooled(np.arange(9), 0)-pooled(np.arange(9), ci),
                        gap_lo=np.percentile(gap, 2.5), gap_hi=np.percentile(gap, 97.5),
                        p_gap_le0=(np.sum(np.array(gap) <= 0)+1)/2001,
                        above_half_lo=np.percentile(ab, 2.5), p_le_half=(np.sum(np.array(ab) <= 0)+1)/2001))
P2 = pd.DataFrame(res); P2.to_csv(OUT+"/part2_splithalf_agreement.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(P1.round(3).to_string()); print(P2.round(3).to_string())

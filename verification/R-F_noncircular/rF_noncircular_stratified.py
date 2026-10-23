"""[POST-HOC, outline stage] R-F 分層版：ref=bias_rmse、獨立 seed 切半（同 rF_noncircular.py Part 2 的選對/一致率定義），
把一致率依 regime(A/B) 與序列拆開；含第五來源 e_mahony_gated（來自 attitude_controls/ate_per_run_gated.csv）。
CI：以序列為 cluster 的 bootstrap（A 5 序列、B 4 序列；B 的 4 序列共用同一批 seed → 序列間相依，CI 偏窄）。
CPU only。用法: python3 rF_noncircular_stratified.py"""
import itertools, os, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), ""); OUT = os.path.dirname(os.path.abspath(__file__))
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
rng = np.random.default_rng(0)
w = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
w = w[w.method.isin(DL)]
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
KEEP = ["a_raw_gyro(paper)","b_gyro-GT-bias(oracle)","c_static-mean-bias","d_mahony(kp=1,ki=0.05)",
        "e_mahony_gated(kp=1,ki=0.05)","s_mahony(kp=5,ki=0.05)"]
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
        if src in KEEP: p = piv(gs, "ate"); assert list(p.index) == seeds; mats[(reg, seq, "ATE:"+src)] = p.values
others = ["IABE"] + ["ATE:"+s for s in KEEP]
def select(M, idx):
    X = M[idx]; D = X[:, PI]-X[:, PJ]
    p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
    return bh(p) < .05, D.mean(0)
ALL = np.arange(10)
halves = [np.array(c) for c in itertools.combinations(range(10), 5)]
NS = 1000; hs = rng.integers(0, len(halves), NS)
names = ["bias_rmse (自身重測上限)"] + others
A = np.zeros((NS, len(UNITS), len(names))); N = np.zeros((NS, len(UNITS)))
for si, h in enumerate(hs):
    h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
    for ui, u in enumerate(UNITS):
        mask, d1 = select(mats[(*u, "bias_rmse")], h1); N[si, ui] = mask.sum()
        if not mask.sum(): continue
        def dir2(k):
            m = mats[(*u, k)][h2].mean(0); return np.sign(m[PI]-m[PJ])
        A[si, ui, 0] = (dir2("bias_rmse") == np.sign(d1))[mask].sum()
        for oi, k in enumerate(others): A[si, ui, 1+oi] = (dir2(k) == np.sign(d1))[mask].sum()
uidx = {u: i for i, u in enumerate(UNITS)}
groups = {"A": [uidx[u] for u in UNITS if u[0] == "A"], "B": [uidx[u] for u in UNITS if u[0] == "B"],
          "ALL": list(range(len(UNITS)))}
for u in UNITS: groups[f"{u[0]}/{u[1]}"] = [uidx[u]]
def pooled(ix, c): return A[:, ix, c].sum()/max(N[:, ix].sum(), 1)
rows = []
for gname, gi in groups.items():
    gi = np.array(gi)
    for ci, nm in enumerate(names):
        # 切分變異（固定單位集合）
        sr = A[:, gi, ci].sum(1)/np.maximum(N[:, gi].sum(1), 1); ok = N[:, gi].sum(1) > 0
        r0 = pooled(gi, ci)
        if len(gi) > 1:  # 序列 cluster bootstrap
            bs, gap, ab = [], [], []
            for _ in range(4000):
                ix = gi[rng.integers(0, len(gi), len(gi))]
                bs.append(pooled(ix, ci)); gap.append(pooled(ix, 0)-pooled(ix, ci)); ab.append(pooled(ix, ci)-.5)
            lo, hi = np.percentile(bs, [2.5, 97.5]); glo, ghi = np.percentile(gap, [2.5, 97.5])
            p_half = (np.sum(np.array(ab) <= 0)+1)/(len(ab)+1)
        else: lo = hi = glo = ghi = p_half = np.nan
        rows.append(dict(group=gname, n_units=len(gi), measure=nm, mean_pairs_per_split=N[:, gi].sum(1)[ok].mean(),
                         pooled_rate=r0, seqboot_lo=lo, seqboot_hi=hi,
                         split_p2_5=np.percentile(sr[ok], 2.5), split_median=np.median(sr[ok]), split_p97_5=np.percentile(sr[ok], 97.5),
                         gap_vs_ceiling=pooled(gi, 0)-r0, gap_lo=glo, gap_hi=ghi, p_rate_le_half_seqboot=p_half))
P = pd.DataFrame(rows); P.to_csv(OUT+"/part2_splithalf_agreement_stratified.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(P[P.group.isin(["A","B","ALL"])].round(3).to_string())
print(P[~P.group.isin(["A","B","ALL"])].pivot(index="measure", columns="group", values="pooled_rate").round(3).to_string())
print(P[~P.group.isin(["A","B","ALL"])].groupby("group").mean_pairs_per_split.first().round(1).to_string())

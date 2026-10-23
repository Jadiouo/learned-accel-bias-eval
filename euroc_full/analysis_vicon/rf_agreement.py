"""[Vicon port of paper/revision_2026-10/verification/R-F_noncircular/rF_noncircular_stratified.py]
Same definitions: ref = bias_rmse_avg; 1000 independent 5+5 seed splits; half 1 selects distinguishable method pairs
(paired t + BH 0.05), half 2 checks sign agreement of the other measure; rate = sum(agree)/sum(selected).
Changes: (1) paths/units are parameters; (2) method set = complete-case deep methods (--min-seeds), K may be < 11;
(3) only regime A exists for Vicon; groups = ALL (6 sequences) + each sequence; (4) ATE source table read from
t1_leakage.py output (t1_ate_per_run_gated.csv); (5) CSV carries PARTIAL/FULL tag line."""
import argparse, itertools, os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as vc
ap = vc.add_args(argparse.ArgumentParser())
ap.add_argument("--ate-csv", default=None, help="per-run ATE table from t1_leakage.py (default <out-dir>/t1_ate_per_run_gated.csv)")
ap.add_argument("--methods", default=None, help="comma list overriding the complete-case method set (used for the MH same-method-subset check)")
ap.add_argument("--regime", default="A")
ap.add_argument("--out-name", default="rf_splithalf_agreement.csv")
ARGS = ap.parse_args()
rng = np.random.default_rng(0)
w, _ = vc.load_wide(ARGS.results_dir)
cfg = vc.load_cfg(ARGS); SEQS = list(cfg["sequences"])
w = w[w.regime == ARGS.regime]
DL, cnt = vc.method_set(w, SEQS, ARGS.min_seeds)
if ARGS.methods: DL = ARGS.methods.split(",")
K = len(DL); assert K >= 3, "need >= 3 complete-case methods"
w = w[w.method.isin(DL)]
att = pd.read_csv(ARGS.ate_csv or os.path.join(ARGS.out_dir, "t1_ate_per_run_gated.csv"), comment="#"); att = att[att.method.isin(DL) & (att.regime == ARGS.regime)]
KEEP = ["a_raw_gyro(paper)","b_gyro-GT-bias(oracle)","c_static-mean-bias","d_mahony(kp=1,ki=0.05)",
        "e_mahony_gated(kp=1,ki=0.05)","s_mahony(kp=5,ki=0.05)"]
UNITS = [(r, s) for r in sorted(w.regime.unique()) for s in SEQS]
piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
PAIRS = np.array(list(itertools.combinations(range(K), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
mats = {}; seeds = None
for reg, seq in UNITS:
    g = w[(w.regime == reg) & (w.eval_sequence == seq)]
    a, b = piv(g, "iabe_rot"), piv(g, "bias_rmse_avg")
    if seeds is None: seeds = list(a.index)
    assert list(a.index) == seeds and a.shape == (10, K), a.shape
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
groups = {"ALL": list(range(len(UNITS)))}
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
P = pd.DataFrame(rows); vc.write_csv(P, ARGS, ARGS.out_name, index=False)
print("methods used (K=%d):" % K, DL)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(P[P.group == "ALL"].round(3).to_string())
print(P[P.group != "ALL"].pivot(index="measure", columns="group", values="pooled_rate").round(3).to_string())
print(P[P.group != "ALL"].groupby("group").mean_pairs_per_split.first().round(1).to_string())

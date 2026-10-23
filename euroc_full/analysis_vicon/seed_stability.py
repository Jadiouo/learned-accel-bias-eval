# -*- coding: utf-8 -*-
"""Seed stability / resolution / ranking reliability for Vicon regime A. CPU only.
Port of (formulas copied, same definitions):
  - flip rate P(two random seeds pick different winners) : paper/revision_2026-10/verification/R-B_scripts/tost_backfill.py (flip part)
  - distinguishable pairs (paired t over seed index, BH within the unit's pairs), |d|>=2, MDE, sigma_w,
    split-half Spearman-Brown reliability, ATE-vs-IABE rho : paper/revision_2026-10/verification/R-E_stats/rE_stats.py
Changes: units = Vicon regime A sequences; K = complete-case deep-method count (<=11); numbers of pairs are K(K-1)/2.
Winner = argmin over methods of the metric for each seed (iabe_rot as in R-B; bias_rmse_avg also reported, Fig. 5 of the paper uses it)."""
import argparse, itertools, os, sys, numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as vc
ap = vc.add_args(argparse.ArgumentParser())
ap.add_argument("--ate-csv", default=None); ap.add_argument("--regime", default="A")
ap.add_argument("--methods", default=None); ap.add_argument("--suffix", default="")
ARGS = ap.parse_args()
w, _ = vc.load_wide(ARGS.results_dir); w = w[w.regime == ARGS.regime]
cfg = vc.load_cfg(ARGS); SEQS = list(cfg["sequences"])
DL, cnt = vc.method_set(w, SEQS, ARGS.min_seeds)
if ARGS.methods: DL = ARGS.methods.split(",")
K = len(DL); w = w[w.method.isin(DL)]
att = pd.read_csv(ARGS.ate_csv or os.path.join(ARGS.out_dir, "t1_ate_per_run_gated.csv"), comment="#")
att = att[att.method.isin(DL) & (att.regime == ARGS.regime)]
piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
PAIRS = list(itertools.combinations(range(K), 2))
mats = {}
for seq in SEQS:
    g = w[w.eval_sequence == seq]
    mats[(seq, "IABE")] = piv(g, "iabe_rot").values; mats[(seq, "bias_rmse")] = piv(g, "bias_rmse_avg").values
    for src, gs in att[att.seq == seq].groupby("source"): mats[(seq, "ATE:" + src)] = piv(gs, "ate").values
nseed = {m.shape[0] for m in mats.values()}; assert nseed == {10}, nseed
# ---- 1. flip rate -------------------------------------------------------------------------
rows = []
for metric in ["IABE", "bias_rmse"]:
    for seq in SEQS:
        M = mats[(seq, metric)]; win = M.argmin(1); c = np.bincount(win, minlength=K); n = len(win)
        rows.append(dict(metric=metric, seq=seq, n_methods=K, n_distinct_winners=int((c > 0).sum()), n_seeds=n,
                         p_two_seeds_disagree=1 - (c * (c - 1)).sum() / (n * (n - 1)), top_share=c.max() / n,
                         random_baseline=1 - 1 / K, winners=";".join(DL[i] for i in win)))
    sub = [r for r in rows if r["metric"] == metric]
    rows.append(dict(metric=metric, seq="MEAN", n_methods=K, n_distinct_winners=np.nan, n_seeds=10,
                     p_two_seeds_disagree=np.mean([r["p_two_seeds_disagree"] for r in sub]), top_share=np.nan,
                     random_baseline=1 - 1 / K, winners=""))
F = pd.DataFrame(rows); vc.write_csv(F, ARGS, f"seed_flip_rate{ARGS.suffix}.csv", index=False)
# ---- 2. resolution: distinguishable pairs, MDE ---------------------------------------------------
rs = []; sdp = []; dtyp = []; pair_rows = []
for seq in SEQS:
    P = mats[(seq, "IABE")]
    for i, j in PAIRS:
        d = P[:, i] - P[:, j]; sd = d.std(ddof=1)
        pair_rows.append(dict(seq=seq, A=DL[i], B=DL[j], mean_diff=d.mean(), sd_diff=sd, d_eff=abs(d.mean()) / sd if sd > 0 else np.inf,
                              p_t=stats.ttest_1samp(d, 0).pvalue))
PR = pd.DataFrame(pair_rows)
PR["p_t_bh"] = PR.groupby("seq").p_t.transform(lambda x: bh(x.values))
PR["distinguishable_bh_t"] = PR.p_t_bh < .05
PR["distinguishable"] = PR.distinguishable_bh_t | (PR.d_eff >= 2)
vc.write_csv(PR, ARGS, f"seed_pairwise_iabe{ARGS.suffix}.csv", index=False)
S = PR.groupby("seq").agg(n_pairs=("p_t", "size"), raw_t=("p_t", lambda x: (x < .05).sum()), bh_t=("p_t_bh", lambda x: (x < .05).sum()),
                          d_ge2=("d_eff", lambda x: (x >= 2).sum()), distinguishable=("distinguishable", "sum"),
                          median_sd_diff=("sd_diff", "median"), median_abs_diff=("mean_diff", lambda x: np.median(np.abs(x)))).reset_index()
tot = dict(seq="TOTAL", **{c: S[c].sum() for c in ["n_pairs", "raw_t", "bh_t", "d_ge2", "distinguishable"]},
           median_sd_diff=S.median_sd_diff.median(), median_abs_diff=S.median_abs_diff.median())
S = pd.concat([S, pd.DataFrame([tot])]); S["frac_distinguishable"] = S.distinguishable / S.n_pairs
def power(d, n, alpha):
    df = n - 1; tc = stats.t.ppf(1 - alpha / 2, df); nc = d * np.sqrt(n)
    v = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc); return 1.0 if np.isnan(v) else v
def mde_d(n, alpha):
    try: return optimize.brentq(lambda d: power(d, n, alpha) - .8, 1e-6, 60)
    except ValueError: return np.inf   # not reachable in range
sd0 = float(S[S.seq != "TOTAL"].median_sd_diff.median()); dt0 = float(S[S.seq != "TOTAL"].median_abs_diff.median())
npair = len(PAIRS); mrows = []
for nm, al in [("unadjusted", .05), (f"Bonferroni_m{npair}", .05 / npair)]:
    for n in [5, 10, 20, 40]: mrows.append(dict(alpha=nm, n_seeds=n, sd_diff=sd0, mde_m=mde_d(n, al) * sd0, typical_abs_diff_m=dt0))
    mrows.append(dict(alpha=nm, n_seeds="n_to_resolve_typical_diff", sd_diff=sd0, mde_m=np.nan, typical_abs_diff_m=dt0,
                      n_required=next((n for n in range(3, 400) if mde_d(n, al) * sd0 <= dt0), 400)))  # 400 = cap
vc.write_csv(S, ARGS, f"seed_resolution_by_seq{ARGS.suffix}.csv", index=False)
vc.write_csv(pd.DataFrame(mrows), ARGS, f"seed_mde{ARGS.suffix}.csv", index=False)
# ---- 3. split-half reliability + ATE-vs-IABE rho ----------------------------------------------
halves = [c for c in itertools.combinations(range(10), 5) if 0 in c]
sp = lambda x, y: stats.spearmanr(x, y)[0]
rel = []
for (seq, k), M in mats.items():
    rs_ = np.array([sp(M[list(h)].mean(0), M[[q for q in range(10) if q not in h]].mean(0)) for h in halves])
    sb = 2 * rs_ / (1 + rs_)
    rel.append(dict(seq=seq, measure=k, SB_rho=np.nanmean(sb), SB_lo=np.nanpercentile(sb, 2.5), SB_hi=np.nanpercentile(sb, 97.5)))
REL = pd.DataFrame(rel); vc.write_csv(REL, ARGS, f"seed_reliability{ARGS.suffix}.csv", index=False)
rho = [dict(seq=seq, measure=k, rho_vs_IABE=sp(M.mean(0), mats[(seq, "IABE")].mean(0))) for (seq, k), M in mats.items() if k != "IABE"]
RH = pd.DataFrame(rho); vc.write_csv(RH, ARGS, f"seed_rho_vs_iabe{ARGS.suffix}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("methods (K=%d):" % K, DL)
print(F.drop(columns="winners").round(3).to_string()); print(S.round(3).to_string()); print(pd.DataFrame(mrows).round(4).to_string())
print(REL.groupby("measure").SB_rho.median().round(2).to_string()); print(RH.groupby("measure").rho_vs_IABE.median().round(2).to_string())

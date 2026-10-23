"""R-E: 修補統計論證 (a-e)。CPU only，只讀既有結果。用法: python3 rE_stats.py"""
import itertools, os, json, numpy as np, pandas as pd
from scipy import stats, optimize
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "")
OUT = os.path.dirname(os.path.abspath(__file__))
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
rng = np.random.default_rng(0)
w = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
w = w[w.method.isin(DL)]
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run.csv"); att = att[att.method.isin(DL)]
UNITS = [(r, s) for r in "AB" for s in sorted(w[w.regime == r].eval_sequence.unique())]
def piv(df, col):  # seeds x methods
    return df.pivot_table(index="seed", columns="method", values=col)[DL]
def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p)-r)*p[i])); out[i] = run
    return out
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
SIGNS = np.array(list(itertools.product([1,-1], repeat=10)))  # 1024 x 10
def perm_p(d):
    obs = abs(d.mean()); allm = np.abs(SIGNS @ d)/10
    return float((allm >= obs-1e-15).mean())
PAIRS = list(itertools.combinations(range(11), 2))
# ---------------- a ----------------
rows = []
for reg, seq in UNITS:
    g = w[(w.regime == reg) & (w.eval_sequence == seq)]; P = piv(g, "iabe_rot").values
    assert P.shape == (10, 11)
    for i, j in PAIRS:
        d = P[:, i]-P[:, j]; sd = d.std(ddof=1)
        rows.append(dict(regime=reg, seq=seq, A=DL[i], B=DL[j], mean_diff=d.mean(), sd_diff=sd,
                         d_eff=abs(d.mean())/sd, p_t=stats.ttest_1samp(d, 0).pvalue, p_perm=perm_p(d)))
A = pd.DataFrame(rows)
for c in ["p_t", "p_perm"]:
    for f, nm in [(holm, "holm"), (bh, "bh")]:
        A[f"{c}_{nm}"] = A.groupby(["regime", "seq"])[c].transform(lambda x: f(x.values))
A["distinguishable_bh_t"] = A.p_t_bh < .05; A["distinguishable"] = A.distinguishable_bh_t | (A.d_eff >= 2)
A.to_csv(OUT+"/a_pairwise_tests.csv", index=False)
summ = A.groupby(["regime", "seq"]).agg(n_pairs=("p_t", "size"), raw_t=("p_t", lambda x: (x < .05).sum()),
    raw_perm=("p_perm", lambda x: (x < .05).sum()), holm_t=("p_t_holm", lambda x: (x < .05).sum()),
    holm_perm=("p_perm_holm", lambda x: (x < .05).sum()), bh_t=("p_t_bh", lambda x: (x < .05).sum()),
    bh_perm=("p_perm_bh", lambda x: (x < .05).sum()), d_ge2=("d_eff", lambda x: (x >= 2).sum()),
    distinguishable=("distinguishable", "sum"), min_p_perm=("p_perm", "min"), min_p_t=("p_t", "min")).reset_index()
summ.to_csv(OUT+"/a_summary_by_sequence.csv", index=False)
m = 55; thr = 0.05/m; tcrit = stats.t.ppf(1-thr/2, 9)
floor = dict(holm_first_step_threshold=thr, exact_signflip_min_p_n10=2/1024, min_n_for_signflip_to_pass=int(np.ceil(np.log2(2/thr))),
             t_crit_df9_at_holm_first=tcrit, min_abs_d_needed_n10=tcrit/np.sqrt(10),
             min_abs_d_needed_unadj_n10=stats.t.ppf(.975, 9)/np.sqrt(10))
# ---------------- b ----------------
def power(d, n, alpha):
    df = n-1; tc = stats.t.ppf(1-alpha/2, df); nc = d*np.sqrt(n)
    v = stats.nct.sf(tc, df, nc)+stats.nct.cdf(-tc, df, nc)
    return 1.0 if np.isnan(v) else v
def mde_d(n, alpha): return optimize.brentq(lambda d: power(d, n, alpha)-.8, 1e-3, 12)
sdpair = A.groupby(["regime", "seq"]).sd_diff.median().reset_index()
dtyp = A.assign(ad=A.mean_diff.abs()).groupby(["regime", "seq"]).ad.median().reset_index()
bt = sdpair.merge(dtyp, on=["regime", "seq"])
bt.columns = ["regime", "seq", "med_sd_diff", "med_abs_diff"]
mrows = []
for reg in "AB":
    sub = bt[bt.regime == reg]; sd = sub.med_sd_diff.median(); dt = sub.med_abs_diff.median()
    for alpha_name, alpha in [("unadjusted", .05), ("Holm_m55_first_step", .05/55)]:
        for n in [5, 10, 20, 40]:
            mrows.append(dict(regime=reg, alpha=alpha_name, n_seeds=n, sd_diff=sd, mde=mde_d(n, alpha)*sd, typical_abs_diff=dt))
        nreq = next(n for n in range(3, 5000) if mde_d(n, alpha)*sd <= dt)
        mrows.append(dict(regime=reg, alpha=alpha_name, n_seeds="req_n_for_typical", sd_diff=sd, mde=np.nan, typical_abs_diff=dt, n_required=nreq))
B_ = pd.DataFrame(mrows); B_.to_csv(OUT+"/b_mde_curve.csv", index=False)
# per-pair required n (Holm alpha) distribution using own observed effect size
def need_n(d, alpha):
    for n in range(3, 400):
        if power(d, n, alpha) >= .8: return n
    return 400
A["n_req_holm"] = [need_n(x, .05/55) if x > 0.05 else 400 for x in A.d_eff]
A["n_req_unadj"] = [need_n(x, .05) if x > 0.05 else 400 for x in A.d_eff]
nreq_tab = A.groupby("regime")[["n_req_holm", "n_req_unadj"]].describe(percentiles=[.25, .5, .75]).T
nreq_tab.to_csv(OUT+"/b_required_n_per_pair.csv")
# ---------------- c ----------------
halves = [c for c in itertools.combinations(range(10), 5) if 0 in c]  # 126 distinct splits
def sp(x, y): return stats.spearmanr(x, y)[0]
def splithalf(M):  # M seeds x methods
    rs = []
    for h in halves:
        o = [k for k in range(10) if k not in h]
        rs.append(sp(M[list(h)].mean(0), M[o].mean(0)))
    rs = np.array(rs); sb = 2*rs/(1+rs)
    return np.nanmean(rs), np.nanmean(sb), np.nanpercentile(sb, 2.5), np.nanpercentile(sb, 97.5)
# 組裝每個 unit 的 seeds x methods 矩陣: IABE, bias_rmse_avg, 各 source ATE
mats = {}
for reg, seq in UNITS:
    g = w[(w.regime == reg) & (w.eval_sequence == seq)]
    mats[(reg, seq, "IABE")] = piv(g, "iabe_rot").values
    mats[(reg, seq, "bias_rmse")] = piv(g, "bias_rmse_avg").values
    ga = att[(att.regime == reg) & (att.seq == seq)]
    for src, gs in ga.groupby("source"):
        mats[(reg, seq, "ATE:"+src)] = piv(gs, "ate").values
rel = []
for (reg, seq, k), M in mats.items():
    r, sb, lo, hi = splithalf(M); rel.append(dict(regime=reg, seq=seq, measure=k, split_half_rho=r, SB_rho=sb, SB_lo=lo, SB_hi=hi))
REL = pd.DataFrame(rel); REL.to_csv(OUT+"/c_reliability.csv", index=False)
relmap = REL.set_index(["regime", "seq", "measure"])
def boot_rho(Ma, Mb, B=2000):
    out = []
    for _ in range(B):
        ix = rng.integers(0, 10, 10); out.append(sp(Ma[ix].mean(0), Mb[ix].mean(0)))
    return np.nanpercentile(out, [2.5, 97.5])
drows = []
for reg, seq in UNITS:
    MI = mats[(reg, seq, "IABE")]
    for k in [k for (r_, s_, k) in mats if r_ == reg and s_ == seq and k != "IABE"]:
        Mx = mats[(reg, seq, k)]; rho = sp(Mx.mean(0), MI.mean(0))
        ra = relmap.loc[(reg, seq, k)]; ri = relmap.loc[(reg, seq, "IABE")]
        rmax = np.sqrt(max(ra.SB_rho, 0)*max(ri.SB_rho, 0))
        rmax_lo = np.sqrt(max(ra.SB_lo, 0)*max(ri.SB_lo, 0)); 
        lo, hi = boot_rho(Mx, MI, 1000)
        drows.append(dict(regime=reg, seq=seq, measure=k, rho_obs=rho, rho_ci_lo=lo, rho_ci_hi=hi, rel_meas=ra.SB_rho,
                          rel_IABE=ri.SB_rho, rho_max=rmax, rho_max_lo=rmax_lo,
                          rho_disattenuated=rho/rmax if rmax > .1 else np.nan,
                          obs_below_ceiling=bool(hi < rmax_lo)))
D = pd.DataFrame(drows); D.to_csv(OUT+"/c_disattenuation.csv", index=False)
# ---------------- d ----------------
A["key"] = A.regime+"/"+A.seq
drows = []
for (reg, seq), g in A.groupby(["regime", "seq"]):
    g = g.reset_index(drop=True)
    MI = mats[(reg, seq, "IABE")]; iab = {m_: MI[:, i].mean() for i, m_ in enumerate(DL)}
    ks = [k for (r_, s_, k) in mats if r_ == reg and s_ == seq and k != "IABE"]
    for k in ks:
        Mx = mats[(reg, seq, k)]; mu = {m_: Mx[:, i].mean() for i, m_ in enumerate(DL)}
        for r in g.itertuples():
            dI = iab[r.A]-iab[r.B]; dX = mu[r.A]-mu[r.B]
            drows.append(dict(regime=reg, seq=seq, measure=k, A=r.A, B=r.B, dist=r.distinguishable, bh=r.distinguishable_bh_t,
                              d2=r.d_eff >= 2, agree=np.sign(dI) == np.sign(dX)))
Dd = pd.DataFrame(drows)
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.96; p = k/n; c = (p+z*z/2/n)/(1+z*z/n); h = z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return (c-h, c+h)
out = []
for k, g in Dd.groupby("measure"):
    for lab, mask in [("distinguishable(BH-t or d>=2)", g.dist), ("BH-t only", g.bh), ("d>=2 only", g.d2), ("not distinguishable", ~g.dist), ("all pairs", g.dist | ~g.dist)]:
        s = g[mask]; n = len(s); kk = int(s.agree.sum()); lo, hi = wilson(kk, n)
        # 單位聚類: 每 unit 一個一致率
        pu = s.groupby(["regime", "seq"]).agree.mean()
        out.append(dict(measure=k, subset=lab, n_pairs=n, agree=kk, rate=kk/n if n else np.nan, wilson_lo=lo, wilson_hi=hi,
                        unit_rate_median=pu.median() if len(pu) else np.nan, n_units=len(pu)))
Dr = pd.DataFrame(out); Dr.to_csv(OUT+"/d_direction_agreement.csv", index=False)
# ---------------- e ----------------
erows = []
for reg, seq in UNITS:
    M = mats[(reg, seq, "IABE")]
    sw = np.sqrt(np.mean(M.var(axis=0, ddof=1)))  # 方法內 seed-to-seed SD (池化)
    swv = M.var(axis=0, ddof=1)
    hd = []
    for h in halves:
        o = [k for k in range(10) if k not in h]; hd.extend(np.abs(M[list(h)].mean(0)-M[o].mean(0)))
    psd = sdpair[(sdpair.regime == reg) & (sdpair.seq == seq)].sd_diff.iloc[0]
    erows.append(dict(regime=reg, seq=seq, orig_2x_median_paired_sd=2*psd, sigma_w=sw, R2_2sqrt2_sigma_w=2*np.sqrt(2)*sw,
                      R3_single_run_SD_diff=np.sqrt(2)*sw, LSD_mean10_parametric=stats.t.ppf(.975, 18)*sw*np.sqrt(2/10),
                      LSD_mean10_splithalf_empirical=np.percentile(hd, 95)/np.sqrt(2), median_pair_sd_over_sqrt2_sigma_w=psd/(np.sqrt(2)*sw)))
E_ = pd.DataFrame(erows)
# leave-one-sequence-out: 以同 regime 其他序列的 sigma_w 預測本序列的 paired SD
for reg in "AB":
    idx = E_.index[E_.regime == reg]
    for i in idx:
        oth = [j for j in idx if j != i]
        E_.loc[i, "loso_pred_pair_sd"] = np.sqrt(2)*np.sqrt(np.mean(E_.loc[oth, "sigma_w"]**2))
E_["loso_pred_2x"] = 2*E_.loso_pred_pair_sd
E_.to_csv(OUT+"/e_resolution.csv", index=False)
json.dump(floor, open(OUT+"/a_power_floor.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(json.dumps(floor, indent=1)); print(summ.to_string()); print(B_.round(4).to_string()); print(nreq_tab.round(1).to_string())
print(REL.groupby("measure")[["split_half_rho", "SB_rho"]].median().round(2).to_string())
print(D.groupby("measure")[["rho_obs", "rel_meas", "rel_IABE", "rho_max", "rho_disattenuated"]].median().round(2).to_string())
print(D.groupby("measure").obs_below_ceiling.sum().to_string(), len(D))
print(Dr.round(3).to_string()); print(E_.round(3).to_string())
# ---- c 補充: 跨 cell 彙整 ----
meta = []
for k, g in D.groupby("measure"):
    gap = g.rho_max-g.rho_obs
    z = np.arctanh(np.clip(g.rho_obs, -.99, .99)); zm = np.arctanh(np.clip(g.rho_max, -.99, .99))
    meta.append(dict(measure=k, n_cells=len(g), median_rho_obs=g.rho_obs.median(), median_rho_max=g.rho_max.median(),
        cells_obs_below_max=int((gap > 0).sum()), wilcoxon_p_gap=stats.wilcoxon(gap, alternative="greater").pvalue,
        cells_ci_below_ceiling=int(g.obs_below_ceiling.sum())))
pd.DataFrame(meta).to_csv(OUT+"/c_cell_meta.csv", index=False)
print(pd.DataFrame(meta).round(3).to_string())

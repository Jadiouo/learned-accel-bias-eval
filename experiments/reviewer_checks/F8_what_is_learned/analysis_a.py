"""F8 items 0-3: reproduction check, time variation of b_ref, offset/fluctuation split, tracking skill. Writes csv files and prints the results."""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
from common import *
from scipy import stats as st
S = pd.read_csv(os.path.join(HERE, "stats_per_run.csv"))
W = pd.read_csv(R+"euroc_full/results/metrics_wide.csv")
M = S.merge(W[["run_id","eval_sequence","bias_rmse_avg"]], left_on=["run_id","seq"], right_on=["run_id","eval_sequence"])
assert len(M) == len(S) == 1026
rel = (M.rmse_avg - M.bias_rmse_avg).abs() / M.bias_rmse_avg
print("[repro a] recomputed bias_rmse_avg vs metrics_wide: max relative difference %.2e (n=%d)" % (rel.max(), len(M)))
ident = ((M.off2 + M.fl2) - M.rmse_avg**2).abs() / M.rmse_avg**2
print("[identity] off2+fl2 = rmse^2, max relative difference %.2e" % ident.max())
M["off"] = np.sqrt(M.off2); M["fl"] = np.sqrt(M.fl2); M["sig_ref"] = np.sqrt(M.var_ref)
M["share_fl"] = M.fl2 / (M.off2 + M.fl2)
M["corr_all"] = M.cov_all / np.sqrt(M.var_hat * M.var_ref)
for a in "xyz": M["corr_"+a] = M["cov_"+a] / np.sqrt(M["vh_"+a] * M["vr_"+a])
M["sd_ratio"] = np.sqrt(M.var_hat / M.var_ref); M["skill"] = 1 - M.fl2 / M.var_ref
M["cluster"] = M.seq
M.to_csv(os.path.join(HERE, "stats_enriched.csv"), index=False)

def tci(vals):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]; k = len(v)
    if k < 2: return np.nan, np.nan, np.nan
    m = v.mean(); se = v.std(ddof=1) / np.sqrt(k); t = st.t.ppf(.975, k-1)
    return m, m - t*se, m + t*se   # jackknife (= standard error) of the cluster mean + t interval

def group_summary(df, cols, groups):
    """For each group: first the median over seeds within a case (regime, seq), then the mean over sequence clusters, with a t interval."""
    out = {}
    for g, regs in groups.items():
        d = df[df.regime.isin(regs)]
        per_cl = d.groupby(["cluster"])[cols].median()   # when regimes A and B of one sequence share a cluster, the median is taken over all its runs
        for c in cols:
            out[(g, c)] = tci(per_cl[c].values) + (len(per_cl),)
    return out
GR = {"A": ["A"], "B": ["B"], "ALL": ["A", "B"]}

# ---------- item 1 ----------
DLd = M[M.method.isin(DL)]
rows = []
for (reg, seq), g in DLd.groupby(["regime", "seq"]):
    rows.append(dict(regime=reg, seq=seq, n_win=int(g.n.iloc[0]), sigma_ref=g.sig_ref.iloc[0], mean_ref_norm=g.mean_ref_norm.iloc[0],
                     dl_rmse_median=g.rmse_avg.median(), dl_rmse_min=g.rmse_avg.min(),
                     const_rmse=M[(M.regime == reg) & (M.seq == seq) & M.method.str.startswith("constant")].rmse_avg.median(),
                     off_med=g.off.median(), fl_med=g.fl.median()))
I1 = pd.DataFrame(rows); I1["sig_over_rmse"] = I1.sigma_ref / I1.dl_rmse_median; I1["mean_over_rmse"] = I1.mean_ref_norm / I1.dl_rmse_median
I1.to_csv(os.path.join(HERE, "item1_case.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n[item 1] per case: time standard deviation sigma_ref of b_ref in the test window, mean norm of m_r, median bias RMSE of the deep methods (m/s^2, 3-axis RMS)"); print(I1.round(5).to_string())
for g, regs in GR.items():
    d = I1[I1.regime.isin(regs)]
    print(f"  {g}: sigma_ref median {d.sigma_ref.median():.5f}, deep-method rmse median {d.dl_rmse_median.median():.5f}, sigma/rmse median {d.sig_over_rmse.median():.3f} (range {d.sig_over_rmse.min():.3f}-{d.sig_over_rmse.max():.3f}), mean/rmse median {d.mean_over_rmse.median():.2f}")
# difference of the window means, regime A vs B of the same sequence
mr = {(u.regime, u.seq): np.array([u.mr_x, u.mr_y, u.mr_z]) for u in M[M.method == "tcn"].drop_duplicates(["regime", "seq"]).itertuples()}
print("  difference of the test-window mean b_ref, regime A vs B of the same sequence, |mA-mB|/sqrt(3) (RMS units), and window counts:")
d1rows = []
for s in ["MH_01", "MH_02", "MH_03", "MH_05"]:
    dd = np.linalg.norm(mr[("A", s)] - mr[("B", s)]) / np.sqrt(3)
    nA = int(I1[(I1.regime == "A") & (I1.seq == s)].n_win.iloc[0]); nB = int(I1[(I1.regime == "B") & (I1.seq == s)].n_win.iloc[0])
    d1rows.append(dict(seq=s, diff_rms=dd, n_A=nA, n_B=nB, mrA=np.round(mr[("A", s)], 5).tolist(), mrB=np.round(mr[("B", s)], 5).tolist()))
    print(f"   {s}: {dd:.5f}  nA={nA} nB={nB}")
pd.DataFrame(d1rows).to_csv(os.path.join(HERE, "item1_A_vs_B_mean.csv"), index=False)
cs = np.array(list(mr.values())); print("  RMS spread of the mean b_ref across cases (A/B, each sequence) around the overall mean: %.5f" % (np.sqrt(((cs - cs.mean(0))**2).sum(1).mean() / 3)))

# ---------- items 2, 3 ----------
methods = DL + ["constant_perseq", "constant_pooled"]
cols = ["off", "fl", "rmse_avg", "share_fl", "corr_all", "corr_x", "corr_y", "corr_z", "sd_ratio", "skill", "sig_ref"]
rows = []
for m in methods:
    d = M[M.method == m]
    for g, regs in GR.items():
        dd = d[d.regime.isin(regs)]
        if dd.empty: continue
        res = group_summary(dd, cols, {g: regs})
        row = dict(method=m, group=g, n_runs=len(dd), n_clusters=res[(g, "off")][3])
        for c in cols:
            mm, lo, hi, _ = res[(g, c)]; row[c] = mm; row[c+"_lo"] = lo; row[c+"_hi"] = hi
        row["pooled_median_share_fl"] = dd.share_fl.median(); row["pooled_median_corr"] = dd.corr_all.median(); row["pooled_median_skill"] = dd.skill.median()
        row["frac_runs_skill_pos"] = (dd.skill > 0).mean(); row["frac_runs_corr_ge_0.3"] = (dd.corr_all >= 0.3).mean()
        rows.append(row)
I23 = pd.DataFrame(rows); I23.to_csv(os.path.join(HERE, "item23_by_method.csv"), index=False)
for g in GR:
    d = I23[I23.group == g]
    print(f"\n[items 2/3] group={g} (cluster = sequence; value = cluster mean of the within-sequence seed medians; interval = t interval)")
    print(d[["method", "n_runs", "off", "fl", "rmse_avg", "share_fl", "share_fl_lo", "share_fl_hi", "corr_all", "corr_all_lo", "corr_all_hi", "sd_ratio", "skill", "skill_lo", "skill_hi", "pooled_median_share_fl", "pooled_median_corr", "pooled_median_skill", "frac_runs_skill_pos"]].round(3).to_string(index=False))
# all 11 deep methods together
print("\n[11 deep methods pooled] per group:")
for g, regs in GR.items():
    d = DLd[DLd.regime.isin(regs)]
    print(f"  {g}: share_fl median {d.share_fl.median():.3f} (IQR {d.share_fl.quantile(.25):.3f}-{d.share_fl.quantile(.75):.3f}); off median {d.off.median():.5f}; fl median {d.fl.median():.5f}; corr_all median {d.corr_all.median():.3f}; per-axis corr median x{d.corr_x.median():.2f} y{d.corr_y.median():.2f} z{d.corr_z.median():.2f}; sd_ratio median {d.sd_ratio.median():.3f}; skill median {d.skill.median():.3f}; share with skill>0 {(d.skill>0).mean():.3f}; share with corr>=0.3 {(d.corr_all>=0.3).mean():.3f}")
# share and skill per case (do the cases agree?)
cs_ = DLd.groupby(["regime", "seq"])[["share_fl", "corr_all", "skill", "sd_ratio"]].median(); print("\n[per-case medians]"); print(cs_.round(3).to_string())

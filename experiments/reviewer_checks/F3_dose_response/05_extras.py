# F3 step 5: values at pool quantiles, share of pool above thresholds, figure.
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DT = pd.read_csv(HERE + "/dose_table_by_level.csv"); TH = pd.read_csv(HERE + "/thresholds.csv"); NZ = pd.read_csv(HERE + "/noise_baselines.csv", index_col=0)
TVT = pd.read_csv(HERE + "/dose_tv_table_by_level.csv"); PR = pd.read_csv(HERE + "/pool_runs_summary.csv"); W = pd.read_csv(os.path.join(HERE, "..", "..", "..", "euroc_full", "results", "metrics_wide.csv"))
POOL = dict(p50=W.bias_rmse_avg.median(), p75=W.bias_rmse_avg.quantile(.75), p95=W.bias_rmse_avg.quantile(.95), max=W.bias_rmse_avg.max())
out = open(HERE + "/extras_summary.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
def interp(t, col, r): return float(10 ** np.interp(np.log10(r), np.log10(t.r), np.log10(np.maximum(t[col], 1e-30))))
rows = []
for L in (5, 10):
    t = DT[(DT.group == "all_rand") & (DT.L == L)].sort_values("r"); nz = NZ.loc[L]
    for nm, r in POOL.items():
        rows.append(dict(L=L, pool_point=nm, r=r, cabs_med=interp(t, "cabs_med", r), cdelta_med=interp(t, "cdelta_med", r), adabs_med=interp(t, "adabs_med", r),
                         adabs_over_Nrep=interp(t, "adabs_med", r) / nz.N_rep, adabs_over_Nseq=interp(t, "adabs_med", r) / nz.N_seq, adabs_over_Cabs0=interp(t, "adabs_med", r) / nz.N_med))
PQ = pd.DataFrame(rows); PQ.to_csv(HERE + "/dose_at_pool_quantiles.csv", index=False); P("DOSE-RESPONSE (constant, all random directions) at the pool's bias-RMSE quantiles"); P(PQ.to_string(float_format=lambda x: "%.3g" % x))
P("\nshare of pool runs (all 1026) with bias RMSE above the thresholds")
for L in (5, 10):
    for nm in ("all_rand",):
        g = TH[(TH.L == L) & (TH.group == nm)].iloc[0]
        for c in ("T_rep", "T_seq", "T_10pct"):
            P(f"L={L} {c}={g[c]:.2e}: share of pool runs with RMSE>= = {(W.bias_rmse_avg >= g[c]).mean():.3f}  (n runs {(W.bias_rmse_avg >= g[c]).sum()})")
# effective gain: pool C_delta / RMSE vs constant-offset model
P("\neffective gain C_delta / bias RMSE: pool (median over runs) vs constant-offset random-direction model (median)")
for L in (5, 10):
    g = PR[(PR.L == L) & PR.learned]; t = DT[(DT.group == "all_rand") & (DT.L == L)]
    P(f"L={L}: pool {np.median(g.cdelta / g.bias_rmse_avg):.1f}   model {np.median(t.cdelta_med / t.r):.1f}")
# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for a, L in zip(ax, (5, 10)):
    t = DT[(DT.group == "all_rand") & (DT.L == L)].sort_values("r"); nz = NZ.loc[L]
    g = PR[(PR.L == L)]
    a.scatter(g.bias_rmse_avg, g.cabs, s=5, c="0.6", alpha=.35, label="actual estimators (per run, RMS over 5 reps)")
    a.fill_between(t.r, t.cabs_p05, t.cabs_p95, color="C0", alpha=.2, label="injected constant offset, 5-95% over directions/units")
    a.plot(t.r, t.cabs_med, c="C0", lw=2, label="injected constant offset, median")
    tv = TVT[(TVT.L == L) & (TVT.variant == "TV") & (TVT.strength == "ALL")].sort_values("r"); a.plot(tv.r, tv.cabs_med, c="C3", ls="--", lw=1.5, label="injected real error shape (TV, median)")
    a.axhline(nz.N_med, c="k", lw=.8); a.axhspan(nz.N_med - nz.N_rep, nz.N_med + nz.N_rep, color="k", alpha=.08)
    for nm, ls in (("p50", ":"), ("p95", "--"), ("max", "-.")): a.axvline(POOL[nm], c="C2", ls=ls, lw=1, label=f"pool bias RMSE {nm}")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("bias RMSE equivalent of the injected error (m/s$^2$)"); a.set_ylabel("C_abs (m)"); a.set_title(f"L = {L} s (grey band: median C_abs(0) ± rep-to-rep SD)")
ax[0].legend(fontsize=6.5, loc="upper left"); fig.tight_layout(); fig.savefig(HERE + "/fig_dose_response.png", dpi=150)
out.close()

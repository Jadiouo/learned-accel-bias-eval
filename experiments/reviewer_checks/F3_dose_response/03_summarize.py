# F3 step 3: dose-response tables, noise baselines, thresholds, pool position, Spearman in the injected grid, observed pool Spearman.
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "closed_loop"))
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "euroc_full"))
C = pd.read_csv(HERE + "/dose_const_long.csv"); T = pd.read_csv(HERE + "/dose_tv_long.csv"); U = pd.read_csv(HERE + "/units.csv")
C["dabs"] = C["abs"] - C.abs0; C["adabs"] = C.dabs.abs()
T["dabs"] = T["abs"] - T.abs0; T["adabs"] = T.dabs.abs()
S3 = np.sqrt(3)
W = pd.read_csv(ROOT + "/results/metrics_wide.csv")
W["method"] = W.run_id.str.split("__").str[1]
learned = W[~W.method.str.startswith("constant")].bias_rmse_avg
POOL = dict(min=W.bias_rmse_avg.min(), p25=W.bias_rmse_avg.quantile(.25), med=W.bias_rmse_avg.median(), p75=W.bias_rmse_avg.quantile(.75),
            p95=W.bias_rmse_avg.quantile(.95), max=W.bias_rmse_avg.max())
POOL_L = dict(med=learned.median(), p95=learned.quantile(.95), max=learned.max())
out = open(HERE + "/summary.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
P("pool bias RMSE (all 1026 runs):", {k: float("%.3g" % v) for k, v in POOL.items()}); P("learned only (no constant_*):", {k: float("%.3g" % v) for k, v in POOL_L.items()})

# ---- noise baselines
NZ = {}
for L, g in U.groupby("L"):
    sd_in = np.sqrt(np.mean([x.abs0.std(ddof=1) ** 2 for _, x in g.groupby("seq")]))
    sd_seq = g.groupby("seq").abs0.mean().std(ddof=1)
    NZ[L] = dict(N_rep=sd_in, N_seq=sd_seq, N_med=g.abs0.median(), mean=g.abs0.mean())
    P(f"L={L} noise baselines of C_abs(delta=0): N_rep={sd_in:.3f}  N_seq={sd_seq:.3f}  median={g.abs0.median():.3f}  per-seq mean:",
      {s: round(v, 3) for s, v in g.groupby('seq').abs0.mean().items()})
pd.DataFrame(NZ).T.to_csv(HERE + "/noise_baselines.csv")

def level_table(df, by):
    g = df.groupby(by + ["k", "r"])
    t = g.agg(n=("abs", "size"), cabs_med=("abs", "median"), cabs_p05=("abs", lambda x: x.quantile(.05)), cabs_p25=("abs", lambda x: x.quantile(.25)),
              cabs_p75=("abs", lambda x: x.quantile(.75)), cabs_p95=("abs", lambda x: x.quantile(.95)),
              cdelta_med=("delta", "median"), dabs_med=("dabs", "median"), adabs_med=("adabs", "median"), adabs_p90=("adabs", lambda x: x.quantile(.9))).reset_index()
    return t
C["cls"] = np.where(C.kind == "axis", "axis", np.where(C.uz.abs() < 0.5, "horiz", np.where(C.uz.abs() > 0.8, "vert", "mid")))
rand = C[C.kind == "rand"]
tabs = []
for nm, df in [("all_rand", rand), ("horiz", rand[rand.cls == "horiz"]), ("vert", rand[rand.cls == "vert"]),
               ("axis_x", C[(C.kind == "axis") & (C.ux.abs() > .9)]), ("axis_y", C[(C.kind == "axis") & (C.uy.abs() > .9)]), ("axis_z", C[(C.kind == "axis") & (C.uz.abs() > .9)])]:
    t = level_table(df, ["L"]); t.insert(0, "group", nm); tabs.append(t)
DT = pd.concat(tabs); DT["delta_norm"] = DT.r * S3
DT.to_csv(HERE + "/dose_table_by_level.csv", index=False)
bs = level_table(rand, ["L", "seq"]); bs.to_csv(HERE + "/dose_table_by_seq.csv", index=False)

def threshold(t, col, thr):
    """smallest level with col>=thr and stays >=thr for all larger levels; log-linear interpolation of r."""
    t = t.sort_values("k"); v = t[col].to_numpy(); r = t.r.to_numpy()
    ok = v >= thr
    if not ok.any() or not ok[-1]: return np.nan
    i = len(ok) - 1
    while i > 0 and ok[i - 1]: i -= 1
    if i == 0: return r[0]
    lr0, lr1 = np.log10(r[i - 1]), np.log10(r[i]); v0, v1 = np.log10(max(v[i - 1], 1e-30)), np.log10(v[i])
    return 10 ** (lr0 + (np.log10(thr) - v0) / (v1 - v0) * (lr1 - lr0))
rows = []
for L in (5, 10):
    nz = NZ[L]
    for nm in ["all_rand", "horiz", "vert", "axis_x", "axis_y", "axis_z"]:
        t = DT[(DT.group == nm) & (DT.L == L)]
        rows.append(dict(L=L, group=nm, T_rep=threshold(t, "adabs_med", nz["N_rep"]), T_seq=threshold(t, "adabs_med", nz["N_seq"]),
                         T_10pct=threshold(t, "adabs_med", 0.1 * nz["N_med"]),
                         T_rep_p90=threshold(t, "adabs_p90", nz["N_rep"]), T_cdelta_rep=threshold(t, "cdelta_med", nz["N_rep"])))
    # per-seq
    for s, t in bs[bs.L == L].groupby("seq"):
        gz = U[(U.L == L) & (U.seq == s)]
        rows.append(dict(L=L, group="seq:" + s, T_rep=threshold(t, "adabs_med", nz["N_rep"]), T_seq=threshold(t, "adabs_med", nz["N_seq"]),
                         T_10pct=threshold(t, "adabs_med", 0.1 * gz.abs0.median())))
TH = pd.DataFrame(rows); TH.to_csv(HERE + "/thresholds.csv", index=False)
P("\nTHRESHOLDS (r = bias RMSE equivalent, m/s^2; ||delta|| = r*sqrt(3)):"); P(TH.to_string(float_format=lambda x: "%.2e" % x))
P("\nDose table (all random directions), selected levels:")
for L in (5, 10):
    t = DT[(DT.group == "all_rand") & (DT.L == L) & (DT.k % 2 == 0)]
    P(f"L={L}"); P(t[["r", "delta_norm", "cabs_med", "cabs_p05", "cabs_p95", "cdelta_med", "adabs_med", "dabs_med"]].to_string(index=False, float_format=lambda x: "%.3g" % x))

# ---- pool position: actual estimators in consequences_long (paired with b_ref)
CLg = pd.read_csv(CL + "/consequences_long.csv"); CLg = CLg[(CLg.cfg == "base") & (CLg["mode"] == "O") & CLg.L.isin([5, 10])]
ref = CLg[CLg.run_id == "CTRL:bref"].set_index(["seq", "L", "rep"])["abs"].rename("abs_ref")
R = CLg[~CLg.run_id.str.startswith("CTRL")].merge(ref.reset_index(), on=["seq", "L", "rep"]); R["adabs"] = (R["abs"] - R.abs_ref).abs()
R["regime"] = R.run_id.str.split("__").str[0]; R["method"] = R.run_id.str.split("__").str[1]
R = R.merge(W[["run_id", "eval_sequence", "bias_rmse_avg"]].rename(columns={"eval_sequence": "seq"}), on=["run_id", "seq"])
R["learned"] = ~R.method.str.startswith("constant")
rr = R.groupby(["run_id", "seq", "L", "learned", "bias_rmse_avg"]).agg(cdelta=("delta", lambda x: np.sqrt(np.mean(x ** 2))), cabs=("abs", lambda x: np.sqrt(np.mean(x ** 2))),
                                                                          adabs=("adabs", "median")).reset_index()
rr.to_csv(HERE + "/pool_runs_summary.csv", index=False)
P("\nPOOL (actual estimators; per-run RMS over 5 reps): quantiles over runs of C_delta, median|dC_abs| (paired), bias RMSE")
for L in (5, 10):
    for lab, g in [("learned", rr[(rr.L == L) & rr.learned]), ("all", rr[rr.L == L])]:
        P(f"L={L} {lab} n={len(g)} bias_rmse q50/q95/max={g.bias_rmse_avg.median():.2e}/{g.bias_rmse_avg.quantile(.95):.2e}/{g.bias_rmse_avg.max():.2e}  "
          f"C_delta q50/q95/max={g.cdelta.median():.3f}/{g.cdelta.quantile(.95):.3f}/{g.cdelta.max():.3f}  |dCabs| paired q50/q95/max={g.adabs.median():.3f}/{g.adabs.quantile(.95):.3f}/{g.adabs.max():.3f}  "
          f"share of runs with median|dCabs|>=N_rep: {(g.adabs >= NZ[L]['N_rep']).mean():.3f}, >=N_seq: {(g.adabs >= NZ[L]['N_seq']).mean():.3f}")

# ---- Spearman inside the injected grid, per unit (seq, L, rep) across 462 conditions
def rho(a, b): return spearmanr(a, b)[0]
rowsS = []
bands = {"full(1e-5..1)": (0, 1e9), "pool range [7e-5,5.8e-2]": (POOL["min"] * 0.99, POOL["max"] * 1.01), "pool med x/ 10 [2.8e-4,2.8e-2]": (POOL["med"] / 10, POOL["med"] * 10),
         "pool med +- 1 level[1.7e-3,4.7e-3]": (POOL["med"] / 1.8, POOL["med"] * 1.8)}
for (s, L, rep), g in C.groupby(["seq", "L", "rep"]):
    for bn, (lo, hi) in bands.items():
        gg = g[(g.r >= lo) & (g.r <= hi)]
        rowsS.append(dict(seq=s, L=L, rep=rep, band=bn, n=len(gg), rho_cabs=rho(gg.r, gg["abs"]), rho_cdelta=rho(gg.r, gg.delta)))
SP = pd.DataFrame(rowsS); SP.to_csv(HERE + "/spearman_in_grid_by_unit.csv", index=False)
SPs = SP.groupby(["L", "band"]).agg(n_units=("rep", "size"), rho_cabs_med=("rho_cabs", "median"), rho_cabs_q25=("rho_cabs", lambda x: x.quantile(.25)),
                                     rho_cabs_q75=("rho_cabs", lambda x: x.quantile(.75)), rho_cdelta_med=("rho_cdelta", "median")).reset_index()
SPs.to_csv(HERE + "/spearman_in_grid.csv", index=False); P("\nSpearman(r, C_abs) across injected conditions, per unit (seq,rep), median/IQR over 25 units"); P(SPs.to_string(float_format=lambda x: "%.3f" % x))

# ---- observed pool Spearman (replicates analyze.scenario S1): reproduce scenario_stats.csv
Pd = pd.read_csv(CL + "/predictors.csv"); Pd = Pd[Pd.regime.isin(["A", "B"])]
SS = pd.read_csv(CL + "/scenario_stats.csv")
obs = []
for L in (5, 10):
    s = CLg[CLg.L == L]
    for cons, col in [("abs", "abs"), ("delta", "delta")]:
        agg = s.groupby(["run_id", "seq"])[col].apply(lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))).rename("c").reset_index()
        m = Pd.merge(agg, on=["run_id", "seq"])
        rh = []
        for (reg, seq), d in m.groupby(["regime", "seq"]):
            meth = d.groupby("method"); X = meth["bias_rmse"].mean(); c = meth.c.apply(lambda x: np.sqrt(np.mean(x ** 2)))
            rh.append(rho(X.loc[c.index].to_numpy(), c.to_numpy()))
        ref_v = SS[(SS.cfg == "base") & (SS["mode"] == "O") & (SS.L == L) & (SS.cons == cons) & (SS.pred == "bias_rmse")].s1_median.iloc[0]
        obs.append(dict(L=L, cons=cons, n_cells=len(rh), rho_median=float(np.median(rh)), scenario_stats_value=ref_v, cell_rhos=" ".join("%.2f" % x for x in rh)))
OB = pd.DataFrame(obs); OB.to_csv(HERE + "/observed_pool_spearman.csv", index=False); P("\nObserved pool Spearman(bias RMSE, consequence), cell median (reproduction of scenario_stats.csv)"); P(OB.to_string())

# ---- TV tables
T["strength_k"] = T.strength
tt = []
for L in (5, 10):
    for var in ["TV", "TVmean", "TVdm"]:
        for st in ["min", "med", "max", "ALL"]:
            d = T[(T.L == L) & (T.variant == var)]
            if st != "ALL": d = d[d.strength == st]
            t = d.groupby(["k", "r"]).agg(cabs_med=("abs", "median"), cdelta_med=("delta", "median"), adabs_med=("adabs", "median")).reset_index()
            nz = NZ[L]
            tt.append(dict(L=L, variant=var, strength=st, T_rep=threshold(t, "adabs_med", nz["N_rep"]), T_seq=threshold(t, "adabs_med", nz["N_seq"]),
                           T_10pct=threshold(t, "adabs_med", 0.1 * nz["N_med"])))
            t.insert(0, "L", L); t.insert(1, "variant", var); t.insert(2, "strength", st); t.to_csv(HERE + "/_tmp_tv.csv", mode="a", header=not os.path.exists(HERE + "/_tmp_tv.csv"), index=False)
os.replace(HERE + "/_tmp_tv.csv", HERE + "/dose_tv_table_by_level.csv")
TT = pd.DataFrame(tt); TT.to_csv(HERE + "/thresholds_tv.csv", index=False); P("\nTV THRESHOLDS"); P(TT.to_string(float_format=lambda x: "%.2e" % x))
# constant vs TV: ratio of thresholds
P("\nTV table (variant TV, ALL strengths) selected levels")
for L in (5, 10):
    t = pd.read_csv(HERE + "/dose_tv_table_by_level.csv"); t = t[(t.L == L) & (t.strength == "ALL") & (t.k % 2 == 0)]
    P(f"L={L}"); P(t.pivot(index="r", columns="variant", values="cabs_med").to_string(float_format=lambda x: "%.3g" % x))
    P("adabs_med"); P(t.pivot(index="r", columns="variant", values="adabs_med").to_string(float_format=lambda x: "%.3g" % x))
out.close()

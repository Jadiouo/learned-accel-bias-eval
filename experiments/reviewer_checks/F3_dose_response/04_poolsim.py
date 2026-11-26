# F3 step 4: pool-like simulation (constant offsets, random directions, the pool's actual bias RMSE spread, scaled) + mean-offset surrogate of the REAL pool.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
U = pd.read_csv(HERE + "/units.csv"); S3 = np.sqrt(3)
Pd = pd.read_csv(CL + "/predictors.csv"); Pd = Pd[Pd.regime.isin(["A", "B"])]
NDRAW = 2000; SCALES = [1, 3, 10, 30, 100, 300, 1000]
out = open(HERE + "/poolsim_summary.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

def unit_arrays(seq, L):
    g = U[(U.seq == seq) & (U.L == L)].sort_values("rep")
    e0 = g[["e00", "e01", "e02"]].to_numpy()                                  # (K,3)
    M = g[[f"M{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    return e0, M

def rank_rho(x, y):                       # x (n,), y (N,n) -> (N,)
    rx = rankdata(x); ry = rankdata(y, axis=1)
    rx = rx - rx.mean(); ry = ry - ry.mean(axis=1, keepdims=True)
    return (ry @ rx) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum(axis=1))

cells = [(reg, seq, d.groupby("method").bias_rmse.mean()) for (reg, seq), d in Pd.groupby(["regime", "seq"])]
rng = np.random.default_rng(20261126)
obs = pd.read_csv(HERE + "/observed_pool_spearman.csv")
rows = []
for L in (5, 10):
    # draws are shared across scales (same directions), different cells independent
    res = {s: {"abs": [], "delta": []} for s in SCALES}
    for reg, seq, X in cells:
        e0, M = unit_arrays(seq, L); r = X.to_numpy()
        u = rng.standard_normal((NDRAW, len(r), 3)); u /= np.linalg.norm(u, axis=2, keepdims=True)
        for s in SCALES:
            dl = u * (r * S3 * s)[None, :, None]                              # (N,m,3)
            Md = np.einsum("kij,nmj->nkmi", M, dl)                             # (N,K,m,3)
            ca = np.sqrt(np.mean(np.sum((e0[None, :, None, :] - Md) ** 2, axis=3), axis=1))   # (N,m) RMS over reps
            cd = np.sqrt(np.mean(np.sum(Md ** 2, axis=3), axis=1))
            res[s]["abs"].append(rank_rho(r, ca)); res[s]["delta"].append(rank_rho(r, cd))
    for s in SCALES:
        for cons in ("abs", "delta"):
            A = np.stack(res[s][cons])                                         # (cells,N)
            med = np.median(A, axis=0)                                         # cell-median per draw
            o = obs[(obs.L == L) & (obs.cons == cons)].rho_median.iloc[0]
            rows.append(dict(L=L, scale=s, cons=cons, pool_median_rmse_scaled=float(np.median(np.concatenate([c[2].to_numpy() for c in cells])) * s),
                             pool_max_rmse_scaled=float(max(c[2].max() for c in cells) * s), rho_med=float(np.median(med)), rho_lo=float(np.quantile(med, .025)),
                             rho_hi=float(np.quantile(med, .975)), observed=o, frac_draws_le_observed=float((med <= o).mean()),
                             cell_rho_mean_over_draws=" ".join("%.2f" % x for x in A.mean(axis=1))))
PS = pd.DataFrame(rows); PS.to_csv(HERE + "/poolsim_spearman.csv", index=False)
P("POOL-LIKE SIMULATION: constant offset, random direction per method, pool's actual bias-RMSE spread x scale; cell-median Spearman(bias RMSE, consequence), 2000 draws")
P(PS.drop(columns="cell_rho_mean_over_draws").to_string(float_format=lambda x: "%.3g" % x))
P("per-cell mean rho (scale 1, abs):"); [P(L, " ".join(f"{c[0]}-{c[1]}" for c in cells), PS[(PS.L == L) & (PS.scale == 1) & (PS.cons == 'abs')].cell_rho_mean_over_draws.iloc[0]) for L in (5, 10)]

# ---- mean-offset surrogate of the REAL pool: replace each real estimator's error by its constant outage-window mean
rows = []
for seq in SEQS:
    D = RC.load_seq(seq)
    bref_h = RC.hold_series(D, D["bref"])
    for L in (5, 10):
        e0, M = unit_arrays(seq, L)
        i1 = D["i1"]; ko = i1 - L * FS
        for rid, b in zip(D["run_ids"], D["bhat"]):
            dh = RC.hold_series(D, b)[ko:i1] - bref_h[ko:i1]; dbar = dh.mean(0)       # outage-window mean error
            Mdb = np.einsum("kij,j->ki", M, dbar)                                      # (K,3)
            rows.append(dict(run_id=rid, seq=seq, L=L, sur_abs=float(np.sqrt(np.mean(np.sum((e0 - Mdb) ** 2, 1)))),
                             sur_delta=float(np.sqrt(np.mean(np.sum(Mdb ** 2, 1)))), mean_norm=float(np.linalg.norm(dbar))))
SUR = pd.DataFrame(rows)
CLg = pd.read_csv(CL + "/consequences_long.csv"); CLg = CLg[(CLg.cfg == "base") & (CLg["mode"] == "O") & CLg.L.isin([5, 10])]
act = CLg.groupby(["run_id", "seq", "L"]).agg(act_abs=("abs", lambda x: np.sqrt(np.mean(x ** 2))), act_delta=("delta", lambda x: np.sqrt(np.mean(x ** 2)))).reset_index()
SUR = SUR.merge(act, on=["run_id", "seq", "L"]); SUR.to_csv(HERE + "/surrogate_by_run.csv", index=False)
from scipy.stats import spearmanr
def rho(a, b): return float(spearmanr(a, b)[0])
rr = []
for L in (5, 10):
    S = SUR[SUR.L == L].merge(Pd[["run_id", "seq", "regime", "method", "bias_rmse"]], on=["run_id", "seq"])
    for nm, a, b in [("C_delta", "sur_delta", "act_delta"), ("C_abs", "sur_abs", "act_abs")]:
        g = S
        rr.append(dict(L=L, what=nm, pearson_log_sur_vs_actual=float(np.corrcoef(np.log(g[a]), np.log(g[b]))[0, 1]),
                       median_ratio_actual_over_surrogate=float((g[b] / g[a]).median()), max_abs_diff=float((g[a] - g[b]).abs().max()), median_abs_diff=float((g[a] - g[b]).abs().median())))
    for cons, scol, acol in [("abs", "sur_abs", "act_abs"), ("delta", "sur_delta", "act_delta")]:
        rs, ra = [], []
        for (reg, seq), d in S.groupby(["regime", "seq"]):
            m = d.groupby("method"); X = m.bias_rmse.mean()
            rs.append(rho(X, m[scol].apply(lambda x: np.sqrt(np.mean(x ** 2))).loc[X.index])); ra.append(rho(X, m[acol].apply(lambda x: np.sqrt(np.mean(x ** 2))).loc[X.index]))
        rr.append(dict(L=L, what=f"cell-median Spearman(bias RMSE, {cons}) surrogate vs actual", surrogate=float(np.median(rs)), actual=float(np.median(ra))))
SR = pd.DataFrame(rr); SR.to_csv(HERE + "/surrogate_check.csv", index=False)
P("\nMEAN-OFFSET SURROGATE of the real pool (each estimator -> constant outage-window mean error, same filter state)"); P(SR.to_string(float_format=lambda x: "%.4g" % x))
out.close()

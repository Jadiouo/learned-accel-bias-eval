# F6 step 3: summaries and decision criteria of PLAN.md. Writes summary.txt and csv tables in F6_ba_overwrite/.
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261126)
C = pd.read_csv(HERE + "/cond_long.csv"); S = pd.read_csv(HERE + "/src_long.csv")
H = pd.read_csv(HERE + "/shapley_units.csv"); O = pd.read_csv(HERE + "/onset_units.csv")
out = []; P = lambda s="": out.append(s)
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
SEQ = sorted(C.seq.unique()); KEY = ["seq", "rep", "q", "L"]
CN = {"a": "(a) b_ref (O_base)", "b": "(b) filter b_a", "c": "(c) b_ref + tilt consistent", "c1": "(c1) first-order version of c", "c2": "(c2) c with 1-s mean specific force",
      "d": "(d) b_ref + GT tilt", "e": "(e) filter b_a + GT tilt", "f": "(f) b_ref + GT all 5 states"}

def boot_med(x, seqs, B=2000):
    """cluster bootstrap over sequences of the median of per-unit values"""
    x = np.asarray(x); seqs = np.asarray(seqs); u = sorted(set(seqs)); g = {s: x[seqs == s] for s in u}; r = []
    for _ in range(B):
        r.append(np.median(np.concatenate([g[s] for s in rng.choice(u, len(u))])))
    return np.percentile(r, [2.5, 97.5])

# ---------- 1. condition table ----------
rows = []
for (q, L), d in C.groupby(["q", "L"]):
    A = d[d.cond == "a"].set_index(["seq", "rep"])
    for cn in ["a", "b", "c", "c1", "c2", "d", "e", "f"]:
        X = d[d.cond == cn].set_index(["seq", "rep"]).loc[A.index]
        ratio = (X["abs"] / A["abs"]).values
        ea = A[["ex", "ey", "ez"]].values; ec = X[["ex", "ey", "ez"]].values
        cos = (ea * ec).sum(1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(ec, axis=1))
        dv = np.linalg.norm(ea - ec, axis=1)
        per_seq = [float(np.median(ratio[A.index.get_level_values(0) == s])) for s in SEQ]
        lo, hi = boot_med(ratio, A.index.get_level_values(0).values)
        rows.append(dict(q=q, L=L, cond=cn, abs_unit_median=float(X["abs"].median()), abs_unit_mean=float(X["abs"].mean()),
                         abs_5seqRMS_median=float(X.reset_index().groupby("seq")["abs"].apply(rms).median()),
                         ratio_vs_a_median=float(np.median(ratio)), ratio_lo=lo, ratio_hi=hi,
                         **{f"ratio_{s}": v for s, v in zip(SEQ, per_seq)}, n_seq_below_0p8=int(sum(v < 0.8 for v in per_seq)),
                         cos_vs_a_median=float(np.median(cos)), diff_vec_vs_a_median=float(np.median(dv))))
T = pd.DataFrame(rows); T.to_csv(HERE + "/table_conditions.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P("== 1. Conditions: C_abs (m). unit median (25 units), paper-style 5-seq RMS median, paired per-unit ratio vs (a) with seq-cluster bootstrap CI ==")
for (q, L), d in T.groupby(["q", "L"]):
    P(f"-- Q_ba={q}  L={L}")
    P(d[["cond", "abs_unit_median", "abs_5seqRMS_median", "ratio_vs_a_median", "ratio_lo", "ratio_hi", "n_seq_below_0p8", "cos_vs_a_median", "diff_vec_vs_a_median"]].round(3).to_string(index=False))

# ---------- 2. Shapley shares ----------
rows = []
for (q, L, src), d in H.groupby(["q", "L", "source"]):
    t = d.total.sum(); r = dict(q=q, L=L, source=src, total_mean=d.total.mean(), total_median=d.total.median(), floor=d.floor.sum() / t)
    for c in ["pos", "vel", "tilt", "yaw", "bg"]: r["shap_" + c] = d["shap_" + c].sum() / t
    r["tilt_by_seq_min"] = min(g.shap_tilt.sum() / g.total.sum() for _, g in d.groupby("seq"))
    r["tilt_by_seq_max"] = max(g.shap_tilt.sum() / g.total.sum() for _, g in d.groupby("seq"))
    rows.append(r)
SH = pd.DataFrame(rows); SH.to_csv(HERE + "/table_shapley.csv", index=False)
P(); P("== 2. Shapley shares (sum phi / sum total). source bref = (a), source ba = (b) =="); P(SH.round(3).to_string(index=False))

# ---------- 3. onset states ----------
rows = []
for (q, L), d in O.groupby(["q", "L"]):
    rows.append(dict(q=q, L=L, tilt_deg_med=d.tilt_deg.median(), yaw_deg_med=d.yaw_deg.median(), tilt_c_deg_med=d.tilt_c_deg.median(),
                     corr_c_deg_med=d.corr_c_deg.median(), corr_c_z_deg_med=d.corr_c_z_deg.median(), yaw_c_deg_med=d.yaw_c_deg.median(),
                     vel_err_med=d.vel_err.median(), ba_minus_bref_norm_med=float(np.median(np.linalg.norm(d[["ba0", "ba1", "ba2"]].values - d[["bref0", "bref1", "bref2"]].values, axis=1)))))
OT = pd.DataFrame(rows); OT.to_csv(HERE + "/table_onset.csv", index=False)
P(); P("== 3. Onset state errors (median over 25 units). tilt_c = tilt error after (c) correction; corr_c = size of (c) rotation =="); P(OT.round(4).to_string(index=False))

# ---------- 4. compensation correlation ----------
from scipy.stats import pearsonr
G = 9.81; rows = []
def reg(x, y):
    r = pearsonr(x, y)[0]; sl = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan
    return r, sl
for (q, L), d in O.groupby(["q", "L"]):
    ell = np.stack([G * d.ewy.values, -G * d.ewx.values], 1)        # leakage acceleration from tilt error (world xy)
    beta = d[["Rba0", "Rba1"]].values; betap = d[["Rbd0", "Rbd1"]].values
    for nm, b in (("beta=(R b_a)_xy", beta), ("beta'=(R(b_a-b_ref))_xy", betap)):
        r_all, s_all = reg(b.ravel(), ell.ravel())                    # slope of ell on beta (comp.: ~1)
        # within-sequence: remove sequence means
        sq = d.seq.values; bd = b.copy(); ed = ell.copy()
        for s in SEQ: m = sq == s; bd[m] -= bd[m].mean(0); ed[m] -= ed[m].mean(0)
        r_in, s_in = reg(bd.ravel(), ed.ravel())
        per = [pearsonr(b[sq == s].ravel(), ell[sq == s].ravel())[0] for s in SEQ]
        res = np.linalg.norm(ell - b, axis=1)
        rows.append(dict(q=q, L=L, quantity=nm, r_pooled=r_all, slope_ell_on_beta=s_all, r_within_seq=r_in, slope_within=s_in,
                         r_per_seq_min=min(per), r_per_seq_max=max(per), ell_norm_med=float(np.median(np.linalg.norm(ell, axis=1))),
                         beta_norm_med=float(np.median(np.linalg.norm(b, axis=1))), resid_over_ell_med=float(np.median(res / np.linalg.norm(ell, axis=1)))))
CR = pd.DataFrame(rows); CR.to_csv(HERE + "/table_onset_corr.csv", index=False)
P(); P("== 4. Onset compensation test: ell = g(e_y, -e_x) (tilt-leakage accel) vs beta. compensation <=> slope ~1, high r =="); P(CR.round(3).to_string(index=False))

# ---------- 5. shared error vs C_delta ratio under (a) and (d) ----------
rows = []
for (q, L, cn), d in S.groupby(["q", "L", "cond"]):
    lm = d[~d.run_id.str.startswith("CTRL:")]
    u = lm.groupby(["seq", "rep"]).delta.median(); b = d[d.run_id == "CTRL:bref"].set_index(["seq", "rep"])["abs"]
    ratio = (b / u).dropna()
    rows.append(dict(q=q, L=L, cond=cn, bref_abs_unit_med=float(b.median()), learned_delta_unit_med=float(u.median()),
                     ratio_med=float(ratio.median()), ratio_iqr_lo=float(ratio.quantile(.25)), ratio_iqr_hi=float(ratio.quantile(.75)),
                     learned_abs_med=float(lm["abs"].median()), zero_delta_med=float(d[d.run_id == "CTRL:zero"].delta.median())))
RT = pd.DataFrame(rows); RT.to_csv(HERE + "/table_ratio.csv", index=False)
P(); P("== 5. shared error / C_delta (per unit: C_abs(b_ref) / median over learned sources of C_delta; median over units). (a) vs (d) =="); P(RT.round(3).to_string(index=False))

# ---------- 6. decision criteria ----------
P(); P("== 6. Decision criteria from PLAN.md (L=10 primary, L=5 for direction) ==")
g = lambda q, L, c, col="ratio_vs_a_median": float(T[(T.q == q) & (T.L == L) & (T.cond == c)][col].iloc[0])
gc = lambda q, L, c, col="abs_unit_median": float(T[(T.q == q) & (T.L == L) & (T.cond == c)][col].iloc[0])
for L in (10, 5):
    Rb, Rc, Rd, Re = g("base", L, "b"), g("base", L, "c"), g("base", L, "d"), g("base", L, "e")
    nseq = int(T[(T.q == "base") & (T.L == L) & (T.cond == "c")].n_seq_below_0p8.iloc[0])
    sup = Rc <= 0.6 and Rb <= 0.8 and nseq >= 4; nsup = Rc >= 0.85 and Rb >= 0.85
    fp = (gc("base", L, "a") - gc("base", L, "c")) / (gc("base", L, "a") - gc("base", L, "d"))
    P(f"L={L}: R_b={Rb:.3f}  R_c={Rc:.3f}  R_d={Rd:.3f}  R_e (vs a)={Re:.3f}  (e)/(b)={gc('base',L,'e')/gc('base',L,'b'):.3f}  n_seq(R_c<0.8)={nseq}/5")
    P(f"      hypothesis (i): {'SUPPORTED' if sup else 'NOT SUPPORTED' if nsup else 'PARTIAL'};  f_proto=(C_a-C_c)/(C_a-C_d) (unit medians) = {fp:.3f}")
for L in (10, 5):
    sh = {q: float(SH[(SH.q == q) & (SH.L == L) & (SH.source == "bref")].shap_tilt.iloc[0]) for q in ("q01", "base", "q10")}
    sha = {q: float(SH[(SH.q == q) & (SH.L == L) & (SH.source == "ba")].shap_tilt.iloc[0]) for q in ("q01", "base", "q10")}
    Rc = {q: g(q, L, "c") for q in sh}; Rb = {q: g(q, L, "b") for q in sh}
    Ca = {q: gc(q, L, "a") for q in sh}; Cc = {q: gc(q, L, "c") for q in sh}; Cb = {q: gc(q, L, "b") for q in sh}
    rng_a = (max(Ca.values()) - min(Ca.values())) / np.median(list(Ca.values())); rng_c = (max(Cc.values()) - min(Cc.values())) / np.median(list(Cc.values()))
    mono_up = sh["q01"] < sh["base"] < sh["q10"]
    mono_Rc = Rc["q01"] > Rc["base"] > Rc["q10"] and (Rc["q01"] - Rc["q10"]) >= 0.15
    mono_Rb = Rb["q01"] > Rb["base"] > Rb["q10"] and (Rb["q01"] - Rb["q10"]) >= 0.15
    flat = (max(Cc.values()) - min(Cc.values())) <= 0.5 * (max(Ca.values()) - min(Ca.values()))
    P(f"L={L}: tilt Shapley (b_ref src) q01/base/q10 = {sh['q01']:.3f}/{sh['base']:.3f}/{sh['q10']:.3f}; (b_a src) = {sha['q01']:.3f}/{sha['base']:.3f}/{sha['q10']:.3f}")
    P(f"      unit-median C_abs: (a) {Ca['q01']:.3f}/{Ca['base']:.3f}/{Ca['q10']:.3f}; (b) {Cb['q01']:.3f}/{Cb['base']:.3f}/{Cb['q10']:.3f}; (c) {Cc['q01']:.3f}/{Cc['base']:.3f}/{Cc['q10']:.3f}")
    P(f"      R_b = {Rb['q01']:.3f}/{Rb['base']:.3f}/{Rb['q10']:.3f};  R_c = {Rc['q01']:.3f}/{Rc['base']:.3f}/{Rc['q10']:.3f};  range/median: (a) {rng_a:.2f}, (c) {rng_c:.2f}")
    cond12 = (mono_up, (mono_Rc or mono_Rb), flat)
    P(f"      hypothesis (ii) conditions (1 tilt share rises, 2 R_c or R_b falls >=0.15, 3 (c) range <= half of (a)): {cond12} -> "
      + ("SUPPORTED" if all(cond12) else "NOT SUPPORTED" if (mono_up and not cond12[1] and not cond12[2]) else "PARTIAL"))
r = CR[(CR.q == "base") & (CR.L == 10) & CR.quantity.str.startswith("beta'")].iloc[0]
P(f"compensation correlation (base, L=10, beta'): r={r.r_pooled:.3f}, slope={r.slope_ell_on_beta:.3f}, within-seq r={r.r_within_seq:.3f} -> "
  + ("compensation present" if r.r_pooled >= 0.7 and 0.5 <= r.slope_ell_on_beta <= 1.5 else "weak" if r.r_pooled >= 0.4 else "absent"))
open(HERE + "/summary.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))

# F6 step 4: construction check of (c), (c)-vs-(b) vector difference, per-sequence ratios, robust compensation correlation (exploratory).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f6core import *
import pandas as pd
from scipy.stats import spearmanr, pearsonr
out = []; P = lambda s="": out.append(s)
# --- construction check on all 25 units, L=10, base
res = []; res_fo = []
for seq in SEQS:
    for rep in range(K):
        D, packs = snapshots(seq, rep, 1.0); src = RC.sources(D)
        for L, (ko, i1, Sn) in packs.items():
            bref0 = src["CTRL:bref"][ko]; Rf = Sn["R"]; fbar = Rf.T @ np.array([0, 0, G])
            Rc = consistent_tilt_R(Sn, bref0)
            lhs = Rc @ (fbar - bref0); rhs = Rf @ (fbar - Sn["ba"])
            # direction mismatch only (norm differs by |b_ref|-|b_a| vertical part)
            ang = np.degrees(np.arccos(np.clip(lhs @ rhs / np.linalg.norm(lhs) / np.linalg.norm(rhs), -1, 1)))
            res.append(dict(seq=seq, L=L, angle_mismatch_deg=ang, horiz_resid=float(np.linalg.norm((lhs - rhs)[:2])), horiz_shift_without=float(np.linalg.norm((Rf @ (fbar - bref0) - rhs)[:2]))))
R = pd.DataFrame(res)
P("== (c) construction: direction mismatch of R_c(fbar-b_ref) vs R_f(fbar-b_a), and horizontal world accel mismatch (m/s^2) with and without the correction ==")
P(R.groupby("L").agg(angle_deg_max=("angle_mismatch_deg", "max"), horiz_resid_max=("horiz_resid", "max"), horiz_resid_med=("horiz_resid", "median"),
                     horiz_shift_without_med=("horiz_shift_without", "median"), horiz_shift_without_max=("horiz_shift_without", "max")).to_string())
# --- vectors
C = pd.read_csv(HERE + "/cond_long.csv"); T = pd.read_csv(HERE + "/table_conditions.csv")
P(); P("== ||e_x - e_y|| medians (m), end-error vectors, base Q ==")
for L in (10, 5):
    d = C[(C.q == "base") & (C.L == L)].set_index(["seq", "rep", "cond"])[["ex", "ey", "ez"]]
    def dist(x, y):
        a = d.xs(x, level="cond"); b = d.xs(y, level="cond"); return float(np.median(np.linalg.norm(a.values - b.values, axis=1)))
    P(f"L={L}: |e_c-e_b|={dist('c','b'):.3f}  |e_c2-e_b|={dist('c2','b'):.3f}  |e_a-e_b|={dist('a','b'):.3f}  |e_a-e_c|={dist('a','c'):.3f}  |e_a-e_c2|={dist('a','c2'):.3f}")
P(); P("== per-sequence median ratio C(cond)/C(a) (paired units), base Q =="); 
for L in (10, 5):
    P(f"L={L}"); P(T[(T.q == "base") & (T.L == L)][["cond", "ratio_vs_a_median", "ratio_lo", "ratio_hi"] + [c for c in T.columns if c.startswith("ratio_MH")]].round(3).to_string(index=False))
    P(T[(T.q == "base") & (T.L == L)][["cond", "abs_unit_median", "abs_5seqRMS_median"]].round(3).to_string(index=False))
P(); P("== mean C_abs (25 units) per condition: robustness of 'median' =="); P(T[T.cond.isin(["a", "b", "c", "c2", "d", "e"])].pivot_table(index=["q", "L"], columns="cond", values="abs_unit_mean").round(3).to_string())
P(); P("== ratio vs (a) for c2 (static-gravity assumption relaxed) =="); P(T[T.cond == "c2"][["q", "L", "abs_unit_median", "ratio_vs_a_median", "ratio_lo", "ratio_hi", "n_seq_below_0p8"]].round(3).to_string(index=False))
# --- robust compensation correlation
O = pd.read_csv(HERE + "/onset_units.csv"); G9 = 9.81; rows = []
for (q, L), d in O.groupby(["q", "L"]):
    ell = np.stack([G9 * d.ewy.values, -G9 * d.ewx.values], 1); bp = d[["Rbd0", "Rbd1"]].values
    nrm = np.linalg.norm(ell, axis=1); keep = nrm < np.sort(nrm)[-3]
    sp = spearmanr(bp.ravel(), ell.ravel())[0]; pr = pearsonr(bp.ravel(), ell.ravel())[0]
    k2 = np.repeat(keep, 2); pr2 = pearsonr(bp.ravel()[k2], ell.ravel()[k2])[0]; sl2 = float((bp.ravel()[k2] * ell.ravel()[k2]).sum() / (bp.ravel()[k2] ** 2).sum())
    rows.append(dict(q=q, L=L, pearson=pr, spearman=sp, pearson_drop_top3=pr2, slope_ell_on_beta_drop_top3=sl2, top3_share_of_sum_ell2=float((np.sort(nrm)[-3:] ** 2).sum() / (nrm ** 2).sum())))
P(); P("== robust compensation correlation (beta' vs ell) =="); P(pd.DataFrame(rows).round(3).to_string(index=False))
open(HERE + "/checks.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))

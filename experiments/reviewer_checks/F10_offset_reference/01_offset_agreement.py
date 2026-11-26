"""F10: the reference ranking is the window-mean offset of the bias error instead of bias RMSE; everything else follows the canonical H0 pipeline
(all 252 seed splits, BH pair selection, leave-one-sequence-out jackknife with t quantiles).
Reads the H0 lib, F8 stats_per_run.csv, attitude_controls and decomposition/f5_decomp_runs.csv; writes only this folder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); H0 = HERE + "/../H0_unified_stats"; sys.path.insert(0, H0)
from lib import *   # noqa
RB = R + "experiments/reviewer_checks/"
units, bias, iabe, seeds = load_mh(); cl = [u[1] for u in units]; G = groups_of(units)
canon = pd.read_csv(H0 + "/canonical_agreement.csv"); canon = canon[(canon.dataset == "Machine Hall") & (canon.selector == "bias RMSE")]

# ---- offset / fluctuation matrices from F8 stats_per_run (off2 = ||mean(bhat)-mean(bref)||^2/3 , fl2 = mean_t ||(bhat-mean)-(bref-mean)||^2/3)
S = pd.read_csv(RB + "F8_what_is_learned/stats_per_run.csv"); S = S[S.method.isin(DL)]
S["offset"] = np.sqrt(S.off2); S["fluct"] = np.sqrt(S.fl2)
chk = np.max(np.abs(np.sqrt(S.off2 + S.fl2) - S.rmse_avg) / S.rmse_avg); print("max rel |sqrt(off2+fl2)-rmse_avg|", chk); assert chk < 1e-9
def smat(col): return {u: S[(S.regime == u[0]) & (S.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
off, flu, rmse2 = smat("offset"), smat("fluct"), smat("rmse_avg")
for u in units:
    assert off[u].shape == (10, 11) and not np.isnan(off[u]).any()
    assert np.allclose(rmse2[u], bias[u], rtol=1e-9), u        # F8 rmse_avg equals bias_rmse_avg of metrics_wide exactly

# ---- measures (MH only)
att = pd.read_csv(R + "experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
v1 = pd.read_csv(R + "experiments/attitude_controls/vqf/ate_per_run_vqf.csv"); v1 = v1[v1.method.isin(DL)]
v2 = pd.read_csv(R + "experiments/attitude_controls/vqf/v2/ate_per_run_v2.csv"); v2 = v2[v2.method.isin(DL)]
FR = {"a_raw_gyro(paper)": "raw", "a2_raw_gyro_200Hz": "raw (200 Hz)", "b_gyro-GT-bias(oracle)": "oracle", "c_static-mean-bias": "static average",
 "d_mahony(kp=1,ki=0.05)": "Mahony", "e_mahony_gated(kp=1,ki=0.05)": "gated Mahony", "s_gyro-bias-comp-50%": "gyro bias comp 50%", "s_gyro-bias-comp-90%": "gyro bias comp 90%",
 "s_gyro-bias-comp-99%": "gyro bias comp 99%", "s_mahony(kp=0.2,ki=0)": "Mahony kp0.2 ki0", "s_mahony(kp=0.2,ki=0.05)": "Mahony kp0.2 ki0.05", "s_mahony(kp=1,ki=0)": "Mahony kp1 ki0",
 "s_mahony(kp=5,ki=0)": "Mahony kp5 ki0", "s_mahony(kp=5,ki=0.05)": "Mahony kp5 ki0.05", "f_vqf_6D(default)": "VQF cold, default clip 2", "g_basicvqf_6D(default)": "BasicVQF cold",
 "h_vqf_6D_GTprimed(post-hoc)": "VQF GT-primed", "f_vqf_6D(default,clip2,cold)": "VQF cold, default clip 2", "v_vqf_clip10(cold)": "VQF cold, clip 10", "i_mahony(warm)": "Mahony warm",
 "j_mahony_gated(warm)": "gated Mahony warm", "k_vqf_clip10(warm)": "VQF warm, clip 10", "l_vqf_default_clip2(warm,extra)": "VQF warm, clip 2"}
items = [("offset own retest (upper bound)", "not applicable", "offset", off), ("bias RMSE (as measure)", "not applicable", "bias_rmse", bias),
         ("fluctuation (as measure)", "not applicable", "fluct", flu), ("IABE", "reference attitude", "iabe_rot", iabe)]
seen = set()
for s in att.source.unique(): items.append(("ATE", FR[s], s, ate_from_long(att, units, seeds, s))); seen.add(s)
for df in (v1, v2):
    for s in df.source.unique():
        if s in seen or s == "f_vqf_6D(default,clip2,cold)": continue
        items.append(("ATE", FR[s], s, ate_from_long(df, units, seeds, s))); seen.add(s)
dec = pd.read_csv(RB + "decomposition/f5_decomp_runs.csv")
def dmat(src, col): return {u: dec[(dec.src == src) & (dec.regime == u[0]) & (dec.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
rotitems = []
for src, lab, sid in (("raw", "raw", "a_raw_gyro(paper)"), ("oracle", "oracle", "b_gyro-GT-bias(oracle)")):
    for v, nm in (("noL", "ATE without shared leakage term"), ("noQ", "ATE without rotation term Q"), ("negL", "ATE with leakage sign reversed")): items.append((nm, lab, sid, dmat(src, v)))
    rotitems.append(("ATE with leakage in random directions", lab, sid, [dmat(src, f"rot{i}") for i in range(30)]))

rows = []; UNIT = []
def emit(selname, measure, a_src, sid, mname, grp, a, n, gi):
    pt, lo, hi, g = jackknife(a, n, cl, gi)
    rows.append(dict(measure=measure, attitude_source=a_src, group=grp, mamba=mname, point=pt, lo=max(lo, 0.0), hi=min(hi, 1.0), n_pairs=n[gi].sum()/252, n_clusters=g,
                     selector=selname, dataset="Machine Hall", source_id=sid, interval_type="jackknife t (leave-one-sequence-out)", clipped=bool(lo < 0 or hi > 1), lo_raw=lo, hi_raw=hi, note=""))
for selname, SEL in (("offset", off), ("bias RMSE", bias)):      # the bias RMSE pass only serves as the reproduction check
    for mname, cols in MAMBA_SETS.items():
        ag = Agree(units, {u: SEL[u][:, cols] for u in units}, len(cols))
        for measure, a_src, sid, M in items:
            a, n = ag.counts({u: M[u][:, cols] for u in units})
            for grp, gi in G.items(): emit(selname, measure, a_src, sid, mname, grp, a, n, gi)
            if measure.startswith("offset own") or (selname == "offset" and measure == "ATE" and a_src == "raw"):
                for ui, u in enumerate(units): UNIT.append(dict(selector=selname, mamba=mname, measure=measure, attitude_source=a_src, unit="%s/%s" % u, agree=a[ui]/n[ui] if n[ui] else np.nan, n_pairs=n[ui]/252))
        for measure, a_src, sid, Ms in rotitems:
            res = [ag.counts({u: M[u][:, cols] for u in units}) for M in Ms]
            for grp, gi in G.items():
                pts = [jackknife(a, n, cl, gi)[0] for a, n in res]; n = res[0][1]
                rows.append(dict(measure=measure, attitude_source=a_src, group=grp, mamba=mname, point=float(np.median(pts)), lo=min(pts), hi=max(pts), n_pairs=n[gi].sum()/252, n_clusters=len(set(np.array(cl)[gi])),
                                 selector=selname, dataset="Machine Hall", source_id=sid, interval_type="range over 30 random rotations (point = median)", clipped=False, lo_raw=min(pts), hi_raw=max(pts), note="mean %.3f" % np.mean(pts)))
P = pd.DataFrame(rows)
# ---- reproduction check: the pass with the bias-RMSE selector must equal canonical_agreement.csv
B = P[P.selector == "bias RMSE"].merge(canon[["measure", "attitude_source", "group", "mamba", "point", "n_pairs"]], on=["measure", "attitude_source", "group", "mamba"], suffixes=("", "_canon"))
B["measure_c"] = B.measure
mm = B[(B.measure != "bias RMSE (as measure)") & (B.measure != "fluctuation (as measure)") & (~B.measure.str.startswith("offset"))]
d = (mm.point - mm.point_canon).abs().max(); dn = (mm.n_pairs - mm.n_pairs_canon).abs().max()
print("REPRO: rows compared", len(mm), "max |point - canonical|", d, "max |n_pairs diff|", dn)
r = P[(P.selector == "bias RMSE") & (P.measure == "ATE") & (P.attitude_source == "raw") & (P.group == "nine cases pooled") & (P.mamba == "included")].iloc[0]
print("REPRO raw ATE nine included (bias RMSE selector): %.4f [%.3f, %.3f]" % (r.point, r.lo, r.hi))
assert d < 1e-9
# ---- output: offset as the reference, with the values under the bias-RMSE reference attached
O = P[P.selector == "offset"].copy()
ref = canon.rename(columns={"point": "bias_rmse_ref_value", "lo": "bias_rmse_ref_lo", "hi": "bias_rmse_ref_hi", "n_pairs": "bias_rmse_ref_n_pairs"})[["measure", "attitude_source", "group", "mamba", "bias_rmse_ref_value", "bias_rmse_ref_lo", "bias_rmse_ref_hi", "bias_rmse_ref_n_pairs"]]
ref = ref.drop_duplicates(["measure", "attitude_source", "group", "mamba"])
# name mapping of bias-RMSE own retest and IABE in the canonical table
ref2 = ref.copy(); O = O.merge(ref2, on=["measure", "attitude_source", "group", "mamba"], how="left")
# new measures (offset / fluctuation / bias RMSE as measure) under the bias-RMSE reference: taken from the second pass of this program (same pipeline)
Bp = P[P.selector == "bias RMSE"][["measure", "attitude_source", "group", "mamba", "point", "lo", "hi", "n_pairs"]].rename(columns={"point": "v", "lo": "l", "hi": "h", "n_pairs": "n"})
O = O.merge(Bp, on=["measure", "attitude_source", "group", "mamba"], how="left")
for a, b in (("bias_rmse_ref_value", "v"), ("bias_rmse_ref_lo", "l"), ("bias_rmse_ref_hi", "h"), ("bias_rmse_ref_n_pairs", "n")): O[a] = O[a].fillna(O[b]); O = O.drop(columns=b)
O["bias_rmse_ref_source"] = np.where(O.measure.isin(["offset own retest (upper bound)", "bias RMSE (as measure)", "fluctuation (as measure)"]), "recomputed here (bias RMSE selector)", "canonical_agreement.csv")
O.insert(9, "old_value", "")
O["delta_vs_bias_rmse_ref"] = O.point - O.bias_rmse_ref_value
# the bias-RMSE own retest is called "bias RMSE own retest" in the canonical table; add it for comparison
cbf = canon[canon.measure == "bias RMSE own retest"].set_index(["group", "mamba"])
m = O.measure == "offset own retest (upper bound)"
O.loc[m, "note"] = "offset measured under bias-RMSE selector (F8 offset-only agreement) = " + O[m].apply(lambda r: "%.3f" % r.bias_rmse_ref_value, axis=1) + "; comparison value below is bias RMSE own retest"
for c_, k_ in (("bias_rmse_ref_value", "point"), ("bias_rmse_ref_lo", "lo"), ("bias_rmse_ref_hi", "hi"), ("bias_rmse_ref_n_pairs", "n_pairs")):
    O.loc[m, c_] = O[m].apply(lambda r: cbf.loc[(r.group, r.mamba), k_], axis=1)
O.loc[m, "bias_rmse_ref_source"] = "canonical_agreement.csv (bias RMSE own retest)"
O["delta_vs_bias_rmse_ref"] = O.point - O.bias_rmse_ref_value
O.drop(columns=["lo_raw", "hi_raw"]).to_csv(HERE + "/offset_reference_agreement.csv", index=False)
pd.DataFrame(UNIT).to_csv(HERE + "/per_unit_agreement.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for mam in ("included", "excluded"):
    for grp in ("nine cases pooled", "regime A", "regime B"):
        print("\n==", mam, grp); x = O[(O.group == grp) & (O.mamba == mam) & ~O.attitude_source.isin(["raw (200 Hz)"])]
        print(x[["measure", "attitude_source", "point", "lo", "hi", "n_pairs", "bias_rmse_ref_value", "bias_rmse_ref_n_pairs", "delta_vs_bias_rmse_ref"]].round(3).to_string(index=False))

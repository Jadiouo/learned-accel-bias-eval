"""H0 step 2: canonical direction-agreement table (new pipeline everywhere). Writes canonical_core.csv (ceiling rows added by 03/04)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib import *
units, bias, iabe, seeds = load_mh(); cl = [u[1] for u in units]; G = groups_of(units)
FRIENDLY = {
 "a_raw_gyro(paper)": "raw", "a2_raw_gyro_200Hz": "raw (200 Hz)", "b_gyro-GT-bias(oracle)": "oracle", "c_static-mean-bias": "static average",
 "d_mahony(kp=1,ki=0.05)": "Mahony", "e_mahony_gated(kp=1,ki=0.05)": "gated Mahony",
 "s_gyro-bias-comp-50%": "gyro bias comp 50%", "s_gyro-bias-comp-90%": "gyro bias comp 90%", "s_gyro-bias-comp-99%": "gyro bias comp 99%",
 "s_mahony(kp=0.2,ki=0)": "Mahony kp0.2 ki0", "s_mahony(kp=0.2,ki=0.05)": "Mahony kp0.2 ki0.05", "s_mahony(kp=1,ki=0)": "Mahony kp1 ki0",
 "s_mahony(kp=5,ki=0)": "Mahony kp5 ki0", "s_mahony(kp=5,ki=0.05)": "Mahony kp5 ki0.05",
 "f_vqf_6D(default)": "VQF cold, default clip 2", "g_basicvqf_6D(default)": "BasicVQF cold", "h_vqf_6D_GTprimed(post-hoc)": "VQF GT-primed",
 "f_vqf_6D(default,clip2,cold)": "VQF cold, default clip 2", "v_vqf_clip10(cold)": "VQF cold, clip 10", "i_mahony(warm)": "Mahony warm",
 "j_mahony_gated(warm)": "gated Mahony warm", "k_vqf_clip10(warm)": "VQF warm, clip 10", "l_vqf_default_clip2(warm,extra)": "VQF warm, clip 2"}
rows = []; CNT = {}
def emit(ds, measure, att, srcid, mamba, grp, a, n, gi, clusters, nspl=252, selector="bias RMSE", interval_type="jackknife t (leave-one-sequence-out)", note=""):
    pt, lo, hi, g = jackknife(a, n, clusters, gi)
    rows.append(dict(dataset=ds, selector=selector, measure=measure, attitude_source=att, source_id=srcid, group=grp, mamba=mamba, point=pt, lo=max(lo, 0.0), hi=min(hi, 1.0),
                     lo_raw=lo, hi_raw=hi, n_pairs=n[gi].sum()/nspl, n_clusters=g, interval_type=interval_type, note=note))

def run_dataset(ds, units, clusters, bias, groups, items, rotitems=(), iabe_sel=None, iabe_items=()):
    """items: list of (measure, attitude, source_id, dict_u->(10,11)); rotitems: list of (measure, attitude, source_id, [dict]*30)"""
    for mname, cols in MAMBA_SETS.items():
        ag = Agree(units, {u: bias[u][:, cols] for u in units}, len(cols))
        for measure, att, sid, M in items:
            a, n = ag.counts({u: M[u][:, cols] for u in units}); CNT[(ds, mname, measure, att)] = (a, n, clusters)
            for grp, gi in groups.items(): emit(ds, measure, att, sid, mname, grp, a, n, gi, clusters)
        if iabe_sel is not None:
            agi = Agree(units, {u: iabe_sel[u][:, cols] for u in units}, len(cols))
            for measure, att, sid, M in iabe_items:
                a, n = agi.counts({u: M[u][:, cols] for u in units})
                for grp, gi in groups.items(): emit(ds, measure, att, sid, mname, grp, a, n, gi, clusters, selector="IABE")
        for measure, att, sid, Ms in rotitems:
            res = [ag.counts({u: M[u][:, cols] for u in units}) for M in Ms]
            for grp, gi in groups.items():
                pts = [jackknife(a, n, clusters, gi)[0] for a, n in res]; n = res[0][1]
                rows.append(dict(dataset=ds, selector="bias RMSE", measure=measure, attitude_source=att, source_id=sid, group=grp, mamba=mname, point=float(np.median(pts)),
                                 lo=min(pts), hi=max(pts), lo_raw=min(pts), hi_raw=max(pts), n_pairs=n[gi].sum()/252, n_clusters=len(set(np.array(clusters)[gi])),
                                 interval_type="range over 30 random rotations (point = median)", note="mean %.3f" % np.mean(pts)))

# ================= Machine Hall
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
v1 = pd.read_csv(R+"experiments/attitude_controls/vqf/ate_per_run_vqf.csv"); v1 = v1[v1.method.isin(DL)]
v2 = pd.read_csv(R+"experiments/attitude_controls/vqf/v2/ate_per_run_v2.csv"); v2 = v2[v2.method.isin(DL)]
items = [("bias RMSE own retest", "not applicable", "bias_rmse", bias), ("IABE", "reference attitude", "iabe_rot", iabe)]
srcs = list(att.source.unique())
for s in srcs: items.append(("ATE", FRIENDLY[s], s, ate_from_long(att, units, seeds, s)))
# VQF sources: take from v1/v2 files, not already in gated csv; assert shared sources identical
for df, nm in ((v1, "vqf v1"), (v2, "vqf v2")):
    for s in df.source.unique():
        if s in srcs: 
            x = ate_from_long(df, units, seeds, s); y = ate_from_long(att, units, seeds, s)
            assert all(np.allclose(x[u], y[u], rtol=1e-9) for u in units), (nm, s); continue
        if s == "f_vqf_6D(default,clip2,cold)":   # same run as v1 'f_vqf_6D(default)': check equal, do not list twice
            x = ate_from_long(df, units, seeds, s); y = ate_from_long(v1, units, seeds, "f_vqf_6D(default)")
            assert all(np.allclose(x[u], y[u], rtol=1e-9) for u in units); continue
        if any(i[2] == s for i in items): 
            x = ate_from_long(df, units, seeds, s); y = [i[3] for i in items if i[2] == s][0]
            assert all(np.allclose(x[u], y[u], rtol=1e-9) for u in units), (nm, s); continue
        items.append(("ATE", FRIENDLY[s], s, ate_from_long(df, units, seeds, s)))
# alignment
al = pd.read_csv(R+"experiments/reviewer_checks/F2_aligned_ate/ate_aligned_per_run.csv"); al = al[al.method.isin(DL)]
for s in al.source.unique():
    for a_, lab in (("4dof", "4DoF"), ("se3", "SE(3)")):
        items.append((f"ATE aligned {lab}", FRIENDLY[s], s, ate_from_long(al, units, seeds, s, col="ate_"+a_)))
# decomposition
dec = pd.read_csv(R+"experiments/reviewer_checks/decomposition/f5_decomp_runs.csv")
def dmat(src, col): return {u: dec[(dec.src == src) & (dec.regime == u[0]) & (dec.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
rotitems = []
for src, lab in (("raw", "raw"), ("oracle", "oracle")):
    sid = {"raw": "a_raw_gyro(paper)", "oracle": "b_gyro-GT-bias(oracle)"}[src]
    for v, nm in (("noL", "ATE without shared leakage term"), ("noQ", "ATE without rotation term Q"), ("negL", "ATE with leakage sign reversed")):
        items.append((nm, lab, sid, dmat(src, v)))
    rotitems.append(("ATE with leakage in random directions", lab, sid, [dmat(src, f"rot{i}") for i in range(30)]))
# sanity: decomposed 'full' equals stored ATE
for src, sid in (("raw", "a_raw_gyro(paper)"), ("oracle", "b_gyro-GT-bias(oracle)")):
    x = dmat(src, "full"); y = ate_from_long(att, units, seeds, sid)
    print("decomp full vs stored ATE max rel diff", src, max(np.max(np.abs(x[u]-y[u])/y[u]) for u in units))
iitems = [("IABE own retest", "reference attitude", "iabe_rot", iabe), ("ATE", "raw", "a_raw_gyro(paper)", ate_from_long(att, units, seeds, "a_raw_gyro(paper)")),
          ("ATE", "oracle", "b_gyro-GT-bias(oracle)", ate_from_long(att, units, seeds, "b_gyro-GT-bias(oracle)"))]
run_dataset("Machine Hall", units, cl, bias, G, items, rotitems, iabe_sel=iabe, iabe_items=iitems)

# ================= Machine Hall, synthetic scaling of the oracle attitude error (F5 matrices; leakage-ratio-matched configs)
sc = pd.read_csv(R+"experiments/reviewer_checks/F5_scaled_attitude/ate_scaled_per_run.csv"); sc = sc[sc.method.isin(DL)].rename(columns={"config": "source"})
sitems = [("ATE, oracle attitude error rescaled to leakage ratio %s" % c.split("=")[1], "oracle, error rescaled (ratio target %s)" % c.split("=")[1], c, ate_from_long(sc, units, seeds, c))
          for c in [x for x in sc.source.unique() if x.startswith("RM|")]]
run_dataset("Machine Hall, synthetic scaling (F5)", units, cl, bias, G, sitems)

# ================= Vicon (regime A, six sequences)
t1 = pd.read_csv(R+"euroc_full/analysis_vicon/full_2026-10-23/t1_ate_per_run_gated.csv", comment="#"); t1 = t1[t1.method.isin(DL)]
vd = pd.read_csv(R+"experiments/reviewer_checks/F7_vicon_decomp/vicon_decomp_runs.csv")
VS = ["V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03"]
def vmat(df, col, **kw):
    return {s: df.pivot_table(index="seed", columns="method", values=col)[DL].values for s, df in ((s, kw["frame"][kw["frame"].seq == s]) for s in VS)}
vb = vmat(None, "bias_rmse", frame=vd[vd.src == "raw"]); vi = vmat(None, "iabe_rot", frame=vd[vd.src == "raw"])
vitems = [("bias RMSE own retest", "not applicable", "bias_rmse", vb), ("IABE", "reference attitude", "iabe_rot", vi)]
for s in t1.source.unique():
    vitems.append(("ATE", FRIENDLY[s], s, vmat(None, "ate", frame=t1[t1.source == s])))
vrot = []
for src, sid in (("raw", "a_raw_gyro(paper)"), ("oracle", "b_gyro-GT-bias(oracle)")):
    x = vmat(None, "full", frame=vd[vd.src == src]); y = vmat(None, "ate", frame=t1[t1.source == sid])
    print("vicon decomp full vs t1 rel diff", src, max(np.max(np.abs(x[s]-y[s])/y[s]) for s in VS))
    for v, nm in (("noL", "ATE without shared leakage term"), ("noQ", "ATE without rotation term Q"), ("negL", "ATE with leakage sign reversed")):
        vitems.append((nm, src, sid, vmat(None, v, frame=vd[vd.src == src])))
    vrot.append(("ATE with leakage in random directions", src, sid, [vmat(None, f"rot{i}", frame=vd[vd.src == src]) for i in range(30)]))
run_dataset("Vicon rooms", VS, VS, vb, {"six sequences (regime A)": list(range(6))}, vitems, vrot)
P = pd.DataFrame(rows); P.to_csv(HERE+"/canonical_core.csv", index=False)
# ---- paired differences (jackknife over the same sequence clusters)
MHg = G; Vg = {"six sequences (regime A)": list(range(6))}
SPEC = []
for att_ in ("raw", "oracle"):
    for m_ in ("ATE without shared leakage term", "ATE without rotation term Q", "ATE with leakage sign reversed"):
        SPEC.append(("Machine Hall", MHg, (m_, att_), ("ATE", att_)))
    SPEC.append(("Machine Hall", MHg, ("IABE", "reference attitude"), ("ATE", att_)))
    for lab in ("4DoF", "SE(3)"): SPEC.append(("Machine Hall", MHg, (f"ATE aligned {lab}", att_), ("ATE", att_)))
    for m_ in ("ATE without shared leakage term", "ATE without rotation term Q", "ATE with leakage sign reversed"):
        SPEC.append(("Vicon rooms", Vg, (m_, att_), ("ATE", att_)))
    SPEC.append(("Vicon rooms", Vg, ("IABE", "reference attitude"), ("ATE", att_)))
SPEC.append(("Machine Hall", MHg, ("ATE", "oracle"), ("ATE", "raw")))
SPEC.append(("Vicon rooms", Vg, ("ATE", "oracle"), ("ATE", "raw")))
for a_, b_ in (("VQF warm, clip 10", "Mahony warm"), ("VQF warm, clip 10", "gated Mahony warm"), ("VQF cold, clip 10", "Mahony"), ("VQF cold, clip 10", "gated Mahony"),
               ("Mahony warm", "Mahony"), ("gated Mahony warm", "gated Mahony"), ("VQF warm, clip 10", "VQF cold, clip 10")):
    SPEC.append(("Machine Hall", MHg, ("ATE", a_), ("ATE", b_)))
drows = []
for mname in MAMBA_SETS:
    for ds, grps, k1, k2 in SPEC:
        a1, n1, clu = CNT[(ds, mname, *k1)]; a2, n2, _ = CNT[(ds, mname, *k2)]
        for grp, gi in grps.items():
            d, lo, hi = jk_diff(a1, n1, a2, n2, clu, gi)
            drows.append(dict(dataset=ds, mamba=mname, group=grp, minuend="%s | %s" % k1, subtrahend="%s | %s" % k2, diff=d, lo=lo, hi=hi))
pd.DataFrame(drows).to_csv(HERE+"/differences.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(len(P)); print(P[(P.group == "nine cases pooled") & (P.mamba == "included")][["measure", "attitude_source", "point", "lo", "hi", "n_pairs"]].round(3).to_string())

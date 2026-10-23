# -*- coding: utf-8 -*-
"""v2 (POST-HOC, see PREREG_addendum.md): VQF with biasClip=10 deg/s, cold-start and warm-start protocols. Derived from ../run_vqf.py.
Phase 1: recompute the 5 existing sources with the same code as ../run_controls_gated.py and check against the
existing CSVs (sanity gate). Phase 2: add VQF, produce tables + R-F agreement. CPU only. Outputs go to this folder."""
import os, sys, json, itertools
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); VQD = os.path.dirname(HERE)
sys.path.append(VQD + "/_pkgs")   # vqf only; numpy/scipy come from the system (no numpy in _pkgs)
import numpy as np, pandas as pd, yaml
from scipy.spatial.transform import Rotation as Rot
from scipy import stats
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
ROOT = os.path.join(REPO, "euroc_full")
AC = os.path.dirname(VQD)
RF = os.path.join(REPO, "verification", "R-F_noncircular")
sys.path.insert(0, ROOT + "/src")
from euroc_full import data_euroc, metrics
cfg = yaml.safe_load(open(ROOT + "/configs/full.yaml"))
cfg["_data_dir_abs"] = os.environ.get("EUROC_DATA_DIR", os.path.join(REPO, "data", "euroc", "machine_hall"))
cfg["seq_folders"] = {"MH_01": "MH_01_easy", "MH_02": "MH_02_easy", "MH_03": "MH_03_medium",
                      "MH_04": "MH_04_difficult", "MH_05": "MH_05_difficult"}
SEQS = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]
STRIDE, FS, G = cfg["stride"], cfg["imu_freq"], 9.81
DT = STRIDE / FS
GATE_F, GATE_W = 0.5, 0.5

# ---- helpers copied verbatim from ../run_controls_gated.py (same numerics) ----
def integ_full(w, R0, dt):
    inc = Rot.from_rotvec(w[:-1] * dt).as_matrix()
    R = np.empty((len(w), 3, 3)); R[0] = R0
    for k in range(len(w) - 1): R[k + 1] = R[k] @ inc[k]
    return R
def mahony(w, f, R0, dt, kp, ki):
    R = R0.copy(); I = np.zeros(3); out = np.empty((len(w), 3, 3)); out[0] = R
    for k in range(len(w) - 1):
        fn = f[k] / (np.linalg.norm(f[k]) + 1e-12); e = np.cross(fn, R[2, :]); I += ki * e * dt
        R = R @ Rot.from_rotvec((w[k] + kp * e + I) * dt).as_matrix()
        if k % 200 == 0: R = Rot.from_matrix(R).as_matrix()
        out[k + 1] = R
    return out
def mahony_gated(w, f, R0, dt, kp, ki):
    R = R0.copy(); I = np.zeros(3); out = np.empty((len(w), 3, 3)); out[0] = R
    for k in range(len(w) - 1):
        nf = np.linalg.norm(f[k]); ok = abs(nf - G) < GATE_F and np.linalg.norm(w[k]) < GATE_W
        if ok:
            e = np.cross(f[k] / (nf + 1e-12), R[2, :]); I += ki * e * dt
        else: e = np.zeros(3)
        R = R @ Rot.from_rotvec((w[k] + kp * e + I) * dt).as_matrix()
        if k % 200 == 0: R = Rot.from_matrix(R).as_matrix()
        out[k + 1] = R
    return out
def static_gyro_bias(m, i_end, win_s=1.0, thr=0.03):
    w = m[["wx", "wy", "wz"]].to_numpy()[:i_end]; f = m[["ax", "ay", "az"]].to_numpy()[:i_end]
    n = int(win_s * FS); best = None
    for s in range(0, len(w) - n, n // 4):
        sd = w[s:s + n].std(0).max()
        if abs(np.linalg.norm(f[s:s + n], axis=1).mean() - G) < 0.3 and sd < thr and (best is None or sd < best[0]): best = (sd, s)
    if best is None: return None
    return w[best[1]:best[1] + n].mean(0)
def tilt_yaw_deg(Cg, C):
    q = Rot.from_matrix(np.einsum("nij,nkj->nik", Cg, C)).as_quat()
    tilt = 2 * np.arctan2(np.hypot(q[:, 0], q[:, 1]), np.hypot(q[:, 3], q[:, 2]))
    return np.degrees(tilt)
def rms3(p): return float(np.sqrt(np.mean(np.sum(p ** 2, 1))))
def dint(a): return np.cumsum(np.cumsum(a * DT, 0) * DT, 0)
def world(R, x): return np.einsum("nij,nj->ni", R, x)

# ---- VQF (new) ----
from vqf import VQF, BasicVQF
def vqf_att(w, f, R0, cls, fs=FS, prime=False, **kw):
    """GT-initialised: feed data expressed in the t0 world frame (so VQF's identity start == R0), then C = C' R0."""
    wp = w @ R0.T; fp = f @ R0.T          # (R0 x) for each row
    v = cls(1.0 / fs, **kw)
    if prime:   # POST-HOC variant: start the acc low-pass at the GT gravity direction (in the t0 frame: +z), so tilt is GT-initialised too
        st = v.state; c = v.coeffs
        st["accLpState"] = np.concatenate([VQF.filterInitialState(G * (i == 2), np.asarray(c["accLpB"], float), np.asarray(c["accLpA"], float)) for i in range(3)])
        st["lastAccLp"] = np.array([0.0, 0.0, G]); v.state = st
    out = v.updateBatch(np.ascontiguousarray(wp), np.ascontiguousarray(fp))
    q = out["quat6D"]; Cp = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return Cp @ R0

# ---------------- per-sequence precompute ----------------
BASE, SRC = {}, {}
FULL = {}
for seq in SEQS:
    d = data_euroc.prepare_regime_a(seq, cfg); m = d.merged
    tws = np.asarray(d.test_win_starts, np.int64); i0, i1 = tws[0], tws[-1]
    mi = metrics.build_metric_inputs(m, tws, np.zeros((len(tws), 3)), d.y_test, STRIDE, FS, G); BASE[seq] = mi
    seg = m.iloc[i0:i1 + 1]
    wf = seg[["wx", "wy", "wz"]].to_numpy(); ff = seg[["ax", "ay", "az"]].to_numpy(); bw = seg[["bwx", "bwy", "bwz"]].to_numpy()
    R0 = Rot.from_quat(seg[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()[0]; sel = tws - i0
    bstat = static_gyro_bias(m, i0)
    SRC[(seq, "a_raw_gyro(paper)")] = integ_full(mi.w_raw, mi.R_ref[0], DT)
    SRC[(seq, "b_gyro-GT-bias(oracle)")] = integ_full(wf - bw, R0, 1 / FS)[sel]
    SRC[(seq, "c_static-mean-bias")] = integ_full(wf - bstat, R0, 1 / FS)[sel]
    SRC[(seq, "d_mahony(kp=1,ki=0.05)")] = mahony(wf, ff, R0, 1 / FS, 1.0, 0.05)[sel]
    SRC[(seq, "e_mahony_gated(kp=1,ki=0.05)")] = mahony_gated(wf, ff, R0, 1 / FS, 1.0, 0.05)[sel]
    sF = m.iloc[0:i1 + 1]   # warm start: causal, from sequence start to END of eval window (no later data)
    wF = sF[["wx", "wy", "wz"]].to_numpy(); fF = sF[["ax", "ay", "az"]].to_numpy()
    R0F = Rot.from_quat(sF[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()[0]
    FULL[seq] = (wf, ff, R0, sel, wF, fF, R0F, i0)
OLD = ["a_raw_gyro(paper)", "b_gyro-GT-bias(oracle)", "c_static-mean-bias", "d_mahony(kp=1,ki=0.05)", "e_mahony_gated(kp=1,ki=0.05)"]

wide = pd.read_csv(ROOT + "/results/metrics_wide.csv")
wide["regime"] = wide.run_id.str.split("__").str[0]; wide["method"] = wide.run_id.str.split("__").str[1]
wide["seed"] = wide.run_id.str.extract(r"seed(\d+)").astype(int)
BH = {}
def load_bhat(run_id, seq):
    return np.load(f"{ROOT}/results/runs/{run_id}/series.npz")[f"b_hat_{seq}"]

def build(srcs):
    """leak/tilt table by (regime, seq, source) and per-run ATE for the given sources."""
    S = {}
    for (seq, src), Cg in SRC.items():
        if src not in srcs: continue
        mi = BASE[seq]; C = mi.R_ref
        S[(seq, src)] = dict(tilt=tilt_yaw_deg(Cg, C), PL=rms3(dint(world(Cg - C, mi.a_raw - mi.b_ref))))
    recs = []
    for _, r in wide.iterrows():
        seq = r.eval_sequence; mi = BASE[seq]
        if (r.run_id, seq) not in BH: BH[(r.run_id, seq)] = load_bhat(r.run_id, seq)
        bh = BH[(r.run_id, seq)]; assert len(bh) == len(mi); pr = dint(world(mi.R_ref, mi.a_raw - mi.b_ref) + mi.g_vec)
        eps = float(np.sqrt(np.mean(np.sum((mi.b_ref - bh) ** 2, 1))))
        for src in srcs:
            Cg = SRC[(seq, src)]
            recs.append(dict(run_id=r.run_id, regime=r.regime, method=r.method, seed=r.seed, seq=seq, source=src,
                             ate=rms3(dint(world(Cg, mi.a_raw - bh) + mi.g_vec) - pr), iabe_rot=r.iabe_rot, eps_rms=eps))
    ate = pd.DataFrame(recs); rows = []
    for reg in "AB":
        for src in srcs:
            for seq in SEQS:
                v = S[(seq, src)]; w_s = wide[(wide.eval_sequence == seq) & (wide.regime == reg)]
                e = ate[(ate.seq == seq) & (ate.source == src) & (ate.regime == reg)].eps_rms.median()
                rows.append(dict(regime=reg, source=src, seq=seq, theta_bar=v["tilt"].mean(), leak=v["PL"], iabe_med=w_s.iabe_rot.median(),
                                 g_theta_over_eps=G * np.radians(v["tilt"].mean()) / e))
    return pd.DataFrame(rows), ate

# ================= PHASE 1: sanity gate on the existing five sources =================
T_old, ate_old = build(OLD)
ref_t = pd.read_csv(AC + "/table_sources_by_regime_gated.csv"); ref_a = pd.read_csv(AC + "/ate_per_run_gated.csv")
mt = T_old.merge(ref_t, on=["regime", "source", "seq"], suffixes=("", "_ref"))
assert len(mt) == len(T_old) == 50, len(mt)
errs = {c: float((abs(mt[c] - mt[c + "_ref" if c != "leak" else "leak_ref"]) / abs(mt[c + "_ref" if c != "leak" else "leak_ref"])).max()) for c in ["theta_bar", "leak", "g_theta_over_eps"]}
ma = ate_old.merge(ref_a[["run_id", "seq", "source", "ate"]], on=["run_id", "seq", "source"], suffixes=("", "_ref"))
assert len(ma) == len(ate_old) == len(ref_a[ref_a.source.isin(OLD)]), (len(ma), len(ate_old))
errs["ate"] = float((abs(ma.ate - ma.ate_ref) / ma.ate_ref).max())
mah_theta = float(T_old[(T_old.source == "d_mahony(kp=1,ki=0.05)") & (T_old.regime == "A")].theta_bar.median())
print("SANITY table/ate max rel err:", errs, "Mahony median theta_bar", round(mah_theta, 3), flush=True)
assert all(v < 1e-6 for v in errs.values()) and abs(mah_theta - 2.22) < 0.01, "sanity FAILED"
sanity = dict(max_rel_err=errs, mahony_theta_bar_median=mah_theta)

# ================= PHASE 2: VQF biasClip=10 (cold + warm), Mahony warm =================
V_DEF = "f_vqf_6D(default,clip2,cold)"; V_C10 = "v_vqf_clip10(cold)"
M_COLD = "d_mahony(kp=1,ki=0.05)"; G_COLD = "e_mahony_gated(kp=1,ki=0.05)"
M_W = "i_mahony(warm)"; G_W = "j_mahony_gated(warm)"; V_W = "k_vqf_clip10(warm)"; V_WDEF = "l_vqf_default_clip2(warm,extra)"
for seq in SEQS:
    wf, ff, R0, sel, wF, fF, R0F, i0 = FULL[seq]
    SRC[(seq, V_DEF)] = vqf_att(wf, ff, R0, VQF)[sel]
    SRC[(seq, V_C10)] = vqf_att(wf, ff, R0, VQF, biasClip=10.0)[sel]
    selw = sel + i0
    SRC[(seq, M_W)] = mahony(wF, fF, R0F, 1 / FS, 1.0, 0.05)[selw]
    SRC[(seq, G_W)] = mahony_gated(wF, fF, R0F, 1 / FS, 1.0, 0.05)[selw]
    SRC[(seq, V_W)] = vqf_att(wF, fF, R0F, VQF, biasClip=10.0)[selw]
    SRC[(seq, V_WDEF)] = vqf_att(wF, fF, R0F, VQF)[selw]
    print(seq, "done", flush=True)
ALLS = OLD + [V_DEF, V_C10, M_W, G_W, V_W, V_WDEF]
# quick calibration check BEFORE anything is written: warm Mahony should be ~1.61 deg
_tm = {s_: np.mean([tilt_yaw_deg(SRC[(q, s_)], BASE[q].R_ref).mean() for q in SEQS]) for s_ in [M_COLD, M_W]}
_med = {s_: float(np.median([tilt_yaw_deg(SRC[(q, s_)], BASE[q].R_ref).mean() for q in SEQS])) for s_ in [M_COLD, M_W]}
print("CALIBRATION median theta_bar: cold Mahony %.3f (want 2.22), warm Mahony %.3f (want ~1.61)" % (_med[M_COLD], _med[M_W]), flush=True)
sanity["calib_cold_mahony"] = _med[M_COLD]; sanity["calib_warm_mahony"] = _med[M_W]
assert abs(_med[M_COLD] - 2.22) < 0.01 and abs(_med[M_W] - 1.61) < 0.02, "CALIBRATION FAILED - stop and report"
T, ate = build(ALLS)
T.to_csv(HERE + "/table_sources_by_regime_v2.csv", index=False); ate.to_csv(HERE + "/ate_per_run_v2.csv", index=False)
iabe_reg = {r: float(wide[wide.regime == r].iabe_rot.median()) for r in "AB"}
rows = []
for reg in "AB":
    for src in ALLS:
        s = T[(T.regime == reg) & (T.source == src)]
        rows.append(dict(regime=reg, source=src, theta_med=s.theta_bar.median(), theta_min=s.theta_bar.min(), theta_max=s.theta_bar.max(),
                         n_lower_than_cold_mahony=int((s.set_index("seq").theta_bar.values < T[(T.regime == reg) & (T.source == M_COLD)].set_index("seq").loc[s.seq].theta_bar.values).sum()),
                         n_lower_than_warm_mahony=int((s.set_index("seq").theta_bar.values < T[(T.regime == reg) & (T.source == M_W)].set_index("seq").loc[s.seq].theta_bar.values).sum()),
                         leak_med=s.leak.median(), leak_min=s.leak.min(), leak_max=s.leak.max(),
                         iabe_regime_med=iabe_reg[reg], leak_over_iabe=s.leak.median() / iabe_reg[reg],
                         gth_eps_med=s.g_theta_over_eps.median(), gth_eps_min=s.g_theta_over_eps.min(), gth_eps_max=s.g_theta_over_eps.max()))
SUM = pd.DataFrame(rows); SUM.to_csv(HERE + "/table_sources_regime_summary_v2.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(SUM.round(3).to_string(), flush=True)

# ================= R-F agreement (same definition as rF_noncircular_stratified.py) =================
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
w2 = wide.merge(pd.read_csv(ROOT + "/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id", suffixes=("_x", ""))
w2 = w2[w2.method.isin(DL)]; att = ate[ate.method.isin(DL)]
UNITS = [(r, s) for r in "AB" for s in sorted(w2[w2.regime == r].eval_sequence.unique())]
piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i] * n / (n - k)); out[i] = run
    return out
PAIRS = np.array(list(itertools.combinations(range(11), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
mats = {}; seeds = None
for reg, seq in UNITS:
    g = w2[(w2.regime == reg) & (w2.eval_sequence == seq)]
    b = piv(g, "bias_rmse_avg")
    if seeds is None: seeds = list(b.index)
    assert list(b.index) == seeds and b.shape == (10, 11)
    mats[(reg, seq, "bias_rmse")] = b.values; mats[(reg, seq, "IABE")] = piv(g, "iabe_rot").values
    for src, gs in att[(att.regime == reg) & (att.seq == seq)].groupby("source"):
        p = piv(gs, "ate"); assert list(p.index) == seeds; mats[(reg, seq, "ATE:" + src)] = p.values
others = ["IABE"] + ["ATE:" + s for s in ALLS]
def select(M, idx):
    X = M[idx]; D = X[:, PI] - X[:, PJ]
    p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
    return bh(p) < .05, D.mean(0)
ALL = np.arange(10); halves = [np.array(c) for c in itertools.combinations(range(10), 5)]
NS = 1000; rng = np.random.default_rng(0); hs = rng.integers(0, len(halves), NS)   # same stream as the original script
names = ["bias_rmse (self-retest ceiling)"] + others
A = np.zeros((NS, len(UNITS), len(names))); N = np.zeros((NS, len(UNITS)))
for si, h in enumerate(hs):
    h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
    for ui, u in enumerate(UNITS):
        mask, d1 = select(mats[(*u, "bias_rmse")], h1); N[si, ui] = mask.sum()
        if not mask.sum(): continue
        def dir2(k):
            mm = mats[(*u, k)][h2].mean(0); return np.sign(mm[PI] - mm[PJ])
        A[si, ui, 0] = (dir2("bias_rmse") == np.sign(d1))[mask].sum()
        for oi, k in enumerate(others): A[si, ui, 1 + oi] = (dir2(k) == np.sign(d1))[mask].sum()
uidx = {u: i for i, u in enumerate(UNITS)}
groups = {"A": [uidx[u] for u in UNITS if u[0] == "A"], "B": [uidx[u] for u in UNITS if u[0] == "B"]}
def pooled(ix, c): return A[:, ix, c].sum() / max(N[:, ix].sum(), 1)
# sanity 3: pooled rates of the 5 existing sources vs the existing R-F CSV
refP = pd.read_csv(RF + "/part2_splithalf_agreement_stratified.csv")
keymap = {"ATE:" + s: "ATE:" + s for s in OLD}
e3 = 0.0
for gname, gi in groups.items():
    for s in OLD:
        r_ref = refP[(refP.group == gname) & (refP.measure == "ATE:" + s)].pooled_rate.iloc[0]
        e3 = max(e3, abs(pooled(np.array(gi), names.index("ATE:" + s)) - r_ref))
print("SANITY R-F pooled-rate max abs diff vs existing CSV:", e3, flush=True)
assert e3 < 1e-9, "R-F sanity FAILED"
sanity["rF_pooled_rate_max_abs_diff"] = e3
brng = np.random.default_rng(1); NB = 4000
rows = []; VC = names.index("ATE:" + V_C10); VW = names.index("ATE:" + V_W)
for gname, gi in groups.items():
    gi = np.array(gi); ixs = [gi[brng.integers(0, len(gi), len(gi))] for _ in range(NB)]
    for ci, nm in enumerate(names):
        bs = np.array([pooled(ix, ci) for ix in ixs]); lo, hi = np.percentile(bs, [2.5, 97.5])
        def dd(Vi):
            if ci == Vi: return (0.0, 0.0, 0.0)
            dv = np.array([pooled(ix, Vi) - pooled(ix, ci) for ix in ixs]); return (pooled(gi, Vi) - pooled(gi, ci), *np.percentile(dv, [2.5, 97.5]))
        c = dd(VC); w_ = dd(VW)
        sr = A[:, gi, ci].sum(1) / np.maximum(N[:, gi].sum(1), 1)
        rows.append(dict(group=gname, n_units=len(gi), measure=nm, mean_pairs_per_split=N[:, gi].sum(1).mean(), pooled_rate=pooled(gi, ci),
                         seqboot_lo=lo, seqboot_hi=hi, split_p2_5=np.percentile(sr, 2.5), split_p97_5=np.percentile(sr, 97.5),
                         vqfcold_minus_this=c[0], vc_lo=c[1], vc_hi=c[2], vqfwarm_minus_this=w_[0], vw_lo=w_[1], vw_hi=w_[2]))
AG = pd.DataFrame(rows); AG.to_csv(HERE + "/agreement_v2.csv", index=False)
print(AG.round(3).to_string(), flush=True)
json.dump(sanity, open(HERE + "/sanity.json", "w"), indent=1)
print("vqf version", __import__("vqf").__file__)

# extra: warm minus cold Mahony (same filter, protocol effect) and warm VQF minus warm Mahony are already in the table above.

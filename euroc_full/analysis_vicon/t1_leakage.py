# -*- coding: utf-8 -*-
"""[Vicon port of experiments/attitude_controls/run_controls_gated.py; paths/seq list parameterised, plot dropped, outputs prefixed t1_]
[POST-HOC, outline stage] Same as run_controls.py + 5th source e_mahony_gated; outputs suffixed _gated.
Attitude-source controls for the gyro-ATE leakage claim (R2). CPU only, zero training.
Read-only use of euroc_full (windowing, MetricInputs). Outputs go to this folder."""
import os, sys, json, itertools
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, pandas as pd, yaml
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import spearmanr

import argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vcommon as vc
ap = vc.add_args(argparse.ArgumentParser()); A = ap.parse_args()
OUT = A.out_dir; os.makedirs(OUT, exist_ok=True)
from euroc_full import data_euroc, metrics
cfg = vc.load_cfg(A)                      # vicon.yaml (window/stride/split identical to MH full.yaml) + seq_folders
SEQS = list(cfg["sequences"])
STRIDE, FS, G = cfg["stride"], cfg["imu_freq"], 9.81
DT = STRIDE / FS

# ---------------- attitude generators (all return (N,3,3) body->world at full rate) -------------
def W(df, name, **kw):
    vc.write_csv(df, A, name, **kw)

def integ_full(w, R0, dt):
    inc = Rot.from_rotvec(w[:-1] * dt).as_matrix()
    R = np.empty((len(w), 3, 3)); R[0] = R0
    for k in range(len(w) - 1): R[k + 1] = R[k] @ inc[k]
    return R

def mahony(w, f, R0, dt, kp, ki, bias0=None):
    """Standard Mahony explicit complementary filter (accel tilt correction, no yaw aiding).
    R body->world; gravity direction in body from accel (specific force = +g up)."""
    R = R0.copy(); I = np.zeros(3); out = np.empty((len(w), 3, 3)); out[0] = R
    for k in range(len(w) - 1):
        fn = f[k] / (np.linalg.norm(f[k]) + 1e-12)
        v = R[2, :]                      # = R^T e_z expressed in body (third row of R)
        e = np.cross(fn, v)
        I += ki * e * dt
        wc = w[k] + kp * e + I
        R = R @ Rot.from_rotvec(wc * dt).as_matrix()
        # re-orthonormalise occasionally
        if k % 200 == 0: R = Rot.from_matrix(R).as_matrix()
        out[k + 1] = R
    return out

# Pre-fixed gate thresholds (NOT tuned on results): tilt correction only when
# | ||f|-g | < GATE_F (m/s^2) AND |w| < GATE_W (rad/s). Same kp, ki as (d); integral term also frozen when gated off.
GATE_F, GATE_W = 0.5, 0.5
def mahony_gated(w, f, R0, dt, kp, ki):
    R = R0.copy(); I = np.zeros(3); out = np.empty((len(w), 3, 3)); out[0] = R; ng = 0
    for k in range(len(w) - 1):
        nf = np.linalg.norm(f[k]); ok = abs(nf - G) < GATE_F and np.linalg.norm(w[k]) < GATE_W
        if ok:
            e = np.cross(f[k] / (nf + 1e-12), R[2, :]); I += ki * e * dt; ng += 1
        else: e = np.zeros(3)
        R = R @ Rot.from_rotvec((w[k] + kp * e + I) * dt).as_matrix()
        if k % 200 == 0: R = Rot.from_matrix(R).as_matrix()
        out[k + 1] = R
    mahony_gated.frac = ng / max(len(w) - 1, 1)
    return out

def static_gyro_bias(m, i_end, win_s=1.0, thr=0.03):
    """Causal practitioner estimate: mean gyro over the quietest 1 s window (max-axis gyro std < thr,
    |f| within 0.3 of g) found anywhere BEFORE the evaluation window starts (index i_end)."""
    w = m[["wx", "wy", "wz"]].to_numpy()[:i_end]; f = m[["ax", "ay", "az"]].to_numpy()[:i_end]
    n = int(win_s * FS); best = None
    for s in range(0, len(w) - n, n // 4):
        sd = w[s:s + n].std(0).max()
        if abs(np.linalg.norm(f[s:s + n], axis=1).mean() - G) < 0.3 and sd < thr and (best is None or sd < best[0]):
            best = (sd, s)
    if best is None: return None, -1.0, np.nan
    s = best[1]
    return w[s:s + n].mean(0), s / FS, best[0]

# ---------------- error angles ---------------------------------------------------------------
def tilt_yaw_deg(Cg, C):
    q = Rot.from_matrix(np.einsum("nij,nkj->nik", Cg, C)).as_quat()  # world-frame error Cg C^T, (x,y,z,w)
    tilt = 2 * np.arctan2(np.hypot(q[:, 0], q[:, 1]), np.hypot(q[:, 3], q[:, 2]))
    yaw = 2 * np.arctan2(q[:, 2], q[:, 3]); yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(tilt), np.degrees(yaw)

def rms3(p): return float(np.sqrt(np.mean(np.sum(p ** 2, 1))))
def dint(a): return np.cumsum(np.cumsum(a * DT, 0) * DT, 0)

def world(R, x): return np.einsum("nij,nj->ni", R, x)

# ---------------- per-sequence precompute ----------------------------------------------------
sources_meta = {}
SRC = {}          # (seq, src) -> dict(Cg strided, tilt, yaw, ...)
BASE = {}         # seq -> MetricInputs
MERGED = {}
for seq in SEQS:
    d = data_euroc.prepare_regime_a(seq, cfg)
    m = d.merged; MERGED[seq] = m
    tws = np.asarray(d.test_win_starts, np.int64)
    i0, i1 = tws[0], tws[-1]
    mi = metrics.build_metric_inputs(m, tws, np.zeros((len(tws), 3)), d.y_test, STRIDE, FS, G)
    BASE[seq] = mi
    seg = m.iloc[i0:i1 + 1]
    wf = seg[["wx", "wy", "wz"]].to_numpy(); ff = seg[["ax", "ay", "az"]].to_numpy()
    bw = seg[["bwx", "bwy", "bwz"]].to_numpy()
    Cf = Rot.from_quat(seg[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    R0 = Cf[0]; dtf = 1.0 / FS; sel = tws - i0
    bstat, tstat, sdstat = static_gyro_bias(m, i0)
    sources_meta[seq] = dict(static_win_start_s=tstat, static_win_std=sdstat, static_bias=None if bstat is None else bstat.tolist(),
                             gt_bias_mean=bw.mean(0).tolist(), gt_bias_at_start=m[["bwx","bwy","bwz"]].to_numpy()[:200].mean(0).tolist(),
                             n_test=len(tws), span_s=(i1 - i0) / FS)
    gens = {}
    gens["a_raw_gyro(paper)"] = integ_full(mi.w_raw, mi.R_ref[0], DT)       # exact paper control (strided)
    SRC[(seq, "a_raw_gyro(paper)")] = dict(Cg=gens["a_raw_gyro(paper)"])
    full = {}
    full["a2_raw_gyro_200Hz"] = integ_full(wf, R0, dtf)
    full["b_gyro-GT-bias(oracle)"] = integ_full(wf - bw, R0, dtf)
    if bstat is not None:
        full["c_static-mean-bias"] = integ_full(wf - bstat, R0, dtf)
    full["d_mahony(kp=1,ki=0.05)"] = mahony(wf, ff, R0, dtf, 1.0, 0.05)
    full["e_mahony_gated(kp=1,ki=0.05)"] = mahony_gated(wf, ff, R0, dtf, 1.0, 0.05)
    sources_meta[seq]["gate_open_frac"] = mahony_gated.frac
    # sensitivity variants
    for frac in (0.5, 0.9, 0.99):
        full[f"s_gyro-bias-comp-{int(frac*100)}%"] = integ_full(wf - frac * bw, R0, dtf)
    for kp, ki in itertools.product((0.2, 1.0, 5.0), (0.0, 0.05)):
        if (kp, ki) != (1.0, 0.05): full[f"s_mahony(kp={kp:g},ki={ki:g})"] = mahony(wf, ff, R0, dtf, kp, ki)
    for k, Rf in full.items(): SRC[(seq, k)] = dict(Cg=Rf[sel])
    print(seq, "n_test", len(tws), "span %.1fs" % ((i1 - i0) / FS), "static-window@%.1fs std %.4f" % (tstat, sdstat), flush=True)

SRC_NAMES = []
for (s, k) in SRC:
    if k not in SRC_NAMES: SRC_NAMES.append(k)
MAIN = ["a_raw_gyro(paper)", "a2_raw_gyro_200Hz", "b_gyro-GT-bias(oracle)", "c_static-mean-bias", "d_mahony(kp=1,ki=0.05)", "e_mahony_gated(kp=1,ki=0.05)"]

# ---------------- paper per-method outputs ---------------------------------------------------
wide, _idx = vc.load_wide(A.results_dir)   # regime/method/seed already attached

def load_bhat(run_id, seq):
    z = np.load(f"{A.results_dir}/runs/{run_id}/series.npz"); return z[f"b_hat_{seq}"]

# leakage + error angles per (seq, source)
rows = []
for (seq, src), v in SRC.items():
    mi = BASE[seq]; Cg = v["Cg"]; C = mi.R_ref
    th, ya = tilt_yaw_deg(Cg, C)
    L = world(Cg - C, mi.a_raw - mi.b_ref)
    PL = dint(L)
    v["tilt"], v["yaw"], v["PL"] = th, ya, rms3(PL)
    v["Cg_f"] = Cg
    rows.append(dict(seq=seq, source=src, tilt_end_deg=th[-1], tilt_mean_deg=th.mean(), tilt_max_deg=th.max(),
                     yaw_end_deg=ya[-1], yaw_abs_mean_deg=np.abs(ya).mean(), leak_PL_m=v["PL"]))
leak = pd.DataFrame(rows)

# per run ATE under each source
recs = []
for _, r in wide.iterrows():
    seq = r.eval_sequence; mi = BASE[seq]
    bh = load_bhat(r.run_id, seq)
    assert len(bh) == len(mi)
    p_ref_acc = world(mi.R_ref, mi.a_raw - mi.b_ref) + mi.g_vec
    for src in SRC_NAMES:
        if (seq, src) not in SRC: continue
        Cg = SRC[(seq, src)]["Cg"]
        pm = dint(world(Cg, mi.a_raw - bh) + mi.g_vec); pr = dint(p_ref_acc)
        recs.append(dict(run_id=r.run_id, regime=r.regime, method=r.method, seed=r.seed, seq=seq, source=src,
                         ate=rms3(pm - pr), iabe_rot=r.iabe_rot, bias_rmse=r.bias_rmse_avg, eps_rms=float(np.sqrt(np.mean(np.sum((mi.b_ref - bh) ** 2, 1))))))
        if src == "a_raw_gyro(paper)":
            recs[-1]["paper_gyro_ate"] = r.gyro_ate
ate = pd.DataFrame(recs)
chk = ate[ate.source == "a_raw_gyro(paper)"]
print("reproduction of paper gyro_ate: max rel err", float((abs(chk.ate - chk.paper_gyro_ate) / chk.paper_gyro_ate).max()), flush=True)
W(ate, "t1_ate_per_run_gated.csv", index=False)

# ---------------- table 1: per-source summary ------------------------------------------------
iabe_seq = wide.groupby(["regime", "eval_sequence"]).iabe_rot.median()
iabe_all = wide.iabe_rot
eps_all = ate[ate.source == "a_raw_gyro(paper)"].eps_rms
rows = []
for seq in SEQS:
    for src in SRC_NAMES:
        if (seq, src) not in SRC: continue
        v = SRC[(seq, src)]
        w_s = wide[wide.eval_sequence == seq]
        e_seq = ate[(ate.seq == seq) & (ate.source == src)].eps_rms.median()
        rows.append(dict(seq=seq, source=src, tilt_end_deg=v["tilt"][-1], tilt_mean_deg=v["tilt"].mean(),
                         yaw_end_deg=v["yaw"][-1], leak_m=v["PL"],
                         iabe_median=w_s.iabe_rot.median(), iabe_min=w_s.iabe_rot.min(), iabe_max=w_s.iabe_rot.max(),
                         leak_over_iabe_median=v["PL"] / w_s.iabe_rot.median(),
                         leak_over_iabe_lo=v["PL"] / w_s.iabe_rot.max(), leak_over_iabe_hi=v["PL"] / w_s.iabe_rot.min(),
                         eps_rms_median=e_seq, g_theta_over_eps=G * np.radians(v["tilt"].mean()) / e_seq))
T1 = pd.DataFrame(rows); W(T1, "t1_table_sources_by_seq_gated.csv", index=False)
agg = T1.groupby("source").agg(tilt_end_med=("tilt_end_deg", "median"), tilt_mean_med=("tilt_mean_deg", "median"),
        tilt_mean_min=("tilt_mean_deg", "min"), tilt_mean_max=("tilt_mean_deg", "max"),
        yaw_end_absmed=("yaw_end_deg", lambda x: np.median(np.abs(x))), leak_med=("leak_m", "median"),
        leak_min=("leak_m", "min"), leak_max=("leak_m", "max"),
        ratio_med=("leak_over_iabe_median", "median"), ratio_min=("leak_over_iabe_lo", "min"), ratio_max=("leak_over_iabe_hi", "max"),
        gth_eps_med=("g_theta_over_eps", "median"), gth_eps_min=("g_theta_over_eps", "min"), gth_eps_max=("g_theta_over_eps", "max"))
W(agg.loc[SRC_NAMES], "t1_table_sources_summary_gated.csv"); print(agg.loc[SRC_NAMES].round(3).T.to_string(), flush=True)

# ---------------- ranking consistency --------------------------------------------------------
def sp(a, b): return spearmanr(a, b)[0]
rk = []
for src in MAIN + [s for s in SRC_NAMES if s not in MAIN]:
    sub = ate[ate.source == src]
    for (reg, seq), g in sub.groupby(["regime", "seq"]):
        gm = g.groupby("method").agg(ate=("ate", "mean"), iabe=("iabe_rot", "mean"), eps=("eps_rms", "mean"), brm=("bias_rmse", "mean"))
        if len(gm) < 4: continue
        iab_range = gm.iabe.max() - gm.iabe.min(); ate_range = gm.ate.max() - gm.ate.min()
        # per-seed Spearman
        ps = [sp(x.ate, x.iabe_rot) for _, x in g.groupby("seed") if len(x) >= 4]
        rk.append(dict(source=src, regime=reg, seq=seq, n_methods=len(gm), spearman_seedmean=sp(gm.ate, gm.iabe),
                       spearman_perseed_median=float(np.nanmedian(ps)), iabe_range=iab_range, ate_range=ate_range,
                       ate_range_over_iabe_range=ate_range / iab_range, leak=SRC[(seq, src)]["PL"],
                       leak_over_iabe_range=SRC[(seq, src)]["PL"] / iab_range,
                       spearman_ate_vs_biasrmse=sp(gm.ate, gm.brm), ate_median=gm.ate.median(), spearman_ate_vs_eps=sp(gm.ate, gm.eps)))
RK = pd.DataFrame(rk); W(RK, "t1_ranking_by_seq_regime_gated.csv", index=False)
# pooled over everything (within-regime, across seq: within-seq-normalised ranks)
pool = []
for src in SRC_NAMES:
    sub = RK[RK.source == src]
    s_all = ate[ate.source == src]
    gm = s_all.groupby(["regime", "seq", "method"]).agg(ate=("ate", "mean"), iabe=("iabe_rot", "mean")).reset_index()
    pool.append(dict(source=src, spearman_median_over_cells=sub.spearman_seedmean.median(),
                     spearman_min=sub.spearman_seedmean.min(), spearman_max=sub.spearman_seedmean.max(),
                     frac_cells_rho_gt_0p8=(sub.spearman_seedmean > 0.8).mean(), n_cells=len(sub),
                     perseed_median_of_medians=sub.spearman_perseed_median.median(),
                     leak_over_iabe_range_median=sub.leak_over_iabe_range.median(),
                     ate_range_over_iabe_range_median=sub.ate_range_over_iabe_range.median(),
                     ate_range_over_iabe_range_min=sub.ate_range_over_iabe_range.min(),
                     ate_range_over_iabe_range_max=sub.ate_range_over_iabe_range.max(),
                     pooled_spearman=sp(gm.ate, gm.iabe)))
POOL = pd.DataFrame(pool); W(POOL, "t1_ranking_summary_gated.csv", index=False)
print(POOL.round(3).to_string(), flush=True)

# ---------------- dose-response plot ---------------------------------------------------------
fit = T1[T1.tilt_mean_deg > 0]
sl, ic = np.polyfit(np.log10(fit.tilt_mean_deg), np.log10(fit.leak_m), 1)   # dose-response slope (plot omitted)
json.dump(dict(tag=A.tag, results_dir=os.path.abspath(A.results_dir), slope=sl, intercept_log10=ic, n=len(fit), spearman=sp(fit.tilt_mean_deg, fit.leak_m), meta=sources_meta), open(OUT + "/t1_meta.json", "w"), indent=1)
print("slope", sl, "spearman", sp(fit.tilt_mean_deg, fit.leak_m))

# ---------------- regime A/B stratified tables (post-hoc, outline stage) -------------------------
rows = []
for reg in sorted(wide.regime.unique()):
    for src in SRC_NAMES:
        for seq in SEQS:
            if (seq, src) not in SRC: continue
            v = SRC[(seq, src)]; w_s = wide[(wide.eval_sequence == seq) & (wide.regime == reg)]
            e = ate[(ate.seq == seq) & (ate.source == src) & (ate.regime == reg)].eps_rms.median()
            rows.append(dict(regime=reg, source=src, seq=seq, theta_bar=v["tilt"].mean(), leak=v["PL"],
                             iabe_med=w_s.iabe_rot.median(), ratio=v["PL"] / w_s.iabe_rot.median(),
                             eps=e, g_theta_over_eps=G * np.radians(v["tilt"].mean()) / e))
S = pd.DataFrame(rows); W(S, "t1_table_sources_by_regime_gated.csv", index=False)
SA = S.groupby(["regime", "source"]).agg(n_seq=("seq", "count"), theta_med=("theta_bar", "median"), theta_min=("theta_bar", "min"), theta_max=("theta_bar", "max"),
        leak_med=("leak", "median"), leak_min=("leak", "min"), leak_max=("leak", "max"),
        ratio_med=("ratio", "median"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"),
        gth_eps_med=("g_theta_over_eps", "median"), gth_eps_min=("g_theta_over_eps", "min"), gth_eps_max=("g_theta_over_eps", "max")).reset_index()
W(SA, "t1_table_sources_regime_summary_gated.csv", index=False)
RR = RK.groupby(["source", "regime"]).agg(n_cells=("seq", "count"), rho_ate_iabe_med=("spearman_seedmean", "median"), rho_ate_iabe_min=("spearman_seedmean", "min"), rho_ate_iabe_max=("spearman_seedmean", "max"),
        rho_ate_bias_med=("spearman_ate_vs_biasrmse", "median"), rho_ate_bias_min=("spearman_ate_vs_biasrmse", "min"), rho_ate_bias_max=("spearman_ate_vs_biasrmse", "max")).reset_index()
W(RR, "t1_ranking_by_regime_gated.csv", index=False)
rall = RK.groupby("source").agg(rho_ate_iabe_med=("spearman_seedmean", "median"), rho_ate_bias_med=("spearman_ate_vs_biasrmse", "median"), rho_ate_bias_min=("spearman_ate_vs_biasrmse", "min"), rho_ate_bias_max=("spearman_ate_vs_biasrmse", "max"))
W(rall, "t1_ranking_bias_all_cells_gated.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(SA.round(3).to_string()); print(RR.round(3).to_string()); print(rall.round(3).to_string())
print("gate open frac", {k: round(v["gate_open_frac"], 3) for k, v in sources_meta.items()})

# ---------------- paper Table I definition (added in the Vicon port) -----------------------------
# leakage multiple = median over sequences of leakage / pooled median IABE of the regime (all ok runs of that regime);
# theta_bar and g*theta/||eps|| = medians over sequences (same as table_sources_regime_summary_gated.csv columns).
pooled = wide.groupby("regime").iabe_rot.median()
T1x = SA[["regime", "source", "n_seq", "theta_med", "leak_med", "gth_eps_med", "gth_eps_min", "gth_eps_max"]].copy()
T1x["pooled_iabe_median_m"] = T1x.regime.map(pooled)
T1x["leakage_multiple"] = T1x.leak_med / T1x.pooled_iabe_median_m
W(T1x[T1x.source.isin(MAIN)], "t1_TABLE_I_paper_definition.csv", index=False)
print(T1x[T1x.source.isin(MAIN)].round(2).to_string())

# -*- coding: utf-8 -*-
"""F2 helper: rebuild the paper's open-loop trajectories and align them.
Attitude generators / dint / world / rms3 / static_gyro_bias are COPIED from
experiments/attitude_controls/run_controls_gated.py (read-only reference; that script is not executed).
Data loading imports euroc_full (read-only)."""
import os, sys, itertools
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import numpy as np, yaml
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ROOT = REPO + "/euroc_full"
sys.path.insert(0, ROOT + "/src")
from euroc_full import data_euroc, metrics

cfg = yaml.safe_load(open(ROOT + "/configs/full.yaml"))
cfg["_data_dir_abs"] = os.environ.get("EUROC_DATA_DIR", os.path.join(REPO, "data", "euroc", "machine_hall"))
cfg["seq_folders"] = {"MH_01": "MH_01_easy", "MH_02": "MH_02_easy", "MH_03": "MH_03_medium",
                      "MH_04": "MH_04_difficult", "MH_05": "MH_05_difficult"}
SEQS = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]
STRIDE, FS, G = cfg["stride"], cfg["imu_freq"], 9.81
DT = STRIDE / FS

# ---- copied from run_controls_gated.py ----
def integ_full(w, R0, dt):
    inc = Rot.from_rotvec(w[:-1] * dt).as_matrix()
    R = np.empty((len(w), 3, 3)); R[0] = R0
    for k in range(len(w) - 1): R[k + 1] = R[k] @ inc[k]
    return R

def mahony(w, f, R0, dt, kp, ki):
    R = R0.copy(); I = np.zeros(3); out = np.empty((len(w), 3, 3)); out[0] = R
    for k in range(len(w) - 1):
        fn = f[k] / (np.linalg.norm(f[k]) + 1e-12)
        e = np.cross(fn, R[2, :]); I += ki * e * dt
        R = R @ Rot.from_rotvec((w[k] + kp * e + I) * dt).as_matrix()
        if k % 200 == 0: R = Rot.from_matrix(R).as_matrix()
        out[k + 1] = R
    return out

GATE_F, GATE_W = 0.5, 0.5
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
        if abs(np.linalg.norm(f[s:s + n], axis=1).mean() - G) < 0.3 and sd < thr and (best is None or sd < best[0]):
            best = (sd, s)
    if best is None: return None
    s = best[1]
    return w[s:s + n].mean(0)

def rms3(p): return float(np.sqrt(np.mean(np.sum(p ** 2, 1))))
def dint(a): return np.cumsum(np.cumsum(a * DT, 0) * DT, 0)
def world(R, x): return np.einsum("nij,nj->ni", R, x)
# ---- end copied ----

SRC_NAMES = ["a_raw_gyro(paper)", "a2_raw_gyro_200Hz", "b_gyro-GT-bias(oracle)", "c_static-mean-bias",
             "d_mahony(kp=1,ki=0.05)", "e_mahony_gated(kp=1,ki=0.05)"]

def load_seq(seq):
    """Return (MetricInputs, {source: Cg strided (N,3,3)})."""
    d = data_euroc.prepare_regime_a(seq, cfg)
    m = d.merged
    tws = np.asarray(d.test_win_starts, np.int64); i0, i1 = tws[0], tws[-1]
    mi = metrics.build_metric_inputs(m, tws, np.zeros((len(tws), 3)), d.y_test, STRIDE, FS, G)
    seg = m.iloc[i0:i1 + 1]
    wf = seg[["wx", "wy", "wz"]].to_numpy(); ff = seg[["ax", "ay", "az"]].to_numpy()
    bw = seg[["bwx", "bwy", "bwz"]].to_numpy()
    R0 = Rot.from_quat(seg[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()[0]
    dtf = 1.0 / FS; sel = tws - i0
    bstat = static_gyro_bias(m, i0)
    S = {}
    S["a_raw_gyro(paper)"] = integ_full(mi.w_raw, mi.R_ref[0], DT)
    S["a2_raw_gyro_200Hz"] = integ_full(wf, R0, dtf)[sel]
    S["b_gyro-GT-bias(oracle)"] = integ_full(wf - bw, R0, dtf)[sel]
    if bstat is not None: S["c_static-mean-bias"] = integ_full(wf - bstat, R0, dtf)[sel]
    S["d_mahony(kp=1,ki=0.05)"] = mahony(wf, ff, R0, dtf, 1.0, 0.05)[sel]
    S["e_mahony_gated(kp=1,ki=0.05)"] = mahony_gated(wf, ff, R0, dtf, 1.0, 0.05)[sel]
    return mi, S

# ---- alignment ----
def align_4dof(P, Q):
    """Move P onto Q with translation + yaw (rotation about z). P,Q:(N,3)."""
    mp, mq = P.mean(0), Q.mean(0); Pc, Qc = P - mp, Q - mq
    th = np.arctan2(np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0]),
                    np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1]))
    c, s = np.cos(th), np.sin(th); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Pc @ R.T + mq

def align_se3(P, Q):
    """Umeyama without scale: Q ~ R P + t."""
    mp, mq = P.mean(0), Q.mean(0); Pc, Qc = P - mp, Q - mq
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)          # cov of (Q,P)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return Pc @ R.T + mq

ALIGN = {"none": lambda P, Q: P, "4dof": align_4dof, "se3": align_se3}

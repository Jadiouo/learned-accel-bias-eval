# -*- coding: utf-8 -*-
"""F9 helpers. Read-only imports of F2 (trajectory rebuild), F5 (scaled attitude), H0 (agreement/jackknife),
and the VQF package folder of experiments/attitude_controls/vqf/_pkgs. Writes nothing."""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(HERE, "..")
for p in ("F2_aligned_ate", "F5_scaled_attitude", "H0_unified_stats"):
    sys.path.insert(0, os.path.join(RB, p))
sys.path.append(os.path.join(HERE, "..", "..", "attitude_controls", "vqf", "_pkgs"))   # optional local vqf folder; otherwise `pip install vqf` (imported lazily in vqf_att)
import numpy as np, pandas as pd, itertools
from scipy.spatial.transform import Rotation as Rot
import f2_lib as L          # data loader, integ_full, mahony, dint, world, rms3
import f5_lib as F5         # scaled_att
import lib as H0            # DL, jackknife, Agree, load_mh, ate_from_long ...

G = L.G; DT = L.DT; FS = L.FS; ROOT = L.ROOT
DL = H0.DL
SEQS = L.SEQS
SQ3 = np.sqrt(3.0)

NAMES = {   # internal id -> (csv source id or None, short label)
    "raw": ("a_raw_gyro(paper)", "raw"),
    "oracle": ("b_gyro-GT-bias(oracle)", "oracle"),
    "static": ("c_static-mean-bias", "static avg"),
    "mahony": ("d_mahony(kp=1,ki=0.05)", "Mahony"),
    "gmahony": ("e_mahony_gated(kp=1,ki=0.05)", "gated Mahony"),
    "vqf": ("v_vqf_clip10(cold)", "VQF"),   # bias clip 10 deg/s, cold start (attitude_controls/vqf/v2)
    "comp50": ("s_gyro-bias-comp-50%", "gyro comp 50%"),
    "comp90": ("s_gyro-bias-comp-90%", "gyro comp 90%"),
    "comp99": ("s_gyro-bias-comp-99%", "gyro comp 99%"),
    "orc30": (None, "oracle err x0.3 (synthetic)"),
    "orc10": (None, "oracle err x0.1 (synthetic)"),
    "orc03": (None, "oracle err x0.03 (synthetic)"),
}

def vqf_att(w, f, R0, biasClip=10.0):   # as experiments/attitude_controls/vqf/v2/run_v2.py: VQF defaults except biasClip=10 deg/s, cold start, GT-initialised
    from vqf import VQF
    wp = w @ R0.T; fp = f @ R0.T
    v = VQF(1.0 / FS, biasClip=biasClip)
    out = v.updateBatch(np.ascontiguousarray(wp), np.ascontiguousarray(fp))
    q = out["quat6D"]; Cp = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return Cp @ R0

def load_seq_sources(seq):
    """(MetricInputs, {source id: Cg (N,3,3)}) for all F9 sources. Same construction as f2_lib.load_seq / run_controls_gated.py."""
    d = L.data_euroc.prepare_regime_a(seq, L.cfg); m = d.merged
    tws = np.asarray(d.test_win_starts, np.int64); i0, i1 = tws[0], tws[-1]
    mi = L.metrics.build_metric_inputs(m, tws, np.zeros((len(tws), 3)), d.y_test, L.STRIDE, FS, G)
    seg = m.iloc[i0:i1 + 1]
    wf = seg[["wx", "wy", "wz"]].to_numpy(); ff = seg[["ax", "ay", "az"]].to_numpy(); bw = seg[["bwx", "bwy", "bwz"]].to_numpy()
    R0 = Rot.from_quat(seg[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()[0]
    dtf = 1.0 / FS; sel = tws - i0
    bstat = L.static_gyro_bias(m, i0)
    S = {}
    S["raw"] = L.integ_full(mi.w_raw, mi.R_ref[0], DT)
    S["oracle"] = L.integ_full(wf - bw, R0, dtf)[sel]
    S["static"] = L.integ_full(wf - bstat, R0, dtf)[sel]
    S["mahony"] = L.mahony(wf, ff, R0, dtf, 1.0, 0.05)[sel]
    S["gmahony"] = L.mahony_gated(wf, ff, R0, dtf, 1.0, 0.05)[sel]
    S["vqf"] = vqf_att(wf, ff, R0)[sel]
    for frac in (0.5, 0.9, 0.99):
        S[f"comp{int(frac*100)}"] = L.integ_full(wf - frac * bw, R0, dtf)[sel]
    for key, s in (("orc30", 0.3), ("orc10", 0.1), ("orc03", 0.03)):
        S[key] = F5.scaled_att(S["oracle"], mi.R_ref, s, "W")
    return mi, S

def tilt_mean_deg(Cg, C):
    th, _ = F5.tilt_yaw_deg(Cg, C); return float(th.mean())

# ---- closed-form open-loop quantities under a constant offset delta (eps = eps_real + delta), vectorised over leading dims ----
def quad(Y2, vY, H, dl):
    """ATE or IABE value: sqrt(Y2 + 2 dl.vY + dl'H dl).  dl (...,3)."""
    q = Y2 + 2 * np.einsum("...i,...i->...", dl, vY) + np.einsum("...i,ij,...j->...", dl, H, dl)
    return np.sqrt(np.maximum(q, 0.0))

def rmse_avg(m2, m1, dl):
    """bias RMSE (paper 'avg': rms over time and axes) of eps_real + dl."""
    return np.sqrt(np.maximum(m2 + 2 * np.einsum("...i,...i->...", dl, m1) + np.einsum("...i,...i->...", dl, dl), 0.0) / 3.0)

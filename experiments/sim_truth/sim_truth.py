# -*- coding: utf-8 -*-
"""有真值的模擬核心。定義見 PREREG.md。只用 numpy/scipy，CPU。
積分慣例與 euroc_full/src/euroc_full/metrics.py 相同：v += a*dt; p += v*dt（先更新 v）。
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import os
DATA = os.path.join(os.environ.get("EUROC_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "euroc", "machine_hall")), "{}", "mav0", "state_groundtruth_estimate0", "data.csv")
G = 9.81
GVEC = np.array([0.0, 0.0, -G])
FS = 100.0
DT = 1.0 / FS
TYPES = ["const", "lowfreq", "rw", "highfreq", "motion", "learned"]


# ---------------------------------------------------------------- trajectory
def load_traj(name, T=100.0):
    d = np.loadtxt(DATA.format(name), delimiter=",", skiprows=1)
    t = (d[:, 0] - d[0, 0]) * 1e-9
    m = t <= T + 0.1
    t, p, q, v = t[m], d[m, 1:4], d[m, 4:8], d[m, 8:11]
    ps = savgol_filter(p, 41, 4, deriv=0, axis=0)
    a = savgol_filter(p, 41, 4, deriv=2, delta=0.005, axis=0)
    Rm = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    sl = slice(0, None, 2)  # 200 Hz -> 100 Hz
    Rm, a, ps, v = Rm[sl], a[sl], ps[sl], v[sl]
    N = int(T * FS)
    Rm, a, ps, v = Rm[:N], a[:N], ps[:N], v[:N]
    # 用姿態差分求 gyro，使 R_{k+1} = R_k Exp(w_k dt) 精確成立
    rel = np.einsum("kji,kjl->kil", Rm[:-1], Rm[1:])
    w = Rotation.from_matrix(rel).as_rotvec() / DT
    w = np.vstack([w, w[-1:]])
    f = np.einsum("kji,kj->ki", Rm, a - GVEC)  # f = R^T (a - g)
    return dict(R=Rm, a=a, f=f, w=w, v0=v[0].copy(), p0=ps[0].copy(), N=N, name=name)


def dint(acc, v0, p0):
    """acc (...,N,3) -> pos (...,N,3)，metrics._double_integrate 同式。"""
    vel = v0 + np.cumsum(acc * DT, axis=-2)
    return p0 + np.cumsum(vel * DT, axis=-2)


def rms3(x):  # x (..., N, 3) -> (...)
    return np.sqrt(np.mean(np.sum(x ** 2, axis=-1), axis=-1))


def so3_exp(w):  # (...,3) -> (...,3,3)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = w / np.where(th < 1e-12, 1.0, th)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = np.sin(th)[..., None]
    c = (1 - np.cos(th))[..., None]
    return np.eye(3) + s * K + c * (K @ K)


# ---------------------------------------------------------------- estimators
def _norm_rms(e, a):
    r = np.sqrt(np.mean(np.sum(e ** 2, axis=-1)))
    return e * (a / max(r, 1e-30))


def make_eps(kind, a, tr, rng):
    N = tr["N"]
    tt = np.arange(N) * DT
    T = N * DT
    if kind == "const":
        u = rng.normal(size=3)
        return np.tile(u / np.linalg.norm(u) * a, (N, 1))
    if kind == "lowfreq":
        e = np.zeros((N, 3))
        for ax in range(3):
            for _ in range(3):
                per = rng.uniform(T / 4, T)
                e[:, ax] += rng.normal() * np.sin(2 * np.pi * tt / per + rng.uniform(0, 2 * np.pi))
        return _norm_rms(e, a)
    if kind == "rw":
        return _norm_rms(np.cumsum(rng.normal(size=(N, 3)), axis=0), a)
    if kind == "highfreq":
        return _norm_rms(rng.normal(size=(N, 3)), a)
    if kind == "motion":
        D = np.diag(rng.normal(size=3))
        return _norm_rms(tr["f"] @ D.T, a)
    if kind == "learned":  # 0.9 const + 0.3 lowfreq + 0.3 white（EuRoC 學習法：序列 offset 為主）
        return _norm_rms(0.9 * make_eps("const", 1.0, tr, rng) + 0.3 * make_eps("lowfreq", 1.0, tr, rng)
                         + 0.3 * make_eps("highfreq", 1.0, tr, rng), a)
    raise ValueError(kind)


def make_pool(tr, rng, n_per_type=15, n_groups=12):
    """回傳 eps (P,N,3)、type idx、amp、group id（混合池 group=-1；匹配 RMS 組 group>=0）。"""
    eps, typ, amp, grp = [], [], [], []
    for ti, k in enumerate(TYPES):
        for _ in range(n_per_type):
            a = 10 ** rng.uniform(-3, np.log10(5e-2))
            eps.append(make_eps(k, a, tr, rng)); typ.append(ti); amp.append(a); grp.append(-1)
    for g in range(n_groups):
        a = 10 ** rng.uniform(-3, np.log10(5e-2))
        for ti, k in enumerate(TYPES):
            eps.append(make_eps(k, a, tr, rng)); typ.append(ti); amp.append(a); grp.append(g)
    return np.array(eps), np.array(typ), np.array(amp), np.array(grp)


def iabe_rot(eps, R):
    """與 metrics.iabe_rot 同式（兩臂差 = ∬ R eps）。eps (P,N,3)。"""
    e_w = np.einsum("nij,pnj->pni", R, eps)
    return rms3(dint(e_w, 0.0, 0.0))


def bias_rmse(eps):
    return np.sqrt(np.mean(np.sum(eps ** 2, axis=-1), axis=-1))


# ---------------------------------------------------------------- attitude error models
def inject_theta(N, sig_deg, rng):
    """Mode I：常數偏移 + OU(tau=20s)，3D RMS = sig_deg。"""
    s_ax = np.deg2rad(sig_deg) / np.sqrt(3) * np.sqrt(0.5)
    c = rng.normal(size=3) * s_ax
    ou = np.zeros((N, 3)); tau = 20.0; al = np.exp(-DT / tau)
    x = rng.normal(size=3) * s_ax
    z = rng.normal(size=(N, 3)) * s_ax * np.sqrt(1 - al ** 2)
    for k in range(N):
        x = al * x + z[k]; ou[k] = x
    return c + ou


def run_mode_I(tr, eps, sig_deg, rng):
    """回傳 truthB 的單一實現平方均值 / ATE 的單一實現值，向量化於估測器。"""
    R, f, N = tr["R"], tr["f"], tr["N"]
    th = inject_theta(N, sig_deg, rng)
    Rh = np.einsum("nij,njk->nik", so3_exp(th), R)
    na = rng.normal(size=(N, 3)) * 0.02
    A = dint(np.einsum("nij,nj->ni", Rh - R, f) + np.einsum("nij,nj->ni", Rh, na), 0.0, 0.0)  # 純姿態+雜訊誤差
    L = dint(np.einsum("nij,pnj->pni", Rh, eps), 0.0, 0.0)
    ate = rms3(A[None] + L)
    d2 = np.mean(np.sum(L ** 2, axis=-1), axis=-1)  # |phat_k - phat_0|^2 的時間平均
    att = np.sqrt(np.mean(np.sum(th ** 2, axis=-1)))
    return d2, ate, np.rad2deg(att), rms3(A)


def run_mode_M(tr, eps, s_g, rng, kp=0.5, ki=0.01):
    """Mode M：Mahony（含 gyro bias 積分項＝吸收效應）。eps (P,N,3)；index 0 為完美 bias 的參考臂。"""
    R, f, w, N = tr["R"], tr["f"], tr["w"], tr["N"]
    P = eps.shape[0] + 1
    E = np.concatenate([np.zeros((1, N, 3)), eps], axis=0)
    ba = rng.normal(size=3) * 0.05  # 真 accel bias（常數 + RW）
    ba_t = ba + np.cumsum(rng.normal(size=(N, 3)) * 3e-4 * np.sqrt(DT), axis=0)
    bg = rng.normal(size=3) * s_g
    na = rng.normal(size=(N, 3)) * 0.02
    ng = rng.normal(size=(N, 3)) * 1.7e-3
    fm = f + ba_t + na
    wm = w + bg + ng
    # 參考真軌跡（同積分式）
    a_true = np.einsum("nij,nj->ni", R, f) + GVEC
    p_ref = dint(a_true, tr["v0"], tr["p0"])
    Rh = np.tile(R[0], (P, 1, 1)); bge = np.zeros((P, 3))
    v = np.tile(tr["v0"], (P, 1)); p = np.tile(tr["p0"], (P, 1))
    pos = np.empty((P, N, 3)); ang = np.empty(N)
    ez = np.array([0.0, 0.0, 1.0])
    for k in range(N):
        tr_ = np.trace(R[k].T @ Rh[0])
        ang[k] = np.arccos(np.clip((tr_ - 1) / 2, -1, 1))
        bhat = ba_t[k] - E[:, k]                 # (P,3) = b_ref - eps
        fc = fm[k] - bhat                        # 補償後比力
        a_w = np.einsum("pij,pj->pi", Rh, fc) + GVEC
        v = v + a_w * DT
        p = p + v * DT
        pos[:, k] = p
        m = fc / np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-9)
        u = np.einsum("pji,j->pi", Rh, ez)       # R^T ez
        e = np.cross(m, u)
        wc = wm[k] - bge + kp * e
        bge = bge - ki * e * DT
        Rh = Rh @ so3_exp(wc * DT)
    d2 = np.mean(np.sum((pos[1:] - pos[:1]) ** 2, axis=-1), axis=-1)
    ate = rms3(pos[1:] - p_ref[None])
    return d2, ate, np.rad2deg(np.sqrt(np.mean(ang ** 2))), rms3(pos[0] - p_ref)

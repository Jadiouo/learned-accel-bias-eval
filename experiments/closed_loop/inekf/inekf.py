# -*- coding: utf-8 -*-
"""Right-invariant EKF on SE_2(3) (R, v, p) with separate biases (bg, ba). CPU/numpy only. See PREREG.md.

State X = [[R v p],[0 1 0],[0 0 1]] (world frame), biases bg, ba. Right-invariant error eta = Xhat X^-1, xi = log(eta)
= [xi_theta, xi_v, xi_p] (world frame); bias errors xi_bg = bg_hat - bg, xi_ba = ba_hat - ba.
Error-state ordering: [theta, v, p, bg, ba]  (NOTE: differs from eskf.py's [p, v, theta, bg, ba]).
Nominal propagation is the same discrete scheme as eskf.run_aided, so the filter-independent outage code is reused.
API mirrors eskf.py: init_state / run_aided (single), plus run_aided_batch (leading batch dim B on every array).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import eskf
from eskf import FS, DT, GV, G

_I3 = np.eye(3)


def skew_b(a):
    a = np.asarray(a, float); z = np.zeros(a.shape[:-1])
    return np.stack([np.stack([z, -a[..., 2], a[..., 1]], -1),
                     np.stack([a[..., 2], z, -a[..., 0]], -1),
                     np.stack([-a[..., 1], a[..., 0], z], -1)], -2)


def exp_so3_b(phi):
    phi = np.asarray(phi, float)
    th2 = np.einsum('...i,...i->...', phi, phi); th = np.sqrt(th2); small = th < 1e-6
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1 - th2 / 6, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th2 / 24, (1 - np.cos(ths)) / ths ** 2)
    K = skew_b(phi)
    return _I3 + a[..., None, None] * K + b[..., None, None] * (K @ K)


def jl_so3(phi):
    """Left Jacobian of SO(3)."""
    phi = np.asarray(phi, float)
    th2 = np.einsum('...i,...i->...', phi, phi); th = np.sqrt(th2); small = th < 1e-6
    ths = np.where(small, 1.0, th)
    a = np.where(small, 0.5 - th2 / 24, (1 - np.cos(ths)) / ths ** 2)
    b = np.where(small, 1 / 6 - th2 / 120, (ths - np.sin(ths)) / ths ** 3)
    K = skew_b(phi)
    return _I3 + a[..., None, None] * K + b[..., None, None] * (K @ K)


def log_so3(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1, 1); th = np.arccos(c)
    ax = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if th < 1e-8: return 0.5 * ax
    return th / (2 * np.sin(th)) * ax


def exp_se23_apply(xi, R, v, p):
    """X <- Exp_SE23(xi) X, xi = [phi, nu, rho] (..., 9). Batched over leading dims."""
    phi, nu, rho = xi[..., 0:3], xi[..., 3:6], xi[..., 6:9]
    Re = exp_so3_b(phi); J = jl_so3(phi)
    return Re @ R, np.einsum('...ij,...j->...i', Re, v) + np.einsum('...ij,...j->...i', J, nu), \
        np.einsum('...ij,...j->...i', Re, p) + np.einsum('...ij,...j->...i', J, rho)


def T_e2i(R, v, p):
    """Linear map from ESKF error coords [dp, dv, dtheta_body, dbg, dba] to InEKF coords [xi_theta, xi_v, xi_p, xi_bg, xi_ba]."""
    sk = lambda a: skew_b(a)
    T = np.zeros((15, 15))
    T[0:3, 6:9] = R
    T[3:6, 3:6] = _I3; T[3:6, 6:9] = sk(v) @ R
    T[6:9, 0:3] = _I3; T[6:9, 6:9] = sk(p) @ R
    T[9:12, 9:12] = _I3; T[12:15, 12:15] = _I3
    return T


def cov_e2i(Pe, R, v, p):
    T = T_e2i(R, v, p); return T @ Pe @ T.T


def cov_i2e(Pi, R, v, p):
    Ti = np.linalg.inv(T_e2i(R, v, p)); return Ti @ Pi @ Ti.T


def A_cont(R, v, p):
    """Continuous error-state matrix (InEKF coords) at nominal (R, v, p)."""
    A = np.zeros((15, 15))
    A[0:3, 9:12] = -R
    A[3:6, 0:3] = skew_b(GV); A[3:6, 9:12] = -skew_b(v) @ R; A[3:6, 12:15] = -R
    A[6:9, 3:6] = _I3; A[6:9, 9:12] = -skew_b(p) @ R
    return A


def G_noise(R, v, p):
    """Noise input matrix (15x12) for [n_g, n_a, n_bg, n_ba]."""
    Gm = np.zeros((15, 12))
    Gm[0:3, 0:3] = R
    Gm[3:6, 0:3] = skew_b(v) @ R; Gm[3:6, 3:6] = R
    Gm[6:9, 0:3] = skew_b(p) @ R
    Gm[9:12, 6:9] = _I3; Gm[12:15, 9:12] = _I3
    return Gm


def init_state(R0_gt, p0, v0, bg_gt0, rep_noise, att_scale):
    """Same nominal init and same sigma as eskf.init_state; P re-expressed in InEKF coordinates (first-order equivalent)."""
    S = eskf.init_state(R0_gt, p0, v0, bg_gt0, rep_noise, att_scale)
    S["P"] = cov_e2i(S["P"], S["R"], S["v"], S["p"])
    return S


def _copy(Sb):
    return {k: x.copy() for k, x in Sb.items()}


def run_aided_batch(S, f, w, p_gt, k0, k1, meas_noise, sig_meas, Q, bofs=None, snaps=(), dt=DT, update_every=FS):
    """Batched filter. S arrays: R (B,3,3), v,p,bg,ba (B,3), P (B,15,15). bofs: None or (B,N,3) known bias added to ba.
    Position update at every frame index multiple of update_every (index>0): z = p_gt + meas_noise[idx//FS]*sig."""
    R, v, p, bg, ba, P = (S[k] for k in ("R", "v", "p", "bg", "ba", "P"))
    R, v, p, bg, ba, P = R.copy(), v.copy(), p.copy(), bg.copy(), ba.copy(), P.copy()
    B = R.shape[0]
    dq = np.diag(Q); q = np.concatenate([dq[6:9], dq[3:6], dq[9:12], dq[12:15]])      # [n_g, n_a, n_bg, n_ba] step variances
    I15 = np.eye(15); Rm = np.eye(3) * sig_meas ** 2
    A = np.zeros((B, 15, 15)); A[:, 3:6, 0:3] = skew_b(GV); A[:, 6:9, 3:6] = _I3
    Gm = np.zeros((B, 15, 12)); Gm[:, 9:12, 6:9] = _I3; Gm[:, 12:15, 9:12] = _I3
    Hm = np.zeros((B, 3, 15)); Hm[:, :, 6:9] = -_I3
    snaps = set(snaps); out = {}
    snap = lambda: dict(p=p.copy(), v=v.copy(), R=R.copy(), bg=bg.copy(), ba=ba.copy(), P=P.copy())
    if k0 in snaps: out[k0] = snap()
    zB = np.zeros(3)
    for k in range(k0, k1):
        btot = ba if bofs is None else ba + bofs[:, k]
        fa = f[k] - btot; wc = w[k] - bg
        aw = np.einsum('bij,bj->bi', R, fa) + GV
        Sv = skew_b(v) @ R; Sp = skew_b(p) @ R
        A[:, 0:3, 9:12] = -R; A[:, 3:6, 9:12] = -Sv; A[:, 3:6, 12:15] = -R; A[:, 6:9, 9:12] = -Sp
        Gm[:, 0:3, 0:3] = R; Gm[:, 3:6, 0:3] = Sv; Gm[:, 3:6, 3:6] = R; Gm[:, 6:9, 0:3] = Sp
        Phi = I15 + A * dt
        Qd = (Gm * q) @ Gm.transpose(0, 2, 1)
        P = Phi @ P @ Phi.transpose(0, 2, 1) + Qd
        p = p + v * dt + 0.5 * aw * dt * dt
        v = v + aw * dt
        R = R @ exp_so3_b(wc * dt)
        kk = k + 1
        if kk % update_every == 0:
            z = p_gt[kk] + meas_noise[kk // FS] * sig_meas
            Hm[:, :, 0:3] = skew_b(p)
            PHt = P @ Hm.transpose(0, 2, 1)
            Sm = Hm @ PHt + Rm
            K = np.linalg.solve(Sm, PHt.transpose(0, 2, 1)).transpose(0, 2, 1)       # (B,15,3)
            r = z - p
            dx = -np.einsum('bij,bj->bi', K, r)
            IKH = I15 - K @ Hm
            P = IKH @ P @ IKH.transpose(0, 2, 1) + K @ Rm @ K.transpose(0, 2, 1)
            P = 0.5 * (P + P.transpose(0, 2, 1))
            R, v, p = exp_se23_apply(dx[:, 0:9], R, v, p)
            bg = bg + dx[:, 9:12]; ba = ba + dx[:, 12:15]
        if kk in snaps: out[kk] = snap()
    S.update(R=R, v=v, p=p, bg=bg, ba=ba, P=P)
    return out


def run_aided(S, f, w, p_gt, k0, k1, meas_noise, sig_meas, Q, bofs=None, snaps=(), dt=DT):
    """Single-filter wrapper with eskf.run_aided's signature; S (unbatched arrays) updated in place."""
    Sb = {k: x[None] for k, x in S.items()}
    out = run_aided_batch(Sb, f, w, p_gt, k0, k1, meas_noise, sig_meas, Q, None if bofs is None else bofs[None], snaps, dt)
    S.update({k: x[0] for k, x in Sb.items()})
    return {i: {k: x[0] for k, x in d.items()} for i, d in out.items()}


def outage_rotations_b(Sb, w, k0, k1):
    """Batched eskf.outage_rotations: (B,n,3,3), attitude by gyro only with frozen bg."""
    R = Sb["R"]; n = k1 - k0; Rs = np.empty((R.shape[0], n, 3, 3))
    for i in range(n):
        Rs[:, i] = R
        R = R @ exp_so3_b((w[k0 + i] - Sb["bg"]) * DT)
    return Rs

# -*- coding: utf-8 -*-
"""15-state ESKF (p,v,R,bg,ba), body-frame right-multiplicative attitude error. CPU/numpy only."""
import numpy as np
FS = 200; DT = 1.0 / FS; G = 9.81
GV = np.array([0.0, 0.0, -G])
NOISE = dict(sa=2.0e-3, sg=1.7e-4, sba=3.0e-3, sbg=1.9e-5)   # EuRoC nominal (PREREG s2)
P0_SIG = dict(p=0.01, v=0.05, att=np.radians([0.5, 0.5, 1.0]), bg=2e-3, ba=0.2)
INIT_ERR = dict(att=np.radians([0.5, 0.5, 1.0]), bg=1e-3)   # true init error sigma (x scale)

def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def exp_so3(phi):
    th = np.sqrt(phi @ phi)
    K = skew(phi)
    if th < 1e-9: return np.eye(3) + K
    return np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / th ** 2) * (K @ K)

def make_Q(qscale=1.0, dt=DT):
    q = np.zeros(15)
    q[3:6] = (NOISE['sa'] * qscale) ** 2 * dt
    q[6:9] = (NOISE['sg'] * qscale) ** 2 * dt
    q[9:12] = (NOISE['sbg'] * qscale) ** 2 * dt
    q[12:15] = (NOISE['sba'] * qscale) ** 2 * dt
    return np.diag(q)

def init_state(R0_gt, p0, v0, bg_gt0, rep_noise, att_scale):
    """rep_noise: dict(att(3), bg(3)) unit normals. Returns (p,v,R,bg,ba,P)."""
    e = rep_noise['att'] * INIT_ERR['att'] * att_scale
    R = R0_gt @ exp_so3(e)
    bg = bg_gt0 + rep_noise['bg'] * INIT_ERR['bg'] * att_scale
    P = np.zeros((15, 15))
    P[0:3, 0:3] = np.eye(3) * P0_SIG['p'] ** 2
    P[3:6, 3:6] = np.eye(3) * P0_SIG['v'] ** 2
    P[6:9, 6:9] = np.diag(P0_SIG['att'] ** 2)
    P[9:12, 9:12] = np.eye(3) * P0_SIG['bg'] ** 2
    P[12:15, 12:15] = np.eye(3) * P0_SIG['ba'] ** 2
    return dict(p=p0.copy(), v=v0.copy(), R=R, bg=bg, ba=np.zeros(3), P=P)

def run_aided(S, f, w, p_gt, k0, k1, meas_noise, sig_meas, Q, bofs=None, snaps=()):
    """Propagate S from frame index k0 to k1 (state at index k1). Position update at every frame
    index multiple of FS (index>0). meas_noise[idx//FS] unit normals. bofs: (N,3) known bias added to ba.
    snaps: indices at which to return a copy of the state. Returns dict idx->state copy."""
    p, v, R, bg, ba, P = S['p'], S['v'], S['R'], S['bg'], S['ba'], S['P']
    out = {}
    snaps = set(snaps)
    if k0 in snaps: out[k0] = dict(p=p.copy(), v=v.copy(), R=R.copy(), bg=bg.copy(), ba=ba.copy(), P=P.copy())
    I15 = np.eye(15); Rm = np.eye(3) * sig_meas ** 2
    for k in range(k0, k1):
        fk = f[k]; wk = w[k]
        btot = ba if bofs is None else ba + bofs[k]
        fa = fk - btot; wc = wk - bg
        aw = R @ fa + GV
        p = p + v * DT + 0.5 * aw * DT * DT
        v = v + aw * DT
        F = I15.copy()
        F[0:3, 3:6] = np.eye(3) * DT
        F[3:6, 6:9] = -R @ skew(fa) * DT
        F[3:6, 12:15] = -R * DT
        F[6:9, 6:9] = np.eye(3) - skew(wc) * DT
        F[6:9, 9:12] = -np.eye(3) * DT
        R = R @ exp_so3(wc * DT)
        P = F @ P @ F.T + Q
        kk = k + 1
        if kk % FS == 0:
            z = p_gt[kk] + meas_noise[kk // FS] * sig_meas
            S_ = P[0:3, 0:3] + Rm
            Kg = np.linalg.solve(S_, P[0:3, :]).T          # 15x3
            dx = Kg @ (z - p)
            IKH = I15.copy(); IKH[:, 0:3] -= Kg
            P = IKH @ P @ IKH.T + Kg @ Rm @ Kg.T
            P = 0.5 * (P + P.T)
            p = p + dx[0:3]; v = v + dx[3:6]; R = R @ exp_so3(dx[6:9]); bg = bg + dx[9:12]; ba = ba + dx[12:15]
        if kk in snaps: out[kk] = dict(p=p.copy(), v=v.copy(), R=R.copy(), bg=bg.copy(), ba=ba.copy(), P=P.copy())
    S.update(p=p, v=v, R=R, bg=bg, ba=ba, P=P)
    return out

def outage_rotations(S, w, k0, k1):
    """R_k for k=k0..k1-1 (used at step k), attitude by gyro only with frozen bg."""
    R = S['R']; n = k1 - k0
    Rs = np.empty((n, 3, 3))
    for i in range(n):
        Rs[i] = R
        R = R @ exp_so3((w[k0 + i] - S['bg']) * DT)
    return Rs

def outage_end_pos(S, f, Rs, k0, k1, bser):
    """End position for bias series bser (M, n, 3) or (n,3); linear form (discrete, matches run_aided)."""
    n = k1 - k0
    wk = DT * DT * (n - np.arange(n) - 0.5)                      # weight of accel at step i on final pos
    base = S['p'] + S['v'] * n * DT + (wk[:, None] * (np.einsum('kij,kj->ki', Rs, f[k0:k1]) + GV)).sum(0)
    bser = np.asarray(bser)
    if bser.ndim == 2: bser = bser[None]
    corr = np.einsum('k,kij,mkj->mi', wk, Rs, bser)
    return base[None] - corr

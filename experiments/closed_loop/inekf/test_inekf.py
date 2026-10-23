# -*- coding: utf-8 -*-
"""Unit tests for inekf.py (PREREG.md section 4). Run: python3 -m pytest -q test_inekf.py   (CPU only, no data except test_real_data_*)."""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pytest
from scipy.linalg import expm
import eskf, inekf
from eskf import DT, FS, GV, skew, exp_so3

rng = np.random.default_rng(7)


def rand_rot():
    return exp_so3(rng.standard_normal(3))


def step_nominal(R, v, p, bg, ba, f, w, dt):
    fa = f - ba; wc = w - bg; aw = R @ fa + GV
    return R @ exp_so3(wc * dt), v + aw * dt, p + v * dt + 0.5 * aw * dt * dt


def synth_traj(T, maneuver=True, dt=DT):
    """Truth trajectory + consistent IMU. Returns f, w (N,3), Rt (N+1,3,3), vt, pt (N+1,3)."""
    N = int(round(T / dt)); t = np.arange(N) * dt
    if maneuver:
        w = np.stack([0.30 * np.sin(0.7 * t), 0.25 * np.cos(0.5 * t), 0.20 * np.sin(0.3 * t + 1)], 1)
        aw_ = np.stack([1.0 * np.sin(0.9 * t), 0.8 * np.cos(0.6 * t), 0.5 * np.sin(0.4 * t)], 1)
    else:
        w = np.zeros((N, 3)); aw_ = np.zeros((N, 3))
    R = exp_so3(np.array([0.3, -0.2, 0.5])); v = np.zeros(3); p = np.array([3.0, -2.0, 1.0])
    Rt = [R]; vt = [v]; pt = [p]; f = np.zeros((N, 3))
    for k in range(N):
        f[k] = R.T @ (aw_[k] - GV)                          # specific force consistent with world acceleration aw_
        R, v, p = step_nominal(R, v, p, np.zeros(3), np.zeros(3), f[k], w[k], dt)
        Rt.append(R); vt.append(v); pt.append(p)
    return f, w, np.array(Rt), np.array(vt), np.array(pt)


def p0_state(R, v, p, ba_sig=0.2):
    S = dict(R=R.copy(), v=v.copy(), p=p.copy(), bg=np.zeros(3), ba=np.zeros(3))
    S["P"] = inekf.cov_e2i(eskf.init_state(R, p, v, np.zeros(3), dict(att=np.zeros(3), bg=np.zeros(3)), 0.0)["P"], R, v, p)
    return S


# ---------------------------------------------------------------- 1. static, noise free
def test_static_noise_free_stays_put():
    f, w, Rt, vt, pt = synth_traj(30.0, maneuver=False)
    N = len(f); pg = np.repeat(pt[:1], N + 1, 0)
    S = p0_state(Rt[0], vt[0], pt[0])
    inekf.run_aided(S, f, w, pg, 0, N, np.zeros((N // FS + 3, 3)), 0.05, eskf.make_Q(1.0))
    assert np.linalg.norm(S["p"] - pt[0]) < 1e-9 and np.linalg.norm(S["v"]) < 1e-9
    assert np.linalg.norm(S["R"] - Rt[0]) < 1e-9
    assert np.linalg.norm(S["bg"]) < 1e-9 and np.linalg.norm(S["ba"]) < 1e-9
    P = S["P"]; assert np.allclose(P, P.T, atol=1e-14) and np.linalg.eigvalsh(P).min() > -1e-12


# ---------------------------------------------------------------- 2. pure propagation vs ESKF
def _prop_compare(f, w, pt0, vt0, Rt0):
    n = len(f); assert n < FS
    nz = dict(att=np.array([0.3, -1.1, 0.7]), bg=np.array([0.5, 0.2, -0.9]))
    Se = eskf.init_state(Rt0, pt0, vt0, np.array([0.01, -0.02, 0.005]), nz, 1.0)
    Si = inekf.init_state(Rt0, pt0, vt0, np.array([0.01, -0.02, 0.005]), nz, 1.0)
    pg = np.zeros((n + 2, 3)); mn = np.zeros((4, 3))
    eskf.run_aided(Se, f, w, pg, 0, n, mn, 0.05, eskf.make_Q(1.0))
    inekf.run_aided(Si, f, w, pg, 0, n, mn, 0.05, eskf.make_Q(1.0))
    for k in ("p", "v", "R", "bg", "ba"): assert np.abs(Se[k] - Si[k]).max() < 1e-12, k
    Pe_back = inekf.cov_i2e(Si["P"], Si["R"], Si["v"], Si["p"])
    # compare attitude/velocity/position/bias blocks (ESKF order p,v,att,bg,ba)
    rel = np.linalg.norm(Pe_back - Se["P"]) / np.linalg.norm(Se["P"])
    return rel


def test_pure_propagation_matches_eskf_synthetic():
    f, w, Rt, vt, pt = synth_traj(1.0)
    rel = _prop_compare(f[:199], w[:199], pt[0], vt[0], Rt[0])
    print("synthetic 1 s cov rel diff", rel); assert rel < 0.05


def test_pure_propagation_matches_eskf_real_data():
    import run_closed_loop as rc
    D = rc.load_seq("MH_03")
    k = 3000; f, w = D["f"][k:k + 199], D["w"][k:k + 199]
    rel = _prop_compare(f, w, D["p"][k], D["v"][k], D["R"][k])
    print("real-data 1 s cov rel diff", rel); assert rel < 0.05


def inekf_log_err(Rh, vh, ph, R, v, p):
    eR = Rh @ R.T; phi = inekf.log_so3(eR)
    J = inekf.jl_so3(phi)
    return np.concatenate([phi, np.linalg.solve(J, vh - eR @ v), np.linalg.solve(J, ph - eR @ p)])


def test_propagation_jacobian_and_noise_input():
    dt = 1e-4; eps = 1e-6
    for trial in range(5):
        R = rand_rot(); v = rng.standard_normal(3) * 3; p = rng.standard_normal(3) * 20
        bg = rng.standard_normal(3) * 0.01; ba = rng.standard_normal(3) * 0.1
        f = rng.standard_normal(3) * 5 + np.array([0, 0, 9.81]); w = rng.standard_normal(3)
        xi = eps * rng.standard_normal(15)
        Rh, vh, ph = inekf.exp_se23_apply(xi[0:9], R, v, p)
        bgh, bah = bg + xi[9:12], ba + xi[12:15]
        R1, v1, p1 = step_nominal(R, v, p, bg, ba, f, w, dt)
        Rh1, vh1, ph1 = step_nominal(Rh, vh, ph, bgh, bah, f, w, dt)
        xi1 = np.concatenate([inekf_log_err(Rh1, vh1, ph1, R1, v1, p1), xi[9:15]])
        xi0 = np.concatenate([inekf_log_err(Rh, vh, ph, R, v, p), xi[9:15]])
        pred = inekf.A_cont(R, v, p) @ xi0 * dt
        inc = xi1 - xi0
        rel = np.linalg.norm(inc - pred) / np.linalg.norm(pred)
        assert rel < 1e-2, (trial, rel)
        # noise input: truth sees gyro/accel noise, estimate does not (same start, zero error)
        ng = rng.standard_normal(3) * 1e-2 / 1e-1 * 1e-1; na = rng.standard_normal(3) * 1e-1
        Rt, vt, pt = step_nominal(R, v, p, bg, ba, f - na, w - ng, dt)
        Re, ve, pe = step_nominal(R, v, p, bg, ba, f, w, dt)
        xin = inekf_log_err(Re, ve, pe, Rt, vt, pt)
        prn = (inekf.G_noise(R, v, p)[:, 0:6] @ np.concatenate([ng, na]))[0:9] * dt
        rel = np.linalg.norm(xin - prn) / np.linalg.norm(prn)
        assert rel < 1e-2, ("noise", trial, rel)


# ---------------------------------------------------------------- 3. update Jacobian
def test_update_jacobian():
    eps = 1e-4
    for _ in range(5):
        R = rand_rot(); v = rng.standard_normal(3); p = rng.standard_normal(3) * 30
        xi = eps * rng.standard_normal(9)
        Rh, vh, ph = inekf.exp_se23_apply(xi, R, v, p)
        r = p - ph
        H = np.hstack([skew(ph), np.zeros((3, 3)), -np.eye(3)])
        assert np.linalg.norm(r - H @ xi) < 100 * eps ** 2 * (1 + np.linalg.norm(p))
        assert np.linalg.norm(r - H @ xi) < 0.05 * np.linalg.norm(H @ xi)


# ---------------------------------------------------------------- 4. observability dimension
SIG = np.array([*np.radians([0.5, 0.5, 0.5]), 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 2e-3, 2e-3, 2e-3, 0.2, 0.2, 0.2])
SIG_E = np.concatenate([SIG[6:9], SIG[3:6], SIG[0:3], SIG[9:15]])      # ESKF order p,v,theta,bg,ba


def F_eskf_cont(R, f, w):
    F = np.zeros((15, 15)); F[0:3, 3:6] = np.eye(3); F[3:6, 6:9] = -R @ skew(f); F[3:6, 12:15] = -R
    F[6:9, 6:9] = -skew(w); F[6:9, 9:12] = -np.eye(3); return F


def gram_rows(kind, f, w, Rn, vn, pn, R_lin=None, p_lin=None, v_lin=None, stride=FS, nstep=None):
    """Rows H_k Phi(t_k,0) (unscaled) for k at 1 Hz. kind 'i' (InEKF) or 'e' (ESKF). Optional frozen linearisation point."""
    n = len(f) if nstep is None else nstep
    Phi = np.eye(15); rows = []
    for k in range(n):
        R = Rn[k] if R_lin is None else R_lin; v = vn[k] if v_lin is None else v_lin; p = pn[k] if p_lin is None else p_lin
        A = inekf.A_cont(R, v, p) if kind == "i" else F_eskf_cont(R, f[k] - 0 * f[k], w[k])
        Phi = expm(A * DT) @ Phi
        if (k + 1) % stride == 0:
            ph = p
            H = np.hstack([skew(ph), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 6))]) if kind == "i" else \
                np.hstack([np.eye(3), np.zeros((3, 12))])
            rows.append(H @ Phi)
    return np.vstack(rows), Phi


def svals(M, kind):
    s = SIG if kind == "i" else SIG_E
    return np.linalg.svd(M * s / 0.05, compute_uv=False)


def n_unobs(M, kind, rel=1e-8):
    sv = svals(M, kind); return int((sv < rel * sv[0]).sum()), sv


def test_observability_static_consistent_dim_is_4():
    f, w, Rt, vt, pt = synth_traj(20.0, maneuver=False)
    Mi, _ = gram_rows("i", f, w, Rt, vt, pt); Me, _ = gram_rows("e", f, w, Rt, vt, pt)
    ni, svi = n_unobs(Mi, "i"); ne, sve = n_unobs(Me, "e")
    print("static consistent: unobs InEKF", ni, "ESKF", ne, "\n sv InEKF", svi, "\n sv ESKF", sve)
    assert ni == 4 and ne == 4


def test_observability_static_linearisation_point_invariance():
    f, w, Rt, vt, pt = synth_traj(20.0, maneuver=False)
    Rtilt = exp_so3(np.radians([4.0, -3.0, 20.0])) @ Rt[0]          # tilted + yawed linearisation attitude
    p_far = np.array([40.0, -25.0, 7.0])
    Mi, _ = gram_rows("i", f, w, Rt, vt, pt, R_lin=Rtilt, p_lin=p_far, v_lin=np.zeros(3))
    ni, svi = n_unobs(Mi, "i")
    ez = np.array([0, 0, 1.0]); nyaw = np.concatenate([ez, np.zeros(3), -np.cross(ez, p_far), np.zeros(6)])
    leak = np.abs(Mi @ nyaw).max()
    Me, _ = gram_rows("e", f, w, Rt, vt, pt, R_lin=Rtilt)
    ne, sve = n_unobs(Me, "e")
    # ESKF: world-z yaw direction in body coords is R^T e_z; how much does it leak into the measurement?
    nyaw_e = np.concatenate([np.zeros(6), Rtilt.T @ ez, np.zeros(6)]); leak_e = np.abs(Me @ nyaw_e).max()
    print("tilted/far linearisation: InEKF unobs", ni, "yaw-null leak", leak, "| ESKF unobs", ne, "pure-yaw leak", leak_e)
    assert ni == 4 and leak < 1e-8


def test_observability_maneuver_full_rank():
    f, w, Rt, vt, pt = synth_traj(20.0, maneuver=True)
    Mi, _ = gram_rows("i", f, w, Rt, vt, pt); Me, _ = gram_rows("e", f, w, Rt, vt, pt)
    ni, svi = n_unobs(Mi, "i"); ne, sve = n_unobs(Me, "e")
    print("maneuver: unobs InEKF", ni, "ESKF", ne, "\n sv InEKF", svi, "\n sv ESKF", sve)
    assert ni == 0 and ne == 0


# ---------------------------------------------------------------- 5. batch == single
def test_batch_equals_single():
    f, w, Rt, vt, pt = synth_traj(2.5)
    N = len(f); pg = pt; mn = np.random.default_rng(3).standard_normal((6, 3)); Q = eskf.make_Q(1.0)
    bo = np.random.default_rng(4).standard_normal((3, N, 3)) * 0.05
    Ss = []
    for j in range(3):
        S = inekf.init_state(Rt[0], pt[0], vt[0], np.zeros(3), dict(att=np.array([1., -1, 0.5]), bg=np.array([0.3, 0.1, -0.2])), 1.0)
        S["P"][12:15, 12:15] = 0.0; Ss.append(S)
    Q2 = Q.copy(); Q2[12:15, 12:15] = 0
    Sb = {k: np.stack([S[k] for S in Ss]) for k in Ss[0]}
    ob = inekf.run_aided_batch(Sb, f, w, pg, 0, N, mn, 0.05, Q2, bofs=bo, snaps=[300])
    for j in range(3):
        S = {k: x.copy() for k, x in Ss[j].items()}
        o = inekf.run_aided(S, f, w, pg, 0, N, mn, 0.05, Q2, bofs=bo[j], snaps=[300])
        for k in S: assert np.abs(S[k] - Sb[k][j]).max() < 1e-10, k
        for k in o[300]: assert np.abs(o[300][k] - ob[300][k][j]).max() < 1e-10, k
    Rsb = inekf.outage_rotations_b(Sb, w, 100, 400)
    import eskf as e
    for j in range(3):
        Sj = {k: Sb[k][j] for k in ("R", "bg")}
        assert np.abs(e.outage_rotations(Sj, w, 100, 400) - Rsb[j]).max() < 1e-12


# ---------------------------------------------------------------- 6. sanity: filter converges (noisy) and keeps consistent
def test_converges_with_noisy_updates():
    f, w, Rt, vt, pt = synth_traj(40.0)
    N = len(f); mn = np.random.default_rng(5).standard_normal((N // FS + 3, 3))
    S = inekf.init_state(Rt[0], pt[0], vt[0], np.zeros(3), dict(att=np.array([1., 1, 1.]), bg=np.zeros(3)), 1.0)
    inekf.run_aided(S, f, w, pt, 0, N, mn, 0.05, eskf.make_Q(1.0))
    assert np.linalg.norm(S["p"] - pt[N]) < 0.5 and np.linalg.norm(S["v"] - vt[N]) < 0.5

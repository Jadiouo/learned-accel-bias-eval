# -*- coding: utf-8 -*-
"""
Metric 層 regression test（SDD §8）—— MVP 驗證過的 5 項 sanity 變成 pytest。

**完全 synthetic、data-independent**：任何人 clone（無 EuRoC、無 torch）都能 `pytest` 驗
metric 層。每項對應 SDD §8 的一條，全部要 `pass_strict`。
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from euroc_full import metrics as M

STRIDE, FREQ, GRAV = 5, 200, 9.81
N_RAW = 8000


# ──────────────────────────────────────────────────────────────────────────
# 合成資料（low rotation：rot≈body 成立；raw gyro 有未修正 roll-rate → gyro 漂、gyro_ate 主導）
# ──────────────────────────────────────────────────────────────────────────
def _make_merged(seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    ts = (np.arange(N_RAW) * (1e9 / FREQ)).astype(np.int64)
    t = np.arange(N_RAW) / FREQ
    # 低旋轉 GT 姿態：微幅 yaw 擺動（< 1°）→ R_ref ≈ I → rot≈body
    yaw = 0.01 * np.sin(2 * np.pi * 0.05 * t)
    rotvec = np.column_stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw])
    q = Rotation.from_rotvec(rotvec).as_quat()           # (N,4) xyzw（繞 z 微幅）
    # 加速度：body frame 主要是重力反作用 [0,0,~9.81] + 小運動
    a = np.zeros((N_RAW, 3)); a[:, 2] = GRAV
    a += 0.05 * rng.standard_normal((N_RAW, 3))
    # raw gyro：未修正的 roll-rate bias（繞 x），整段積分會把重力傾斜 → gyro_ate 主導
    w = np.zeros((N_RAW, 3)); w[:, 0] = 0.1
    w += 0.001 * rng.standard_normal((N_RAW, 3))
    # reference accel bias：近常數小偏置
    ba = np.tile(np.array([0.02, -0.015, 0.03]), (N_RAW, 1)) + 0.001 * rng.standard_normal((N_RAW, 3))
    df = pd.DataFrame({
        "ts": ts,
        "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2],
        "wx": w[:, 0], "wy": w[:, 1], "wz": w[:, 2],
        "qw": q[:, 3], "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2],
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "px": 0.0, "py": 0.0, "pz": 0.0,
        "bax": ba[:, 0], "bay": ba[:, 1], "baz": ba[:, 2],
    })
    return df


def _mi(merged, n=400, noise=0.3, seed=1):
    """建一個 MetricInputs：pred = b_ref + noise×σ（noise=0 → 完美）。"""
    starts = np.arange(n) * STRIDE
    seg = merged.iloc[starts].reset_index(drop=True)
    gt = seg[["bax", "bay", "baz"]].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    sigma = gt.std(0) + 1e-6
    pred = gt + noise * sigma * rng.standard_normal(gt.shape)
    mi = M.build_metric_inputs(merged, starts, pred, gt, STRIDE, FREQ, GRAV)
    return mi, merged, starts


@pytest.fixture(scope="module")
def merged():
    return _make_merged()


# ── §8 #1 ──────────────────────────────────────────────────────────────────
def test_body_iabe_equals_reference(merged):
    mi, mdf, starts = _mi(merged, noise=0.3)
    c = M.check_body_iabe_equals_reference(mi, mdf, starts, STRIDE, FREQ, tol=1e-6)
    assert c["pass_strict"], c


# ── §8 #2 ──────────────────────────────────────────────────────────────────
def test_gyro_ate_dominates(merged):
    suite = [_mi(merged, noise=lvl, seed=10 + i)[0] for i, lvl in enumerate([0.0, 0.5, 1.0, 2.0])]
    c = M.check_gyro_dominates(suite, min_ratio=10.0)
    assert c["pass_strict"], c
    assert c["gyro_ate_abs_median_m"] > 1.0, "gyro_ate 應有物理量級（非 0 strawman）"


# ── §8 #3（≥30 合成擾動，門檻 0.3）────────────────────────────────────────
def test_gyro_ate_bias_insensitive(merged):
    mi, _, _ = _mi(merged, noise=0.5)
    c = M.check_gyro_bias_insensitive(mi, n_samples=40, levels_max=4.0, max_abs_corr=0.3, seed=0)
    assert c["n_samples"] >= 30, c
    assert c["threshold_strict"] == 0.3
    assert c["pass_strict"], c


# ── §8 #4 ──────────────────────────────────────────────────────────────────
def test_rot_iabe_approx_body(merged):
    suite = [_mi(merged, noise=lvl, seed=20 + i)[0] for i, lvl in enumerate([0.2, 0.5, 1.0])]
    c = M.check_rot_approx_body(suite, max_rel_diff=0.20)
    assert c["pass_strict"], c


# ── §8 #5 ──────────────────────────────────────────────────────────────────
def test_lemma_constant_bias(merged):
    mi, _, _ = _mi(merged, noise=0.3)
    c = M.check_lemma_constant_bias(mi, tol_m=1e-4)
    assert c["pass_strict"], c


# ── gate 組裝 + relaxed 留痕（§3.4 機制保留）────────────────────────────────
def test_phase0_gate_strict_and_relaxed_recorded(merged):
    mi, mdf, starts = _mi(merged, noise=0.3)
    suite = [_mi(merged, noise=lvl, seed=30 + i)[0] for i, lvl in enumerate([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])]
    cfg = {"stride": STRIDE, "imu_freq": FREQ, "seed_base": 42,
           "sanity": {"legacy_tol": 1e-6, "gyro_dominates_min_ratio": 10.0,
                      "gyro_bias_insensitive_max_corr": 0.3, "gyro_bias_insensitive_n_samples": 40,
                      "gyro_bias_insensitive_levels_max": 4.0,
                      "gyro_bias_insensitive_relax_to": None, "gyro_bias_insensitive_relax_reason": None,
                      "rot_body_max_rel_diff": 0.20, "lemma_tol_m": 1e-4}}
    payload = M.run_phase0_sanity(cfg, mi, suite, mdf, starts, "test", "test")
    assert payload["gate_passed_strict"] is True, payload
    assert "gate_passed_relaxed" in payload
    for name, c in payload["checks"].items():
        assert "pass_strict" in c and "relaxed_for_mvp" in c, (name, c)
        assert c["relaxed_for_mvp"] is False     # 乾淨過關，不偽裝

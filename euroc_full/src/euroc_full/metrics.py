# -*- coding: utf-8 -*-
"""
Metric 層（SDD §5）—— MVP 最脆、最先寫、最先驗的一環。

四個指標 / four metrics:
  - iabe_body  : body-frame 二次積分（rotation = identity）= legacy 機制的 sanity 變體
  - iabe_rot   : ★ canonical，rotation-aware（world frame），重力/旋轉相減對消（Lemma）
  - gyro_ate   : ★ 非隔離對照——model 臂用會漂的 gyro-integrated 姿態（isolation 支柱）
  - bias_rmse  : window-level 偏置 RMSE（不積分）
  + slie / ate_reduction（作用在 rotation-aware 軌跡上）

關鍵不變式 / key invariants:
  * 所有指標吃同一份 per-sample 輸入（`MetricInputs`），由 `build_metric_inputs` 從 merged_df
    依 legacy 完全相同的 strided 取樣組出 → 保證 `iabe_body` 數值等於 legacy `compute_3d_ate`。
  * iabe_rot / gyro_ate 兩臂共用 R_ref（或各自姿態）、g_vec、a_raw、v0、p0；差異只在 bias 與姿態。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────────────────────────────────
# Per-sample 輸入容器 / per-sample metric inputs
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MetricInputs:
    """測試段沿時間軸的 per-sample 量（N = 測試 window 數）。"""
    a_raw: np.ndarray      # (N, 3) raw accel，body/sensor frame（含重力反作用）
    w_raw: np.ndarray      # (N, 3) raw gyro，body frame (rad/s)
    b_hat: np.ndarray      # (N, 3) 模型預測偏置 / predicted accel bias
    b_ref: np.ndarray      # (N, 3) reference 偏置 (GT b_a)
    R_ref: np.ndarray      # (N, 3, 3) reference 姿態 body→world (from GT quaternion)
    v0: np.ndarray         # (3,) 初速 seed (GT)
    p0: np.ndarray         # (3,) 初位置 seed (GT)
    dt: float              # 取樣間隔 = stride / imu_freq
    g_vec: np.ndarray      # (3,) world 重力向量 = [0, 0, -gravity]

    def __len__(self) -> int:
        return len(self.a_raw)


def build_metric_inputs(merged_df, test_win_starts, pred_bias, gt_bias,
                        stride: int, imu_freq: int, gravity: float) -> MetricInputs:
    """從 merged_df 依 legacy 慣例取 strided 測試段，組出 `MetricInputs`。

    `test_win_starts` 為每個測試 window 的「起始 raw-frame index」。連續序列時等於
    ``arange(n_offset, n_total) * stride``，與 legacy ``iloc[start:end:stride]`` 完全一致；
    gap-clean 丟窗後也能精確對位（傳實際保留的 frame index 即可）。
    """
    dt = float(stride) / float(imu_freq)
    idx = np.asarray(test_win_starts, dtype=np.int64)
    N = len(pred_bias)
    idx = idx[:N]
    seg = merged_df.iloc[idx].reset_index(drop=True)

    a_raw = seg[["ax", "ay", "az"]].to_numpy(np.float64)
    w_raw = seg[["wx", "wy", "wz"]].to_numpy(np.float64)
    # scipy 四元數順序為 (x, y, z, w)；GT 欄位是 qw,qx,qy,qz（q_RS: body→world）。
    quat_xyzw = seg[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    R_ref = Rotation.from_quat(quat_xyzw).as_matrix()           # (N, 3, 3) body→world
    v0 = seg[["vx", "vy", "vz"]].to_numpy(np.float64)[0]
    p0 = seg[["px", "py", "pz"]].to_numpy(np.float64)[0]
    g_vec = np.array([0.0, 0.0, -float(gravity)])

    return MetricInputs(
        a_raw=a_raw, w_raw=w_raw,
        b_hat=np.asarray(pred_bias, np.float64)[:N],
        b_ref=np.asarray(gt_bias, np.float64)[:N],
        R_ref=R_ref, v0=v0.astype(np.float64), p0=p0.astype(np.float64),
        dt=dt, g_vec=g_vec,
    )


# ──────────────────────────────────────────────────────────────────────────
# 積分工具 / integration helpers
# ──────────────────────────────────────────────────────────────────────────
def _double_integrate(acc: np.ndarray, v0: np.ndarray, p0: np.ndarray, dt: float):
    """逐軸 cumsum 二次積分。acc:(N,3) → pos:(N,3)。與 legacy 完全同式。"""
    vel = v0 + np.cumsum(acc * dt, axis=0)
    pos = p0 + np.cumsum(vel * dt, axis=0)
    return pos


def _rmse_3d(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """3D 逐點歐氏誤差的 RMSE。"""
    return float(np.sqrt(np.mean(np.sum((p_a - p_b) ** 2, axis=1))))


# ──────────────────────────────────────────────────────────────────────────
# 5.1  iabe_body — body frame，rotation = identity（legacy 機制）
# ──────────────────────────────────────────────────────────────────────────
def iabe_body(mi: MetricInputs) -> dict:
    """body-frame IABE：a_corr = a_raw − b_hat，逐軸積分，輸出 RMSE(p_corr − p_ideal)。

    這在數值上必須等於 legacy `compute_3d_ate`（同一計算）。
    """
    a_corr = mi.a_raw - mi.b_hat
    a_ideal = mi.a_raw - mi.b_ref
    p_corr = _double_integrate(a_corr, mi.v0, mi.p0, mi.dt)
    p_ideal = _double_integrate(a_ideal, mi.v0, mi.p0, mi.dt)
    return {"value": _rmse_3d(p_corr, p_ideal), "p_corr": p_corr, "p_ideal": p_ideal}


# ──────────────────────────────────────────────────────────────────────────
# 5.2  iabe_rot — ★ canonical，rotation-aware（world frame）
# ──────────────────────────────────────────────────────────────────────────
def _world_accel(R: np.ndarray, a_body: np.ndarray, g_vec: np.ndarray) -> np.ndarray:
    """a_world = R · a_body + g_vec（R:(N,3,3) body→world）。"""
    return np.einsum("nij,nj->ni", R, a_body) + g_vec


def iabe_rot(mi: MetricInputs) -> dict:
    """rotation-aware IABE。兩臂共用 R_ref/g_vec → 相減 = ∬ R_ref·(b_ref − b_hat)。"""
    a_corr_w = _world_accel(mi.R_ref, mi.a_raw - mi.b_hat, mi.g_vec)
    a_ideal_w = _world_accel(mi.R_ref, mi.a_raw - mi.b_ref, mi.g_vec)
    p_corr = _double_integrate(a_corr_w, mi.v0, mi.p0, mi.dt)
    p_ideal = _double_integrate(a_ideal_w, mi.v0, mi.p0, mi.dt)
    return {"value": _rmse_3d(p_corr, p_ideal), "p_corr": p_corr, "p_ideal": p_ideal}


# ──────────────────────────────────────────────────────────────────────────
# 5.3  gyro_ate — ★ 非隔離對照：model 臂用會漂的 gyro 積分姿態
# ──────────────────────────────────────────────────────────────────────────
def integrate_gyro(w_raw: np.ndarray, R0: np.ndarray, dt: float) -> np.ndarray:
    """從 R0 起、用 raw body-rate 以 SO(3) 正確更新積分姿態（會漂）。

    R(k+1) = R(k) · Exp(w(k)·dt)，用 Rodrigues / quaternion-exp（非小角度近似累積）。
    回傳 (N,3,3)；R[0] = R0。
    """
    N = len(w_raw)
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[0] = R0
    # body-frame 角速度 → 右乘增量旋轉
    incr = Rotation.from_rotvec(w_raw[:-1] * dt).as_matrix()    # (N-1, 3, 3)
    for k in range(N - 1):
        R[k + 1] = R[k] @ incr[k]
    return R


def gyro_ate(mi: MetricInputs) -> dict:
    """gyro-attitude ATE：model 臂姿態用 integrate_gyro（漂），ref 臂用乾淨 R_ref。

    兩臂姿態不同 → (R_gyro − R_ref)·a_raw（含 ~9.8 重力反作用）漏入並主導，
    對 bias 品質近乎不敏感。
    """
    R_gyro = integrate_gyro(mi.w_raw, mi.R_ref[0], mi.dt)
    a_model_w = _world_accel(R_gyro, mi.a_raw - mi.b_hat, mi.g_vec)
    a_ref_w = _world_accel(mi.R_ref, mi.a_raw - mi.b_ref, mi.g_vec)
    p_model = _double_integrate(a_model_w, mi.v0, mi.p0, mi.dt)
    p_ref = _double_integrate(a_ref_w, mi.v0, mi.p0, mi.dt)
    return {"value": _rmse_3d(p_model, p_ref), "p_corr": p_model, "p_ideal": p_ref}


# ──────────────────────────────────────────────────────────────────────────
# 5.5  bias_rmse — window-level（不積分）
# ──────────────────────────────────────────────────────────────────────────
def bias_rmse(b_hat: np.ndarray, b_ref: np.ndarray) -> dict:
    b_hat = np.asarray(b_hat, np.float64)
    b_ref = np.asarray(b_ref, np.float64)
    out = {}
    for i, ax in enumerate("xyz"):
        out[ax] = float(np.sqrt(np.mean((b_hat[:, i] - b_ref[:, i]) ** 2)))
    out["avg"] = float(np.sqrt(np.mean((b_hat - b_ref) ** 2)))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 5.6  slie / ate_reduction
# ──────────────────────────────────────────────────────────────────────────
def slie(p_corr: np.ndarray, p_ideal: np.ndarray, tau: float, dt: float):
    """Segment-Level Integration Error：τ 秒分段、50% overlap、origin-aligned RMS。"""
    N = len(p_corr)
    L = int(np.floor(tau / dt))
    if L < 2 or L >= 0.8 * N:
        return None
    step = max(L // 2, 1)
    starts = list(range(0, N - L + 1, step))
    if not starts:
        return None
    errs = []
    for s in starts:
        offset = p_ideal[s] - p_corr[s]
        p_aligned = p_corr[s:s + L] + offset
        errs.append(_rmse_3d(p_aligned, p_ideal[s:s + L]))
    return float(np.mean(errs))


def ate_reduction(mi: MetricInputs) -> float:
    """相對未補償（b_hat=0）的 % 減少，在 canonical rot frame 上算。

    僅為 counterexample 展示飽和，不當主指標（SDD §5.6）。
    """
    corrected = iabe_rot(mi)["value"]
    zero = MetricInputs(mi.a_raw, mi.w_raw, np.zeros_like(mi.b_hat), mi.b_ref,
                        mi.R_ref, mi.v0, mi.p0, mi.dt, mi.g_vec)
    nocomp = iabe_rot(zero)["value"]
    if nocomp <= 0:
        return 0.0
    return float((1.0 - corrected / nocomp) * 100.0)


# ──────────────────────────────────────────────────────────────────────────
# 便利：算一個 run 的全部指標 / compute all metrics for one run
# ──────────────────────────────────────────────────────────────────────────
def compute_all_metrics(mi: MetricInputs, slie_taus) -> dict:
    """回傳 {metric_name: value}，metric 字串與 SDD §3.2 對齊。"""
    rot = iabe_rot(mi)
    body = iabe_body(mi)
    gyro = gyro_ate(mi)
    brm = bias_rmse(mi.b_hat, mi.b_ref)
    out = {
        "iabe_rot": rot["value"],
        "iabe_body": body["value"],
        "gyro_ate": gyro["value"],
        "bias_rmse_x": brm["x"], "bias_rmse_y": brm["y"],
        "bias_rmse_z": brm["z"], "bias_rmse_avg": brm["avg"],
        "ate_reduction": ate_reduction(mi),
    }
    for tau in slie_taus:
        val = slie(rot["p_corr"], rot["p_ideal"], float(tau), mi.dt)
        out[f"slie_{int(tau)}"] = val if val is not None else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────
# 5.7  Phase 0 解析 sanity（五項，全過才放行）
# ──────────────────────────────────────────────────────────────────────────
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """純 numpy Spearman ρ（避免硬依賴 scipy.stats）。"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / denom) if denom > 0 else 0.0


def _mi_with_bhat(mi: MetricInputs, b_hat: np.ndarray) -> MetricInputs:
    """以同一份幾何（a_raw/w_raw/R_ref/v0/p0/dt/g）換上不同 b_hat，組新 MetricInputs。"""
    return MetricInputs(mi.a_raw, mi.w_raw, np.asarray(b_hat, np.float64), mi.b_ref,
                        mi.R_ref, mi.v0, mi.p0, mi.dt, mi.g_vec)


def check_body_iabe_equals_reference(mi: MetricInputs, merged_df, test_win_starts,
                                     stride, imu_freq, tol: float) -> dict:
    """(1)【self-contained】body-IABE == 手算 cumsum 雙積分（SDD §8 test #1）。

    euroc_full 不依賴 legacy（§1.2）。以「內嵌、逐軸手算的 cumsum 雙積分」當參考真值，
    驗證 `iabe_body` 的向量化實作數值一致 —— 這正是 MVP 已 bit-identical 對上 legacy
    `compute_3d_ate` 的同一條計算。
    """
    new_val = iabe_body(mi)["value"]
    pred = mi.b_hat
    gt = mi.b_ref
    seg = merged_df.iloc[test_win_starts[:len(pred)]].reset_index(drop=True)
    ref_p_corr, ref_p_ideal = [], []
    for i, ax in enumerate("xyz"):
        a_raw = seg[f"a{ax}"].to_numpy(np.float64)
        v0 = float(seg[f"v{ax}"].iloc[0]); p0 = float(seg[f"p{ax}"].iloc[0])
        vc = v0 + np.cumsum((a_raw - pred[:, i]) * (stride / imu_freq))
        vi = v0 + np.cumsum((a_raw - gt[:, i]) * (stride / imu_freq))
        ref_p_corr.append(p0 + np.cumsum(vc * (stride / imu_freq)))
        ref_p_ideal.append(p0 + np.cumsum(vi * (stride / imu_freq)))
    ref_val = _rmse_3d(np.stack(ref_p_corr, 1), np.stack(ref_p_ideal, 1))
    rel = abs(new_val - ref_val) / (abs(ref_val) + 1e-12)
    return {"pass_strict": bool(rel < tol), "max_rel_err": float(rel), "tol": float(tol),
            "new_iabe_body": float(new_val), "reference_ate": float(ref_val),
            "note": "compared against inlined hand-computed cumsum double-integral (self-contained)"}


def check_gyro_dominates(mi_list, min_ratio: float) -> dict:
    """(2) median(gyro_ate / iabe_rot) > min_ratio；附 gyro_ate 絕對量級（確認物理合理）。"""
    ratios = []; g_abs = []
    for mi in mi_list:
        r = iabe_rot(mi)["value"]; g = gyro_ate(mi)["value"]
        g_abs.append(g)
        if r > 0:
            ratios.append(g / r)
    med = float(np.median(ratios)) if ratios else 0.0
    return {"pass_strict": bool(med > min_ratio), "ratio_gyro_over_iabe_median": med,
            "min_ratio_threshold": float(min_ratio),
            "gyro_ate_abs_median_m": float(np.median(g_abs)) if g_abs else 0.0,
            "gyro_ate_abs_max_m": float(np.max(g_abs)) if g_abs else 0.0}


def check_gyro_bias_insensitive(mi_base: MetricInputs, n_samples: int, levels_max: float,
                                max_abs_corr: float, seed: int = 0) -> dict:
    """(3)【v1.1 Fix 1】gyro_ate 對 bias 品質不敏感 —— 用 ≥30 個合成擾動樣本檢定。

    run-1 用 ~7 個樣本測 Spearman，噪聲撐到 0.77 被迫放寬門檻；根因是**測試沒鑑別力**，
    不是門檻太嚴。v1.1：把 `b_ref` 加上強度從 ~0 線性掃到 `levels_max × per-axis σ` 的
    高斯擾動，產生 `n_samples`(≥30) 個 (gyro_ate, bias_rmse_avg) 樣本。

      - gyro_ate 由 (R_gyro − R_ref)·a_raw（含重力）主導 → 幾乎不隨 bias 擾動變。
      - bias_rmse_avg 隨擾動單調增。
      → 兩者 Spearman ≈ 0；n 大則 ρ 穩定（真值 ≈ 0.18–0.22）。

    `pass_strict = |Spearman| < max_abs_corr`（strict 門檻 0.3，非 0.8）。
    """
    rng = np.random.default_rng(seed)
    b_ref = mi_base.b_ref
    sigma = np.std(b_ref, axis=0) + 1e-9
    n = max(int(n_samples), 30)
    levels = np.linspace(0.02, float(levels_max), n)   # 從近乎完美到很差，跨 bias_rmse 量級
    g_list, b_list = [], []
    for lvl in levels:
        b_hat = b_ref + lvl * sigma * rng.standard_normal(b_ref.shape)
        mi_i = _mi_with_bhat(mi_base, b_hat)
        g_list.append(gyro_ate(mi_i)["value"])
        b_list.append(bias_rmse(b_hat, b_ref)["avg"])
    rho = abs(_spearman(np.array(g_list), np.array(b_list)))
    return {"pass_strict": bool(rho < max_abs_corr),
            "abs_spearman_gyroate_vs_biasrmse": float(rho),
            "threshold_strict": float(max_abs_corr),
            "n_samples": int(n),
            "bias_rmse_span": [float(np.min(b_list)), float(np.max(b_list))],
            "gyro_ate_cv": float(np.std(g_list) / (np.mean(g_list) + 1e-12))}


def check_rot_approx_body(mi_list, max_rel_diff: float) -> dict:
    """(4) 低旋轉序列上 median(|iabe_rot − iabe_body|/iabe_rot) < max_rel_diff。"""
    diffs = []
    for mi in mi_list:
        r = iabe_rot(mi)["value"]; b = iabe_body(mi)["value"]
        if r > 0:
            diffs.append(abs(r - b) / r)
    med = float(np.median(diffs)) if diffs else 1.0
    return {"pass_strict": bool(med < max_rel_diff), "median_rel_diff": med,
            "max_rel_diff_threshold": float(max_rel_diff)}


def check_lemma_constant_bias(mi: MetricInputs, tol_m: float) -> dict:
    """(5) 合成 b_ref − b_hat = δ（常數 per-axis）→ body frame 解析 p_diff = 0.5·δ·t²。

    用離散 cumsum 等價基準（消除 0.5t² 的離散化偏差），驗證 Lemma 的數值實現。
    """
    N = len(mi)
    dt = mi.dt
    delta = np.array([0.02, -0.015, 0.03])               # 任意常數 per-axis bias 差
    b_hat = mi.b_ref - delta                              # → b_ref − b_hat = δ
    test = MetricInputs(mi.a_raw, mi.w_raw, b_hat, mi.b_ref,
                        mi.R_ref, mi.v0, mi.p0, dt, mi.g_vec)
    got = iabe_body(test)
    p_diff_num = got["p_corr"] - got["p_ideal"]          # 應 = ∬ δ（body frame）
    # 離散等價解析解：∬δ via 同樣的 cumsum（避免連續 0.5δt² 的離散偏差）
    vel = np.cumsum(np.tile(delta, (N, 1)) * dt, axis=0)
    p_analytic = np.cumsum(vel * dt, axis=0)
    max_err = float(np.max(np.abs(p_diff_num - p_analytic)))
    return {"pass_strict": bool(max_err < tol_m), "max_abs_err_m": max_err, "tol_m": float(tol_m)}


# ──────────────────────────────────────────────────────────────────────────
# 【v1.1 Fix 2】relaxed_for_mvp 留痕機制（SDD §3.4）
# ──────────────────────────────────────────────────────────────────────────
def _attach_relax(check: dict, relax_to=None, relax_reason=None,
                  relaxed_metric: float = None) -> dict:
    """為一項 check 補上 §3.4 的 relaxed 欄位，並回傳「放寬後是否過」。

    規則（SDD §3.4）：
      * 預設 `relaxed_for_mvp=False`，三個 relaxed 欄位為 null —— 乾淨跑 strict。
      * 只有「strict 沒過、且 config 指定放寬」時才 `relaxed_for_mvp=True`，
        此時 `pass_strict` 維持 False（**不得偽裝成乾淨過關**），另記 relaxed 門檻/理由。
      * 回傳 effective_pass：strict 過、或（放寬且通過放寬門檻）。
    """
    check.setdefault("relaxed_for_mvp", False)
    check.setdefault("relaxed_threshold", None)
    check.setdefault("relax_reason", None)
    pass_strict = bool(check["pass_strict"])
    if pass_strict or relax_to is None:
        return check, pass_strict
    # strict 沒過 → 啟用放寬留痕
    relaxed_pass = (relaxed_metric is not None) and (relaxed_metric < float(relax_to))
    check["relaxed_for_mvp"] = True
    check["relaxed_threshold"] = float(relax_to)
    check["relax_reason"] = relax_reason or "relaxed for MVP pipeline; needs re-verification"
    return check, bool(relaxed_pass)


def run_phase0_sanity(cfg, mi_real: MetricInputs, mi_suite, merged_df,
                      test_win_starts, git_hash: str, timestamp: str) -> dict:
    """組裝 phase0_sanity.json payload（SDD §3.4，含 strict/relaxed 分記）。

    mi_suite = 6 階品質預測（給 dominates / rot≈body 兩項 median 用）。
    gyro_ate_bias_insensitive 改由 `check_gyro_bias_insensitive` 自建 ≥30 合成擾動樣本。
    """
    s = cfg["sanity"]
    stride, freq = cfg["stride"], cfg["imu_freq"]

    c_legacy = check_body_iabe_equals_reference(
        mi_real, merged_df, test_win_starts, stride, freq, s["legacy_tol"])
    c_dom = check_gyro_dominates(mi_suite, s["gyro_dominates_min_ratio"])
    c_ins = check_gyro_bias_insensitive(
        mi_real, s.get("gyro_bias_insensitive_n_samples", 40),
        s.get("gyro_bias_insensitive_levels_max", 4.0),
        s["gyro_bias_insensitive_max_corr"], seed=cfg.get("seed_base", 42))
    c_rot = check_rot_approx_body(mi_suite, s["rot_body_max_rel_diff"])
    c_lemma = check_lemma_constant_bias(mi_real, s["lemma_tol_m"])

    # 每項補 relaxed 欄位；目前只有 gyro_ate_bias_insensitive 可由 config 放寬。
    effective = {}
    c_legacy, effective["body_iabe_equals_reference"] = _attach_relax(c_legacy)
    c_dom, effective["gyro_ate_dominates"] = _attach_relax(c_dom)
    c_ins, effective["gyro_ate_bias_insensitive"] = _attach_relax(
        c_ins, relax_to=s.get("gyro_bias_insensitive_relax_to"),
        relax_reason=s.get("gyro_bias_insensitive_relax_reason"),
        relaxed_metric=c_ins["abs_spearman_gyroate_vs_biasrmse"])
    c_rot, effective["rot_iabe_approx_body"] = _attach_relax(c_rot)
    c_lemma, effective["lemma_constant_bias"] = _attach_relax(c_lemma)

    checks = {
        "body_iabe_equals_reference": c_legacy,
        "gyro_ate_dominates": c_dom,
        "gyro_ate_bias_insensitive": c_ins,
        "rot_iabe_approx_body": c_rot,
        "lemma_constant_bias": c_lemma,
    }
    gate_strict = all(c["pass_strict"] for c in checks.values())
    gate_relaxed = all(effective.values())
    return {"git_hash": git_hash, "timestamp": timestamp,
            "checks": checks,
            "gate_passed_strict": bool(gate_strict),
            "gate_passed_relaxed": bool(gate_relaxed),
            # 向後相容鍵：gate_passed = strict（run_mvp gate 以 strict 為準）
            "gate_passed": bool(gate_strict)}

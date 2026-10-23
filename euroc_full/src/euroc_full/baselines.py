# -*- coding: utf-8 -*-
"""
Baselines（SDD §5.4）。

  - constant_pooled   : 所有 pooled training windows 的 mean bias（regime B 用，★ global）
  - constant_perseq   : 單序列 train split mean（regime A 用，展示 near-tie pathology）
  - bias_spread       : 序列間 bias spread（pairwise L2 + 全域 std）→ 寫進 SUMMARY
  - integrate_gyro    : gyro-attitude helper（轉呼叫 metrics 的 SO(3) 積分，單一真相源）
  - kalman_filter_bias: 簡單常數偏置 KF（bookend，非主角）
  - particle_filter_bias: bootstrap PF（§4；確認失敗非線性化 artifact，**PF≈KF，勿 oversell**）
  - LinearStatsRegressor: window 特徵線性回歸（§4，~3K 參數，statistical baseline）
"""
from __future__ import annotations

import numpy as np

# gyro-attitude helper 的唯一實作放在 metrics（SO(3) 正確積分），這裡轉出避免重複。
from .metrics import integrate_gyro  # noqa: F401  (re-exported per SDD §5.4)


def constant_pooled(pooled_y_train: np.ndarray, n_pred: int) -> np.ndarray:
    """所有 pooled training windows 的全域 mean，平鋪成 (n_pred, 3)。"""
    mean = np.asarray(pooled_y_train, np.float64).mean(axis=0)
    return np.tile(mean, (n_pred, 1))


def constant_perseq(y_train: np.ndarray, n_pred: int) -> np.ndarray:
    """單序列 train split 的 mean，平鋪成 (n_pred, 3)。"""
    mean = np.asarray(y_train, np.float64).mean(axis=0)
    return np.tile(mean, (n_pred, 1))


def bias_spread(per_seq_train_mean: dict) -> dict:
    """序列間 bias spread：各序列 train-mean 的兩兩 L2 距離 + 全域 std。"""
    seqs = list(per_seq_train_mean.keys())
    means = np.stack([np.asarray(per_seq_train_mean[s], np.float64) for s in seqs], axis=0)
    pair = {}
    dists = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            d = float(np.linalg.norm(means[i] - means[j]))
            pair[f"{seqs[i]}|{seqs[j]}"] = d
            dists.append(d)
    return {
        "pairwise_l2": pair,
        "mean_pairwise_l2": float(np.mean(dists)) if dists else 0.0,
        "max_pairwise_l2": float(np.max(dists)) if dists else 0.0,
        "global_std_per_axis": means.std(axis=0).tolist(),
        "global_std_norm": float(np.linalg.norm(means.std(axis=0))),
    }


def kalman_filter_bias(imu_accel: np.ndarray, b_init: np.ndarray,
                       q: float = 1e-6, r: float = 1e-2) -> np.ndarray:
    """sensor-only 常數偏置隨機遊走 KF（per-axis 獨立），回傳 (N, 3) 偏置估計。

    【bookend 角色 / v1.1 Fix 4】這是「無外部位置/速度修正」的 KF：只有原始 specific force
    可觀測，accel 偏置與真實運動不可分（unobservability）。因此濾波器收斂到 `b_init`
    （= train/global mean），結果 **≈ constant baseline**，而非「災難性發散」。

    run-1 在 regime A 已驗證 `kf ≡ constant_perseq`（iabe_rot 完全相同）。v1.1 把它也跑在
    regime B（`b_init = global pooled mean` → `kf ≈ constant_pooled`），讓 SUMMARY bookend
    能據實對帳：規劃書的「484–1596 m 災難性失敗」是 world-frame / 含姿態的 ATE，**不是 IABE**；
    IABE 量級下 KF 的真相是「不可觀測 → 打不贏 constant」。

    `imu_accel` 傳 `zeros((N,3))` 時觀測項退化為常數 `b_init`（純 bookend 對照，非主指標）。
    """
    accel = np.asarray(imu_accel, np.float64)
    N = len(accel)
    b = np.asarray(b_init, np.float64).copy()
    P = np.ones(3) * 1e-2
    out = np.empty((N, 3))
    for k in range(N):
        # predict（常數模型）
        P = P + q
        # update（以瞬時 accel 減去長期平均當粗觀測）
        z = accel[k] - accel.mean(axis=0) + b_init
        K = P / (P + r)
        b = b + K * (z - b)
        P = (1 - K) * P
        out[k] = b
    return out


def particle_filter_bias(imu_accel: np.ndarray, b_init: np.ndarray,
                         n_particles: int = 400, q: float = 1e-8, r: float = 1e-4,
                         seed: int = 0) -> np.ndarray:
    """sensor-only bootstrap 粒子濾波偏置估計（§4）。

    **角色：確認 KF 的失敗不是線性化 artifact** —— 同樣的不可觀測觀測模型下，非線性 PF
    與 KF 給出**幾乎相同**的估計（都收斂到 `b_init`）。SUMMARY 標明 `PF≈KF`，**勿 oversell**。

    觀測模型與 KF 一致（degenerate，純 bookend）：z_k = accel_k − mean(accel) + b_init。
    粒子緊圍 `b_init` 初始化、小過程/觀測噪聲 → 加權估計快速收斂到 `b_init`（≈ KF ≈ constant）。
    """
    accel = np.asarray(imu_accel, np.float64)
    N = len(accel)
    b_init = np.asarray(b_init, np.float64)
    rng = np.random.default_rng(seed)
    part = b_init[None, :] + rng.normal(0.0, np.sqrt(r), size=(n_particles, 3))
    out = np.empty((N, 3))
    z_mean = accel.mean(axis=0)
    for k in range(N):
        part = part + rng.normal(0.0, np.sqrt(q), size=part.shape)     # RW 過程
        z = accel[k] - z_mean + b_init                                  # 同 KF 的退化觀測
        ll = -0.5 * np.sum((part - z[None, :]) ** 2, axis=1) / r        # 高斯似然 → 權重
        w = np.exp(ll - ll.max()); w /= w.sum()
        out[k] = np.sum(part * w[:, None], axis=0)
        idx = rng.choice(n_particles, size=n_particles, p=w)            # 系統重採樣
        part = part[idx]
    return out


class LinearStatsRegressor:
    """window 特徵線性回歸（§4，statistical baseline，~3K 參數）。

    特徵 = 每個 window 逐通道的 [mean, std, min, max]（6 通道 × 4 = 24 維）→ 線性映射到 3 軸
    bias。閉式最小二乘（含 bias 項），不需 torch；fit on train、predict on test。
    """

    def __init__(self):
        self.W = None  # (n_feat+1, 3)

    @staticmethod
    def _feats(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)                      # (N, W, 6)
        f = np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)  # (N, 24)
        return f

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearStatsRegressor":
        F = self._feats(X_train)
        A = np.concatenate([F, np.ones((len(F), 1))], axis=1)   # 加 bias 項
        self.W, *_ = np.linalg.lstsq(A, np.asarray(y_train, np.float64), rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("LinearStatsRegressor 未 fit")
        F = self._feats(X)
        A = np.concatenate([F, np.ones((len(F), 1))], axis=1)
        return (A @ self.W).astype(np.float64)

    @property
    def n_params(self) -> int:
        return int(self.W.size) if self.W is not None else 0

# -*- coding: utf-8 -*-
"""
資料層（SDD §6）—— 複製自驗證過的 MVP `data_euroc.py`，**windowing / regime 邏輯不改**，
另加 §6 要求的 **dataset adapter 介面**（TUM-VI 日後 drop-in，本套件只實作 EuRoC）。

  - DatasetAdapter (ABC) / EurocAdapter : load_sequence(name) -> 標準化 dict
  - align_and_merge / make_windows_clean: ★ gap-clean 切窗（丟橫跨 gap 的 window + assert）
  - prepare_regime_a                    : 單序列時序 70/15/15（regime A）
  - prepare_regime_b_pooled             : ★ 跨序列 head 池化、scaler 只 fit pooled-train（regime B）

關鍵不變式：metric 層用 `test_win_starts`（每個測試 window 的起始 raw-frame index）精確對位。
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .common import seq_csv_paths

FEAT_COLS = ["wx", "wy", "wz", "ax", "ay", "az"]
LABEL_COLS = ["bax", "bay", "baz"]
GT_COLS = ["ts", "px", "py", "pz", "qw", "qx", "qy", "qz",
           "vx", "vy", "vz", "bwx", "bwy", "bwz", "bax", "bay", "baz"]


# ──────────────────────────────────────────────────────────────────────────
# 容器 / containers
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SeqData:
    seq: str
    merged: pd.DataFrame
    X_train: np.ndarray; y_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray
    test_win_starts: np.ndarray
    scaler: StandardScaler
    win_starts_all: np.ndarray


@dataclass
class PooledData:
    train_seqs: list
    X_train: np.ndarray; y_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray
    scaler: StandardScaler
    pooled_y_train_raw: np.ndarray
    per_seq_eval: dict = field(default_factory=dict)
    per_seq_train_mean: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# §6  Dataset adapter 介面（TUM-VI drop-in 用；本套件只實作 EuRoC）
# ──────────────────────────────────────────────────────────────────────────
class DatasetAdapter(abc.ABC):
    """抽象資料來源。日後加 `TumviRoomAdapter` 即可沿用全部 loader/metric/analysis。"""

    @abc.abstractmethod
    def load_sequence(self, name: str) -> dict:
        """回傳標準化 dict：
        {imu: DataFrame[ts,wx,wy,wz,ax,ay,az],
         gt:  DataFrame[ts,px,py,pz,qw,qx,qy,qz, ...(可選 v/b)]}
        timestamps 為 ns int；frame/sign/gravity 慣例由各 adapter 自負。
        """
        raise NotImplementedError

    def derive_labels(self, merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """align 後的後處理 hook。預設 no-op（GT 已含 v/b_a，如 EuRoC）。

        TUM-VI 覆寫此方法,從 mocap p/q 導出 v 與 reference accel bias（§4.3）。
        EuRoC 走 no-op → 既有 metric 計算與 684 runs 結果完全不受影響。
        """
        return merged


class EurocAdapter(DatasetAdapter):
    """EuRoC ASL：imu0 6 軸 + state_groundtruth_estimate0 17 欄（含 b_a reference bias）。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load_sequence(self, name: str) -> dict:
        paths = seq_csv_paths(name, self.cfg)
        return {"imu": load_imu(paths["imu"]), "gt": load_gt(paths["gt"])}


# room1–6 唯一有 full mocap coverage（corridor/magistrale/outdoors 排除，§2）
TUMVI_ROOMS = {"room1", "room2", "room3", "room4", "room5", "room6"}


class TumviRoomAdapter(DatasetAdapter):
    """TUM-VI room（§4）：BMI160 IMU + OptiTrack mocap（只有 p/q）。

    reference bias **導出**（§4.3）：v=savgol(p, deriv=1)、a_world=savgol(p, deriv=2)，
    b_a = a_meas − R(q)^T(a_world − g_world)。導出在 `derive_labels`（align 後、開窗前）做。
    corridor/magistrale/outdoors 一律拒絕（mocap 有大洞，跨洞二次微分無意義）。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _paths(self, name: str) -> dict:
        if name not in TUMVI_ROOMS:
            raise ValueError(
                f"TumviRoomAdapter 只支援 room1–6,拒絕 {name!r}"
                "（corridor/magistrale/outdoors 排除：mocap 覆蓋不足/有大洞）")
        base = os.path.join(self.cfg["_data_dir_abs"], name, "mav0")
        return {"imu": os.path.join(base, "imu0", "data.csv"),
                "mocap": os.path.join(base, "mocap0", "data.csv")}

    def load_sequence(self, name: str) -> dict:
        p = self._paths(name)
        return {"imu": load_imu(p["imu"]), "gt": load_mocap(p["mocap"])}

    def derive_labels(self, merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        return derive_mocap_labels(merged, cfg)


def get_adapter(cfg: dict) -> DatasetAdapter:
    """config 的 `dataset` 欄選 adapter（euroc / tumvi_room）。"""
    name = str(cfg.get("dataset", "euroc")).lower()
    if name == "euroc":
        return EurocAdapter(cfg)
    if name in ("tumvi_room", "tumvi"):
        return TumviRoomAdapter(cfg)
    raise ValueError(f"未知 dataset adapter: {name}（支援 euroc / tumvi_room）")


# ──────────────────────────────────────────────────────────────────────────
# CSV 載入 / loaders
# ──────────────────────────────────────────────────────────────────────────
def load_imu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", header=0)
    df.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
    df["ts"] = df["ts"].astype(np.int64)
    return df.reset_index(drop=True)


def load_gt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", header=0)
    df.columns = GT_COLS[:df.shape[1]]
    df["ts"] = df["ts"].astype(np.int64)
    return df.reset_index(drop=True)


def load_mocap(path: str) -> pd.DataFrame:
    """TUM-VI mocap0：只有 ts + p(3) + q(4)（無 v / b_a / b_w）。"""
    df = pd.read_csv(path, comment="#", header=0)
    df.columns = ["ts", "px", "py", "pz", "qw", "qx", "qy", "qz"][:df.shape[1]]
    df["ts"] = df["ts"].astype(np.int64)
    return df.reset_index(drop=True)


def derive_mocap_labels(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """從 mocap p/q 導出 v 與 reference accel bias（§4.3；port 自 legacy，不 import）。

        v_world = d/dt   p_world            (Savitzky-Golay, deriv=1)
        a_world = d²/dt² p_world            (Savitzky-Golay, deriv=2)
        f_body  = R(q)^T (a_world − g_world)              # 預期比力（g_world=[0,0,−g]）
        b_a     = a_meas_body − f_body                    # accel bias（body frame）

    ⚠ 二次微分對噪聲災難性放大：`mocap_smoothing.window`(savgol 視窗) 直接決定「什麼叫 bias」。
    room1 derisk（§7.1）必驗 b_ref 平穩（smoothing 後 std « 量級，非噪聲主導）。
    在 align 後的 `merged`（imu 時序）上做，與 legacy 一致。
    """
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as _R

    imu_freq = cfg.get("imu_freq", 200)
    gravity = cfg.get("gravity", 9.81)
    sm = cfg.get("mocap_smoothing", {}) or {}
    polyorder = int(sm.get("polyorder", 3))
    pos_window = sm.get("window") or ((imu_freq // 2) | 1)
    pos_window = int(pos_window) | 1                         # 強制奇數
    pos_window = min(pos_window, (len(merged) // 2) * 2 - 1)
    if pos_window <= polyorder:
        raise ValueError(f"序列太短無法導出 mocap label（n={len(merged)}）")
    dt = 1.0 / imu_freq

    p = merged[["px", "py", "pz"]].to_numpy(np.float64)
    v = np.empty_like(p); a = np.empty_like(p)
    for i in range(3):
        v[:, i] = savgol_filter(p[:, i], pos_window, polyorder, deriv=1, delta=dt)
        a[:, i] = savgol_filter(p[:, i], pos_window, polyorder, deriv=2, delta=dt)

    R_bw = _R.from_quat(merged[["qx", "qy", "qz", "qw"]].to_numpy(np.float64))
    g_world = np.array([0.0, 0.0, -gravity])
    f_body = R_bw.inv().apply(a - g_world)                   # world→body 比力
    a_meas = merged[["ax", "ay", "az"]].to_numpy(np.float64)
    b_a = a_meas - f_body

    bsw = sm.get("bias_smooth_window")
    if bsw:
        bsw = int(bsw) | 1
        for i in range(3):
            b_a[:, i] = savgol_filter(b_a[:, i], bsw, 1)

    out = merged.copy()
    out[["vx", "vy", "vz"]] = v
    out[["bax", "bay", "baz"]] = b_a
    # a_world（位置二次微分）僅供 TUM-VI 的 motion-tail mask 用；
    # ⚠ 絕不改動 bax/bay/baz（b_ref）的計算（並行分析 agent 依賴它）。
    out[["awx", "awy", "awz"]] = a
    return out


def align_and_merge(imu_df: pd.DataFrame, gt_df: pd.DataFrame, tol_ms: float) -> pd.DataFrame:
    """nearest merge_asof，容忍 tol_ms 毫秒；dropna 後 reset。"""
    merged = pd.merge_asof(
        imu_df.sort_values("ts"), gt_df.sort_values("ts"),
        on="ts", direction="nearest", tolerance=int(tol_ms * 1e6))
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def _merged_for(adapter: DatasetAdapter, seq: str, cfg: dict) -> pd.DataFrame:
    raw = adapter.load_sequence(seq)
    merged = align_and_merge(raw["imu"], raw["gt"], cfg.get("merge_tol_ms", 10))
    return adapter.derive_labels(merged, cfg)        # EuRoC: no-op；TUM-VI: 導出 v/b_a


# ──────────────────────────────────────────────────────────────────────────
# §6.1  gap-clean windowing（強制；EuRoC 連續、為 TUM-VI 鋪路）
# ──────────────────────────────────────────────────────────────────────────
def _interp_small_gaps(merged: pd.DataFrame, imu_freq: int, small_gap_sec: float) -> pd.DataFrame:
    nominal_dt = 1.0 / imu_freq
    ts = merged["ts"].to_numpy(np.int64)
    if len(ts) < 4:
        return merged
    dt = np.diff(ts) / 1e9
    small = (dt > 1.5 * nominal_dt) & (dt <= small_gap_sec)
    if not small.any():
        return merged
    out = merged.copy()
    gt_cols = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "bax", "bay", "baz",
               "awx", "awy", "awz"]
    t = (ts - ts[0]) / 1e9
    for c in gt_cols:
        if c in out.columns:
            s = pd.Series(out[c].to_numpy(np.float64), index=t)
            out[c] = s.interpolate(method="cubic").to_numpy()
    return out


def make_windows_clean(merged: pd.DataFrame, window: int, stride: int,
                       imu_freq: int, gap_tol_factor: float, small_gap_sec: float,
                       motion_tail_quantile: float = None):
    """切窗並丟棄橫跨 GT gap 的 window。回傳 (X, y, win_starts)，含 assert。

    motion_tail_quantile（TUM-VI only）：若提供，額外丟棄「窗內 mean|a_world| 超過該
    分位（quantile，預設口徑 0.95 → top 5%）」的窗。門檻**逐序列一次性**從 gap-clean
    後留存的窗的 motion 分位算出，對所有方法/所有 seed 完全一致（與方法/seed 無關）。
    需 merged 含 awx/awy/awz（只有 TUM-VI 的 derive_mocap_labels 會存）；EuRoC 無此欄、
    且 EuRoC config 不提供此鍵 → 此 mask 完全不啟用，windowing 行為 byte-identical。
    """
    merged = _interp_small_gaps(merged, imu_freq, small_gap_sec)
    feats = merged[FEAT_COLS].to_numpy(np.float64)
    labels = merged[LABEL_COLS].to_numpy(np.float64)
    ts = merged["ts"].to_numpy(np.int64)
    nominal_dt = 1.0 / imu_freq
    max_allow = gap_tol_factor * nominal_dt

    # motion-tail mask 只在「明確要求此鍵」且「merged 真有 a_world」時啟用（TUM-VI）。
    use_motion_tail = (motion_tail_quantile is not None
                       and all(c in merged.columns for c in ("awx", "awy", "awz")))
    aw_norm = None
    if use_motion_tail:
        aw = merged[["awx", "awy", "awz"]].to_numpy(np.float64)
        aw_norm = np.linalg.norm(aw, axis=1)            # |a_world| per raw frame

    X_list, y_list, starts, motion_list = [], [], [], []
    n = len(feats)
    for start in range(0, n - window, stride):
        end = start + window
        seg_ts = ts[start:end]
        max_dt = float(np.max(np.diff(seg_ts)) / 1e9) if window > 1 else 0.0
        if max_dt > max_allow:
            continue
        X_list.append(feats[start:end])
        y_list.append(labels[start:end].mean(axis=0))
        starts.append(start)
        if use_motion_tail:
            motion_list.append(float(aw_norm[start:end].mean()))   # 窗內 mean|a_world|

    if not X_list:
        raise ValueError("make_windows_clean: 無有效 window（全被 gap 丟棄？）")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    win_starts = np.asarray(starts, dtype=np.int64)

    # motion-tail mask：逐序列一次性閾值（quantile），方法/seed 無關。
    if use_motion_tail and len(motion_list) > 1:
        motion = np.asarray(motion_list, dtype=np.float64)
        thr = float(np.quantile(motion, motion_tail_quantile))
        keep = motion <= thr
        if keep.any():
            X = X[keep]; y = y[keep]; win_starts = win_starts[keep]

    for s in win_starts:
        seg_ts = ts[s:s + window]
        assert float(np.max(np.diff(seg_ts)) / 1e9) < max_allow, \
            f"gap-clean assert failed at window start {s}"

    return X, y, win_starts


# ──────────────────────────────────────────────────────────────────────────
# config helpers（容納 full.yaml 的 nested regimes 與扁平 fallback）
# ──────────────────────────────────────────────────────────────────────────
def _regimeA_split(cfg):
    return cfg.get("regimes", {}).get("A", {}).get("split", cfg.get("split_ratio", [0.70, 0.15, 0.15]))


def _regimeB_pooled(cfg):
    return cfg.get("regimes", {}).get("B", {}).get(
        "pooled_train", cfg.get("phaseB_pooled_train", ["MH_01", "MH_02", "MH_03", "MH_05"]))


def _win_cfg(cfg):
    # motion_tail_mask_quantile：只有 TUM-VI config 會帶此鍵；EuRoC（full.yaml/smoke.yaml）
    # 沒有此鍵 → 回傳 None → make_windows_clean 完全不啟用 motion-tail mask（行為不變）。
    mtq = cfg.get("motion_tail_mask_quantile")
    if mtq is not None and str(cfg.get("dataset", "euroc")).lower() not in ("tumvi_room", "tumvi"):
        mtq = None                                       # 防呆：非 TUM-VI dataset 一律不啟用
    return (cfg["window"], cfg["stride"], cfg.get("imu_freq", 200),
            cfg.get("gap_tol_factor", 2.0), cfg.get("small_gap_sec", 0.1), mtq)


def _standardize(X, scaler):
    n, w, c = X.shape
    return scaler.transform(X.reshape(-1, c)).reshape(n, w, c).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# §6.2  regime-A loader（單序列時序 70/15/15）
# ──────────────────────────────────────────────────────────────────────────
def prepare_regime_a(seq: str, cfg: dict, adapter: DatasetAdapter = None) -> SeqData:
    adapter = adapter or get_adapter(cfg)
    merged = _merged_for(adapter, seq, cfg)
    X, y, win_starts = make_windows_clean(merged, *_win_cfg(cfg))

    n = len(X)
    r = _regimeA_split(cfg)
    n_train = int(n * r[0]); n_val = int(n * r[1])
    sl_tr = slice(0, n_train); sl_va = slice(n_train, n_train + n_val); sl_te = slice(n_train + n_val, n)

    scaler = StandardScaler().fit(X[sl_tr].reshape(-1, 6))
    return SeqData(
        seq=seq, merged=merged,
        X_train=_standardize(X[sl_tr], scaler), y_train=y[sl_tr],
        X_val=_standardize(X[sl_va], scaler), y_val=y[sl_va],
        X_test=_standardize(X[sl_te], scaler), y_test=y[sl_te],
        test_win_starts=win_starts[sl_te],
        scaler=scaler, win_starts_all=win_starts,
    )


# ──────────────────────────────────────────────────────────────────────────
# §6.3  regime-B pooled loader（跨序列 head 池化；constant = global mean）
# ──────────────────────────────────────────────────────────────────────────
def prepare_regime_b_pooled(cfg: dict, adapter: DatasetAdapter = None) -> PooledData:
    adapter = adapter or get_adapter(cfg)
    seqs = _regimeB_pooled(cfg)
    r = _regimeA_split(cfg)

    pooled_Xtr, pooled_ytr = [], []
    pooled_Xva, pooled_yva = [], []
    per_seq_eval = {}
    per_seq_train_mean = {}

    raw_cache = {}
    for seq in seqs:
        merged = _merged_for(adapter, seq, cfg)
        X, y, win_starts = make_windows_clean(merged, *_win_cfg(cfg))
        n = len(X)
        n_train = int(n * r[0]); n_val = int(n * r[1])
        raw_cache[seq] = (X, y, win_starts, merged, n_train, n_val)
        pooled_Xtr.append(X[:n_train]); pooled_ytr.append(y[:n_train])
        pooled_Xva.append(X[n_train:n_train + n_val]); pooled_yva.append(y[n_train:n_train + n_val])
        per_seq_train_mean[seq] = y[:n_train].mean(axis=0)

    Xtr = np.concatenate(pooled_Xtr, axis=0)
    ytr = np.concatenate(pooled_ytr, axis=0)
    Xva = np.concatenate(pooled_Xva, axis=0)
    yva = np.concatenate(pooled_yva, axis=0)

    scaler = StandardScaler().fit(Xtr.reshape(-1, 6))

    for seq in seqs:
        X, y, win_starts, merged, n_train, n_val = raw_cache[seq]
        sl_te = slice(n_train + n_val, len(X))
        per_seq_eval[seq] = {
            "merged": merged,
            "X_test": _standardize(X[sl_te], scaler),
            "y_test": y[sl_te],
            "test_win_starts": win_starts[sl_te],
        }

    return PooledData(
        train_seqs=list(seqs),
        X_train=_standardize(Xtr, scaler), y_train=ytr,
        X_val=_standardize(Xva, scaler), y_val=yva,
        scaler=scaler, pooled_y_train_raw=ytr,
        per_seq_eval=per_seq_eval, per_seq_train_mean=per_seq_train_mean,
    )

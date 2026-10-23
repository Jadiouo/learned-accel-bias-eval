# -*- coding: utf-8 -*-
"""
聚合與判決（SDD §9 / §10）。

讀 results/runs/*/ + runs_index.csv →
  metrics_long.csv / metrics_wide.csv（tidy；無絕對路徑）
  SUMMARY.md（架構家族 rollup + L1 + L2(§9 收緊) + L3(10-seed) + Bookend(KF/PF/const) + counterexample）
  plots/*.png

L2（§9 收緊判準，本套件關鍵改動）：
  separated_iabe = (p_IABE<α) 且 (gap/σ_IABE ≥ min_gap_over_sigma=2.0)
  separated_rmse = (p_RMSE<α)
  robust_inversion = separated_iabe 且 separated_rmse 且 sign(ΔIABE)≠sign(ΔRMSE)
  L2 PASS = 存在 ≥1 robust_inversion（且該 inversion 在 IABE 上分得開）；否則 FAIL。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:                                   # pragma: no cover
    _wilcoxon = None

from . import common, metrics as M, band_decomp

UNIT = {
    "iabe_rot": "m", "iabe_body": "m", "gyro_ate": "m",
    "slie_5": "m", "slie_10": "m", "slie_15": "m",
    "bias_rmse_x": "m_s2", "bias_rmse_y": "m_s2", "bias_rmse_z": "m_s2", "bias_rmse_avg": "m_s2",
    "ate_reduction": "percent",
}

L1_RATIO_PASS, L1_RATIO_WEAK, L1_CORR_PASS = 10.0, 3.0, 0.3
L3_FLIP_PASS, L3_FLIP_WEAK = 0.5, 1e-9


# ──────────────────────────────────────────────────────────────────────────
# TUM-VI 事前註冊分流鎖（唯讀；只在 dataset=tumvi 時讀取/套用）
# ──────────────────────────────────────────────────────────────────────────
def _is_tumvi(cfg) -> bool:
    return str(cfg.get("dataset", "")).lower() in ("tumvi_room", "tumvi")


def load_prereg_lock(cfg):
    """讀 results/<tumvi>/preregistration.json（唯讀）。失敗回 None（不崩，僅退回全序列）。"""
    if not _is_tumvi(cfg):
        return None
    root = cfg.get("_results_dir_abs", "")
    path = os.path.join(root, "preregistration.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:                                   # pragma: no cover
        return None


def _render_prereg_block(lock):
    """把鎖渲染成 SUMMARY 頂部的「PRE-REGISTERED ANALYSIS SPLIT」區塊（行列表）。"""
    L = []
    ts = lock.get("created_utc", "?")
    L.append(f"## 🔒 PRE-REGISTERED ANALYSIS SPLIT（locked {ts}, before --full）")
    L.append("> 此分流規則在 `--full` 跑完之前就鎖死（方法論誠信鎖,防事後挑格子）。"
             "**full 跑完未改任何門檻或格子清單**;本 SUMMARY 純套用此鎖。")
    L.append(f"- clean 判準:{lock.get('clean_rule', '?')}")
    cells = lock.get("clean_cells", [])
    cell_str = " · ".join(f"({s},{ax})" for s, ax in cells)
    L.append(f"- **乾淨格子 {lock.get('n_clean_cells', len(cells))}/{lock.get('n_total_cells', 18)}**:{cell_str}")
    L.append(f"- **main_sequences（含 ≥1 乾淨格子,進主張）= {lock.get('main_sequences')}**"
             f"  ·  room4(3/3)扛主力,room1/3/5(各 1/3 軸乾淨,另 2 軸已污染)為輔 caveat")
    L.append(f"- **appendix_sequences（0 乾淨格子,只進附錄 scope evidence,不進任何主張）= "
             f"{lock.get('appendix_sequences')}**")
    L.append(f"- per-axis caveat:{lock.get('per_axis_caveat', '?')}")
    L.append(f"- L1/L3 判準:{lock.get('L1_L3', '?')}")
    L.append(f"- cross-dataset ordering:{lock.get('cross_dataset_ordering', '?')}")
    L.append(f"- δθ 措辭（鎖）:{lock.get('delta_theta_wording', '?')}")
    L.append(f"- L2:{lock.get('L2', 'N/A')}")
    L.append(f"- tripod framing:{lock.get('tripod_framing', '?')}")
    L.append("")
    return L


# 鎖之乾淨格子（per-axis 對照表用；亦由 load_prereg_lock 動態取得,此為 fallback 排版）
def _clean_cell_table(lock):
    cells = lock.get("clean_cells", [])
    by_seq = {}
    for s, ax in cells:
        by_seq.setdefault(s, []).append(ax)
    L = ["| 序列 | 乾淨軸 | 角色 |", "|---|---|---|"]
    role = {"room4": "主力(3/3)", "room1": "caveat(1/3,x；y/z 污染)",
            "room3": "caveat(1/3,x；y/z 污染)", "room5": "caveat(1/3,z；x/y 污染)"}
    for s in lock.get("main_sequences", []):
        axes = ", ".join(sorted(by_seq.get(s, []))) or "—"
        L.append(f"| {s} | {axes} | {role.get(s, '')} |")
    return L


# ──────────────────────────────────────────────────────────────────────────
# load
# ──────────────────────────────────────────────────────────────────────────
def load_runs(cfg) -> pd.DataFrame:
    root = cfg["_results_dir_abs"]
    idx_path = os.path.join(root, "runs_index.csv")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"{idx_path} 不存在；請先跑 --full。")
    idx = pd.read_csv(idx_path)
    rows = []
    for _, r in idx.iterrows():
        if str(r.get("status")) != "ok":
            continue
        mpath = os.path.join(root, "runs", r["run_id"], "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            evblock = json.load(f)
        for eval_seq, mt in evblock.items():
            for metric, value in (mt or {}).items():
                if value is None:
                    continue
                rows.append({"run_id": r["run_id"], "method": r["method"],
                             "family": r.get("family"), "regime": r["regime"],
                             "seed": int(r["seed"]), "eval_sequence": eval_seq,
                             "metric": metric, "value": float(value),
                             "unit": UNIT.get(metric, "dimensionless")})
    return pd.DataFrame(rows)


def write_long_wide(df, root):
    long_cols = ["run_id", "eval_sequence", "metric", "value", "unit"]
    df[long_cols].to_csv(os.path.join(root, "metrics_long.csv"), index=False)
    wide = df.pivot_table(index=["run_id", "eval_sequence"], columns="metric",
                          values="value", aggfunc="first").reset_index()
    wide.to_csv(os.path.join(root, "metrics_wide.csv"), index=False)
    return wide


def _pivot(df, regime):
    sub = df[df["regime"] == regime]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index=["run_id", "eval_sequence", "method", "family", "seed"],
                           columns="metric", values="value", aggfunc="first").reset_index()


def _dl_methods(cfg):
    from .registry import is_dl
    return set(m for m in cfg["methods"] if is_dl(m))


# ──────────────────────────────────────────────────────────────────────────
# L1 — Isolation
# ──────────────────────────────────────────────────────────────────────────
def analyze_L1(df, sequences=None):
    """sequences=None → 不篩(EuRoC 行為完全不變)；給 list → 只在這些 eval_sequence 上算。"""
    out = {}
    for regime in ["A", "B"]:
        w = _pivot(df, regime)
        if w.empty or "iabe_rot" not in w or "gyro_ate" not in w:
            out[regime] = None; continue
        if sequences is not None:
            w = w[w["eval_sequence"].isin(sequences)]
            if w.empty:
                out[regime] = None; continue
        w = w.dropna(subset=["iabe_rot", "gyro_ate"]); w = w[w["iabe_rot"] > 0]
        if w.empty:
            out[regime] = None; continue
        sp = abs(M._spearman(w["gyro_ate"].to_numpy(), w["bias_rmse_avg"].to_numpy())) \
            if "bias_rmse_avg" in w else float("nan")
        out[regime] = {"median_gyro_over_iabe": float(np.median(w["gyro_ate"] / w["iabe_rot"])),
                       "gyro_ate_abs_median_m": float(np.median(w["gyro_ate"])),
                       "abs_spearman_gyroate_vs_biasrmse": sp, "n": int(len(w))}
    return out


def _verdict_L1(L1):
    ref = L1.get("B") or L1.get("A")
    if not ref:
        return "FAIL", "無資料"
    r, sp = ref["median_gyro_over_iabe"], ref["abs_spearman_gyroate_vs_biasrmse"]
    if r >= L1_RATIO_PASS and sp < L1_CORR_PASS:
        return "PASS", f"ratio={r:.1f}≥{L1_RATIO_PASS}, |ρ|={sp:.2f}<{L1_CORR_PASS}"
    if r >= L1_RATIO_WEAK:
        return "WEAK", f"ratio={r:.1f} 但 |ρ|={sp:.2f}"
    return "FAIL", f"ratio={r:.1f}<{L1_RATIO_WEAK}"


# ──────────────────────────────────────────────────────────────────────────
# L2 — §9 收緊 robust inversion
# ──────────────────────────────────────────────────────────────────────────
def _paired_p(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    if _wilcoxon is None or len(d) < 1 or np.allclose(d, 0.0):
        return 1.0
    try:
        return float(_wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return 1.0


def _wilcoxon_p_floor(n):
    return 1.0 if n < 1 else min(1.0, 2.0 / (2 ** n))


def analyze_L2(df, cfg, regime="B"):
    a = cfg.get("analysis", {}).get("l2_robust", {})
    alpha = a.get("alpha", 0.05)
    min_gap = a.get("min_gap_over_sigma", 2.0)
    dlm = _dl_methods(cfg)
    w = _pivot(df, regime)
    if w.empty or "iabe_rot" not in w or "bias_rmse_avg" not in w:
        return None
    methods = sorted([m for m in w["method"].unique() if m in dlm])
    per_seq = {}
    n_robust = n_dir = n_near = 0
    n_seeds = []
    for seq, g in w.groupby("eval_sequence"):
        g = g[g["method"].isin(methods)]
        by = {m: {int(r.seed): (float(r.iabe_rot), float(r.bias_rmse_avg)) for r in g[g["method"] == m].itertuples()}
              for m in methods}
        pairs = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                mi, mj = methods[i], methods[j]
                seeds = sorted(set(by[mi]) & set(by[mj]))
                if len(seeds) < 2:
                    continue
                n_seeds.append(len(seeds))
                ai = np.array([by[mi][s][0] for s in seeds]); aj = np.array([by[mj][s][0] for s in seeds])
                ri = np.array([by[mi][s][1] for s in seeds]); rj = np.array([by[mj][s][1] for s in seeds])
                dI, dR = ai - aj, ri - rj
                mdI, mdR = float(dI.mean()), float(dR.mean())
                sdI = float(dI.std(ddof=1)) if len(dI) > 1 else 0.0
                gap_std = abs(mdI) / (sdI + 1e-12)
                p_iabe, p_rmse = _paired_p(ai, aj), _paired_p(ri, rj)
                separated_iabe = (p_iabe < alpha) and (gap_std >= min_gap)
                separated_rmse = (p_rmse < alpha)
                sign_opp = (np.sign(mdI) != np.sign(mdR)) and mdI != 0 and mdR != 0
                robust = bool(separated_iabe and separated_rmse and sign_opp)
                near_tie = gap_std < min_gap
                if sign_opp:
                    n_dir += 1
                if sign_opp and near_tie:
                    n_near += 1
                if robust:
                    n_robust += 1
                cls = ("robust" if robust else
                       ("dir-only(near-tie)" if (sign_opp and near_tie) else
                        ("dir-only" if sign_opp else "concordant")))
                pairs.append({"models": [mi, mj], "n_seeds": len(seeds), "class": cls,
                              "mean_dIABE": mdI, "mean_dRMSE": mdR, "gap_over_sigma": float(gap_std),
                              "p_IABE": p_iabe, "p_RMSE": p_rmse,
                              "separated_iabe": bool(separated_iabe), "separated_rmse": bool(separated_rmse),
                              "robust_inversion": robust})
        per_seq[seq] = {"pairs": pairs,
                        "n_robust": sum(p["robust_inversion"] for p in pairs),
                        "n_directional": sum(p["class"].startswith("dir") for p in pairs)}
    if not per_seq:
        return None
    n_typ = int(np.median(n_seeds)) if n_seeds else 0
    return {"regime": regime, "alpha": alpha, "min_gap_over_sigma": min_gap, "methods": methods,
            "n_seeds_typical": n_typ, "wilcoxon_min_p_floor": _wilcoxon_p_floor(n_typ),
            "per_seq": per_seq, "total_robust_inversions": n_robust,
            "total_directional": n_dir, "total_near_tie_directional": n_near}


def _verdict_L2(L2):
    if not L2:
        return "FAIL", "無 regime-B 資料"
    nr = L2["total_robust_inversions"]
    floor = L2["wilcoxon_min_p_floor"]
    powered = floor <= L2["alpha"]
    if not powered:
        return "DEFERRED", (f"n_seeds≈{L2['n_seeds_typical']} → Wilcoxon p 下限 {floor:.3f} > α；"
                            f"power 不足（需 ≥8 seeds）。robust={nr}")
    if nr >= 1:
        return "PASS", (f"{nr} 個 robust inversion（IABE 分得開 gap/σ≥{L2['min_gap_over_sigma']} "
                        f"且 RMSE 顯著、方向相反）→ 1/ω² 加權有實證價值")
    return "FAIL", (f"robust inversion = 0（dir-only={L2['total_directional']}，其中 near-tie="
                    f"{L2['total_near_tie_directional']}）→ §4.3 的 1/ω² 降為純理論論證；論文靠 L1+L3")


# ──────────────────────────────────────────────────────────────────────────
# L3 — Reproducibility（10-seed winner stability）
# ──────────────────────────────────────────────────────────────────────────
def analyze_L3(df, cfg, regime="A", sequences=None):
    """sequences=None → 不篩(EuRoC 行為完全不變)；給 list → 只在這些 eval_sequence 上算。"""
    dlm = _dl_methods(cfg)
    w = _pivot(df, regime)
    if w.empty or "iabe_rot" not in w:
        return None
    w = w[w["method"].isin(dlm)]
    if sequences is not None:
        w = w[w["eval_sequence"].isin(sequences)]
        if w.empty:
            return None
    per_seq = {}; flips = 0; n_seq = 0
    for seq, g in w.groupby("eval_sequence"):
        winners = {}
        for seed, gs in g.groupby("seed"):
            agg = gs.groupby("method")["iabe_rot"].mean()
            if len(agg):
                winners[int(seed)] = agg.idxmin()
        if not winners:
            continue
        n_seq += 1
        uniq = set(winners.values()); flip = len(uniq) > 1; flips += int(flip)
        mean_rank = g.groupby("method")["iabe_rot"].mean().sort_values()
        mean_winner = mean_rank.index[0]
        agree = float(np.mean([v == mean_winner for v in winners.values()]))
        gap_std = float("nan"); near_tie = False
        if len(mean_rank) >= 2:
            top2 = list(mean_rank.index[:2])
            gap = float(mean_rank.iloc[1] - mean_rank.iloc[0])
            seed_std = float(np.mean([g[g["method"] == m]["iabe_rot"].std(ddof=1) for m in top2]))
            gap_std = gap / (seed_std + 1e-12); near_tie = gap_std < 1.0
        per_seq[seq] = {"winner_by_seed": winners, "n_distinct_winners": len(uniq),
                        "mean_winner": mean_winner, "single_seed_agree_rate": agree,
                        "gap_over_seedstd": gap_std, "near_tie": bool(near_tie)}
    if not n_seq:
        return None
    return {"per_seq": per_seq, "flip_rate": flips / n_seq, "n_eval_seqs": n_seq}


def _verdict_L3(L3):
    if not L3:
        return "FAIL", "無 regime-A 資料"
    fr = L3["flip_rate"]
    if fr >= L3_FLIP_PASS:
        return "PASS", f"flip_rate={fr:.2f}≥{L3_FLIP_PASS}（單 seed 不可靠，已證明）"
    if fr > L3_FLIP_WEAK:
        return "WEAK", f"flip_rate={fr:.2f}"
    return "FAIL", "flip_rate=0"


# ──────────────────────────────────────────────────────────────────────────
# Bookend（KF / PF / constant；PF≈KF 勿 oversell）
# ──────────────────────────────────────────────────────────────────────────
def analyze_bookend(df, cfg):
    dlm = _dl_methods(cfg)
    const = {"constant_pooled", "constant_perseq"}
    out = {}
    for regime in ["A", "B"]:
        w = _pivot(df, regime)
        if w.empty or "iabe_rot" not in w:
            out[regime] = None; continue
        groups = {"kf": w[w["method"] == "kf"]["iabe_rot"],
                  "pf": w[w["method"] == "pf"]["iabe_rot"],
                  "constant": w[w["method"].isin(const)]["iabe_rot"],
                  "DL": w[w["method"].isin(dlm)]["iabe_rot"]}
        rec = {}
        for k, s in groups.items():
            s = s.dropna()
            rec[k] = None if s.empty else {"min": float(s.min()), "median": float(s.median()),
                                           "max": float(s.max()), "n": int(len(s))}
        recon = []
        if rec.get("kf") and rec.get("constant"):
            rel = abs(rec["kf"]["median"] - rec["constant"]["median"]) / (abs(rec["constant"]["median"]) + 1e-12)
            recon.append("kf≈constant (unobservable)" if rel < 0.05 else f"kf 偏 constant {rel*100:.1f}%")
        if rec.get("pf") and rec.get("kf"):
            rel = abs(rec["pf"]["median"] - rec["kf"]["median"]) / (abs(rec["kf"]["median"]) + 1e-12)
            recon.append("pf≈kf (非線性化非主因)" if rel < 0.10 else f"pf 偏 kf {rel*100:.1f}%")
        rec["reconciliation"] = "；".join(recon) if recon else "n/a"
        out[regime] = rec
    return out


# ──────────────────────────────────────────────────────────────────────────
# 架構家族 rollup（結構性陳述，非排行榜）
# ──────────────────────────────────────────────────────────────────────────
def analyze_family_rollup(df, regime="B"):
    w = _pivot(df, regime)
    if w.empty or "iabe_rot" not in w:
        return None
    out = {}
    for fam, g in w.groupby("family"):
        s = g["iabe_rot"].dropna()
        if s.empty:
            continue
        out[fam] = {"min": float(s.min()), "median": float(s.median()),
                    "max": float(s.max()), "n": int(len(s)),
                    "methods": sorted(g["method"].unique().tolist())}
    return out


# ──────────────────────────────────────────────────────────────────────────
# SUMMARY.md
# ──────────────────────────────────────────────────────────────────────────
def write_summary(cfg, df, L1, L2, L3, bookend, rollup, plots_made, band=None, cross=None,
                  lock=None, L1_appendix=None, L3_appendix=None):
    root = cfg["_results_dir_abs"]
    is_tumvi = str(cfg.get("dataset", "")).lower() in ("tumvi_room", "tumvi")
    v1, r1 = _verdict_L1(L1); v2, r2 = _verdict_L2(L2); v3, r3 = _verdict_L3(L3)
    n_runs = df["run_id"].nunique() if not df.empty else 0
    L = []
    L.append("# EuRoC Full Matrix — Main Result & Tripod Verdict\n")
    # ── TUM-VI：SUMMARY 最上方嵌入事前註冊分流鎖 ──
    if is_tumvi and lock:
        L.extend(_render_prereg_block(lock))
    if is_tumvi:
        L.append("## ★ TUM-VI room framing（label 品質與可檢定範圍的誠實揭露）")
        L.append("- 資料：TUM-VI **room1–6**，低動態手持室內 + OptiTrack MoCap（只 p/q）；"
                 "reference accel bias 由位置二次微分導出（§4.3）。**不宣稱 high-dynamic**。")
        L.append("- **L2：MoCap label 下不可檢定**——低頻被姿態相關共模項污染（R(q) 與低頻運動殘差耦合），"
                 "MoCap-derived label 無法乾淨分離 trajectory-weighting。**本資料集不對 L2 下任何 PASS/FAIL，待 OKVIS**。")
        L.append("- **label 品質逐軸不同**：水平軸（x/y）吃姿態 + 運動洩漏，z 較乾淨；"
                 "**L1/L3 結論建議逐軸看**。L1/L3 對共模污染較免疫（看 isolation 與 seed 穩定性，"
                 "非低頻能量絕對值），但仍誠實揭露此局限。")
        L.append("- **跨資料集 ordering 只 claim ordering、非數值**：BMI160（TUM-VI）vs ADIS16448（EuRoC）"
                 "bias 量級差 1–2 數量級，數值不可直接比；可比的是 IABE 方法**排序**是否一致（Kendall/Spearman）。"
                 "此為候選新腿。\n")
    L.append("> Production / GitHub-ready；10 seeds；L2 用 §9 收緊 robust-inversion；"
             "承接 SDD_EuRoC_MVP_11（MVP 凍結保存於 `euroc_mvp/`）。\n")
    L.append(f"- git_hash: `{common.git_hash()}`  ·  generated: {common.now_iso()}")
    L.append(f"- config: `{os.path.basename(cfg['_config_path'])}`  ·  methods={len(cfg['methods'])}  ·  "
             f"seeds={len(cfg['seeds'])}  ·  total runs(ok)={n_runs}\n")

    if rollup:
        L.append("## 架構家族 rollup（regime B；結構性陳述，非排行榜）")
        L.append("> protocol 把各 family 放進 IABE 區間 X、bookend 在區間外——呈現結構，不選冠軍。")
        for fam, d in sorted(rollup.items()):
            L.append(f"- **{fam}**: iabe_rot ∈ [{d['min']:.3f}, {d['max']:.3f}] m "
                     f"(median {d['median']:.3f}, n={d['n']})  ·  {d['methods']}")
        L.append("")

    L.append("## L1 — Isolation (IABE vs gyro-attitude ATE)")
    if is_tumvi and lock:
        L.append(f"> **正文判決只建立在 main_sequences {lock.get('main_sequences')}**（含 ≥1 乾淨格子）;"
                 "room2/room6 在下方附錄「label-unusable」表,不進主張。逐軸 label 強弱見對照表。")
    for reg in ["A", "B"]:
        d = L1.get(reg)
        if d:
            L.append(f"- regime {reg}: median(gyro_ate/iabe_rot) = **{d['median_gyro_over_iabe']:.1f}×**  ·  "
                     f"gyro_ate 絕對中位 ≈ {d['gyro_ate_abs_median_m']:.1f} m  ·  "
                     f"|Spearman(gyro_ate, bias_rmse)| = **{d['abs_spearman_gyroate_vs_biasrmse']:.2f}**  (n={d['n']})")
    L.append(f"- **判決：{v1}** — {r1}\n")
    if is_tumvi and lock:
        L.append("### per-axis 乾淨格子對照（PREREGISTRATION,讀者對照用）")
        L.append("> 水平軸(x/y)label 較弱、**y 最弱**;z 較乾淨。下表為鎖定之乾淨格子。")
        L.extend(_clean_cell_table(lock))
        L.append("")

    L.append("## L2 — Trajectory Weighting (§9 收緊 robust inversion)  [主看 regime B]")
    L.append("> separated_iabe = p<α 且 gap/σ≥min；robust = separated_iabe 且 separated_rmse 且方向相反。")
    if L2:
        L.append(f"- methods={L2['methods']}  ·  n_seeds≈{L2['n_seeds_typical']}  ·  α={L2['alpha']}  ·  "
                 f"min_gap/σ={L2['min_gap_over_sigma']}  ·  Wilcoxon p 下限={L2['wilcoxon_min_p_floor']:.3f}")
        for seq, d in L2["per_seq"].items():
            L.append(f"- **{seq}**: robust={d['n_robust']} · directional={d['n_directional']}")
            for p in d["pairs"]:
                if p["class"] != "concordant":
                    L.append(f"    - {p['models'][0]} vs {p['models'][1]} [{p['class']}]: "
                             f"ΔIABE={p['mean_dIABE']:+.4f} (p={p['p_IABE']:.3f}, gap/σ={p['gap_over_sigma']:.2f}) · "
                             f"ΔRMSE={p['mean_dRMSE']:+.4f} (p={p['p_RMSE']:.3f})")
        L.append(f"- **total robust inversions = {L2['total_robust_inversions']}**  ·  "
                 f"directional={L2['total_directional']}（near-tie={L2['total_near_tie_directional']}）")
    if is_tumvi:
        L.append("- **判決：不可檢定（N/A）** — MoCap label 下低頻被姿態相關共模項污染，"
                 "本資料集不對 L2 下 PASS/FAIL，待 OKVIS（上方 framing）。\n")
    else:
        L.append(f"- **判決：{v2}** — {r2}\n")

    L.append("## L3 — Reproducibility (10-seed winner stability)  [主看 regime A]")
    if is_tumvi and lock:
        L.append(f"> **flip rate 正文判決只算 main_sequences {lock.get('main_sequences')}**;"
                 "room2/room6 在附錄表,不進主張。")
    if L3:
        for seq, d in L3["per_seq"].items():
            wbs = ", ".join(f"s{k}:{v}" for k, v in sorted(d["winner_by_seed"].items()))
            nt = f" · gap/seed-std={d['gap_over_seedstd']:.2f}{'（near-tie★）' if d['near_tie'] else ''}"
            L.append(f"- {seq}: winners [{wbs}] · distinct={d['n_distinct_winners']} · "
                     f"mean-winner={d['mean_winner']} · agree={d['single_seed_agree_rate']:.2f}{nt}")
        L.append(f"- winner flip rate = **{L3['flip_rate']:.2f}** over {L3['n_eval_seqs']} eval seqs")
    L.append(f"- **判決：{v3}** — {r3}\n")

    # ── TUM-VI 附錄：label-unusable（scope evidence,不進主張）──
    if is_tumvi and lock and (L1_appendix is not None or L3_appendix is not None):
        L.append("## 📎 附錄 — appendix_sequences「label-unusable」(scope evidence,不進任何主張)")
        L.append(f"> {lock.get('appendix_sequences')} = 0 乾淨格子,label 被高動態運動洩漏主導。"
                 "以下數字僅供 scope/透明度,**正文 L1 ratio/ρ、L3 flip rate 不基於這些**。")
        if L1_appendix:
            L.append("### L1（附錄序列,逐 regime）")
            for reg in ["A", "B"]:
                d = (L1_appendix or {}).get(reg)
                if d:
                    L.append(f"- regime {reg}: median(gyro_ate/iabe_rot) = {d['median_gyro_over_iabe']:.1f}×  ·  "
                             f"|Spearman(gyro_ate, bias_rmse)| = {d['abs_spearman_gyroate_vs_biasrmse']:.2f}  (n={d['n']})")
        if L3_appendix:
            L.append("### L3（附錄序列,逐序列）")
            for seq, d in L3_appendix["per_seq"].items():
                wbs = ", ".join(f"s{k}:{v}" for k, v in sorted(d["winner_by_seed"].items()))
                L.append(f"- {seq}: winners [{wbs}] · distinct={d['n_distinct_winners']} · "
                         f"mean-winner={d['mean_winner']} · agree={d['single_seed_agree_rate']:.2f}")
            L.append(f"- (附錄) winner flip rate = {L3_appendix['flip_rate']:.2f} over "
                     f"{L3_appendix['n_eval_seqs']} eval seqs — **不進主張**")
        L.append("")

    L.append("## Bookend (KF / PF / constant 在 IABE 上的位置)")
    for reg in ["A", "B"]:
        d = (bookend or {}).get(reg)
        if not d:
            continue
        parts = []
        for k in ("kf", "pf", "constant", "DL"):
            rr = d.get(k)
            if rr:
                parts.append(f"{k}: [{rr['min']:.3f},{rr['max']:.3f}] (med {rr['median']:.3f})")
        L.append(f"- regime {reg} — " + "  ·  ".join(parts))
        L.append(f"    - 對帳：**{d.get('reconciliation')}**")
    L.append("- ★ KF≈constant（不可觀測）已知；**PF≈KF，勿 oversell**（確認失敗非線性化 artifact）。\n")

    L.append("## ★ Bias-error 頻帶分解（§8；讓 L2 有機制）")
    if band:
        b = band
        L.append(f"- ε(t)=b_ref−b̂ 的 PSD（Welch），fs={b['fs_hz']:.0f}Hz（=imu_freq/stride），"
                 f"bands={b['bands_hz']}；regime {b['regime']}，{b['n_methods']} 方法 / {b['n_rows']} runs")
        L.append(f"- **(a) 各方法低頻能量分不分得開**：跨方法低頻能量 CV = **{b['low_energy_cross_method_cv']:.3f}**"
                 f"（越大越分得開；接近 0 → 方法低頻殘差太像 → IABE 無從區分）")
        L.append(f"- **(b) corr(iabe_rot, 低頻能量) = {b['corr_iabe_lowband']:.2f}**  vs  "
                 f"corr(iabe_rot, bias_rmse) = {b['corr_iabe_biasrmse']:.2f}  → "
                 f"{'IABE 較跟低頻能量走（與 RMSE 脫鉤）' if b['iabe_tracks_lowband_more_than_rmse'] else 'IABE 與兩者相關度相近'}")
        L.append(f"- **(c) 低頻佔比排序 vs 總能量(∝RMSE)排序**：Kendall τ = **{b['ordering_kendall_tau']:.2f}**"
                 f"（{'兩排序不同 → 有 inversion 的頻域根源' if b['orderings_differ'] else '兩排序一致 → 無頻域 inversion，呼應 L2 robust=0'}）")
        L.append("- 機制結論：" + ("低頻能量分得開且排序不同 → L2 inversion 有頻域根據（支持 1/ω² 經驗版）。"
                 if (b["orderings_differ"] and b["low_energy_cross_method_cv"] > 0.3)
                 else "方法低頻殘差能量太接近、排序一致 → 即使有方向差也分不開 → **L2 FAIL 的機制解釋**（benign 資料上方法低頻殘差相近），1/ω² 維持理論版 + 本頻帶圖佐證。"))
    else:
        L.append("- （無 series.npz 或方法不足，頻帶分解略過）")
    L.append("")

    if cross:
        L.append("## 跨資料集 ordering（EuRoC vs 本資料集；★ 只 claim ordering 非數值）")
        if is_tumvi and lock:
            L.append(f"> **TUM-VI 側只取 main_sequences {lock.get('main_sequences')} 的 IABE 方法排序**"
                     "(不拿 room2/6 髒排序去比);EuRoC 全乾淨。")
        L.append("> BMI160（TUM-VI）vs ADIS16448（EuRoC）bias 量級差 1–2 數量級，**數值不可比**；"
                 "可比的是 IABE 方法**排序**是否一致。此為候選新腿。")
        L.append(f"- IABE 方法排序 Spearman = **{cross['spearman_of_rankings']:.2f}**（n={cross['n_methods']} 方法）")
        L.append(f"- 本資料集序：{cross['order_current']}")
        L.append(f"- EuRoC 序：{cross['order_euroc']}\n")

    L.append("## Counterexample 雛形 (C5)")
    L.append("- 同數據下「single-seed + ate_reduction（飽和）」會給樂觀結論；"
             "「IABE + 10-seed + TOST」則揭露 near-tie/翻盤 → 兩者結論不同（見 L3 翻盤 + ate_reduction 飽和）。\n")

    # ── TUM-VI：δθ / 姿態洩漏措辭（鎖定固定文字,不重算）──
    if is_tumvi:
        L.append("## δθ / 姿態相關低頻成分（措辭鎖定,不重算)")
        L.append("- 姿態相關低頻成分**存在**,但來源在本資料**無法區分** (i) mocap-IMU 外參失準 "
                 "vs (ii) 加速度計刻度因數誤差 ε·g_body。")
        L.append("- δθ 僅在**低動態序列(room1/4/5)乾淨可見(~1.3° 一致)**;高動態被運動蓋掉。"
                 "**不宣稱真發散**,亦不用跨序列 δθ 的 CV 宣稱「偏向刻度誤差/發散」。\n")

    L.append("## 三腳架最終狀態")
    if is_tumvi:
        L.append(f"- L1={v1} · L2=N/A（MoCap 下不可檢定，待 OKVIS） · L3={v3}  "
                 f"→  **本資料集只服務 L1/L3 + 跨資料集 ordering**（L2 不計入）。")
        L.append("- L1/L3 結論建議逐軸看（label 品質逐軸不同;水平軸 x/y 較弱,y 最弱）；不宣稱 high-dynamic。")
        L.append("- **tripod framing**:三腳架已在 **EuRoC 站穩**;TUM-VI 是 **C4 跨感測器穩定性的輕量加分,"
                 "非腿**。乾淨格子 6/18(room4 扛大樑)→ L1/L3 證據預期比 EuRoC 弱,角色是 **robustness**。"
                 "label 污染主因是**運動**(高動態 room),姿態 g_body 是**次要共模項**。")
    else:
        legs = sum(v == "PASS" for v in (v1, v2, v3))
        L.append(f"- L1={v1} · L2={v2} · L3={v3}  →  PASS 腳數 **{legs}/3**")
        nxt = "三腳架成立 → 進 TUM-VI room 階段（§12）" if (v1 == "PASS" and v3 == "PASS") \
            else "L1/L3 未同時 PASS → 回 Planning §4.4 檢討"
        L.append(f"- 下一步：{nxt}")
    L.append(f"\n_plots: {'已產生於 plots/' if plots_made else 'matplotlib 不可用，已跳過'}_")

    with open(os.path.join(root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    if is_tumvi:
        print(f"  ✓ SUMMARY.md  (TUM-VI: L1={v1} L2=N/A L3={v3}；只服務 L1/L3 + ordering)")
    else:
        legs = sum(v == "PASS" for v in (v1, v2, v3))
        print(f"  ✓ SUMMARY.md  (L1={v1} L2={v2} L3={v3} → {legs}/3 PASS)")


# ──────────────────────────────────────────────────────────────────────────
# plots
# ──────────────────────────────────────────────────────────────────────────
def make_plots(cfg, df, L2=None, L3=None, bookend=None, rollup=None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    root = cfg["_results_dir_abs"]; pdir = common.ensure_dir(os.path.join(root, "plots"))
    made = False
    try:
        w = df.pivot_table(index=["run_id", "eval_sequence", "method", "family"],
                           columns="metric", values="value", aggfunc="first").reset_index()
    except Exception:
        return False
    # L1
    try:
        if {"iabe_rot", "gyro_ate"}.issubset(w.columns):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(w["iabe_rot"], w["gyro_ate"], s=10, alpha=0.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("iabe_rot (m)"); ax.set_ylabel("gyro_ate (m)"); ax.set_title("L1: gyro-ATE vs IABE")
            fig.tight_layout(); fig.savefig(os.path.join(pdir, "L1_gyroate_vs_iabe.png"), dpi=120); plt.close(fig); made = True
    except Exception:
        pass
    # family rollup
    try:
        if rollup:
            fams = sorted(rollup)
            fig, ax = plt.subplots(figsize=(7, 0.5 * len(fams) + 2))
            for yi, fam in enumerate(fams):
                d = rollup[fam]
                ax.plot([d["min"], d["max"]], [yi, yi], lw=4, alpha=0.6)
                ax.scatter([d["median"]], [yi], c="k", zorder=3)
            ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
            ax.set_xlabel("iabe_rot (m)"); ax.set_title("Architecture family IABE ranges (regime B)")
            fig.tight_layout(); fig.savefig(os.path.join(pdir, "family_rollup.png"), dpi=120); plt.close(fig); made = True
    except Exception:
        pass
    # L2 robust
    try:
        if L2 and L2.get("per_seq"):
            fig, ax = plt.subplots(figsize=(6, 5))
            for seq, d in L2["per_seq"].items():
                for p in d["pairs"]:
                    c = {"robust": "red", "dir-only": "orange", "dir-only(near-tie)": "gold"}.get(p["class"], "gray")
                    ax.scatter(p["mean_dRMSE"], p["mean_dIABE"], c=c, s=30, alpha=0.7)
            ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
            ax.set_xlabel("ΔRMSE"); ax.set_ylabel("ΔIABE"); ax.set_title("L2 robust inversion (red=robust)")
            fig.tight_layout(); fig.savefig(os.path.join(pdir, "L2_robust_inversions.png"), dpi=120); plt.close(fig); made = True
    except Exception:
        pass
    # L3 winner-by-seed
    try:
        if L3 and L3.get("per_seq"):
            seqs = list(L3["per_seq"].keys())
            fig, ax = plt.subplots(figsize=(8, 0.5 * len(seqs) + 2))
            mods = sorted({v for d in L3["per_seq"].values() for v in d["winner_by_seed"].values()})
            cm = {m: i for i, m in enumerate(mods)}
            for yi, seq in enumerate(seqs):
                for s, m in sorted(L3["per_seq"][seq]["winner_by_seed"].items()):
                    ax.scatter(s, yi, c=f"C{cm[m]%10}", s=90)
            ax.set_yticks(range(len(seqs))); ax.set_yticklabels(seqs)
            ax.set_xlabel("seed"); ax.set_title("L3 winner by seed (regime A)")
            fig.tight_layout(); fig.savefig(os.path.join(pdir, "L3_winner_by_seed.png"), dpi=120); plt.close(fig); made = True
    except Exception:
        pass
    # bookend
    try:
        if bookend:
            fig, ax = plt.subplots(figsize=(7, 4)); yi = 0; labels = []
            for reg in ["A", "B"]:
                d = bookend.get(reg)
                if not d:
                    continue
                for k in ("kf", "pf", "constant", "DL"):
                    rr = d.get(k)
                    if not rr:
                        continue
                    ax.plot([rr["min"], rr["max"]], [yi, yi], lw=3, alpha=.7)
                    ax.scatter([rr["median"]], [yi], c="k", s=16, zorder=3)
                    labels.append(f"{reg}/{k}"); yi += 1
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            ax.set_xlabel("iabe_rot (m)"); ax.set_title("Bookend ranges (KF≈const, PF≈KF)")
            fig.tight_layout(); fig.savefig(os.path.join(pdir, "bookend_iabe_ranges.png"), dpi=120); plt.close(fig); made = True
    except Exception:
        pass
    return made


# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# 跨資料集 ordering（§9/§10；只 claim ordering 非數值）
# ──────────────────────────────────────────────────────────────────────────
def analyze_cross_dataset_ordering(cfg, df_cur, dl_methods, cur_sequences=None):
    """比較 IABE 方法**排序**在當前資料集 vs EuRoC 之間是否穩定（Kendall/Spearman of rankings）。

    cur_sequences=None → 當前資料集側不篩(EuRoC 行為不變)；給 list → 當前(TUM-VI)側
    只取這些 eval_sequence 的排序(只比 main_sequences,不拿髒序列去比)。EuRoC 側永遠全乾淨不篩。
    """
    other = cfg.get("analysis", {}).get("cross_dataset_ordering", {}).get("euroc_results_dir")
    if not other or str(cfg.get("dataset", "")).lower() == "euroc":
        return None
    if not os.path.isabs(other):
        other = os.path.join(os.getcwd(), other)
    idx = os.path.join(other, "runs_index.csv")
    if not os.path.exists(idx):
        return None

    def _rank(df, sequences=None):
        w = _pivot(df, "B")
        if w.empty or "iabe_rot" not in w:
            return None
        w = w[w["method"].isin(dl_methods)]
        if sequences is not None:
            w = w[w["eval_sequence"].isin(sequences)]
            if w.empty:
                return None
        return w.groupby("method")["iabe_rot"].mean().sort_values()

    cur = _rank(df_cur, cur_sequences)
    # 載入 EuRoC（簡化：直接讀其 runs_index + metrics.json）
    ei = pd.read_csv(idx); rows = []
    for _, r in ei.iterrows():
        if str(r.get("status")) != "ok":
            continue
        mp = os.path.join(other, "runs", r["run_id"], "metrics.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            ev = json.load(f)
        for seq, mt in ev.items():
            if r["regime"] == "B" and mt and "iabe_rot" in mt:
                rows.append({"method": r["method"], "iabe_rot": float(mt["iabe_rot"])})
    if cur is None or not rows:
        return None
    euroc = pd.DataFrame(rows).groupby("method")["iabe_rot"].mean().sort_values()
    common_m = [m for m in cur.index if m in euroc.index]
    if len(common_m) < 3:
        return None
    rc = {m: i for i, m in enumerate([m for m in cur.index if m in common_m])}
    re = {m: i for i, m in enumerate([m for m in euroc.index if m in common_m])}
    sp = M._spearman(np.array([rc[m] for m in common_m]), np.array([re[m] for m in common_m]))
    return {"order_current": [m for m in cur.index if m in common_m],
            "order_euroc": [m for m in euroc.index if m in common_m],
            "spearman_of_rankings": float(sp), "n_methods": len(common_m)}


def run_analysis(cfg):
    root = cfg["_results_dir_abs"]
    df = load_runs(cfg)
    if df.empty:
        print("⚠ 無 ok 的 run 資料，先跑 --full。"); return
    write_long_wide(df, root)
    dlm = _dl_methods(cfg)

    # ── TUM-VI 事前註冊分流：主分析只用 main_sequences;room2/6 另算附錄 ──
    lock = load_prereg_lock(cfg)
    main_seqs = lock.get("main_sequences") if lock else None
    appendix_seqs = lock.get("appendix_sequences") if lock else None

    L1 = analyze_L1(df, sequences=main_seqs)
    L2 = analyze_L2(df, cfg, "B")
    L3 = analyze_L3(df, cfg, "A", sequences=main_seqs)
    L1_appendix = analyze_L1(df, sequences=appendix_seqs) if appendix_seqs else None
    L3_appendix = analyze_L3(df, cfg, "A", sequences=appendix_seqs) if appendix_seqs else None
    bookend = analyze_bookend(df, cfg)
    rollup = analyze_family_rollup(df, "B")
    band = band_decomp.analyze(cfg, df, root, dlm, regime="B",
                               label=str(cfg.get("dataset", "dataset")))
    cross = analyze_cross_dataset_ordering(cfg, df, dlm, cur_sequences=main_seqs)
    plots = make_plots(cfg, df, L2=L2, L3=L3, bookend=bookend, rollup=rollup)
    write_summary(cfg, df, L1, L2, L3, bookend, rollup, plots, band=band, cross=cross,
                  lock=lock, L1_appendix=L1_appendix, L3_appendix=L3_appendix)
    print(f"  ✓ metrics_long.csv / metrics_wide.csv → {root}")

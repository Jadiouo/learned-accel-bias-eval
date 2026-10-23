# -*- coding: utf-8 -*-
"""
Run-plan 展開（SDD §7.2）。由 config **deterministic** 展開成 method × regime × seq × seed
的完整清單。run_id 純由內容決定 → 可中斷續跑（已存在的 run skip）。

run 粒度：
  - regime A，DL          : 每 (method, seq, seed) 一個 run（train 該 seq、eval 該 seq）
  - regime A，baseline/filter: 每 (method, seq) 一個 run（deterministic，單 seed）
  - regime B，DL          : 每 (method, seed) 一個 run（train pooled、eval 各序列自己 tail）
  - regime B，baseline/filter: 每 method 一個 run（deterministic，eval 全部 tail）

方法 × regime 適用性：
  - constant_perseq 只在 regime A（per-seq floor）
  - constant_pooled 只在 regime B（global floor）
  - 其餘（DL / linear_stats / kf / pf）兩個 regime 都跑
"""
from __future__ import annotations

from .registry import kind_of, method_info


def _regimeA_methods(methods):
    return [m for m in methods if m != "constant_pooled"]


def _regimeB_methods(methods):
    return [m for m in methods if m != "constant_perseq"]


def build_plan(cfg: dict) -> list:
    """回傳 run-plan：list[dict]，每個 dict 是一個 run 的完整規格。"""
    methods = cfg["methods"]
    seeds = cfg["seeds"]
    seed0 = seeds[0]
    regimes = cfg.get("regimes", {})
    seqs_A = cfg.get("sequences", ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"])
    pooled = regimes.get("B", {}).get("pooled_train", ["MH_01", "MH_02", "MH_03", "MH_05"])

    plan = []

    # ---- regime A ----
    if "A" in regimes:
        for method in _regimeA_methods(methods):
            dl = kind_of(method) == "dl"
            for seq in seqs_A:
                run_seeds = seeds if dl else [seed0]
                for seed in run_seeds:
                    plan.append({
                        "run_id": f"A__{method}__{seq}__seed{seed}",
                        "method": method, "regime": "A", "seed": int(seed),
                        "train_sequences": [seq], "eval_sequences": [seq],
                        "kind": kind_of(method), "deterministic": not dl,
                        "approx_params": method_info(method)["approx_params"]})

    # ---- regime B（pooled）----
    if "B" in regimes:
        for method in _regimeB_methods(methods):
            dl = kind_of(method) == "dl"
            run_seeds = seeds if dl else [seed0]
            for seed in run_seeds:
                plan.append({
                    "run_id": f"B__{method}__pooled__seed{seed}",
                    "method": method, "regime": "B", "seed": int(seed),
                    "train_sequences": list(pooled), "eval_sequences": list(pooled),
                    "kind": kind_of(method), "deterministic": not dl,
                    "approx_params": method_info(method)["approx_params"]})

    return plan


def plan_summary(plan: list) -> dict:
    """給 run.py / README 報規模。"""
    by_regime = {}
    by_kind = {}
    for e in plan:
        by_regime[e["regime"]] = by_regime.get(e["regime"], 0) + 1
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    return {"total_runs": len(plan), "by_regime": by_regime, "by_kind": by_kind}

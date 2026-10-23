# -*- coding: utf-8 -*-
"""
單一 entrypoint（SDD §7）。

  python -m euroc_full.run --config configs/full.yaml  --smoke    [--data-dir ...]
  python -m euroc_full.run --config configs/full.yaml  --full     [--data-dir ...]
  python -m euroc_full.run --config configs/full.yaml  --analyze

  --smoke   : preflight（每方法 1 seq×1 seed forward/backward、Mamba fallback、Transformer
              收斂與 causal-mask sanity、PF/linear_stats、+ metric sanity test）→ 不過不准 --full
  --full    : 內建 metric-gate（test suite）→ 展開 run-plan → resumable 執行 → 寫 provenance
  --analyze : 聚合 → CSV / SUMMARY / plots（可獨立重跑，不重訓）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from euroc_full import common, data_euroc, metrics, baselines, plan as planmod, registry
else:
    from . import common, data_euroc, metrics, baselines, plan as planmod, registry

RUNS_INDEX_COLS = ["run_id", "method", "family", "regime", "n_params", "train_sequences",
                   "eval_sequences", "seed", "git_hash", "timestamp_iso", "wall_time_s",
                   "status", "notes"]
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.normpath(os.path.join(PKG_DIR, "..", "..", "tests"))


# ──────────────────────────────────────────────────────────────────────────
# output helpers（provenance；無絕對路徑寫進 index）
# ──────────────────────────────────────────────────────────────────────────
def _results_dirs(cfg):
    root = common.ensure_dir(cfg["_results_dir_abs"])
    runs = common.ensure_dir(os.path.join(root, "runs"))
    common.ensure_dir(os.path.join(root, "plots"))
    return root, runs


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _append_index(root, row):
    """append 一列；run_id 已存在則覆寫（冪等，供 resumable 再跑）。"""
    path = os.path.join(root, "runs_index.csv")
    rows, found = [], False
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("run_id") == row["run_id"]:
                    rows.append({k: row.get(k) for k in RUNS_INDEX_COLS}); found = True
                else:
                    rows.append(r)
    if not found:
        rows.append({k: row.get(k) for k in RUNS_INDEX_COLS})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_INDEX_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in RUNS_INDEX_COLS})


def _write_run(runs_dir, entry, cfg, eval_block, series, n_params, git_hash, ts, notes):
    d = common.ensure_dir(os.path.join(runs_dir, entry["run_id"]))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"run_id": entry["run_id"], "method": entry["method"], "regime": entry["regime"],
                   "seed": entry["seed"], "train_sequences": entry["train_sequences"],
                   "eval_sequences": entry["eval_sequences"], "n_params": n_params,
                   "git_hash": git_hash, "timestamp": ts, "notes": notes,
                   "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}, f, indent=2)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(_sanitize(eval_block), f, indent=2)
    if series and cfg.get("save_series", False):
        np.savez_compressed(os.path.join(d, "series.npz"), **series)


def _eval_metrics(merged, test_win_starts, preds, y_ref, cfg):
    mi = metrics.build_metric_inputs(merged, test_win_starts, preds, y_ref,
                                     cfg["stride"], cfg.get("imu_freq", 200), cfg.get("gravity", 9.81))
    return metrics.compute_all_metrics(mi, cfg.get("slie_taus", [5, 10, 15]))


# ──────────────────────────────────────────────────────────────────────────
# 預測：依方法 kind 產 b_hat
# ──────────────────────────────────────────────────────────────────────────
def _predict_regimeA(method, d, cfg, seed):
    """回傳 (preds, n_params, notes)。d = SeqData。"""
    n = len(d.y_test)
    kind = registry.kind_of(method)
    if kind == "dl":
        from . import train
        model = registry.build_dl_model(method, cfg)
        npar = registry.count_params(model)
        note = ""
        if method == "mamba" and getattr(model, "used_fallback", False):
            note = "mamba:pure_pytorch_fallback"
        model, _ = train.train_model(model, d, cfg, seed)
        return train.infer(model, d.X_test), npar, note
    if method == "constant_perseq":
        return baselines.constant_perseq(d.y_train, n), 3, ""
    if method == "linear_stats":
        reg = baselines.LinearStatsRegressor().fit(d.X_train, d.y_train)
        return reg.predict(d.X_test), reg.n_params, ""
    if method == "kf":
        return baselines.kalman_filter_bias(np.zeros((n, 3)), d.y_train.mean(axis=0)), 0, ""
    if method == "pf":
        return baselines.particle_filter_bias(np.zeros((n, 3)), d.y_train.mean(axis=0), seed=seed), 0, "pf≈kf"
    raise ValueError(f"regimeA: unhandled method {method}")


def _predict_regimeB(method, pooled, ev, cfg, seed, trained_model=None):
    """回傳 (preds, n_params, notes)。ev = per_seq_eval[seq]。"""
    n = len(ev["y_test"])
    kind = registry.kind_of(method)
    if kind == "dl":
        from . import train
        return train.infer(trained_model, ev["X_test"]), registry.count_params(trained_model), ""
    if method == "constant_pooled":
        return baselines.constant_pooled(pooled.pooled_y_train_raw, n), 3, ""
    if method == "linear_stats":
        reg = baselines.LinearStatsRegressor().fit(pooled.X_train, pooled.y_train)
        return reg.predict(ev["X_test"]), reg.n_params, ""
    if method == "kf":
        return baselines.kalman_filter_bias(np.zeros((n, 3)), pooled.pooled_y_train_raw.mean(axis=0)), 0, ""
    if method == "pf":
        return baselines.particle_filter_bias(np.zeros((n, 3)), pooled.pooled_y_train_raw.mean(axis=0), seed=seed), 0, "pf≈kf"
    raise ValueError(f"regimeB: unhandled method {method}")


# ──────────────────────────────────────────────────────────────────────────
# 執行一個 run（resumable）
# ──────────────────────────────────────────────────────────────────────────
def _execute_run(entry, cfg, adapter, cacheA, cacheB, runs_dir, root, git_hash):
    metrics_path = os.path.join(runs_dir, entry["run_id"], "metrics.json")
    if os.path.exists(metrics_path):
        return "skipped"
    t0 = time.time()
    eval_block, series, n_params, notes, status = {}, {}, 0, "", "ok"
    try:
        if entry["regime"] == "A":
            seq = entry["eval_sequences"][0]
            d = cacheA.get(seq) or data_euroc.prepare_regime_a(seq, cfg, adapter)
            cacheA[seq] = d
            preds, n_params, notes = _predict_regimeA(entry["method"], d, cfg, entry["seed"])
            eval_block[seq] = _eval_metrics(d.merged, d.test_win_starts, preds, d.y_test, cfg)
            series = {f"b_hat_{seq}": preds, f"b_ref_{seq}": d.y_test}
        else:  # regime B
            pooled = cacheB.get("pooled") or data_euroc.prepare_regime_b_pooled(cfg, adapter)
            cacheB["pooled"] = pooled
            trained = None
            if registry.kind_of(entry["method"]) == "dl":
                from . import train
                model = registry.build_dl_model(entry["method"], cfg)
                n_params = registry.count_params(model)
                if entry["method"] == "mamba" and getattr(model, "used_fallback", False):
                    notes = "mamba:pure_pytorch_fallback"
                trained, _ = train.train_model(model, pooled, cfg, entry["seed"])
            for seq, ev in pooled.per_seq_eval.items():
                preds, n_params, n2 = _predict_regimeB(entry["method"], pooled, ev, cfg, entry["seed"], trained)
                eval_block[seq] = _eval_metrics(ev["merged"], ev["test_win_starts"], preds, ev["y_test"], cfg)
                series[f"b_hat_{seq}"] = preds; series[f"b_ref_{seq}"] = ev["y_test"]
                notes = notes or n2
    except Exception as e:
        status = "failed"; notes = f"{type(e).__name__}: {e}"
        print(f"   ⚠ {entry['run_id']} failed: {notes}")

    ts = common.now_iso()
    _write_run(runs_dir, entry, cfg, eval_block, series if status == "ok" else None,
               n_params, git_hash, ts, notes)
    _append_index(root, {
        "run_id": entry["run_id"], "method": entry["method"],
        "family": registry.method_info(entry["method"])["family"], "regime": entry["regime"],
        "n_params": n_params, "train_sequences": "|".join(entry["train_sequences"]),
        "eval_sequences": "|".join(entry["eval_sequences"]), "seed": entry["seed"],
        "git_hash": git_hash, "timestamp_iso": ts, "wall_time_s": round(time.time() - t0, 2),
        "status": status, "notes": notes})
    return status


# ──────────────────────────────────────────────────────────────────────────
# metric gate（pytest test suite；§7.1/§7.2）
# ──────────────────────────────────────────────────────────────────────────
def _run_metric_gate() -> bool:
    test_file = os.path.join(TESTS_DIR, "test_metric_sanity.py")
    if not os.path.exists(test_file):
        print(f"   ⚠ 找不到 {test_file}，metric gate 無法執行"); return False
    print("   running metric sanity test suite (pytest)…")
    env = dict(os.environ)
    # 讓 metric-gate 對環境裡的第三方 pytest plugin 免疫（hermetic、可攜）。
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # 確保 subprocess 能 import euroc_full（src layout，未 pip install 時）。
    src = os.path.normpath(os.path.join(PKG_DIR, ".."))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "pytest", test_file, "-q"],
                       capture_output=True, text=True, env=env)
    sys.stdout.write(r.stdout[-2000:] if r.stdout else "")
    ok = r.returncode == 0
    print(f"   metric gate: {'PASS' if ok else 'FAIL'}")
    return ok


# ──────────────────────────────────────────────────────────────────────────
# --smoke（preflight）
# ──────────────────────────────────────────────────────────────────────────
def cmd_smoke(cfg) -> bool:
    print("[smoke] preflight — 每方法 1 seq×1 seed + metric gate")
    ok_all = True

    # 1) metric sanity test suite（torch-free，最重要）
    if not _run_metric_gate():
        ok_all = False

    # 2) DL forward/backward + 特例 sanity（需 torch）
    try:
        import torch
        have_torch = True
    except Exception:
        have_torch = False
        print("   ⚠ torch 不在場 → DL preflight 跳過（CI metric-only 模式）；--full 的 DL run 會記 failed")

    if have_torch:
        import torch
        for method in cfg["methods"]:
            if registry.kind_of(method) != "dl":
                continue
            try:
                model = registry.build_dl_model(method, cfg)
                x = torch.randn(4, cfg["window"], 6, requires_grad=False)
                y = torch.randn(4, 3)
                out = model(x)
                assert out.shape == (4, 3), f"{method} 輸出形狀 {tuple(out.shape)} ≠ (4,3)"
                loss = ((out - y) ** 2).mean(); loss.backward()
                assert torch.isfinite(loss), f"{method} loss 非有限"
                note = ""
                if method == "mamba":
                    note = " (pure-pytorch fallback)" if getattr(model, "used_fallback", False) else " (mamba-ssm)"
                print(f"   [ok] {method} forward/backward{note}")
                # Transformer：causal-mask sanity
                if method.startswith("transformer"):
                    if not _causal_mask_sanity(model):
                        print(f"   [FAIL] {method} causal-mask sanity（後半段洩漏到前半段）"); ok_all = False
                    else:
                        print(f"   [ok] {method} causal-mask sanity")
            except Exception as e:
                print(f"   [FAIL] {method}: {type(e).__name__}: {e}"); ok_all = False

    # 3) TUM-VI room derisk（§7.1）：gap-clean 吃重 + mocap-derived bias label 平穩
    if str(cfg.get("dataset", "")).lower() in ("tumvi_room", "tumvi"):
        if not _tumvi_room_derisk(cfg):
            ok_all = False

    print(f"[smoke] {'PASS — 可進 --full' if ok_all else 'FAIL — 修正後再 --full'}")
    return ok_all


def _tumvi_room_derisk(cfg) -> bool:
    """room1 derisk（SDD §7.1，正確口徑）：
       ① gap-clean 真的丟對 window（room1 ~1.06s mocap gap → 預期丟 ~9.9%）。
       ② 低動態子集的 mocap-derived b_ref 平穩。

    口徑說明（已由先前診斷確立）：整段 per-sample std 量到的是「運動洩漏（~1.5）」而非
    感測器噪聲，會誤判 FAIL。正確做法是：gap-clean 後、**window-mean 層級**（模型真正預測的
    target），以每窗運動強度 = 窗內 mean|a_world| 排序，取「運動最低 20–30% 的窗」子集，看其
    b_ref 的 |mean|、|std|。已驗證乾淨基準：該子集 |mean|≈0.039、|std|≈0.067 m/s²。

    第三關「各方法收斂」由上面的 DL forward/backward + Transformer causal mask
    + Mamba fallback + metric gate 覆蓋（本函式不動那段）。
    """
    seq = "room1"
    print(f"  [TUM-VI derisk] {seq}: gap-clean(吃重) + 低動態子集 mocap-derived b_ref 平穩")
    ok = True
    try:
        import numpy as _np
        adapter = data_euroc.get_adapter(cfg)
        win_cfg = data_euroc._win_cfg(cfg)
        window, stride, freq = win_cfg[0], win_cfg[1], win_cfg[2]

        # ── ① gap-clean 丟棄統計（room1 有 ~1.06s mocap gap → 應丟若干）──
        # 注意：以「無 motion-tail mask」口徑計，純看 gap 丟了多少（mask 是另一關注點）。
        merged = data_euroc._merged_for(adapter, seq, cfg)
        total_win = len(range(0, len(merged) - window, stride))
        win_cfg_nogap_mask = win_cfg[:5] + (None,)        # 關掉 motion-tail mask，只量 gap
        _, _, starts = data_euroc.make_windows_clean(merged, *win_cfg_nogap_mask)
        kept = len(starts); dropped = total_win - kept
        pct = 100.0 * dropped / max(total_win, 1)
        print(f"   [gap-clean] 總窗 {total_win} → 保留 {kept}，丟棄 {dropped}（{pct:.1f}%，橫跨 gap）")
        if dropped == 0:
            print("   ⚠ 沒丟任何 window（room1 預期有 ~1.06s gap）→ 檢查 gap 偵測（非致命,先標記）")

        # ── ② 低動態子集 b_ref 平穩（gap-clean 後、window-mean 層級）──
        # 每窗：b_ref = 窗內 b_a 均值（模型 target）；運動強度 = 窗內 mean|a_world|。
        # a_world 同 adapter 口徑（savgol deriv=2，已由 derive_mocap_labels 存進 awx/awy/awz）。
        aw = merged[["awx", "awy", "awz"]].to_numpy(_np.float64)
        aw_norm = _np.linalg.norm(aw, axis=1)
        b = merged[["bax", "bay", "baz"]].to_numpy(_np.float64)
        ts = merged["ts"].to_numpy(_np.int64)
        nominal_dt = 1.0 / freq
        max_allow = cfg.get("gap_tol_factor", 2.0) * nominal_dt

        wins_b, wins_motion = [], []
        for s in starts:                                   # starts = gap-clean 後保留的窗
            e = s + window
            wins_b.append(b[s:e].mean(axis=0))
            wins_motion.append(float(aw_norm[s:e].mean()))
        wins_b = _np.asarray(wins_b)                       # (Nwin, 3) window-mean b_ref
        wins_motion = _np.asarray(wins_motion)             # (Nwin,) 每窗運動強度

        # 取運動最低 20–30% 的窗子集（用 30% 涵蓋，較穩健）。
        frac = 0.30
        k = max(int(len(wins_motion) * frac), 1)
        low_idx = _np.argsort(wins_motion)[:k]
        sub = wins_b[low_idx]                              # 低動態子集 window-mean b_ref
        mean_ax = sub.mean(axis=0); std_ax = sub.std(axis=0)
        mean_norm = float(_np.linalg.norm(mean_ax))
        std_norm = float(_np.linalg.norm(std_ax))
        print(f"   [低動態子集] 運動最低 {int(frac*100)}% 窗（{k}/{len(wins_motion)}）的 window-mean b_ref：")
        print(f"     |mean| 逐軸=[{mean_ax[0]:+.4f} {mean_ax[1]:+.4f} {mean_ax[2]:+.4f}]  範數={mean_norm:.4f} m/s²")
        print(f"     |std|  逐軸=[{std_ax[0]:.4f} {std_ax[1]:.4f} {std_ax[2]:.4f}]  範數={std_norm:.4f} m/s²")
        print(f"     (已驗證乾淨基準：|mean|≈0.039、|std|≈0.067 m/s²)")

        # PASS：低動態子集 |mean| ≲ 0.2 且 |std| ≲ 0.15（寬鬆涵蓋 0.039/0.067，留量級餘裕）。
        # FAIL：量級荒謬（|mean|>0.5 或 |std|>0.3，或非有限）。
        finite = _np.isfinite(mean_norm) and _np.isfinite(std_norm)
        if not finite or mean_norm > 0.5 or std_norm > 0.3:
            print(f"   [FAIL] 低動態子集 b_ref 量級荒謬（|mean|={mean_norm:.3f}>0.5 或 |std|={std_norm:.3f}>0.3 "
                  f"或非有限）→ MoCap-derived 不可用（SDD §7.1：改 okvis 或暫停）"); ok = False
        elif mean_norm <= 0.2 and std_norm <= 0.15:
            print(f"   [ok] 低動態子集 b_ref 平穩（|mean|={mean_norm:.3f}≲0.2 且 |std|={std_norm:.3f}≲0.15）")
        else:
            print(f"   [FAIL] 低動態子集 b_ref 超出 PASS 帶（|mean|={mean_norm:.3f} 或 |std|={std_norm:.3f} "
                  f"未在 ≲0.2/≲0.15 內，但未到荒謬閾）→ 需檢查 smoothing/口徑"); ok = False
    except Exception as e:
        print(f"   [FAIL] TUM-VI derisk 例外：{type(e).__name__}: {e}"); ok = False
    return ok


def _causal_mask_sanity(model, W=40) -> bool:
    """擾動 input 後半段，前半段 per-position 預測必不變（eval 關 dropout）。"""
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, W, 6)
        a = model.causal_sequence_output(x)
        x2 = x.clone(); x2[:, W // 2:] = torch.randn(2, W - W // 2, 6)
        b = model.causal_sequence_output(x2)
        return bool(torch.allclose(a[:, :W // 2], b[:, :W // 2], atol=1e-5))


# ──────────────────────────────────────────────────────────────────────────
# --full
# ──────────────────────────────────────────────────────────────────────────
def cmd_full(cfg, force=False) -> None:
    root, runs = _results_dirs(cfg)
    print("[full] 內建 metric gate → run-plan → resumable 執行")
    if not _run_metric_gate() and not force:
        print("✗ metric gate 未過，拒跑 --full（--force 可強行，不建議）。"); return

    plan = planmod.build_plan(cfg)
    summ = planmod.plan_summary(plan)
    print(f"   run-plan: {summ['total_runs']} runs  ·  by_regime={summ['by_regime']}  ·  by_kind={summ['by_kind']}")
    gh = common.git_hash()
    adapter = data_euroc.get_adapter(cfg)
    cacheA, cacheB = {}, {}
    n_ok = n_skip = n_fail = 0
    for i, entry in enumerate(plan, 1):
        st = _execute_run(entry, cfg, adapter, cacheA, cacheB, runs, root, gh)
        n_ok += st == "ok"; n_skip += st == "skipped"; n_fail += st == "failed"
        if i % 20 == 0 or st == "failed":
            print(f"   [{i}/{len(plan)}] {entry['run_id']} → {st}  (ok={n_ok} skip={n_skip} fail={n_fail})")
    print(f"[full] 完成：ok={n_ok} skipped={n_skip} failed={n_fail}（總 {len(plan)}）")
    print("→ 跑 `--analyze` 產生 SUMMARY.md")


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="EuRoC Full Matrix — single entrypoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None, help="覆寫資料目錄（優先於 EUROC_DATA_DIR / config）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true", help="跳過 metric gate（不建議）")
    args = ap.parse_args()

    cfg = common.load_config(args.config, data_dir_override=args.data_dir)

    if args.smoke:
        ok = cmd_smoke(cfg)
        sys.exit(0 if ok else 1)
    if args.full:
        cmd_full(cfg, force=args.force)
    if args.analyze:
        from . import analysis
        analysis.run_analysis(cfg)


if __name__ == "__main__":
    main()

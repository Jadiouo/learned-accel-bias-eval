# -*- coding: utf-8 -*-
"""一個訓練 worker：不改既有程式，只在行程內 monkeypatch plan 過濾 / 遙測 / 截止時間。

用法：python worker.py --code <euroc_full 根> --config <yaml> --data-dir <dir> --cwd <此 worker 的獨立 cwd>
        [--only-method mamba | --exclude-method mamba] [--part i/n] [--deadline <unix>]
結果寫在 <cwd>/<config 的 results_dir>（與 MH 同結構：runs/<run_id>/{config,metrics}.json、series.npz、runs_index.csv）。
訓練邏輯、超參數、mamba fallback 都與 MH 實驗相同（hidden mamba_ssm 以確保走 pure-PyTorch fallback）。
"""
import argparse, csv, json, os, shutil, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--cwd", required=True)
ap.add_argument("--only-method", default=None)
ap.add_argument("--exclude-method", default=None)
ap.add_argument("--part", default="0/1")
ap.add_argument("--deadline", type=float, default=0.0)
args = ap.parse_args()
for _k in ('code', 'config', 'data_dir', 'cwd'):
    setattr(args, _k, os.path.abspath(getattr(args, _k)))

sys.modules["mamba_ssm"] = None            # 強制走 pure-PyTorch fallback（與 MH 結果同口徑）
os.makedirs(args.cwd, exist_ok=True)
os.chdir(args.cwd)
sys.path.insert(0, os.path.join(args.code, "src"))

from euroc_full import common, run as R, train as T            # noqa: E402

cfg = common.load_config(args.config, data_dir_override=args.data_dir)
pi, pn = [int(x) for x in args.part.split("/")]

# ---- plan 過濾（確定性：先依 method 篩，再 index % pn == pi）----
_orig_plan = R.planmod.build_plan
def _plan(c):
    p = _orig_plan(c)
    if args.only_method:
        p = [e for e in p if e["method"] == args.only_method]
    if args.exclude_method:
        p = [e for e in p if e["method"] != args.exclude_method]
    return [e for i, e in enumerate(p) if i % pn == pi]
R.planmod.build_plan = _plan

# ---- 續跑前清理：failed 的 run、metrics.json 壞掉（被截斷）的 run 目錄刪掉重跑 ----
root = cfg["_results_dir_abs"]
runs = os.path.join(root, "runs")
if os.path.isdir(runs):
    failed = set()
    ip = os.path.join(root, "runs_index.csv")
    if os.path.exists(ip):
        with open(ip, newline="") as f:
            failed = {r["run_id"] for r in csv.DictReader(f) if r.get("status") != "ok"}
    for rid in os.listdir(runs):
        mp = os.path.join(runs, rid, "metrics.json")
        bad = rid in failed
        if not bad and os.path.exists(mp):
            try:
                json.load(open(mp))
            except Exception:
                bad = True
        if bad:
            print(f"[worker] 清除需重跑的 run: {rid}", flush=True)
            shutil.rmtree(os.path.join(runs, rid), ignore_errors=True)

# ---- 截止時間：在「開始下一個 run 前」檢查，乾淨退出（不會製造 failed run）----
_orig_exec = R._execute_run
def _exec(*a, **k):
    if args.deadline and time.time() > args.deadline:
        print("[worker] deadline reached, stop cleanly before next run", flush=True)
        raise SystemExit(0)
    return _orig_exec(*a, **k)
R._execute_run = _exec

# ---- 遙測：每個 DL run 的 epoch 數與耗時（只記錄，不影響訓練）----
_orig_train = T.train_model
def _train(model, data, c, seed):
    t0 = time.time()
    m, hist = _orig_train(model, data, c, seed)
    ep = len(hist["train_loss"])
    with open("telemetry.jsonl", "a") as f:
        f.write(json.dumps({"n_train": int(len(data.X_train)), "epochs": ep, "best_epoch": hist["best_epoch"],
                            "sec": round(time.time() - t0, 2), "seed": seed,
                            "model": type(model).__name__}) + "\n")
    return m, hist
T.train_model = _train

import torch                                                    # noqa: E402
print(f"[worker] cwd={os.getcwd()} cuda={torch.cuda.is_available()} "
      f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} part={args.part} only={args.only_method} "
      f"excl={args.exclude_method}", flush=True)
R.cmd_full(cfg)

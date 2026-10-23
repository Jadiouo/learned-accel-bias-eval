# s4：只補缺的 10 個 mamba run（以 run_id 清單指定，不 restore、不靠 resume）。smoke -> bench -> 4 個 mamba worker（每張 T4 兩個）-> 打包。
import glob, json, os, shutil, subprocess, sys, tarfile, time, yaml

T_START = time.time()
SOFT_STOP_H = float(os.environ.get("SOFT_STOP_H", "1.5"))      # 超過此時數就不再開新 run
HARD_STOP_H = float(os.environ.get("HARD_STOP_H", "1.8"))     # 超過就終止還在跑的 run，丟掉未完成的，打包已完成的
DEADLINE = T_START + SOFT_STOP_H * 3600
W = "/kaggle/working"
CODE_SRC = [os.path.dirname(os.path.dirname(p)) for p in
            glob.glob("/kaggle/input/*/configs/vicon.yaml") + glob.glob("/kaggle/input/*/*/configs/vicon.yaml")
            + glob.glob("/kaggle/input/*/*/*/configs/vicon.yaml")][0]
DATA = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))) for p in
        glob.glob("/kaggle/input/*/V1_01_easy/mav0/imu0/data.csv") + glob.glob("/kaggle/input/*/*/V1_01_easy/mav0/imu0/data.csv")
        + glob.glob("/kaggle/input/*/*/*/V1_01_easy/mav0/imu0/data.csv")][0]
CODE = f"{W}/code/euroc_full"
shutil.copytree(CODE_SRC, CODE, dirs_exist_ok=True)
CFG = f"{CODE}/configs/vicon.yaml"
WORKER = f"{CODE}/kaggle_vicon/train/worker.py"
BENCH = f"{CODE}/kaggle_vicon/train/bench.py"
ENV = dict(os.environ, EUROC_DATA_DIR=DATA, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
LOGS = f"{W}/logs"; os.makedirs(LOGS, exist_ok=True)

# 缺的 10 個 run（與 missing_runs.txt 相同；順序即優先序，依序輪流發給 4 個 worker）
RUN_IDS = [
    "A__mamba__V2_01__seed49", "A__mamba__V2_02__seed43", "A__mamba__V2_03__seed43", "A__mamba__V2_02__seed47",
    "A__mamba__V2_03__seed45", "A__mamba__V2_02__seed49", "A__mamba__V2_03__seed47", "A__mamba__V2_02__seed51",
    "A__mamba__V2_03__seed49", "A__mamba__V2_03__seed51",
]
EXPECTED = len(RUN_IDS)


def assign(ids, n):
    return [ids[i::n] for i in range(n)]


# worker 包裝：只在行程內把 plan 換成指定的 run_id（順序照清單），其餘全交給 train/worker.py。
WRAPPER_SRC = r'''
import argparse, json, os, runpy, sys
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--run-ids", required=True)
ap.add_argument("--worker-path", required=True)
ap.add_argument("--dry-run", action="store_true")
own, rest = ap.parse_known_args()
ids = [x for x in own.run_ids.split(",") if x]
sys.path.insert(0, os.path.join(os.path.abspath(rest[rest.index("--code") + 1]), "src"))
sys.modules["mamba_ssm"] = None
from euroc_full import run as R
_orig = R.planmod.build_plan
def _plan(c):
    full = {e["run_id"]: e for e in _orig(c)}
    miss = [i for i in ids if i not in full]
    if miss:
        print("[wrapper] run_id 不在 plan:", miss, flush=True); raise SystemExit(4)
    return [full[i] for i in ids]
R.planmod.build_plan = _plan
if own.dry_run:
    def _dry(cfg, force=False):
        print("PLAN_IDS " + json.dumps([e["run_id"] for e in R.planmod.build_plan(cfg)]), flush=True)
    R.cmd_full = _dry
sys.argv = [own.worker_path] + rest
runpy.run_path(own.worker_path, run_name="__main__")
'''



def log(*a):
    print(f"[{(time.time()-T_START)/60:7.1f}m]", *a, flush=True)


def sh(cmd, **k):
    return subprocess.run(cmd, text=True, capture_output=True, **k)


log("code:", CODE_SRC, "data:", DATA)
log(sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]).stdout.strip())
log(sh([sys.executable, "-c", "import torch;print(torch.__version__,torch.cuda.device_count())"]).stdout.strip())
for p in ("einops", "pytest", "sklearn", "yaml", "scipy", "pandas"):
    r = sh([sys.executable, "-c", f"import {p}"])
    if r.returncode:
        log("pip install", p); sh([sys.executable, "-m", "pip", "install", "-q", {"sklearn": "scikit-learn", "yaml": "pyyaml"}.get(p, p)])

# ---------- s4：不 restore（輸出只放新跑的 run）----------
WRAPPER = f"{W}/worker_ids.py"
open(WRAPPER, "w").write(WRAPPER_SRC)
WORKERS = [  # name, gpu, args
    (n, g, ["--run-ids", ",".join(ids), "--worker-path", WORKER])
    for (n, g), ids in zip([("g0m0", 0), ("g0m1", 0), ("g1m0", 1), ("g1m1", 1)], assign(RUN_IDS, 4))
]
# ---------- smoke（V1_01 / m1_cnn / seed 42 / 2 epoch）----------
if not os.environ.get("SKIP_SMOKE"):
    c = yaml.safe_load(open(CFG))
    c.update(sequences=["V1_01"], methods=["m1_cnn"], seeds=[42], results_dir="results_smoke")
    c["train"]["epochs"] = 2
    sp = f"{W}/smoke.yaml"; yaml.safe_dump(c, open(sp, "w"))
    r = sh([sys.executable, WORKER, "--code", CODE, "--config", sp, "--data-dir", DATA, "--cwd", f"{W}/smoke"], env=ENV)
    open(f"{LOGS}/smoke.log", "w").write(r.stdout + r.stderr)
    log("SMOKE rc", r.returncode); log(r.stdout[-1500:]); log(r.stderr[-800:])
    mj = glob.glob(f"{W}/smoke/results_smoke/runs/*/metrics.json")
    good = ("metric gate: PASS" in r.stdout) and ("ok=1 skipped=0 failed=0" in r.stdout) and len(mj) == 1
    if good:
        m = json.load(open(mj[0]))["V1_01"]
        import math
        vals = [v for v in m.values() if isinstance(v, (int, float))]
        good = len(vals) > 0 and all(math.isfinite(v) for v in vals)
        log("SMOKE metrics", json.dumps(m)[:600])
    if not good:
        log("SMOKE FAILED -> abort before training"); sys.exit(3)
    log("SMOKE PASS (metric gate PASS, 1 ok, metrics finite)")
    shutil.rmtree(f"{W}/smoke", ignore_errors=True)

# ---------- bench：50-100 step 速度 ----------
r = sh([sys.executable, BENCH, CODE, CFG, DATA, "60"], env=dict(ENV, CUDA_VISIBLE_DEVICES="0"), cwd=W)
open(f"{LOGS}/bench.log", "w").write(r.stdout + r.stderr)
log("BENCH rc", r.returncode); log(r.stdout); log(r.stderr[-600:])

# ---------- 啟動 4 個 mamba worker（每張 T4 兩個，無 r worker，同時啟動）----------
procs = {}
PENDING = {}


def launch(name, gpu, a):
    lf = open(f"{LOGS}/{name}.log", "a")
    procs[name] = subprocess.Popen(
        [sys.executable, WRAPPER, "--code", CODE, "--config", CFG, "--data-dir", DATA, "--cwd", f"{W}/res/{name}",
         "--deadline", str(DEADLINE)] + a,
        env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=lf, stderr=subprocess.STDOUT)
    log("started", name, "gpu", gpu, a)


for name, gpu, a in WORKERS:
    if name not in PENDING:
        launch(name, gpu, a)


def progress():
    n = 0; per = {}
    for name, *_ in WORKERS:
        k = len(glob.glob(f"{W}/res/{name}/results_vicon/runs/*/metrics.json"))
        per[name] = k; n += k
    return n, per


def projection(n):
    """依已完成 run 的實測 wall_time 推算剩餘時間（每方法平均 × 剩餘數，除以並行度）。"""
    import csv, collections
    rows = []
    for name, *_ in WORKERS:
        p = f"{W}/res/{name}/results_vicon/runs_index.csv"
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    return len(rows)


last = 0
while any(p.poll() is None for p in procs.values()) or any(k not in procs for k in PENDING):
    time.sleep(30)
    if time.time() > T_START + HARD_STOP_H * 3600:
        for k, p in procs.items():
            if p.poll() is None:
                log("HARD STOP: terminate", k); p.terminate()
        time.sleep(10)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        break
    for name, gpu, a in WORKERS:
        if name in PENDING and name not in procs and procs[PENDING[name]].poll() is not None:
            launch(name, gpu, a)
    if time.time() - last > 600:
        last = time.time()
        n, per = progress()
        log(f"progress {n}/{EXPECTED}", per, "rc:", {k: p.poll() for k, p in procs.items()})
        log(sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]).stdout.strip().replace("\n", " | "))
        log(sh(["bash", "-c", "free -g | sed -n 2p; df -h /kaggle/working | tail -1"]).stdout.strip().replace("\n", " | "))

n, per = progress()
log(f"workers exited. progress {n}/{EXPECTED}", per, {k: p.returncode for k, p in procs.items()})

# ---------- 驗證 + 打包（每 worker 一個 tar；刪掉原目錄以免 output 檔數爆炸）----------
import csv
stat = {"ok": 0, "failed": 0, "other": 0}
for name, *_ in WORKERS:
    p = f"{W}/res/{name}/results_vicon/runs_index.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            stat["ok" if r["status"] == "ok" else "failed" if r["status"] == "failed" else "other"] += 1
log("index status counts", stat, "metrics.json total", n)
# 被硬截止中斷的 run：沒有 metrics.json 的目錄一律刪掉，不放進 tar
for name, *_ in WORKERS:
    for d in glob.glob(f"{W}/res/{name}/results_vicon/runs/*"):
        if not os.path.exists(f"{d}/metrics.json"):
            log("drop incomplete run", d); shutil.rmtree(d, ignore_errors=True)
for name, *_ in WORKERS:
    d = f"{W}/res/{name}"
    if os.path.isdir(d):
        with tarfile.open(f"{W}/results_{name}.tar", "w") as tf:
            tf.add(d, arcname=".")
        shutil.copy(f"{d}/telemetry.jsonl", f"{LOGS}/telemetry_{name}.jsonl") if os.path.exists(f"{d}/telemetry.jsonl") else None
        shutil.rmtree(d)
shutil.rmtree(f"{W}/code", ignore_errors=True)
json.dump({"progress": n, "status": stat, "elapsed_h": (time.time() - T_START) / 3600}, open(f"{W}/SUMMARY.json", "w"))
log("DONE", (time.time() - T_START) / 3600, "h")

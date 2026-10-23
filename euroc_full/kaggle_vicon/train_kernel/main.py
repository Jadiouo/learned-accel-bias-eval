# Kaggle 訓練 kernel（GPU T4x2）：smoke -> bench -> 6 個 worker（每張 T4：2 mamba + 1 其餘）-> 打包。
import glob, json, os, shutil, subprocess, sys, tarfile, time, yaml

T_START = time.time()
SOFT_STOP_H = float(os.environ.get("SOFT_STOP_H", "9.0"))      # 超過此時數就不再開新 run
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
ENV = dict(os.environ, EUROC_DATA_DIR=DATA, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
LOGS = f"{W}/logs"; os.makedirs(LOGS, exist_ok=True)
EXPECTED = 684


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

# ---------- 還原先前批次的結果（續跑）----------
WORKERS = [  # name, gpu, args
    ("g0m0", 0, ["--only-method", "mamba", "--part", "0/4"]),
    ("g0m1", 0, ["--only-method", "mamba", "--part", "1/4"]),
    ("g0r",  0, ["--exclude-method", "mamba", "--part", "0/2"]),
    ("g1m0", 1, ["--only-method", "mamba", "--part", "2/4"]),
    ("g1m1", 1, ["--only-method", "mamba", "--part", "3/4"]),
    ("g1r",  1, ["--exclude-method", "mamba", "--part", "1/2"]),
]
for t in glob.glob("/kaggle/input/*/results_*.tar") + glob.glob("/kaggle/input/*/*/results_*.tar"):
    name = os.path.basename(t)[len("results_"):-4]
    if name in [w[0] for w in WORKERS]:
        log("restore", t)
        os.makedirs(f"{W}/res/{name}", exist_ok=True)
        with tarfile.open(t) as tf:
            tf.extractall(f"{W}/res/{name}")

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

# ---------- 啟動 6 個 worker ----------
procs = {}
for name, gpu, a in WORKERS:
    cwd = f"{W}/res/{name}"
    lf = open(f"{LOGS}/{name}.log", "a")
    procs[name] = subprocess.Popen(
        [sys.executable, WORKER, "--code", CODE, "--config", CFG, "--data-dir", DATA, "--cwd", cwd,
         "--deadline", str(DEADLINE)] + a,
        env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=lf, stderr=subprocess.STDOUT)
    log("started", name, "gpu", gpu, a)


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
while any(p.poll() is None for p in procs.values()):
    time.sleep(30)
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

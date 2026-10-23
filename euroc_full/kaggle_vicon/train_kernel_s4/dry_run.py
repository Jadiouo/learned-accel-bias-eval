"""s4 本機 dry-run（只建 plan、不訓練）：用 main.py 內嵌的 WRAPPER_SRC / assign / RUN_IDS，
經 worker.py 的真實過濾鏈取得每個 worker 的 plan，驗證聯集 = missing_runs.txt、無重疊、都不在 runs_index.csv。
用法：python dry_run.py <euroc_full 根> <scratch 目錄>"""
import ast, csv, json, os, subprocess, sys
root, scratch = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
tree = ast.parse(open(f"{here}/main.py").read())
ns = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("RUN_IDS", "WRAPPER_SRC"):
        exec(compile(ast.Module([n], []), "m", "exec"), ns)
    if isinstance(n, ast.FunctionDef) and n.name == "assign":
        exec(compile(ast.Module([n], []), "m", "exec"), ns)
os.makedirs(scratch, exist_ok=True)
wp = f"{scratch}/worker_ids.py"; open(wp, "w").write(ns["WRAPPER_SRC"])
parts = ns["assign"](ns["RUN_IDS"], 4)
names = ["g0m0", "g0m1", "g1m0", "g1m1"]
got = {}
for nme, ids in zip(names, parts):
    r = subprocess.run([sys.executable, wp, "--dry-run", "--run-ids", ",".join(ids), "--worker-path", f"{root}/kaggle_vicon/train/worker.py",
                        "--code", root, "--config", f"{root}/configs/vicon.yaml", "--data-dir", f"{scratch}/nodata", "--cwd", f"{scratch}/{nme}"],
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("PLAN_IDS ")]
    assert r.returncode == 0 and line, (nme, r.returncode, r.stdout[-500:], r.stderr[-500:])
    got[nme] = json.loads(line[0][9:])
    print(nme, len(got[nme]), "runs:"); [print("   ", i) for i in got[nme]]
allids = [i for v in got.values() for i in v]
want = [l.strip() for l in open(f"{here}/missing_runs.txt") if l.strip()]
idx = {r["run_id"] for r in csv.DictReader(open(f"{root}/results_vicon/runs_index.csv"))}
print("union", len(set(allids)), "total", len(allids), "overlap-free", len(allids) == len(set(allids)))
print("union == missing_runs.txt:", set(allids) == set(want) and len(want) == 10)
print("in runs_index.csv:", sorted(set(allids) & idx), "(count %d, runs_index rows %d)" % (len(set(allids) & idx), len(idx)))
assert set(allids) == set(want) and len(allids) == len(set(allids)) == 10 and not (set(allids) & idx)
print("DRY-RUN OK")

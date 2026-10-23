import numpy as np, sys, os, time, itertools
from multiprocessing import Pool
from sim_truth import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
TRAJ = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]
SEEDS = [0, 1, 2, 3]
LEV_I = [0.05, 0.1, 0.25, 0.5, 1, 2, 5]
LEV_M = [1e-4, 5e-4, 2e-3]
R_TRUTH, R_TEST = 12, 30

def job(args):
    ti, sd, mode, li = args
    fn = f"{OUT}/{mode}_{TRAJ[ti]}_s{sd}_L{li}.npz"
    if os.path.exists(fn): return fn
    tr = load_traj(TRAJ[ti])
    rng = np.random.default_rng(1000 * ti + sd)
    eps, typ, amp, grp = make_pool(tr, rng)
    lev = (LEV_I if mode == "I" else LEV_M)[li]
    d2s, ates, atts, leaks = [], [], [], []
    for r in range(R_TRUTH + R_TEST):
        rr = np.random.default_rng([ti, sd, li, r, 7 if mode == "I" else 8])
        d2, ate, att, A = (run_mode_I(tr, eps, lev, rr) if mode == "I" else run_mode_M(tr, eps, lev, rr))
        d2s.append(d2); ates.append(ate); atts.append(att); leaks.append(A)
    d2s, ates = np.array(d2s), np.array(ates)           # (R,P)
    truthB = np.sqrt(d2s[:R_TRUTH].mean(0))
    np.savez(fn, rmse=bias_rmse(eps), iabe=iabe_rot(eps, tr["R"]), truthB=truthB,
             ate=ates[R_TRUTH:].T, ate_truthreal=ates[:R_TRUTH].T, typ=typ, amp=amp, grp=grp,
             att=np.mean(atts), leak=np.mean(leaks), lev=lev)
    return fn

if __name__ == "__main__":
    jobs = [(ti, sd, "I", li) for ti in range(3) for sd in SEEDS for li in range(len(LEV_I))] + \
           [(ti, sd, "M", li) for ti in range(3) for sd in SEEDS for li in range(len(LEV_M))]
    t0 = time.time()
    with Pool(14) as p:
        for i, f in enumerate(p.imap_unordered(job, jobs)):
            if i % 10 == 0: print(i, len(jobs), round(time.time() - t0), flush=True)
    print("done", time.time() - t0)

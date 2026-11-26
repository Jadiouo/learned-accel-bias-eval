# -*- coding: utf-8 -*-
"""Step 2b. Brute-force check of the closed-form quantities at NONZERO offsets: rebuild trajectories directly
(dint of world(C, a_raw - b_hat) + g minus the reference) for random (run, offset) and compare with f9_lib.quad / rmse_avg."""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f9_lib as F
L = F.L
D = pickle.load(open(HERE + "/ops.pkl", "rb")); OPS, HR, RUNS = D["OPS"], D["HR"], D["RUNS"]
rng = np.random.default_rng(20261126); worst = dict(ate=0.0, iabe=0.0, rmse=0.0, ate_const=0.0); n = 0
for seq in ("MH_01", "MH_03", "MH_05"):
    mi, S = F.load_seq_sources(seq)
    pref = L.dint(L.world(mi.R_ref, mi.a_raw - mi.b_ref) + mi.g_vec)
    cand = [r for r in RUNS if r["seq"] == seq]
    for _ in range(6):
        r = cand[rng.integers(len(cand))]
        bh = np.load(f"{L.ROOT}/results/runs/{r['run_id']}/series.npz")[f"b_hat_{seq}"]
        u = rng.normal(size=3); u /= np.linalg.norm(u); dl = F.SQ3 * 10 ** rng.uniform(-4, 0) * u
        bh2 = bh - dl                              # eps = b_ref - b_hat  ->  eps_real + dl
        for src in ("raw", "oracle", "vqf", "orc10"):
            C = S[src]; o = OPS[(seq, src)]
            direct = L.rms3(L.dint(L.world(C, mi.a_raw - bh2) + mi.g_vec) - pref)
            mine = float(F.quad(r["Y2"][src], r["vY"][src], o["H"], dl))
            worst["ate"] = max(worst["ate"], abs(direct - mine) / direct)
            # pure constant estimator b_ref + dl' (eps = -dl' ... sign irrelevant): Y = L
            direct_c = L.rms3(L.dint(L.world(C, mi.a_raw - (mi.b_ref - dl)) + mi.g_vec) - pref)
            mine_c = float(F.quad(o["L2"], o["vL"], o["H"], dl)); worst["ate_const"] = max(worst["ate_const"], abs(direct_c - mine_c) / direct_c)
        direct_i = L.rms3(L.dint(L.world(mi.R_ref, mi.b_ref - bh2)))
        worst["iabe"] = max(worst["iabe"], abs(direct_i - float(F.quad(r["I2"], r["vI"], HR[seq], dl))) / direct_i)
        direct_r = L.metrics.bias_rmse(bh2, mi.b_ref)["avg"]
        worst["rmse"] = max(worst["rmse"], abs(direct_r - float(F.rmse_avg(r["m2"], r["m1"], dl))) / direct_r)
        n += 1
print("selftest max rel err over", n, "random (run, offset):", worst)
json.dump(dict(n=n, max_rel_err=worst), open(HERE + "/selftest_offsets.json", "w"), indent=1)

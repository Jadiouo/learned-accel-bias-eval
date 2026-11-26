# -*- coding: utf-8 -*-
"""Step 3. Open-loop dose-response of direction agreement. Post hoc, CPU only, no retraining.
Pair (a,b) of synthetic estimators; delta = offset in the bias error eps (|delta| = sqrt(3)*d so that bias RMSE (paper 'avg') = d).
Design delta_x = d_b - d_a on a log grid (bias-RMSE units, same units as the 0.01 S15 threshold and the pool quantiles).
Truth ordering = bias-RMSE ordering. Agreement = P(sign(ATE_b-ATE_a) == truth).
Modes:
  C-pool : both constant. d_a drawn from the case's real bias-RMSE pool (110 DL runs), d_b = d_a + Dx.    [MAIN]
  C-zero : both constant, d_a = 0, d_b = Dx.
  C-fix  : both constant, d_a = 2.8e-3 (pool median), d_b = d_a + Dx.
  R-same : real run r (jitter kept): a = r, b = r + offset Dx. Truth = actual bias RMSE order.
  R-diff : a = real run r, b = other real run r' + offset Dx. Truth = actual bias RMSE order.
Directions: uniform on the sphere, fresh for every pair (3000 pairs / case = 30 chunks x 100 pairs, same draws reused for all Dx and all sources).
Writes dose_counts.csv (agree counts per mode, case, source, Dx) and dose_chunks.npz (per-chunk counts)."""
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f9_lib as F
D = pickle.load(open(HERE + "/ops.pkl", "rb")); OPS, HR, RUNS = D["OPS"], D["HR"], D["RUNS"]
units, *_ = F.H0.load_mh()
SRCS = list(F.NAMES)
GRID = 10 ** np.linspace(-4, 1, 51)      # Dx in bias-RMSE units (m/s^2)
NCH, PER = 30, 100; NP = NCH * PER
MODES = ["C-pool", "C-zero", "C-fix", "R-same", "R-diff"]
D_FIX = 2.8e-3
def unit(rng, n):
    u = rng.normal(size=(n, 3)); return u / np.linalg.norm(u, axis=1, keepdims=True)
t0 = time.time()
rows = []; CH = np.zeros((len(MODES), len(units), len(SRCS) + 1, len(GRID), NCH), np.int32)    # last source index = IABE
chunk = np.arange(NP) // PER
for ci, (reg, seq) in enumerate(units):
    rng = np.random.default_rng([20261126, ci])
    cr = [r for r in RUNS if r["regime"] == reg and r["seq"] == seq]; assert len(cr) == 110
    pool = np.array([r["rmse_avg_ref"] for r in cr])
    ua, ub = unit(rng, NP), unit(rng, NP)
    da_pool = pool[rng.integers(0, len(pool), NP)]
    ra = rng.integers(0, len(cr), NP); rb = (ra + rng.integers(1, len(cr), NP)) % len(cr)
    M1 = np.array([r["m1"] for r in cr]); M2 = np.array([r["m2"] for r in cr]); I2 = np.array([r["I2"] for r in cr]); VI = np.array([r["vI"] for r in cr])
    HRs = HR[seq]
    Dg = GRID[None, :, None]
    for mode in MODES:
        if mode.startswith("C-"):
            d_a = {"C-pool": da_pool, "C-zero": np.zeros(NP), "C-fix": np.full(NP, D_FIX)}[mode][:, None]
            d_b = d_a + GRID[None, :]                                                    # (NP, nG)
            ea = F.SQ3 * d_a[:, :, None] * ua[:, None, :] * np.ones_like(Dg)             # (NP,nG,3)
            eb = F.SQ3 * d_b[:, :, None] * ub[:, None, :]
            truth = np.ones((NP, len(GRID)), bool)                                       # d_b > d_a always
            I_a = F.quad(0.0, np.zeros(3), HRs, ea); I_b = F.quad(0.0, np.zeros(3), HRs, eb)
            ate = lambda s: (F.quad(OPS[(seq, s)]["L2"], OPS[(seq, s)]["vL"], OPS[(seq, s)]["H"], ea),
                             F.quad(OPS[(seq, s)]["L2"], OPS[(seq, s)]["vL"], OPS[(seq, s)]["H"], eb))
        else:
            r2 = ra if mode == "R-same" else rb
            zero = np.zeros((NP, len(GRID), 3)); db = F.SQ3 * GRID[None, :, None] * ub[:, None, :] * np.ones((NP, 1, 1))
            m1a, m2a = M1[ra][:, None, :], M2[ra][:, None]; m1b, m2b = M1[r2][:, None, :], M2[r2][:, None]
            truth = F.rmse_avg(m2b, m1b, db) > F.rmse_avg(m2a, m1a, zero)
            I_a = np.sqrt(I2[ra])[:, None] * np.ones((1, len(GRID)))
            I_b = F.quad(I2[r2][:, None], VI[r2][:, None, :], HRs, db)
            def ate(s, ra=ra, r2=r2, db=db):
                Ya = np.array([RUNS_c["Y2"][s] for RUNS_c in cr]); Va = np.array([RUNS_c["vY"][s] for RUNS_c in cr])
                return (np.sqrt(Ya[ra])[:, None] * np.ones((1, len(GRID))), F.quad(Ya[r2][:, None], Va[r2][:, None, :], OPS[(seq, s)]["H"], db))
        def record(name, v_a, v_b, si):
            ag = ((v_b > v_a) == truth)                                                  # (NP,nG)
            for ch in range(NCH): CH[MODES.index(mode), ci, si, :, ch] = ag[chunk == ch].sum(0)
            for gi, dx in enumerate(GRID):
                rows.append(dict(mode=mode, regime=reg, seq=seq, source=name, dx=dx, agree=int(ag[:, gi].sum()), n=NP))
        for si, s in enumerate(SRCS): record(s, *ate(s), si)
        record("IABE", I_a, I_b, len(SRCS))
    print(reg, seq, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(rows).to_csv(HERE + "/dose_counts.csv", index=False)
np.savez_compressed(HERE + "/dose_chunks.npz", CH=CH, modes=MODES, sources=SRCS + ["IABE"], grid=GRID,
                    units=[f"{r}|{s}" for r, s in units])
print("done %.0fs" % (time.time() - t0))

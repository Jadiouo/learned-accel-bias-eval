# -*- coding: utf-8 -*-
"""Step 1: per-run ATE for scaled oracle attitude; per-(case,config) leakage diagnostics. Reads only; writes into this folder."""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import f5_lib as F
L = F.L
t0 = time.time()
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
S_GRID = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0]
RM_TARGETS = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]
wide = pd.read_csv(L.ROOT + "/results/metrics_wide.csv").merge(
    pd.read_csv(L.ROOT + "/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
att = pd.read_csv(os.path.join(HERE, "..", "..", "attitude_controls", "ate_per_run_gated.csv"))
eps_unit = att[att.source == F.ORACLE].groupby(["regime", "seq"]).eps_rms.median()   # all runs of the case
iabe_unit = wide.groupby(["regime", "eval_sequence"]).iabe_rot.median()
dl = wide[wide.method.isin(DL)]
units = [(r, s) for r in "AB" for s in sorted(dl[dl.regime == r].eval_sequence.unique())]
BASE, ORA, PREF = {}, {}, {}
for seq in L.SEQS:
    mi, S = L.load_seq(seq); BASE[seq] = mi; ORA[seq] = S[F.ORACLE]
    PREF[seq] = L.dint(L.world(mi.R_ref, mi.a_raw - mi.b_ref) + mi.g_vec)
    print(seq, len(mi), "%.0fs" % (time.time() - t0), flush=True)

def diag(seq, Cg):
    mi = BASE[seq]; th, ya = F.tilt_yaw_deg(Cg, mi.R_ref)
    PL = L.rms3(L.dint(L.world(Cg - mi.R_ref, mi.a_raw - mi.b_ref)))
    return th.mean(), ya.mean(), PL

def solve_s(seq, eps, target, variant="W"):
    """s such that g*radians(mean tilt)/eps == target (log bisection; mean tilt monotone in s)."""
    mi = BASE[seq]
    f = lambda s: F.G * np.radians(F.tilt_yaw_deg(F.scaled_att(ORA[seq], mi.R_ref, s, variant), mi.R_ref)[0].mean()) / eps
    lo, hi = 1e-8, 1e4
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if f(mid) < target: lo = mid
        else: hi = mid
    return np.sqrt(lo * hi)

configs = {}      # (config name) -> {(regime,seq): Cg}
def add(name, fn, per_unit=False):
    configs[name] = {}
    for (reg, seq) in units: configs[name][(reg, seq)] = fn(reg, seq)
for var in "WBT":
    for s in S_GRID:
        # attitude is regime independent -> cache per seq
        cache = {seq: F.scaled_att(ORA[seq], BASE[seq].R_ref, s, var) for seq in L.SEQS}
        add(f"{var}|s={s:g}", lambda reg, seq, c=cache: c[seq])
RMS = {}
for tg in RM_TARGETS:
    def fn(reg, seq, tg=tg):
        s = solve_s(seq, eps_unit[(reg, seq)], tg); RMS[(tg, reg, seq)] = s
        return F.scaled_att(ORA[seq], BASE[seq].R_ref, s, "W")
    add(f"RM|r={tg:g}", fn)
print("configs", len(configs), "%.0fs" % (time.time() - t0), flush=True)

drows = []
for name, d in configs.items():
    for (reg, seq), Cg in d.items():
        th, ya, PL = diag(seq, Cg); eps = eps_unit[(reg, seq)]; ib = iabe_unit[(reg, seq)]
        drows.append(dict(config=name, regime=reg, seq=seq, theta_bar_deg=th, yaw_mean_deg=ya, leak_PL_m=PL, eps_rms=eps,
                          leak_ratio=F.G * np.radians(th) / eps, iabe_median=ib, leak_multiple=PL / ib,
                          s_used=RMS.get((float(name[5:]), reg, seq)) if name.startswith("RM|") else float(name.split("=")[1])))
pd.DataFrame(drows).to_csv(HERE + "/diag_by_case.csv", index=False)

recs = []
for n, (_, r) in enumerate(dl.iterrows()):
    seq = r.eval_sequence; mi = BASE[seq]; u = (r.regime, seq)
    bh = np.load(f"{L.ROOT}/results/runs/{r.run_id}/series.npz")[f"b_hat_{seq}"]; assert len(bh) == len(mi)
    x = mi.a_raw - bh
    for name, d in configs.items():
        Cg = d[u]
        recs.append((r.run_id, r.regime, r.method, int(r.seed), seq, name, L.rms3(L.dint(L.world(Cg, x) + mi.g_vec) - PREF[seq])))
    if n % 200 == 0: print("run", n, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(recs, columns=["run_id", "regime", "method", "seed", "seq", "config", "ate"]).to_csv(HERE + "/ate_scaled_per_run.csv", index=False)
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in RMS.items()}, open(HERE + "/rm_scales.json", "w"), indent=1)
print("done %.0fs" % (time.time() - t0))

"""H0 step 1: reproduce known numbers before anything else.
New pipeline (252 splits, jackknife t): raw nine 0.559 [0.460,0.657], IABE 0.802, noL 0.799, retest 0.987.
Old pipeline (1000 rng(0) splits, case bootstrap, original bootstrap RNG order): raw nine 0.554 [0.444, 0.674] (copy of F1 reproduce.py).
Also K=11 noise-ceiling check is in 03_ceiling.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib import *
units, bias, iabe, seeds = load_mh(); cl = [u[1] for u in units]; G = groups_of(units)
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
raw = ate_from_long(att, units, seeds, "a_raw_gyro(paper)"); ora = ate_from_long(att, units, seeds, "b_gyro-GT-bias(oracle)")
dec = pd.read_csv(R+"experiments/reviewer_checks/decomposition/f5_decomp_runs.csv")
def dmat(src, col): return {u: dec[(dec.src == src) & (dec.regime == u[0]) & (dec.seq == u[1])].pivot_table(index="seed", columns="method", values=col)[DL].values for u in units}
ag = Agree(units, bias, 11)
rows = []
for nm, m, target in (("bias RMSE retest", bias, 0.987), ("IABE", iabe, 0.802), ("raw ATE", raw, "0.559 [0.460, 0.657]"), ("oracle ATE", ora, "0.560 [0.423,0.697] (F7)"), ("raw noL", dmat("raw", "noL"), 0.799)):
    a, n = ag.counts(m); pt, lo, hi, g = jackknife(a, n, cl, G["nine cases pooled"])
    rows.append(dict(pipeline="new (252, jackknife t4)", measure=nm, point=pt, lo=lo, hi=hi, target=target)); print(rows[-1])
# ---- old pipeline (copy of F1 reproduce.py logic; pairs selection from F1 252-split table indexed by rng(0) draws)
rng = np.random.default_rng(0); hs = rng.integers(0, len(HALVES), 1000)
def counts_by_split(meas):
    A = np.zeros((252, 9)); N = np.zeros((252, 9))
    for hi, h1 in enumerate(HALVES):
        h2 = np.setdiff1d(ALL10, h1)
        for ui, u in enumerate(units):
            mask, s1 = ag.sel[(hi, u)]; N[hi, ui] = mask.sum()
            if mask.sum():
                m = meas[u][h2].mean(0); A[hi, ui] = (np.sign(m[ag.PI]-m[ag.PJ]) == s1)[mask].sum()
    return A, N
A, N = counts_by_split(raw)
gnames = ["A", "B", "ALL"]; Gd = {"A": G["regime A"], "B": G["regime B"], "ALL": G["nine cases pooled"]}
for i, u in enumerate(units): Gd[f"{u[0]}/{u[1]}"] = [i]; gnames.append(f"{u[0]}/{u[1]}")
measures = ["bias", "IABE", "raw", "oracle", "static", "mahony", "gated", "mahony5"]
Ao = counts_by_split(ora)[0]
store = {}
for g in gnames:
    gi = np.array(Gd[g])
    for m in measures:
        if len(gi) > 1:
            ix_all = [gi[rng.integers(0, len(gi), len(gi))] for _ in range(4000)]
            if m == "raw": store[g] = [A[hs][:, ix].sum()/N[hs][:, ix].sum() for ix in ix_all]
            if m == "oracle": store[g+"_o"] = [Ao[hs][:, ix].sum()/N[hs][:, ix].sum() for ix in ix_all]
for g in ["A", "B", "ALL"]:
    gi = np.array(Gd[g]); lo, hi = np.percentile(store[g], [2.5, 97.5]); lo2, hi2 = np.percentile(store[g+"_o"], [2.5, 97.5])
    rows.append(dict(pipeline="old (1000 rng0 splits, case bootstrap)", measure="raw ATE "+g, point=A[hs][:, gi].sum()/N[hs][:, gi].sum(), lo=lo, hi=hi,
                     target="0.554 [0.444, 0.674]" if g == "ALL" else "A 0.542 [0.450,0.629]/B 0.598 [0.270,0.918] (R-F csv)")); print(rows[-1])
    rows.append(dict(pipeline="old (1000 rng0 splits, case bootstrap)", measure="oracle ATE "+g, point=Ao[hs][:, gi].sum()/N[hs][:, gi].sum(), lo=lo2, hi=hi2,
                     target="0.558 [0.433,0.659]" if g == "ALL" else "A 0.573 / B 0.503")); print(rows[-1])
pd.DataFrame(rows).to_csv(HERE+"/reproduce.csv", index=False)

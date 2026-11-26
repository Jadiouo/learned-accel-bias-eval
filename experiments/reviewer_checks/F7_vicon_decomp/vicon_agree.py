"""F7 step 2: Vicon agreement of each decomposition variant with the bias-RMSE ranking. Reads vicon_decomp_runs.csv (own)."""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from agree_lib import *
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
SEQS = ["V1_01","V1_02","V1_03","V2_01","V2_02","V2_03"]
D = pd.read_csv(HERE + "/vicon_decomp_runs.csv"); units = SEQS
NROT = 30; VAR = ["full", "noL", "noQ", "I", "negL"] + [f"rot{j}" for j in range(NROT)]
def mat(src, col, cols):
    d = D[D.src == src]
    return {s: d[d.seq == s].pivot_table(index="seed", columns="method", values=col)[DL].values[:, cols] for s in SEQS}
biasm = {s: D[(D.src == "raw") & (D.seq == s)].pivot_table(index="seed", columns="method", values="bias_rmse")[DL].values for s in SEQS}
iabe_csv = {s: D[(D.src == "raw") & (D.seq == s)].pivot_table(index="seed", columns="method", values="iabe_rot")[DL].values for s in SEQS}
for s in SEQS: assert biasm[s].shape == (10, 11)
SETS = {"with_mamba(K=11)": list(range(11)), "no_mamba(K=10)": [i for i, m in enumerate(DL) if m != "mamba"]}
SPL = {"252": HALVES252, "rng0_1000": halves_rng0(1000)}
out = []; lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
# ---- reproduction of the paper's Vicon numbers (rng(0), 1000 splits, K=11, original t1 csv ATE)
ref = pd.read_csv(os.path.join(HERE, "..", "..", "..", "euroc_full", "analysis_vicon", "full_2026-10-23", "t1_ate_per_run_gated.csv"), comment="#")
refm = lambda src: {s: ref[(ref.source == src) & (ref.seq == s)].pivot_table(index="seed", columns="method", values="ate")[DL].values for s in SEQS}
cols = SETS["with_mamba(K=11)"]
ag = Agree(units, biasm, 11, SPL["rng0_1000"])
P("== REPRODUCTION (paper pipeline: rng(0), 1000 random splits, 11 methods; paper vicon_numbers.tex: raw .604 / oracle .535 / IABE .700 / ceiling .993)")
for nm, m in (("raw ATE (t1 csv)", refm("a_raw_gyro(paper)")), ("oracle ATE (t1 csv)", refm("b_gyro-GT-bias(oracle)")), ("IABE (metrics iabe_rot)", iabe_csv),
              ("raw ATE (my recon)", mat("raw", "full", cols)), ("oracle ATE (my recon)", mat("oracle", "full", cols)), ("IABE (my recon I)", mat("raw", "I", cols)), ("bias RMSE self-retest", biasm)):
    a, n = ag.counts(m); P("  %-26s pooled %.3f   pairs/split %.1f" % (nm, a.sum() / n.sum(), n.sum() / len(SPL["rng0_1000"])))
# ---- decomposition
for setn, cols in SETS.items():
    for spn, hv in SPL.items():
        ag = Agree(units, biasm, len(cols), hv)
        bsub = {s: biasm[s][:, cols] for s in SEQS}
        ag = Agree(units, bsub, len(cols), hv)
        for src in ("raw", "oracle"):
            res = {}
            for v in VAR: res[v] = ag.counts(mat(src, v, cols))
            a0, n0 = res["full"]
            for v in VAR:
                a, n = res[v]; pt, lo, hi, se, loo = jackknife(a, n, units)
                dpt, dlo, dhi = jackknife_diff(a, n, a0, n0, units)
                out.append(dict(set=setn, splits=spn, src=src, variant=v, agree=pt, jk_lo=lo, jk_hi=hi, d_vs_full=dpt, d_lo=dlo, d_hi=dhi,
                                n_pairs_per_split=n.sum() / len(hv), per_seq=";".join(f"{s}:{a[i]/n[i]:.2f}" if n[i] else f"{s}:na" for i, s in enumerate(SEQS))))
O = pd.DataFrame(out); O.to_csv(HERE + "/vicon_agree.csv", index=False)
pd.set_option("display.width", 220)
for setn in SETS:
    for src in ("raw", "oracle"):
        P(f"\n== Vicon {setn}  {src} attitude  (252 splits, leave-one-sequence-out jackknife t5, 6 sequences)")
        x = O[(O["set"] == setn) & (O.splits == "252") & (O.src == src)]
        P(x[~x.variant.str.startswith("rot")][["variant", "agree", "jk_lo", "jk_hi", "d_vs_full", "d_lo", "d_hi", "n_pairs_per_split"]].round(3).to_string(index=False))
        r = x[x.variant.str.startswith("rot")].agree
        P("  random-direction L (30 rotations): mean %.3f  min %.3f  25%% %.3f  median %.3f  75%% %.3f  max %.3f  | fraction >= noL-0.05: %.2f" %
          (r.mean(), r.min(), r.quantile(.25), r.median(), r.quantile(.75), r.max(), (r >= x[x.variant == "noL"].agree.iloc[0] - .05).mean()))
        P("  per-sequence (252 splits): ")
        for v in ["full", "noL", "noQ", "I", "negL"]: P("    %-5s %s" % (v, x[x.variant == v].per_seq.iloc[0]))
open(HERE + "/vicon_agree_stdout.txt", "w").write("\n".join(lines))

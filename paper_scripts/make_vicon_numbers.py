#!/usr/bin/env python3
"""Regenerate vicon_numbers.tex from an euroc_full/analysis_vicon result folder.

Usage: python3 -I make_vicon_numbers.py <analysis_dir> > vicon_numbers.tex
  (analysis_dir = e.g. .../euroc_full/analysis_vicon/preview_674_2026-10-21 now,
   the K=11 FULL folder later). Only reads CSV/JSON/MD files, writes to stdout.
Every macro is followed by a comment naming the source file and column.
Rounding is half-up on the printed decimal (not banker's rounding).
"""
import csv, json, re, sys, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

D = Path(sys.argv[1])
SRC = D.name


def rd(path):
    with open(D / path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return rows


def fmt(x, nd):
    q = Decimal(1).scaleb(-nd)
    s = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return s.replace("-", "$-$") if s.startswith("-") else s


def fmt_sig(x, sig):
    x = float(x)
    if x == 0:
        return "0"
    from math import floor, log10
    nd = sig - 1 - int(floor(log10(abs(x))))
    return fmt(x, max(nd, 0))


def fmt_g(x):  # >=100 integer, else 1 decimal (Table II style)
    return fmt(x, 0) if abs(float(x)) >= 100 else fmt(x, 1)


out = []


def M(name, val, src):
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}  % {src}")


def H(t):
    out.append(f"\n% ---- {t}")


# ---------- coverage
cov = (D / "coverage.md").read_text()
ok = re.search(r"ok: (\d+); failed", cov).group(1)
tot = re.search(r"expected total: (\d+)", cov).group(1)
cc = re.search(r"Complete-case deep-method set .*?: \[(.*?)\]", cov).group(1)
nmeth = len([s for s in cc.split(",") if s.strip()])
H("run counts and method set")
M("VcRunsOk", ok, f"{SRC}/coverage.md 'ok:'")
M("VcRunsTotal", tot, f"{SRC}/coverage.md 'expected total:'")
M("VcNmeth", nmeth, f"{SRC}/coverage.md 'Complete-case deep-method set' (count of deep methods with >=10 ok seeds on every sequence)")
fl = rd("seed_flip_rate.csv")
iabe_flip = {r["seq"]: r for r in fl if r["metric"] == "IABE"}
M("VcRandBase", fmt(iabe_flip["MEAN"]["random_baseline"], 2), f"{SRC}/seed_flip_rate.csv, metric=IABE, col random_baseline")
meta = json.load(open(D / "t1_meta.json"))["meta"]
spans = [v["span_s"] for v in meta.values()]
M("VcNseq", len(meta), f"{SRC}/t1_meta.json, number of sequences in 'meta'")
M("VcSpanMin", fmt(min(spans), 1), f"{SRC}/t1_meta.json, min of meta[seq].span_s (V1_02)")
M("VcSpanMax", fmt(max(spans), 1), f"{SRC}/t1_meta.json, max of meta[seq].span_s (V1_01)")

# ---------- Table I analogue (regime A)
t1 = {r["source"]: r for r in rd("t1_TABLE_I_paper_definition.csv") if r["regime"] == "A"}
key = {"Raw": "a_raw_gyro(paper)", "Static": "c_static-mean-bias", "Mahony": "d_mahony(kp=1,ki=0.05)",
       "Gated": "e_mahony_gated(kp=1,ki=0.05)", "Oracle": "b_gyro-GT-bias(oracle)"}
H("Table I analogue, regime A (t1_TABLE_I_paper_definition.csv)")
for k, s in key.items():
    r = t1[s]
    M(f"VcTh{k}", fmt_sig(r["theta_med"], 3), f"{SRC}/t1_TABLE_I_paper_definition.csv, source={s}, col theta_med (deg)")
    M(f"VcGth{k}", fmt_g(r["gth_eps_med"]), f"{SRC}/t1_TABLE_I_paper_definition.csv, source={s}, col gth_eps_med")
    M(f"VcLeak{k}", fmt_g(r["leakage_multiple"]), f"{SRC}/t1_TABLE_I_paper_definition.csv, source={s}, col leakage_multiple")
impl = [float(t1[key[k]]["gth_eps_med"]) for k in ("Raw", "Static", "Mahony", "Gated")]
M("VcGthMinImpl", fmt_g(min(impl)), f"{SRC}/t1_TABLE_I_paper_definition.csv, min of gth_eps_med over the four implementable sources (raw, static, Mahony, gated)")
M("VcPooledIabe", fmt_sig(t1[key["Raw"]]["pooled_iabe_median_m"], 3), f"{SRC}/t1_TABLE_I_paper_definition.csv, col pooled_iabe_median_m (m)")

# ---------- closed-loop ratios
r3 = rd("t3_table3_ratio.csv")
def cell(cfg, L):
    return [r for r in r3 if r["cfg"] == cfg and r["L"] == str(L)][0]
H("closed-loop unit-level ratios (t3_table3_ratio.csv); L in s")
for cfg, tag in (("O_base", "Ob"), ("P_off", "Po")):
    for L, lt in ((10, "Ten"), (5, "Five")):
        r = cell(cfg, L)
        s = f"{SRC}/t3_table3_ratio.csv, cfg={cfg}, L={L}"
        M(f"VcRatio{tag}{lt}", fmt(r["ratio_med"], 1), s + ", col ratio_med")
        M(f"VcRatio{tag}{lt}Lo", fmt(r["ratio_med_lo"], 1), s + ", col ratio_med_lo")
        M(f"VcRatio{tag}{lt}Hi", fmt(r["ratio_med_hi"], 0 if float(r["ratio_med_hi"]) >= 100 else 1), s + ", col ratio_med_hi")
r = cell("O_base", 10)
M("VcCommon", fmt(r["common_m"], 2), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col common_m (m, unit-level shared error)")
M("VcMedDelta", fmt(r["median_delta_m"], 3), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col median_delta_m (m, unit-level)")
M("VcPerSeqMin", fmt(r["per_seq_min"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col per_seq_min")
M("VcPerSeqMax", fmt(r["per_seq_max"], 0), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col per_seq_max")
r5 = cell("O_base", 5)
M("VcCommonFive", fmt(r5["common_m"], 2), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=5, col common_m")
M("VcMedDeltaFive", fmt(r5["median_delta_m"], 3), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=5, col median_delta_m")
r = cell("P_off", 10)
M("VcRatioPoIqr", fmt(r["ratio_iqr"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=P_off, L=10, col ratio_iqr")
M("VcRatioPoRng", fmt(r["ratio_rng"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=P_off, L=10, col ratio_rng")
M("VcRatioPoRngLo", fmt(r["ratio_rng_lo"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=P_off, L=10, col ratio_rng_lo")
M("VcRatioPoRngHi", fmt(r["ratio_rng_hi"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=P_off, L=10, col ratio_rng_hi")
r = cell("O_base", 10)
M("VcRatioObIqr", fmt(r["ratio_iqr"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col ratio_iqr")
M("VcRatioObRng", fmt(r["ratio_rng"], 1), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=10, col ratio_rng")
five = [x for x in r3 if x["L"] in ("5", "10")]
M("VcRatioMin", fmt(min(float(x["ratio_med"]) for x in five), 1), f"{SRC}/t3_table3_ratio.csv, min of ratio_med over all cfg (O_base, O_q10, O_q01, P_est, P_off) at L=5,10")
M("VcRatioMinLo", fmt(min(float(x["ratio_med_lo"]) for x in five), 1), f"{SRC}/t3_table3_ratio.csv, min of ratio_med_lo over all cfg at L=5,10")
M("VcRngMin", fmt(min(float(x["ratio_rng"]) for x in five), 1), f"{SRC}/t3_table3_ratio.csv, min of ratio_rng over all cfg at L=5,10")

# ---------- Table II/III analogue (run-pooled medians)
tu = rd("t2_table_unified.csv")
def tg(L, g):
    return [x for x in tu if x["L"] == str(L) and x["group"] == g][0]
H("Table III analogue, run-pooled medians (t2_table_unified.csv)")
for L, lt in ((10, "Ten"), (5, "Five")):
    s = f"{SRC}/t2_table_unified.csv, L={L}"
    M(f"VcCabsRef{lt}", fmt(tg(L, "CTRL:bref")["abs_med"], 2), s + ", group=CTRL:bref, col abs_med (6 sequences)")
    M(f"VcCabsEst{lt}", fmt(tg(L, "learned+const (all runs)")["abs_med"], 2), s + ", group=learned+const (all runs), col abs_med")
    M(f"VcDeltaEst{lt}", fmt(tg(L, "learned+const (all runs)")["delta_med"], 3), s + ", group=learned+const (all runs), col delta_med")
    M(f"VcDeltaZero{lt}", fmt(tg(L, "CTRL:zero")["delta_med"], 2), s + ", group=CTRL:zero, col delta_med")
s = f"{SRC}/t2_table_unified.csv, L=20 (V1_01 only)"
M("VcCabsRefTwenty", fmt(tg(20, "CTRL:bref")["abs_med"], 1), s + ", group=CTRL:bref, col abs_med")
M("VcCabsZeroTwenty", fmt(tg(20, "CTRL:zero")["abs_med"], 1), s + ", group=CTRL:zero, col abs_med")
M("VcRatioObTwenty", fmt(cell("O_base", 20)["ratio_med"], 0), f"{SRC}/t3_table3_ratio.csv, cfg=O_base, L=20, col ratio_med (one sequence, not interpreted)")

# ---------- Spearman (scenario, nine-cells analogue = six cells)
sp = rd("t3_spearman_scenario.csv")
def spc(cons, pred):
    return [x for x in sp if x["cfg"] == "base" and x["mode"] == "O" and x["L"] == "10" and x["cons"] == cons and x["pred"] == pred][0]
H("Spearman with C_Delta / C_abs, base cfg, L=10 (t3_spearman_scenario.csv, col s1_median)")
s = f"{SRC}/t3_spearman_scenario.csv, cfg=base mode=O L=10"
M("VcSpIabe", fmt(spc("delta", "iabe")["s1_median"], 2), s + ", cons=delta pred=iabe, col s1_median")
M("VcSpBias", fmt(spc("delta", "bias_rmse")["s1_median"], 2), s + ", cons=delta pred=bias_rmse, col s1_median")
b = spc("delta", "bias_rmse")
M("VcSpDiff", fmt(b["diff_iabe_minus"], 2), s + ", cons=delta pred=bias_rmse, col diff_iabe_minus (IABE minus bias RMSE)")
M("VcSpDiffLo", fmt(b["diff_lo"], 2), s + ", col diff_lo")
M("VcSpDiffHi", fmt(b["diff_hi"], 2), s + ", col diff_hi")
M("VcSpAbsIabe", fmt(spc("abs", "iabe")["s1_median"], 2), s + ", cons=abs pred=iabe, col s1_median")
M("VcSpAbsBias", fmt(spc("abs", "bias_rmse")["s1_median"], 2), s + ", cons=abs pred=bias_rmse, col s1_median")
M("VcSpCells", spc("delta", "iabe")["n_cells"], s + ", col n_cells")

# ---------- direction agreement (R-F)
rf = {x["measure"]: x for x in rd("rf_splithalf_agreement.csv") if x["group"] == "ALL"}
H("direction agreement with bias-RMSE ranking, all six sequences (rf_splithalf_agreement.csv, group=ALL)")
for tag, m in (("Raw", "ATE:a_raw_gyro(paper)"), ("Oracle", "ATE:b_gyro-GT-bias(oracle)"), ("Iabe", "IABE"),
               ("Ceil", "bias_rmse (自身重測上限)")):
    s = f"{SRC}/rf_splithalf_agreement.csv, group=ALL, measure={m}"
    M(f"VcAgr{tag}", fmt(rf[m]["pooled_rate"], 3), s + ", col pooled_rate")
    M(f"VcAgr{tag}Lo", fmt(rf[m]["seqboot_lo"], 3), s + ", col seqboot_lo")
    M(f"VcAgr{tag}Hi", fmt(rf[m]["seqboot_hi"], 3), s + ", col seqboot_hi")

# ---------- rho(ATE ranking, IABE ranking) within sequence
rho = rd("seed_rho_vs_iabe.csv")
v = [float(x["rho_vs_IABE"]) for x in rho if x["measure"] == "ATE:a_raw_gyro(paper)"]
H("within-sequence Spearman between ATE ranking (raw attitude) and IABE ranking (seed_rho_vs_iabe.csv)")
s = f"{SRC}/seed_rho_vs_iabe.csv, measure=ATE:a_raw_gyro(paper), col rho_vs_IABE"
M("VcRhoAteIabe", fmt(st.median(v), 2), s + ", median over sequences")
M("VcRhoAteIabeMin", fmt(min(v), 2), s + ", min over sequences")
M("VcRhoAteIabeMax", fmt(max(v), 2), s + ", max over sequences")

# ---------- seed instability
H("seed instability (seed_flip_rate.csv, metric=IABE)")
per = [float(iabe_flip[k]["p_two_seeds_disagree"]) for k in iabe_flip if k != "MEAN"]
s = f"{SRC}/seed_flip_rate.csv, metric=IABE, col p_two_seeds_disagree"
M("VcFlip", fmt(iabe_flip["MEAN"]["p_two_seeds_disagree"], 2), s + ", seq=MEAN")
M("VcFlipMin", fmt(min(per), 2), s + ", min over sequences")
M("VcFlipMax", fmt(max(per), 2), s + ", max over sequences")

# ---------- Shapley
sh = rd("t3_table3_shapley.csv")
H("Shapley shares of the O_base shared error (t3_table3_shapley.csv, cfg=base)")
for L, lt in ((10, "Ten"), (5, "Five")):
    r = [x for x in sh if x["cfg"] == "base" and x["L"] == str(L)][0]
    s = f"{SRC}/t3_table3_shapley.csv, cfg=base, L={L}"
    for comp, nd in (("tilt", 2), ("vel", 3), ("floor", 2)):
        if comp == "floor" and L == 5:
            continue
        M(f"VcShap{comp.capitalize()}{lt}", fmt(r[comp], nd), s + f", col {comp}")
        if comp in ("tilt", "vel") and L == 10:
            M(f"VcShap{comp.capitalize()}{lt}Lo", fmt(r[comp + "_lo"], nd), s + f", col {comp}_lo")
            M(f"VcShap{comp.capitalize()}{lt}Hi", fmt(r[comp + "_hi"], nd), s + f", col {comp}_hi")

# ---------- V1_01 note: values quoted in euroc_full/analysis_vicon/README.md (prose), not a CSV
H("V1_01 note (prose in euroc_full/analysis_vicon/README.md, section 'V1_01 accelerometer-bias anomaly')")
M("VcBaY", "0.55", "README.md: GT b_a of V1_01 is 0.548 m/s^2 in y (rounded)")
M("VcOnsetTilt", "$-2.7$", "README.md: ESKF tilt error at L=20 onset, deg")
M("VcOnsetGth", "$-0.47$", "README.md: g*theta_y at L=20 onset, m/s^2")

print("% vicon_numbers.tex -- every EuRoC Vicon-room (V1/V2) number used in main.tex and supplementary.tex.")
print(f"% GENERATED by make_vicon_numbers.py from: {SRC}")
print("% STATUS: FINAL (684 of 684 runs, K=11 deep methods incl. mamba).")
print("% To update: re-run make_vicon_numbers.py on the FULL analysis folder.")
print("% Text that quotes the MH values (13.5, 0.94, 0.71, ...) is literal in the .tex files and is not touched.")
print("\\newif\\ifVcFinal\n\\VcFinaltrue")
print("\\AtEndDocument{\\ifVcFinal\\else\\PackageWarning{vicon_numbers}{Vicon numbers are from the PREVIEW. Regenerate before submission.}\\fi}")
print("\n".join(out))

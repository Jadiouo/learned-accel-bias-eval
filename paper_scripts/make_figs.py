"""Build all figures for main.tex and supplementary.tex from existing CSV files.
No experiment is rerun. Output: vector PDF in figs/. Run: python3 make_figs.py
Colours: Okabe-Ito palette (colour-blind safe); marker shapes/hatches add redundancy.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

R = Path(__file__).resolve().parents[1]  # repo root
CL = R / "experiments/closed_loop"
AC = R / "experiments/attitude_controls"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", pink="#CC79A7", grey="#7F7F7F")
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif",
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
})
W = 3.5  # IEEE single column, inches


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.4); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="black", ls="-", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arr(x0, y0, x1, y1, c="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c))
    box(0.0, 5.4, 2.5, 1.6, "bias error\n$\\varepsilon$", fc="#DCEEF9")
    box(0.0, 2.3, 2.5, 1.6, "tilt error\n$\\theta$", fc="#FBE3CF")
    box(3.5, 5.4, 2.5, 1.6, "double\nintegration")
    box(3.5, 2.3, 2.5, 1.6, "double\nintegration")
    arr(2.5, 6.2, 3.5, 6.2); arr(2.5, 3.1, 3.5, 3.1)
    box(7.3, 3.7, 2.7, 1.8, "ATE:\nboth lanes", fc="#EEEEEE")
    arr(6.0, 6.2, 7.3, 5.0); arr(6.0, 3.1, 7.3, 4.2)
    ax.text(4.75, 7.75, "shared attitude: IABE\n(common terms cancel)", fontsize=8, ha="center", va="center", color="#007A5A")
    ax.add_patch(Rectangle((3.4, 5.3), 2.7, 1.8, fill=False, ec=OI["green"], lw=1.2))
    box(0.0, 0.0, 10.0, 1.2, "ESKF onset states (IV-B): different quantity", ls="--", fs=8)
    save(fig, "fig1_error_flow.pdf")


# ------------------------------------------------------------------ Fig. 2 (scatter)
def fig_scatter():
    c = pd.read_csv(CL / "consequences_long.csv")
    c = c[(c.cfg == "base") & (c["mode"] == "O")]
    ctrl = c.run_id.str.startswith("CTRL:")
    bref = c[c.run_id == "CTRL:bref"].set_index(["seq", "L", "rep"])["abs"]
    est = c[~ctrl].copy()
    est["bref"] = [bref.get((s, l, r), np.nan) for s, l, r in zip(est.seq, est.L, est.rep)]
    est = est.dropna(subset=["bref"])
    cols = [OI["blue"], OI["verm"], OI["green"], OI["pink"], OI["orange"]]
    mk = ["o", "s", "^", "D", "v"]
    seqs = sorted(est.seq.unique())
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.75), sharey=True)
    for ax, L in zip(axs, (5, 10, 20)):
        d = est[est.L == L]
        for i, s in enumerate(seqs):
            ds = d[d.seq == s]
            if ds.empty:
                continue
            ax.scatter(ds.delta.clip(lower=1e-4), ds.bref, s=1.5, color=cols[i], alpha=0.12, lw=0, rasterized=True)
            ax.scatter(ds.delta.median(), ds.bref.median(), s=22, color=cols[i], marker=mk[i],
                       ec="black", lw=0.5, zorder=3, label=s if L == 5 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-3, 3); ax.set_ylim(0.2, 30)
        xs = np.array([1e-3, 3])
        ax.plot(xs, 10 * xs, color=OI["grey"], lw=0.7, ls="--")
        ax.set_title(f"$L={L}$ s" + (" (2 seq.)" if L == 20 else ""), fontsize=8)
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["$10^{-3}$", "", "$10^{-1}$", "1"])
    axs[0].set_ylabel("$\\mathbf{b}_{\\mathrm{ref}}$ $C_{\\mathrm{abs}}$ (m)")
    fig.supxlabel("per-source $C_{\\Delta}$ (m)", fontsize=8, y=0.08)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=8,
               frameon=False, handletextpad=0.1, bbox_to_anchor=(0.5, 1.09), columnspacing=0.4)
    fig.subplots_adjust(wspace=0.08, bottom=0.27, top=0.82, left=0.13, right=0.99)
    save(fig, "fig2_scatter.pdf")


# ------------------------------------------------------------------ Fig. 3 (Shapley)
def fig_shapley():
    s = pd.read_csv(CL / "table3_shapley.csv")
    parts = [("tilt", "tilt"), ("vel", "velocity"), ("pos", "position"), ("yaw", "yaw"), ("bg", "gyro bias"), ("floor", "unattrib.")]
    cfgs = [("base", "base", OI["blue"], None), ("q10", "$Q\\times10$", OI["orange"], "//"), ("q01", "$Q\\times0.1$", OI["green"], "xx")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    for ax, L in zip(axs, (5, 10)):
        for j, (cfg, lab, col, h) in enumerate(cfgs):
            r = s[(s.cfg == cfg) & (s.L == L)].iloc[0]
            v = np.array([r[p] for p, _ in parts])
            lo = v - np.array([r[p + "_lo"] for p, _ in parts]); hi = np.array([r[p + "_hi"] for p, _ in parts]) - v
            x = np.arange(len(parts)) + (j - 1) * 0.26
            ax.bar(x, v, width=0.25, color=col, hatch=h, ec="black", lw=0.4, label=lab if L == 5 else None,
                   yerr=[lo, hi], error_kw=dict(lw=0.5, capsize=1))
        ax.set_xticks(range(len(parts))); ax.set_xticklabels([n for _, n in parts], rotation=50, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"$L={L}$ s", fontsize=8)
    axs[0].set_ylabel("Shapley share")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.55, 1.12))
    fig.subplots_adjust(wspace=0.06, bottom=0.30, top=0.88, left=0.13, right=0.99)
    save(fig, "fig3_shapley.pdf")


# ------------------------------------------------------------------ Fig. 4 (agreement)
def fig_agree():
    a = pd.read_csv(R / "verification/R-F_noncircular/part2_splithalf_agreement_stratified.csv")
    items = [("IABE", "IABE", OI["grey"], "..."),
             ("ATE:a_raw_gyro(paper)", "raw", OI["blue"], None),
             ("ATE:b_gyro-GT-bias(oracle)", "oracle", OI["verm"], "//"),
             ("ATE:c_static-mean-bias", "static", OI["green"], "xx"),
             ("ATE:d_mahony(kp=1,ki=0.05)", "Mahony", OI["pink"], "\\\\"),
             ("ATE:e_mahony_gated(kp=1,ki=0.05)", "gated M.", OI["orange"], "--")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True)
    for ax, g in zip(axs, ("A", "B")):
        d = a[a.group == g].set_index("measure")
        for i, (m, lab, col, h) in enumerate(items):
            r = d.loc[m]
            ax.bar(i, r.pooled_rate, color=col, hatch=h, ec="black", lw=0.4, width=0.75,
                   yerr=[[r.pooled_rate - r.seqboot_lo], [r.seqboot_hi - r.pooled_rate]],
                   error_kw=dict(lw=0.6, capsize=1.5))
            if g == "B":
                per = a[a.group.str.startswith("B/") & (a.measure == m)].pooled_rate
                ax.scatter([i] * len(per), per, s=7, color="black", marker="_", lw=0.8, zorder=4)
        ax.axhline(0.988, color="black", lw=0.7, ls="--")
        ax.axhline(0.5, color=OI["grey"], lw=0.5, ls=":")
        ax.set_xticks(range(len(items))); ax.set_xticklabels([l for _, l, _, _ in items], rotation=50, ha="right", fontsize=8)
        ax.set_title(("Regime A (5 seq.)" if g == "A" else "Regime B (4 seq.)"), fontsize=8)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("direction agreement")
    axs[0].text(5.4, 0.93, "0.99", fontsize=8, ha="right", va="top")
    fig.subplots_adjust(wspace=0.06, bottom=0.28, top=0.9, left=0.13, right=0.99)
    save(fig, "fig4_agreement.pdf")


# ------------------------------------------------------------------ Fig. 5 (heatmap)
DEEP = ["tcn", "transformer_s", "transformer_m", "transformer_l", "mamba", "m1_cnn", "m2", "m3_lstm", "m4", "m5", "m6"]


def fig_heatmap(seqs=("MH_01", "MH_02", "MH_03", "MH_04", "MH_05"), name="fig5_heatmap.pdf"):
    m = pd.read_csv(R / "euroc_full/results/metrics_wide.csv")
    ri = pd.read_csv(R / "euroc_full/results/runs_index.csv")[["run_id", "method", "regime", "seed"]]
    m = m.merge(ri, on="run_id")
    m = m[(m.regime == "A") & m.method.isin(DEEP)]
    fig, axs = plt.subplots(1, len(seqs), figsize=(2 * W + 0.1, 2.5), sharey=True)
    for ax, sq in zip(np.atleast_1d(axs), seqs):
        d = m[m.eval_sequence == sq].pivot_table(index="method", columns="seed", values="bias_rmse_avg").loc[DEEP]
        rk = d.rank(axis=0, method="min")
        im = ax.imshow(rk.values, cmap="viridis_r", vmin=1, vmax=11, aspect="auto")
        ax.set_xticks([0, 4, 9]); ax.set_xticklabels([str(rk.columns[k]) for k in (0, 4, 9)], fontsize=8)
        ax.set_title(sq.replace("_", "\\_") if False else sq, fontsize=8)
        ax.set_xlabel("seed", fontsize=8)
    np.atleast_1d(axs)[0].set_yticks(range(len(DEEP)))
    np.atleast_1d(axs)[0].set_yticklabels([x for x in DEEP], fontsize=8)
    cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.02); cb.set_label("rank (1 = best)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    save(fig, name)


# ------------------------------------------------------------------ Supplement: dose-response
def fig_dose():
    d = pd.read_csv(AC / "dose_points_gated.csv")
    x = d.tilt_mean_deg.values; y = d.leak_m.values
    k, b = np.polyfit(np.log10(x), np.log10(y), 1)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    gated = d.source.str.contains("gated")
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.scatter(x[~gated], y[~gated], s=12, color=OI["blue"], ec="black", lw=0.3, label="sources and variants")
    ax.scatter(x[gated], y[gated], s=22, marker="D", color=OI["verm"], ec="black", lw=0.4, label="gated Mahony")
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 20)
    ax.plot(xx, 10 ** (b + k * np.log10(xx)), color="black", lw=0.8, ls="--", label=f"fit, slope {k:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean tilt error $\\bar\\theta$ (deg)"); ax.set_ylabel("leakage $\\|\\mathcal{P}[\\mathbf{l}]\\|$ (m)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "figS_dose.pdf")
    return dict(slope=k, rho=rho, n=len(d))


# ------------------------------------------------------------------ Supplement: simulation
def fig_sim():
    # IABE / Truth_B median and 5-95% from results_tables.md T1 (Mode M rows) is constant 1.71 [1.02, 10.2];
    # plot ATE/Truth_B and IABE/Truth_B versus attitude error RMS (Mode I and M).
    rows = [("I", 0.05, 0.05, 2.2, 1.00, 1.00, 1.00), ("I", 0.1, 0.10, 4.2, 1.00, 1.00, 1.00),
            ("I", 0.25, 0.24, 10.0, 1.00, 1.00, 1.00), ("I", 0.5, 0.48, 19.4, 1.00, 1.00, 1.00),
            ("I", 1, 0.97, 38.9, 1.00, 1.00, 1.00), ("I", 2, 1.94, 76.2, 1.00, 1.00, 1.00),
            ("I", 5, 4.77, 198.4, 1.00, 0.99, 1.01), ("M", 1e-4, 1.79, 6.9, 1.71, 1.02, 10.20),
            ("M", 5e-4, 2.33, 8.8, 1.71, 1.02, 10.20), ("M", 2e-3, 5.87, 21.9, 1.71, 1.02, 10.20)]
    # parse directly from the table file to avoid transcription errors
    txt = (R / "experiments/sim_truth/results_tables.md").read_text().split("## T2")[0]
    parsed = []
    for ln in txt.splitlines():
        mm = re.match(r"\|\s*([IM])\s*\|\s*([\d.e-]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
        if mm:
            parsed.append((mm.group(1),) + tuple(float(g) for g in mm.groups()[1:]))
    assert len(parsed) == 10, parsed
    P = pd.DataFrame(parsed, columns=["mode", "lev", "att", "ate", "iabe", "lo", "hi"])
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for mode, col, mk, lab in (("I", OI["blue"], "o", "Mode I"), ("M", OI["verm"], "s", "Mode M")):
        d = P[P["mode"] == mode]
        ax.plot(d.att, d.ate, marker=mk, color=col, lw=0.8, ms=3.5, label=f"ATE / truth, {lab}")
    d = P[P["mode"] == "M"]
    ax.errorbar(d.att, d.iabe, yerr=[d.iabe - d.lo, d.hi - d.iabe], fmt="D", color=OI["black"], ms=3.5, lw=0.7,
                capsize=2, label="IABE / truth, Mode M (5--95%)")
    ax.axhline(1, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("attitude error RMS (deg)"); ax.set_ylabel("ratio to truth consequence")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "figS_sim.pdf")


# ------------------------------------------------------------------ Graphical abstract
def fig_ga():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 24); ax.set_ylim(0, 8.7); ax.axis("off")
    fs = 8

    def box(x, y, w, h, t, fc="white", ls="-", fsz=fs):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15", fc=fc, ec="black", lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fsz)

    def arr(x0, y0, x1, y1, c="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=c))
    # left: accelerometer icon and two lanes
    box(0.2, 3.2, 2.6, 2.2, "MEMS\naccelerometer", fc="#EEEEEE")
    box(4.2, 5.4, 2.6, 1.5, "double\nintegration", fc="#DCEEF9")
    box(4.2, 1.2, 2.6, 1.5, "double\nintegration", fc="#FBE3CF")
    arr(2.8, 4.9, 4.2, 6.1); arr(2.8, 3.7, 4.2, 2.0)
    ax.text(3.35, 6.1, "bias", fontsize=fs, ha="center", color=OI["blue"], fontweight="bold")
    ax.text(3.35, 2.2, "tilt", fontsize=fs, ha="center", color=OI["verm"], fontweight="bold")
    ax.text(5.5, 4.05, "two lanes,\nnot merged", fontsize=8, ha="center", style="italic")
    # middle: position-error bar
    arr(6.8, 6.1, 8.4, 4.9); arr(6.8, 2.0, 8.4, 3.6)
    bx, by, bw, bh = 8.6, 2.2, 3.9, 4.2
    ax.add_patch(Rectangle((bx, by), bw, bh * 0.82, fc="#E8E8E8", ec="black", lw=0.8))
    ax.add_patch(Rectangle((bx, by + bh * 0.82), bw, bh * 0.18, fc=OI["green"], ec="black", lw=0.8))
    ax.add_patch(Rectangle((bx, by), bw, bh * 0.82 * 0.55, fc=OI["orange"], ec="black", lw=0.8, alpha=0.9))
    ax.text(bx + bw / 2, by + bh * 0.82 * 0.27, "outage-onset\ntilt", ha="center", va="center", fontsize=8)
    ax.text(bx + bw / 2, by + bh * 0.82 * 0.78, "outage-onset\nvelocity", ha="center", va="center", fontsize=8)
    ax.text(bx + bw + 0.2, by + bh * 0.35, "shared by all\nestimators", fontsize=8, va="center")
    ax.text(bx + bw + 0.2, by + bh * 0.91, "between-estimator\ndifference", fontsize=8, va="center", color=OI["green"])
    ax.text(bx + bw / 2, by - 0.55, "position error at outage end", ha="center", fontsize=8)
    # right: ranking diagram
    ax.text(19.6, 7.75, "bias-error\nranking", ha="center", fontsize=8, va="center")
    ax.text(23.0, 7.75, "ATE\nranking", ha="center", fontsize=8, va="center")
    lx, rx = 19.6, 23.0
    ys = [6.7, 5.9, 5.1, 4.3]; perm = [2, 0, 3, 1]
    for i, y in enumerate(ys):
        ax.scatter([lx], [y], s=22, color=OI["blue"], ec="black", lw=0.5, zorder=3)
        ax.scatter([rx], [ys[perm[i]]], s=22, color=OI["orange"], marker="s", ec="black", lw=0.5, zorder=3)
        ax.plot([lx, rx], [y, ys[perm[i]]], color=OI["grey"], lw=0.8)
    ax.annotate("", xy=(23.8, 4.1), xytext=(23.8, 6.9), arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.text(21.3, 3.7, "agreement shown with\nconfidence interval", fontsize=8, ha="center", va="top")
    box(16.4, 0.2, 7.4, 1.8, "navigation endpoint error:\nclosed-loop evaluation", ls="--", fsz=8)
    save(fig, "graphical_abstract.pdf")


if __name__ == "__main__":
    fig1(); fig_scatter(); fig_shapley(); fig_agree(); fig_heatmap()
    info = fig_dose(); fig_sim(); fig_ga()
    print("dose fit:", info)


# ------------------------------------------------------------------ Supplement tables (generated TeX)
def supp_tables():
    d = pd.read_csv(AC / "table_sources_regime_summary_gated.csv")
    IABE = {"A": 0.06946, "B": 0.791}   # regime IABE medians (m), as in attitude_controls/REPORT.md table 2
    rows = [("a_raw_gyro(paper)", "Raw gyro integration"), ("b_gyro-GT-bias(oracle)", "Oracle (GT gyro bias)"),
            ("s_gyro-bias-comp-99%", "Gyro bias 99\\% compensated"), ("s_gyro-bias-comp-90%", "Gyro bias 90\\% compensated"),
            ("s_gyro-bias-comp-50%", "Gyro bias 50\\% compensated"), ("c_static-mean-bias", "Static average"),
            ("d_mahony(kp=1,ki=0.05)", "Mahony ($k_p{=}1$, $k_i{=}0.05$)"), ("s_mahony(kp=5,ki=0.05)", "Mahony ($k_p{=}5$, $k_i{=}0.05$)"),
            ("e_mahony_gated(kp=1,ki=0.05)", "Gated Mahony")]
    out = ["\\begin{table}[!t]", "\\caption{Full version of Table~I: gravity leakage by attitude source, including compensation variants (five-sequence medians). Leakage multiple = median leakage $\\div$ regime median IABE (A: 0.0695~m; B: 0.791~m).}",
           "\\label{tab:leakfull}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
           "Attitude source & $\\bar\\theta$ (\\textdegree) & Leakage/IABE (A / B) & $g\\bar\\theta/\\lVert\\boldsymbol\\varepsilon\\rVert$ (A / B) \\\\", "\\midrule"]
    for key, lab in rows:
        a = d[(d.regime == "A") & (d.source == key)].iloc[0]; b = d[(d.regime == "B") & (d.source == key)].iloc[0]
        f1 = lambda v: f"{v:.0f}" if v >= 20 else f"{v:.1f}"
        def f2(v):
            if v >= 100: return f"{v:.0f}"
            t = f"{v:.1f}"
            return t[:-2] if (v >= 10 and t.endswith(".0")) else t
        th = f"{a.theta_med:.1f}" if a.theta_med >= 10 else (f"{a.theta_med:.2f}" if a.theta_med >= 0.5 or a.theta_med >= 1 else f"{a.theta_med:.3f}")
        out.append(f"{lab} & {th} & {f1(a.leak_med / IABE['A'])} / {f1(b.leak_med / IABE['B'])} & {f2(a.gth_eps_med)} / {f2(b.gth_eps_med)} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (Path(__file__).resolve().parent / "supp_table_leak.tex").write_text("\n".join(out) + "\n")

    g = pd.read_csv(CL / "table_gated.csv"); x = pd.read_csv(CL / "decomp_summary.csv")
    sp = pd.read_csv(CL / "table_gated_spearman.csv")
    out = ["\\begin{table}[!t]", "\\caption{Gravity-gated tilt update versus the original protocol (medians). Gate: pass rate of the post hoc gravity-gated tilt update. $b_{\\mathrm{ref}}$ $C_{\\mathrm{abs}}$ and estimated $C_{\\Delta}$ in m.}",
           "\\label{tab:gated}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}clcccc@{}}", "\\toprule",
           "$L$ (s) & Variant & Gate pass & $\\mathbf{b}_{\\mathrm{ref}}$ $C_{\\mathrm{abs}}$ & Est.\\ $C_{\\Delta}$ & Zero $C_{\\Delta}$ \\\\", "\\midrule"]
    for L in (5, 10, 20):
        for v, lab in (("nogate", "ungated"), ("gated", "gated")):
            r = g[(g.L == L) & (g.variant == v)].iloc[0]
            gp = "---" if v == "nogate" else f"{100 * r.gate_frac:.0f}\\%"
            out.append(f"{L}{'$^{\\mathrm a}$' if L == 20 else ''} & {lab} & {gp} & {r.bref_abs_med:.2f} & {r.learn_delta_med:.3f} & {r.zero_delta_med:.2f} \\\\")
    out += ["\\bottomrule", "\\multicolumn{6}{@{}l@{}}{\\scriptsize $^{\\mathrm a}$ Two sequences.}", "\\end{tabular}", "\\end{table}"]
    (Path(__file__).resolve().parent / "supp_table_gated.tex").write_text("\n".join(out) + "\n")

    # Spearman and Shapley (L=10 shares) used in the text
    def S(var, cons, pred):
        r = sp[(sp.variant == var) & (sp.L == 10) & (sp.cons == cons) & (sp.pred == pred)].iloc[0]
        return f"{r.rho_med:.2f}"
    out = ["\\begin{table}[!t]", "\\caption{Rank correlation and component shares, ungated versus gated ($L=10$~s, nine cells for Spearman; base configuration for shares).}",
           "\\label{tab:gatedsp}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Quantity & Ungated & Gated \\\\", "\\midrule"]
    for lab, cons, pred in (("Spearman IABE vs.\\ $C_{\\Delta}$", "delta", "iabe"), ("Spearman bias RMSE vs.\\ $C_{\\Delta}$", "delta", "bias_rmse"),
                            ("Spearman IABE vs.\\ $C_{\\mathrm{abs}}$", "abs", "iabe"), ("Spearman bias RMSE vs.\\ $C_{\\mathrm{abs}}$", "abs", "bias_rmse")):
        out.append(f"{lab} & {S('nogate', cons, pred)} & {S('gated', cons, pred)} \\\\")
    for part, lab in (("tilt", "Shapley share, tilt"), ("vel", "Shapley share, velocity")):
        a = x[(x.variant == "nogate") & (x.L == 10) & (x.part == part)].iloc[0].share
        b = x[(x.variant == "gated") & (x.L == 10) & (x.part == part)].iloc[0].share
        out.append(f"{lab} & {a:.2f} & {b:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (Path(__file__).resolve().parent / "supp_table_gatedsp.tex").write_text("\n".join(out) + "\n")

    f = pd.read_csv(R / "verification/R-B_scripts/flip_rate.csv")
    f = f[f.dataset.isin(["EuRoC", "TUMVI_main"])]
    out = ["\\begin{table}[!t]", "\\caption{Seed instability of the best method among 11 deep estimators (10 seeds). $p$: probability that two randomly chosen seeds select different best methods.}",
           "\\label{tab:flip}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Dataset & Sequence & Distinct winners & $p$ \\\\", "\\midrule"]
    for _, r in f.iterrows():
        if str(r.eval_sequence).startswith("RATE"):
            out.append(f"{r.dataset.replace('TUMVI_main', 'TUM-VI')} & mean over sequences & --- & {r.p_two_seeds_disagree:.2f} \\\\ \\midrule" if r.dataset == "EuRoC" else f"{r.dataset.replace('TUMVI_main', 'TUM-VI')} & mean over sequences & --- & {r.p_two_seeds_disagree:.2f} \\\\")
        else:
            out.append(f"{r.dataset.replace('TUMVI_main', 'TUM-VI')} & {str(r.eval_sequence).replace('_', chr(92) + '_')} & {int(r.n_distinct_winners)} & {r.p_two_seeds_disagree:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (Path(__file__).resolve().parent / "supp_table_flip.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    supp_tables()

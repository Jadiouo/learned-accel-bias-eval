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
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 8.4); ax.axis("off")

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
    ax.text(4.75, 7.75, "shared attitude: IABE\n(shared terms cancel)", fontsize=8, ha="center", va="center", color="#007A5A")
    ax.add_patch(Rectangle((3.4, 5.3), 2.7, 1.8, fill=False, ec=OI["green"], lw=1.2))
    box(0.0, 0.0, 10.0, 1.2, "ESKF onset states (IV-C): different quantity", ls="--", fs=8)
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
    cfgs = [("base", "O_base", OI["blue"], None), ("q10", "O_q10", OI["orange"], None), ("q01", "O_q01", OI["green"], None)]  # colors only; the three differ in lightness, so they stay distinct in grayscale
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


# ------------------------------------------------------------------ Fig. 4 (decomposition chain of the agreement)
RC = R / "experiments/reviewer_checks"
_CHAIN = {}


def chain_data():
    """Numbers behind Fig. 4 and the supplementary tables S13-S16, read from the stored result files (nothing typed in).
    Agreement values use the 252-split enumeration of the F1 pipeline (all pairs selected on one seed half by bias RMSE, BH q=0.05);
    95% CIs of the nine-case pool are leave-one-sequence-out jackknife intervals with t_4 quantiles (five sequence clusters)."""
    if _CHAIN:
        return _CHAIN
    import sys
    from scipy import stats
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RC / "F1_noise_ceiling"))
    import common as cm
    units, bias, ate = cm.load()
    sel = cm.precompute_selection(units, bias)
    runs = pd.read_csv(RC / "decomposition/f5_decomp_runs.csv")
    seqs = sorted({u[1] for u in units})

    def mat(src, col):
        return {u: runs[(runs.src == src) & (runs.regime == u[0]) & (runs.seq == u[1])]
                .pivot_table(index="seed", columns="method", values=col)[cm.DL].values for u in units}

    def jk(m):
        A, N = cm.agree_counts(m, units, sel)
        a, n = A.sum(0), N.sum(0)
        pt = a.sum() / n.sum()
        th = np.array([a[[u[1] != s for u in units]].sum() / n[[u[1] != s for u in units]].sum() for s in seqs])
        se = np.sqrt((len(seqs) - 1) / len(seqs) * ((th - th.mean()) ** 2).sum())
        t = stats.t.ppf(0.975, len(seqs) - 1)
        return dict(pt=pt, lo=max(pt - t * se, 0.0), hi=min(pt + t * se, 1.0))

    d = {}
    for src in ("raw", "oracle"):
        d[src, "full"] = jk(ate[src])
        for v in ("noL", "noQ", "I", "negL"):
            d[src, v] = jk(mat(src, v))
        rot = [cm_pt for cm_pt in (jk(mat(src, f"rot{i}"))["pt"] for i in range(30))]
        d[src, "rot"] = dict(lo=min(rot), hi=max(rot), mean=float(np.mean(rot)), n=len(rot))
    f4 = pd.read_csv(RC / "F4_seqcluster/f4_results.csv")
    d["retest"] = float(f4[f4.source == "bias RMSE own retest / ALL"].point.iloc[0])
    ce = pd.read_csv(RC / "F1_noise_ceiling/results_ceiling_N2000.csv")
    ce = ce[(ce.version == "fixed_cases")]
    d["ceil"] = ce
    r = ce[(ce.source == "raw") & (ce.group == "ALL") & ce.variant.isin(["V1", "V2", "V3"])].ceil_median
    d["ceil_band"] = (float(r.min()), float(r.max()))
    al = pd.read_csv(RC / "F2_aligned_ate/agreement_aligned.csv")
    d["align"] = al[al.group == "ALL"].set_index("measure")
    d["align_mag"] = pd.read_csv(RC / "F2_aligned_ate/ate_magnitudes.csv")
    _CHAIN.update(d)
    return _CHAIN


def fig_agree():
    c = chain_data()
    al = c["align"].loc["a_raw_gyro(paper)|se3"]
    r2 = lambda v: round(float(v), 2)
    rows = [  # (label, kind, point/low, high, CI, colour)
        ("bias RMSE retest", "pt", c["retest"], None, None, OI["grey"]),
        ("IABE", "pt", c["raw", "I"]["pt"], None, (c["raw", "I"]["lo"], c["raw", "I"]["hi"]), OI["green"]),
        ("ATE, leakage removed", "pt", c["raw", "noL"]["pt"], None, (c["raw", "noL"]["lo"], c["raw", "noL"]["hi"]), OI["green"]),
        ("ATE, raw attitude", "pt", c["raw", "full"]["pt"], None, (c["raw", "full"]["lo"], c["raw", "full"]["hi"]), OI["blue"]),
        ("leakage, random dir.", "band", c["raw", "rot"]["lo"], c["raw", "rot"]["hi"], None, OI["orange"]),
        ("ATE, SE(3)-aligned", "pt", al.pooled_rate, None, (al.seqboot_lo, al.seqboot_hi), OI["pink"]),
        ("ATE noise ceiling", "band", c["ceil_band"][0], c["ceil_band"][1], None, OI["sky"]),
    ]
    fig, ax = plt.subplots(figsize=(W, 1.85))
    for i, (lab, kind, v0, v1, ci, col) in enumerate(rows):
        if kind == "pt":
            if ci is not None:
                ax.plot(ci, [i, i], color=col, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.scatter([v0], [i], s=26, color=col, ec="black", lw=0.5, zorder=3)
        else:
            ax.add_patch(Rectangle((v0, i - 0.28), v1 - v0, 0.56, fc=col, ec=col, alpha=0.55, lw=0.8, zorder=2))
    ax.axvline(0.5, color=OI["grey"], lw=0.5, ls=":", zorder=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0.25, 1.02); ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("direction agreement (nine cases pooled)")
    # step labels: the two drops of the chain, as differences of the rounded points
    d1 = r2(c["retest"]) - r2(c["raw", "I"]["pt"]); d2 = r2(c["raw", "noL"]["pt"]) - r2(c["raw", "full"]["pt"])
    ax.text(0.27, 0.5, f"double integration\n$-${d1:.2f}", fontsize=8, va="center", ha="left", color=OI["black"], linespacing=0.95,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(0.635, 2.5, f"shared leakage term $-${d2:.2f}", fontsize=7.5, va="center", ha="left", color=OI["black"], linespacing=0.95,
            bbox=dict(fc="white", ec="none", pad=0.6))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.37, right=0.985, bottom=0.2, top=0.98)
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
    d = pd.read_csv(AC / "table_sources_regime_summary_gated.csv")   # theta_med, gth_eps_med only (per-sequence ratios first, then median; B has 4 sequences)
    ps = pd.read_csv(AC / "table_sources_by_regime_gated.csv")       # per-sequence leakage (regime, source, seq)
    w = pd.read_csv(R / "euroc_full/results/metrics_wide.csv"); w["reg"] = w.run_id.str[0]
    IABE = {r: float(w[w.reg == r].iabe_rot.median()) for r in "AB"}  # regime IABE medians (m), computed (0.069466 and 0.791333), not hard-coded
    SEQ = {r: sorted(w[w.reg == r].eval_sequence.unique()) for r in "AB"}   # A: MH_01-05; B: MH_01, 02, 03, 05
    leak_mult = lambda key, r: ps[(ps.regime == r) & (ps.source == key) & ps.seq.isin(SEQ[r])].leak.median() / IABE[r]   # numerator over the same sequences as the denominator
    rows = [("a_raw_gyro(paper)", "Raw gyro integration"), ("b_gyro-GT-bias(oracle)", "Oracle (GT gyro bias)"),
            ("s_gyro-bias-comp-99%", "Gyro bias 99\\% compensated"), ("s_gyro-bias-comp-90%", "Gyro bias 90\\% compensated"),
            ("s_gyro-bias-comp-50%", "Gyro bias 50\\% compensated"), ("c_static-mean-bias", "Static average"),
            ("d_mahony(kp=1,ki=0.05)", "Mahony ($k_p{=}1$, $k_i{=}0.05$)"), ("s_mahony(kp=5,ki=0.05)", "Mahony ($k_p{=}5$, $k_i{=}0.05$)"),
            ("e_mahony_gated(kp=1,ki=0.05)", "Gated Mahony")]
    out = ["\\begin{table}[!t]", "\\caption{Full version of Table~III: gravity leakage by attitude source, including compensation variants (medians over five sequences in regime~A and four in regime~B). Leakage multiple = median leakage $\\div$ regime median IABE (A: " + f"{IABE['A']:.4f}" + "~m; B: " + f"{IABE['B']:.3f}" + "~m).}",
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
        out.append(f"{lab} & {th} & {f1(leak_mult(key, 'A'))} / {f1(leak_mult(key, 'B'))} & {f2(a.gth_eps_med)} / {f2(b.gth_eps_med)} \\\\")
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
    out = ["\\begin{table}[!t]", "\\caption{Rank correlation and component shares, ungated versus gated ($L=10$~s, nine cases for Spearman; base configuration for shares).}",
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
    f = f[f.dataset == "EuRoC"]
    out = ["\\begin{table}[!t]", "\\caption{Seed instability of the best method among 11 deep estimators (10 seeds), EuRoC Machine Hall, regime~A. $p$: probability that two randomly chosen seeds select different best methods.}",
           "\\label{tab:flip}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Sequence & Distinct winners & $p$ \\\\", "\\midrule"]
    for _, r in pd.concat([f[~f.eval_sequence.astype(str).str.startswith("RATE")], f[f.eval_sequence.astype(str).str.startswith("RATE")]]).iterrows():
        if str(r.eval_sequence).startswith("RATE"):
            out.append(f"mean over sequences & --- & {r.p_two_seeds_disagree:.2f} \\\\")
        else:
            out.append(f"{str(r.eval_sequence).replace('_', chr(92) + '_')} & {int(r.n_distinct_winners)} & {r.p_two_seeds_disagree:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (Path(__file__).resolve().parent / "supp_table_flip.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    supp_tables()


# ------------------------------------------------------------------ Supplement tables S13-S16 (generated TeX; every number is read from the stored result files)
def supp_tables_g():
    here = Path(__file__).resolve().parent
    c = chain_data()
    top = lambda cap, lab, cols, tc="3pt": ["\\begin{table}[!t]", f"\\caption{{{cap}}}", f"\\label{{{lab}}}", "\\centering", "\\scriptsize",
                                            f"\\setlength{{\\tabcolsep}}{{{tc}}}", f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule"]
    end = ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    ci = lambda d, nd=2: f"{d['pt']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"

    # S13: seed-noise ceiling (F1)
    ce = c["ceil"]
    out = top("Seed-noise ceiling of the direction agreement. Cases: nine pooled, or the cases of regime A or B. Observed: all 252 splits. V1--V3: median [2.5th, 97.5th percentile] of 2000 simulations in which the trajectory-error ranking follows bias error exactly and carries the observed seed noise; the case set is fixed. V1 mixes the residuals of all methods before reshuffling, V2 keeps each method's residual scale, and V3 takes the signal from the bias means.",
              "tab:ceiling", "llcccc", "1.8pt")
    out += ["Attitude & Cases & Observed & V1 & V2 & V3 \\\\", "\\midrule"]
    for src, sl in (("raw", "raw"), ("oracle", "oracle")):
        for g, gl in (("ALL", "nine"), ("A", "A"), ("B", "B")):
            sub = ce[(ce.source == src) & (ce.group == g)].set_index("variant")
            cells = [f"{sub.loc[v].ceil_median:.2f} [{sub.loc[v].ceil_p2_5:.2f}, {sub.loc[v].ceil_p97_5:.2f}]" for v in ("V1", "V2", "V3")]
            out.append(f"{sl} & {gl} & {sub.loc['V1'].observed:.3f} & " + " & ".join(cells) + " \\\\")
    out += end
    (here / "supp_table_ceiling.tex").write_text("\n".join(out) + "\n")

    # S14: aligned ATE (F2)
    al, mg = c["align"], c["align_mag"]
    mg = mg[mg.regime == "ALL"].set_index("source")
    out = top("Direction agreement and median ATE with and without alignment of the open-loop trajectory to the reference trajectory (nine cases pooled; 1000 random splits of the original pipeline, 95\\% sequence-cluster bootstrap CI). 4DoF: translation and yaw; SE(3): rigid transform.",
              "tab:align", "llcc", "4pt")
    out += ["Attitude & Alignment & Median ATE (m) & Agreement [95\\% CI] \\\\", "\\midrule"]
    out.append(f"bias RMSE & retest & --- & {al.loc['bias_rmse'].pooled_rate:.3f} [{al.loc['bias_rmse'].seqboot_lo:.3f}, {al.loc['bias_rmse'].seqboot_hi:.3f}] \\\\")
    out.append(f"IABE (reference attitude) & --- & {mg.loc['a_raw_gyro(paper)'].iabe_rot_median:.3f} & {al.loc['iabe_rot'].pooled_rate:.3f} [{al.loc['iabe_rot'].seqboot_lo:.3f}, {al.loc['iabe_rot'].seqboot_hi:.3f}] \\\\")
    for key, sl in (("a_raw_gyro(paper)", "raw"), ("b_gyro-GT-bias(oracle)", "oracle")):
        for tag, tl, col in (("none", "none", "ate_none_median"), ("4dof", "4DoF", "ate_4dof_median"), ("se3", "SE(3)", "ate_se3_median")):
            r = al.loc[f"{key}|{tag}"]
            m = mg.loc[key][col]
            out.append(f"{sl} & {tl} & {m:.2f} & {r.pooled_rate:.3f} [{r.seqboot_lo:.3f}, {r.seqboot_hi:.3f}] \\\\")
    out += end
    (here / "supp_table_align.tex").write_text("\n".join(out) + "\n")

    # S15: dose response (F3)
    F3 = RC / "F3_dose_response"
    th = pd.read_csv(F3 / "thresholds.csv"); th = th[th.group == "all_rand"].set_index("L")
    nb = pd.read_csv(F3 / "noise_baselines.csv", index_col=0)
    pr = pd.read_csv(F3 / "pool_runs_summary.csv")
    q = pr[pr.L == 10].bias_rmse_avg
    e = lambda v: f"{v:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
    out = top("Detection thresholds of the closed-loop dose response (original protocol, random-direction constant offsets injected into the outage, 25 units). Thresholds are bias RMSE in m/s$^2$ (per-axis rms; a constant offset has $\\lVert\\boldsymbol\\delta\\rVert=\\sqrt3$ times this). $N_{\\mathrm{rep}}$ and $N_{\\mathrm{seq}}$: spread of the shared endpoint error across repetitions and across sequences (m). $T_{\\mathrm{rep}}$, $T_{\\mathrm{seq}}$: smallest size at which the median change of $C_{\\mathrm{abs}}$ stays above $N_{\\mathrm{rep}}$ or $N_{\\mathrm{seq}}$; $T_{10\\%}$: above 10\\% of the shared endpoint error. Last two columns: share of the 1026 learned and baseline runs whose bias RMSE exceeds $T_{\\mathrm{rep}}$ or $T_{\\mathrm{seq}}$.",
              "tab:dosethr", "cccccccc", "2pt")
    out += ["$L$ (s) & $N_{\\mathrm{rep}}$ & $N_{\\mathrm{seq}}$ & $T_{\\mathrm{rep}}$ & $T_{\\mathrm{seq}}$ & $T_{10\\%}$ & $>T_{\\mathrm{rep}}$ & $>T_{\\mathrm{seq}}$ \\\\", "\\midrule"]
    for L in (5, 10):
        qq = pr[pr.L == L].bias_rmse_avg
        out.append(f"{L} & {nb.loc[L].N_rep:.2f} & {nb.loc[L].N_seq:.2f} & ${e(th.loc[L].T_rep)}$ & ${e(th.loc[L].T_seq)}$ & ${e(th.loc[L].T_10pct)}$ & "
                   f"{100 * (qq > th.loc[L].T_rep).mean():.1f}\\% & {100 * (qq > th.loc[L].T_seq).mean():.1f}\\% \\\\")
    out += end
    (here / "supp_table_dosethr.tex").write_text("\n".join(out) + "\n")
    c["pool_q"] = dict(med=float(q.median()), p95=float(q.quantile(.95)), max=float(q.max()), n=int(len(q)))

    ps = pd.read_csv(F3 / "poolsim_spearman.csv"); ps = ps[ps.cons == "abs"]
    out = top("Expected rank correlation between bias RMSE and the shared endpoint error $C_{\\mathrm{abs}}$ under the pool-like simulation: each method receives a constant offset of its actual bias RMSE (times the scale) in a random direction; cell-median Spearman correlation over nine cases, median [2.5th, 97.5th percentile] of 2000 draws. Observed: values of the experiment.",
              "tab:dosesim", "lccc", "3pt")
    out += ["Scale & $L=5$~s & $L=10$~s & Pool max (m/s$^2$) \\\\", "\\midrule"]
    obs = {L: float(ps[ps.L == L].observed.iloc[0]) for L in (5, 10)}
    for sc in (1, 3, 10, 30, 100):
        cells = []
        for L in (5, 10):
            r = ps[(ps.L == L) & (ps.scale == sc)].iloc[0]
            cells.append(f"{r.rho_med:.2f} [{r.rho_lo:.2f}, {r.rho_hi:.2f}]".replace("-", "$-$"))
        pm = ps[(ps.L == 5) & (ps.scale == sc)].iloc[0].pool_max_rmse_scaled
        out.append(f"$\\times${sc} & {cells[0]} & {cells[1]} & {pm:.2f} \\\\")
    out.append("\\midrule")
    out.append(f"Observed & {obs[5]:.2f} & {obs[10]:.2f} & --- \\\\")
    out += end
    (here / "supp_table_dosesim.tex").write_text("\n".join(out) + "\n")

    # S16.1: decomposition (decomposition, nine cases)
    out = top("Direction agreement when one term of Proposition~1(i) is changed (nine cases pooled, 252 splits, 95\\% leave-one-sequence-out jackknife CI with $t_4$ quantiles). $\\mathcal P[\\mathbf l]$: shared leakage term; $\\mathcal P[(C^{\\mathrm g}-C)\\boldsymbol\\varepsilon]$: bias error rotated by the attitude error; $\\mathcal P[C\\boldsymbol\\varepsilon]$: IABE trajectory. Random: the shared term rotated by 30 random fixed rotations (range of the 30 agreements).",
              "tab:decomp", ">{\\raggedright\\arraybackslash}p{0.40\\columnwidth}cc", "3pt")
    out += ["Trajectory error built from & Raw attitude & Oracle attitude \\\\", "\\midrule"]
    for lab, key in (("all three terms (observed ATE)", "full"), ("without $\\mathcal P[\\mathbf l]$", "noL"),
                     ("without $\\mathcal P[(C^{\\mathrm g}-C)\\boldsymbol\\varepsilon]$", "noQ"),
                     ("$\\mathcal P[C\\boldsymbol\\varepsilon]$ only (IABE)", "I"), ("sign of $\\mathcal P[\\mathbf l]$ reversed", "negL")):
        out.append(f"{lab} & {ci(c['raw', key])} & {ci(c['oracle', key])} \\\\")
    out.append(f"$\\mathcal P[\\mathbf l]$ in random directions & {c['raw', 'rot']['lo']:.2f}--{c['raw', 'rot']['hi']:.2f} & {c['oracle', 'rot']['lo']:.2f}--{c['oracle', 'rot']['hi']:.2f} \\\\")
    out += end
    (here / "supp_table_decomp.tex").write_text("\n".join(out) + "\n")

    # S16.2: synthetic scaling (F5), leakage-ratio-matched version
    F5 = RC / "F5_scaled_attitude"
    ra = pd.read_csv(F5 / "results_agreement.csv"); rs = json.load(open(F5 / "rm_scales.json"))
    out = top("Synthetic scaling of the oracle attitude error: direction agreement when the attitude error of each case is rescaled so that its leakage ratio $g\\bar\\theta/\\lVert\\boldsymbol\\varepsilon\\rVert$ equals the target (median over the nine cases). Extrapolated: cases (of nine) whose scale factor exceeds 1, so that the error was enlarged. Intervals are not shown because five sequence clusters understate their width; the point estimates are consistent across the five sequences (text).",
              "tab:scale", "cccccc", "3pt")
    out += ["Target ratio & $\\bar\\theta$ (\\textdegree) & Agreement & Regime A & Regime B & Extrap. \\\\", "\\midrule"]
    for t in ("30", "10", "3", "1", "0.3", "0.1"):
        g = ra[ra.config == f"RM|r={t}"].set_index("group")
        ext = sum(1 for k, v in rs.items() if k.split("|")[0] == t and v > 1.0)
        out.append(f"{t} & {g.loc['ALL'].theta_bar_med:.3f} & {g.loc['ALL'].agree:.2f} & {g.loc['A'].agree:.2f} & {g.loc['B'].agree:.2f} & {ext} \\\\")
    s1 = ra[ra.config == "W|s=1"].set_index("group")
    s0 = ra[ra.config == "W|s=0"].set_index("group")
    out.append("\\midrule")
    out.append(f"Unscaled (oracle) & {s1.loc['ALL'].theta_bar_med:.3f} & {s1.loc['ALL'].agree:.2f} & {s1.loc['A'].agree:.2f} & {s1.loc['B'].agree:.2f} & --- \\\\")
    out.append(f"Attitude error 0 (IABE) & 0 & {s0.loc['ALL'].agree:.2f} & {s0.loc['A'].agree:.2f} & {s0.loc['B'].agree:.2f} & --- \\\\")
    out += end
    (here / "supp_table_scale.tex").write_text("\n".join(out) + "\n")
    c["scale_ext"] = {t: sum(1 for k, v in rs.items() if k.split("|")[0] == t and v > 1.0) for t in ("30", "10", "3", "1", "0.3", "0.1")}

    # S16.3: overwriting the bias state (F6)
    tc = pd.read_csv(RC / "F6_ba_overwrite/table_conditions.csv")
    desc = {"a": "(a) inject $\\mathbf b_{\\mathrm{ref}}$ (original)", "b": "(b) keep the filter's own $b_a$",
            "c": "(c) $\\mathbf b_{\\mathrm{ref}}$, onset tilt made consistent with it", "d": "(d) $\\mathbf b_{\\mathrm{ref}}$, ground-truth tilt",
            "e": "(e) filter's $b_a$, ground-truth tilt", "f": "(f) $\\mathbf b_{\\mathrm{ref}}$, all five states at truth"}
    out = top("Shared endpoint error $C_{\\mathrm{abs}}$ (m) under conditions that keep or replace the filter's own accelerometer-bias state $b_a$ at outage onset (original protocol, five sequences $\\times$ five repetitions; unit median). Ratio: median over units of the ratio to condition (a). Last column: sequences (of five) with a ratio below 0.8 at $L=10$~s.",
              "tab:overwrite", ">{\\raggedright\\arraybackslash}p{0.40\\columnwidth}ccccc", "2.5pt")
    out += ["Condition & $C_{\\mathrm{abs}}$, $L{=}5$ & Ratio & $C_{\\mathrm{abs}}$, $L{=}10$ & Ratio & $<0.8$ \\\\", "\\midrule"]
    b5 = tc[(tc.q == "base") & (tc.L == 5)].set_index("cond"); b10 = tc[(tc.q == "base") & (tc.L == 10)].set_index("cond")
    for k in "abcdef":
        out.append(f"{desc[k]} & {b5.loc[k].abs_unit_median:.2f} & {b5.loc[k].ratio_vs_a_median:.2f} & {b10.loc[k].abs_unit_median:.2f} & {b10.loc[k].ratio_vs_a_median:.2f} & {int(b10.loc[k].n_seq_below_0p8)} \\\\")
    out += end
    (here / "supp_table_overwrite.tex").write_text("\n".join(out) + "\n")
    c["overwrite"] = tc


if __name__ == "__main__":
    supp_tables_g()

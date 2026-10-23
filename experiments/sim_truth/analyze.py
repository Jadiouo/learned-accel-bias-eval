import numpy as np, glob, os, json, zlib
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_all import TRAJ, SEEDS, LEV_I, LEV_M, OUT
from sim_truth import TYPES

K, NTR, MS = 4, 334, [1, 5, 20]
rng0 = np.random.default_rng(12345)
NB = 2000

def load(mode, li):
    return [np.load(f"{OUT}/{mode}_{t}_s{s}_L{li}.npz") for t in TRAJ for s in SEEDS]

def trials(u, scen, rng):
    """回傳每個 trial 的結果列：misrank(ATE m=1,5,20, IABE, RMSE)、pair inversion、IABE-vs-RMSE disagreement。"""
    m = u["grp"] == -1
    idx_all = np.where(m)[0]; typ = u["typ"]
    ate = u["ate"]  # (P, R_TEST)
    nT = ate.shape[1]
    out = np.zeros((NTR, 6)); pc = np.zeros((NTR, 2 * 6))  # 對 inversion: [wrong, total] x6 metric
    for i in range(NTR):
        if scen == "mixed": pool = idx_all
        elif scen == "same": pool = idx_all[typ[idx_all] == rng.integers(6)]
        else: pool = idx_all[np.isin(typ[idx_all], [0, 5])]  # euroc-like: const + learned
        c = rng.choice(pool, K, replace=False)
        tb = u["truthB"][c]; best = np.argmin(tb)
        mets = []
        for mm in MS:
            js = rng.choice(nT, mm, replace=False)
            mets.append(np.sqrt(np.mean(ate[c][:, js] ** 2, axis=1)))
        mets += [u["iabe"][c], u["rmse"][c]]
        for q, mt in enumerate(mets):
            out[i, q] = np.argmin(mt) != best
            for a in range(K):
                for b in range(a + 1, K):
                    r = tb[a] / tb[b]
                    if max(r, 1 / r) >= 1.25:
                        pc[i, 2 * q + 1] += 1
                        pc[i, 2 * q] += (mt[a] < mt[b]) != (tb[a] < tb[b])
        out[i, 5] = np.argmin(u["iabe"][c]) != np.argmin(u["rmse"][c])
    return out, pc

def boot(per_unit_num, per_unit_den, rng, nb=NB):
    """per_unit_*: (U,...) -> 估計與 bootstrap 95% CI（對 unit 重抽）"""
    U = len(per_unit_num)
    est = per_unit_num.sum(0) / per_unit_den.sum(0)
    bs = []
    for _ in range(nb):
        s = rng.integers(0, U, U)
        bs.append(per_unit_num[s].sum(0) / per_unit_den[s].sum(0))
    return est, np.percentile(bs, [2.5, 97.5], axis=0), np.array(bs)

def r2(x, y):
    x, y = np.log(x), np.log(y)
    return np.corrcoef(x, y)[0, 1] ** 2

def main():
    rows_h1, rows_h2, rows_h3, rows_ratio, rows_info = [], [], [], [], []
    store = {}
    cells = [("I", li, LEV_I[li]) for li in range(len(LEV_I))] + [("M", li, LEV_M[li]) for li in range(len(LEV_M))]
    names = ["ATE_m1", "ATE_m5", "ATE_m20", "IABE", "RMSE"]
    for mode, li, lev in cells:
        us = load(mode, li)
        att = np.mean([float(u["att"]) for u in us])
        leak = np.median(np.concatenate([u["ate"].mean(1) / u["truthB"] for u in us]))
        rr = np.concatenate([np.log(u["iabe"] / u["truthB"]) for u in us])
        rows_info.append((mode, lev, att, leak, np.exp(np.median(rr)), np.exp(np.percentile(rr, 5)), np.exp(np.percentile(rr, 95))))
        for ti, tn in enumerate(TYPES):
            rt = np.concatenate([np.log(u["iabe"][u["typ"] == ti] / u["truthB"][u["typ"] == ti]) for u in us])
            rows_ratio.append((mode, lev, tn, np.exp(np.median(rt)), np.exp(np.percentile(rt, 5)), np.exp(np.percentile(rt, 95))))
        for scen in ["mixed", "same", "euroc"]:
            rng = np.random.default_rng(zlib.crc32(f"{mode}{li}{scen}".encode()))
            res = [trials(u, scen, rng) for u in us]
            num = np.array([r[0][:, :5].sum(0) for r in res]); den = np.full_like(num, NTR, dtype=float)
            est, ci, bs = boot(num, den, rng)
            pn = np.array([[r[1][:, 2 * q].sum() for q in range(5)] for r in res])
            pd = np.array([[r[1][:, 2 * q + 1].sum() for q in range(5)] for r in res])
            pest, pci, _ = boot(pn, pd, rng)
            dn = np.array([r[0][:, 5].sum() for r in res]); dis, dci, _ = boot(dn, np.full(len(dn), NTR, float), rng)
            d_ate_iabe = bs[:, 0] - bs[:, 3]
            dlo, dhi = np.percentile(d_ate_iabe, [2.5, 97.5])
            rows_h1.append(dict(mode=mode, lev=lev, att=att, scen=scen, **{n: (est[i], ci[0, i], ci[1, i]) for i, n in enumerate(names)},
                                pair={n: (pest[i], pci[0, i], pci[1, i]) for i, n in enumerate(names)}, d_ate_iabe=(est[0] - est[3], dlo, dhi),
                                dis=(float(dis), float(dci[0]), float(dci[1]))))
        # H2：混合池 R^2
        def pooled(f):
            return np.array([f(u) for u in us])
        X = {k: [u[k][u["grp"] == -1] for u in us] for k in ["iabe", "rmse", "truthB"]}
        def r2d(sel):
            a = np.concatenate([X["iabe"][i] for i in sel]); b = np.concatenate([X["rmse"][i] for i in sel]); t = np.concatenate([X["truthB"][i] for i in sel])
            return r2(a, t), r2(b, t)
        allsel = np.arange(len(us)); a0, b0 = r2d(allsel)
        d = []
        for _ in range(NB):
            s = rng0.integers(0, len(us), len(us)); x, y = r2d(s); d.append(x - y)
        # 匹配 RMSE 組
        ratios, sp_i, sp_a = [], [], []
        for u in us:
            for g in np.unique(u["grp"][u["grp"] >= 0]):
                m = u["grp"] == g
                tb = u["truthB"][m]; ratios.append(tb.max() / tb.min())
                sp_i.append(spearmanr(u["iabe"][m], tb)[0])
                sp_a.append(np.median([spearmanr(u["ate"][m][:, j], tb)[0] for j in range(u["ate"].shape[1])]))
        rows_h2.append((mode, lev, a0, b0, a0 - b0, *np.percentile(d, [2.5, 97.5]), np.median(ratios), np.nanmedian(sp_i), np.nanmedian(sp_a)))
        store[(mode, li)] = us
    json.dump(dict(h1=rows_h1, h2=rows_h2, info=rows_info, ratio=rows_ratio), open("results_summary.json", "w"), default=float, indent=1)
    write_md(rows_h1, rows_h2, rows_info, rows_ratio)
    figs(rows_h1, rows_h2, store)

def f3(t): return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"

def write_md(h1, h2, info, ratio):
    L = []
    L.append("# 自動產生的結果表 (analyze.py)\n")
    L.append("## T1 各格姿態誤差、ATE/Truth_B 中位比、IABE/Truth_B 比值 (中位, 5%, 95%)\n")
    L.append("| Mode | level (deg 或 s_g) | 實測姿態誤差 RMS (deg) | ATE/TruthB 中位 | IABE/TruthB 中位 | 5% | 95% |\n|---|---|---|---|---|---|---|")
    for r in info: L.append(f"| {r[0]} | {r[1]:g} | {r[2]:.2f} | {r[3]:.1f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} |")
    for scen in ["mixed", "same", "euroc"]:
        L.append(f"\n## T2 top-1 mis-rank（K=4，對 Truth_B），情境={scen}；隨機基準 0.75\n")
        L.append("| Mode | level | 姿態RMS(deg) | ATE m=1 | ATE m=5 | ATE m=20 | IABE | RMSE | ATE1-IABE | IABE vs RMSE top1 不一致 |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in h1:
            if r["scen"] != scen: continue
            L.append(f"| {r['mode']} | {r['lev']:g} | {r['att']:.2f} | " + " | ".join(f3(r[n]) for n in ["ATE_m1", "ATE_m5", "ATE_m20", "IABE", "RMSE"]) + f" | {f3(r['d_ate_iabe'])} | {f3(r['dis'])} |")
        L.append(f"\n成對反轉率（僅 Truth 比值>=1.25 的對），情境={scen}\n")
        L.append("| Mode | level | ATE m=1 | ATE m=5 | ATE m=20 | IABE | RMSE |\n|---|---|---|---|---|---|---|")
        for r in h1:
            if r["scen"] != scen: continue
            L.append(f"| {r['mode']} | {r['lev']:g} | " + " | ".join(f3(r['pair'][n]) for n in ["ATE_m1", "ATE_m5", "ATE_m20", "IABE", "RMSE"]) + " |")
    L.append("\n## T3 H2：混合池 log-log R^2 與匹配 RMSE 組\n")
    L.append("| Mode | level | R2(IABE) | R2(RMSE) | 差 | CI lo | CI hi | 匹配組 Truth max/min 中位 | 組內 Spearman(IABE,Truth) 中位 | 組內 Spearman(ATE單次,Truth) 中位 |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in h2: L.append(f"| {r[0]} | {r[1]:g} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.2f} | {r[8]:.2f} | {r[9]:.2f} |")
    L.append("\n## T4 IABE/Truth_B 比值依誤差型態（中位, 5%, 95%）；>1 = IABE 高估真實後果\n")
    L.append("| Mode | level | 型態 | 中位 | 5% | 95% |\n|---|---|---|---|---|---|")
    for r in ratio: L.append(f"| {r[0]} | {r[1]:g} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} |")
    L.append("\n## T5 H3：滿足「IABE 無額外價值」(不一致<0.05 且 ATE_m1 mis-rank<0.10) 的格子\n")
    L.append("| 情境 | Mode | level | 不一致 | ATE m=1 mis-rank | 滿足 |\n|---|---|---|---|---|---|")
    for r in h1:
        ok = r["dis"][0] < 0.05 and r["ATE_m1"][0] < 0.10
        L.append(f"| {r['scen']} | {r['mode']} | {r['lev']:g} | {r['dis'][0]:.3f} | {r['ATE_m1'][0]:.3f} | {'是' if ok else '否'} |")
    open("results_tables.md", "w").write("\n".join(L))

def figs(h1, h2, store):
    # Fig1: mis-rank vs 姿態誤差
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, scen in zip(axs, ["mixed", "same", "euroc"]):
        for n, c, ls in [("ATE_m1", "C3", "-"), ("ATE_m5", "C1", "--"), ("ATE_m20", "C2", ":"), ("IABE", "C0", "-"), ("RMSE", "k", "-.")]:
            for mode, mk in [("I", "o"), ("M", "s")]:
                rs = [r for r in h1 if r["scen"] == scen and r["mode"] == mode]
                x = [max(r["att"], 0.03) for r in rs]; y = [r[n][0] for r in rs]
                lo = [r[n][1] for r in rs]; hi = [r[n][2] for r in rs]
                if mode == "I": ax.plot(x, y, color=c, ls=ls, marker=mk, ms=4, label=n)
                else: ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], color=c, marker=mk, ms=6, ls="none", mfc="none")
        ax.set_xscale("log"); ax.set_title(f"scenario: {scen}"); ax.set_xlabel("realized attitude error RMS (deg)  [o: Mode I, square: Mode M]"); ax.axhline(0.75, color="gray", lw=.5)
    axs[0].set_ylabel("top-1 mis-rank vs Truth_B (K=4)"); axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("fig1_misrank_vs_attitude.png", dpi=130); plt.close()
    # Fig2: 散點
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for row, (mode, li) in enumerate([("I", 4), ("M", 1)]):
        us = store[(mode, li)]
        t = np.concatenate([u["truthB"][u["grp"] == -1] for u in us]); ty = np.concatenate([u["typ"][u["grp"] == -1] for u in us])
        for col, (k, lab) in enumerate([("rmse", "bias RMSE (m/s^2)"), ("iabe", "IABE (m)"), ("ate", "ATE single realization (m)")]):
            v = np.concatenate([(u[k][u["grp"] == -1] if k != "ate" else u[k][u["grp"] == -1][:, 0]) for u in us])
            for ti, tn in enumerate(TYPES): axs[row, col].scatter(v[ty == ti], t[ty == ti], s=6, label=tn)
            axs[row, col].set_xscale("log"); axs[row, col].set_yscale("log"); axs[row, col].set_xlabel(lab); axs[row, col].set_ylabel("Truth_B (m)")
            axs[row, col].set_title(f"Mode {mode} level {(LEV_I if mode=='I' else LEV_M)[li]:g}")
    axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig("fig2_scatter_truth.png", dpi=120); plt.close()
    # Fig3: IABE/Truth_B 比值依型態
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (mode, li) in zip(ax, [("I", 4), ("M", 1)]):
        us = store[(mode, li)]
        data = [np.concatenate([u["iabe"][u["typ"] == ti] / u["truthB"][u["typ"] == ti] for u in us]) for ti in range(6)]
        a.boxplot(data, labels=TYPES, showfliers=False); a.set_yscale("log"); a.axhline(1, color="gray"); a.set_title(f"IABE / Truth_B, Mode {mode}")
    ax[0].set_ylabel(">1: IABE overstates deployed consequence"); plt.tight_layout(); plt.savefig("fig3_iabe_over_truth.png", dpi=120); plt.close()

if __name__ == "__main__": main()

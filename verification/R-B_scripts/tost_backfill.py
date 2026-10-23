"""R-B: 補算配對差/CI/Wilcoxon/TOST 與 flip rate。只讀 euroc_full 的 metrics_wide.csv，CPU only。
用法: python3 tost_backfill.py   (輸出到本目錄)"""
import itertools, os, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "euroc_full", "results")
OUT = os.path.dirname(os.path.abspath(__file__))
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
SETS = {"EuRoC": (ROOT, None), "TUMVI_main": (ROOT + "/tumvi", ["room1","room3","room4","room5"]),
        "TUMVI_all": (ROOT + "/tumvi", None)}
ABS = [0.05, 0.10]; REL = 0.25  # delta 候選: 絕對 0.05/0.10 m；相對 25% of 該 regime×序列之 DL 方法 IABE 中位數
def tost(d, delta):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0: return 0.0 if abs(m) < delta else 1.0
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    return float(max(p1, p2))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); out[i] = run
    return out
rows = []; flips = []
for name, (path, seqs) in SETS.items():
    w = pd.read_csv(path + "/metrics_wide.csv"); ri = pd.read_csv(path + "/runs_index.csv")[["run_id","regime","method","seed"]]
    w = w.merge(ri, on="run_id"); w = w[w.method.isin(DL)]
    if seqs: w = w[w.eval_sequence.isin(seqs)]
    for reg in ["A", "B"]:
        g0 = w[w.regime == reg]
        # pooled-over-sequence 單位 (每 seed 對序列取平均) + 各序列
        units = {s: g0[g0.eval_sequence == s] for s in sorted(g0.eval_sequence.unique())}
        pool = g0.groupby(["method","seed"], as_index=False).iabe_rot.mean(); pool["eval_sequence"] = "POOLED"
        units["POOLED"] = pool
        for seq, g in units.items():
            piv = g.pivot_table(index="seed", columns="method", values="iabe_rot")
            med = float(np.median(piv.values)); pstd = []
            pairs = list(itertools.combinations([m for m in DL if m in piv], 2))
            recs = []
            for a, b in pairs:
                d = (piv[a] - piv[b]).dropna().values; n = len(d)
                if n < 3: continue
                m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
                t90 = stats.t.ppf(.95, n - 1)
                try: pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except Exception: pw = np.nan
                recs.append(dict(dataset=name, regime=reg, eval_sequence=seq, A=a, B=b, n=n, mean_diff=m, sd_diff=sd,
                                 ci95_lo=m - t * se, ci95_hi=m + t * se, ci90_lo=m - t90 * se, ci90_hi=m + t90 * se,
                                 p_t=float(stats.ttest_1samp(d, 0).pvalue), p_wilcoxon=float(pw),
                                 effect_size=abs(m) / sd if sd > 0 else np.inf, min_delta_equiv=abs(m) + t90 * se))
                pstd.append(sd)
            floor = 2 * float(np.median(pstd)); rel = REL * med
            deltas = {"abs0.05": 0.05, "abs0.10": 0.10, f"rel{int(REL*100)}%medIABE": rel, "floor2xMedPairedSD": floor}
            df = pd.DataFrame(recs)
            df["p_wilcoxon_holm"] = holm(df.p_wilcoxon.fillna(1).values)
            for k, dl in deltas.items():
                df[f"delta_{k}"] = dl
                df[f"pTOST_{k}"] = [tost((piv[r.A] - piv[r.B]).dropna().values, dl) for r in df.itertuples()]
                df[f"equiv_{k}"] = df[f"pTOST_{k}"] < 0.05
            rows.append(df)
        # flip rate (只對 regime A 且僅非 POOLED 序列)
        if reg == "A":
            seqlist = [s for s in units if s != "POOLED"]; flipped = 0; pdis = []
            for s in seqlist:
                piv = units[s].pivot_table(index="seed", columns="method", values="iabe_rot")
                win = piv.idxmin(axis=1); k = win.nunique(); flipped += k > 1
                cnt = win.value_counts().values; n = len(win)
                pdis.append(1 - (cnt * (cnt - 1)).sum() / (n * (n - 1)))  # 兩個不同 seed 贏家不同的機率
                flips.append(dict(dataset=name, eval_sequence=s, n_distinct_winners=k, n_seeds=n,
                                  p_two_seeds_disagree=pdis[-1], top_share=cnt.max() / n, winners=";".join(win.astype(str))))
            flips.append(dict(dataset=name, eval_sequence="RATE(any>1 winner)", n_distinct_winners=flipped, n_seeds=len(seqlist),
                              p_two_seeds_disagree=float(np.mean(pdis)), top_share=np.nan, winners=""))
res = pd.concat(rows); res.to_csv(OUT + "/paired_tost_all_pairs.csv", index=False)
pd.DataFrame(flips).to_csv(OUT + "/flip_rate.csv", index=False)
# 摘要
eq = [c for c in res.columns if c.startswith("equiv_")]
summ = []
for (d, r, s), g in res.groupby(["dataset","regime","eval_sequence"]):
    row = dict(dataset=d, regime=r, eval_sequence=s, n_pairs=len(g), frac_wilcoxon_sig=(g.p_wilcoxon < .05).mean(),
               frac_holm_sig=(g.p_wilcoxon_holm < .05).mean(), frac_effect_ge2_and_sig=((g.effect_size >= 2) & (g.p_wilcoxon < .05)).mean(),
               median_abs_diff=g.mean_diff.abs().median(), median_min_delta_equiv=g.min_delta_equiv.median())
    for c in eq: row[c] = g[c].mean(); row["delta_" + c[6:]] = g["delta_" + c[6:]].iloc[0]
    summ.append(row)
pd.DataFrame(summ).to_csv(OUT + "/summary_by_sequence.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(summ).round(3).to_string()); print(pd.DataFrame(flips).drop(columns="winners").round(3).to_string())

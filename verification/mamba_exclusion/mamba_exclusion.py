"""Mamba 排除敏感度。複製（不修改）R-F stratified / R-E / R-B flip / closed_loop analyze_3 的邏輯，DL 可參數化。CPU only。
用法: python3 mamba_exclusion.py  -> 輸出本目錄 CSV。每個指標先用 DL=11 重現原值，再用 DL=10(去 mamba)。"""
import itertools, os, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), ""); OUT = os.path.dirname(os.path.abspath(__file__))
DL11 = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
DL10 = [m for m in DL11 if m != 'mamba']
w0 = pd.read_csv(R+"euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R+"euroc_full/results/runs_index.csv")[["run_id","regime","method","seed"]], on="run_id")
attg = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv")
KEEP = ["a_raw_gyro(paper)","b_gyro-GT-bias(oracle)","c_static-mean-bias","d_mahony(kp=1,ki=0.05)",
        "e_mahony_gated(kp=1,ki=0.05)","s_mahony(kp=5,ki=0.05)"]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
def build(DL):
    w = w0[w0.method.isin(DL)]; att = attg[attg.method.isin(DL)]
    UNITS = [(r, s) for r in "AB" for s in sorted(w[w.regime == r].eval_sequence.unique())]
    piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]
    mats = {}
    for reg, seq in UNITS:
        g = w[(w.regime == reg) & (w.eval_sequence == seq)]
        mats[(reg, seq, "IABE")] = piv(g, "iabe_rot").values; mats[(reg, seq, "bias_rmse")] = piv(g, "bias_rmse_avg").values
        for src, gs in att[(att.regime == reg) & (att.seq == seq)].groupby("source"):
            if src in KEEP: mats[(reg, seq, "ATE:"+src)] = piv(gs, "ate").values
        assert mats[(reg, seq, "IABE")].shape == (10, len(DL))
    return UNITS, mats
def agreement(DL):  # 同 rF_noncircular_stratified.py（rng seed 0、1000 切分、4000 bootstrap）
    UNITS, mats = build(DL); rng = np.random.default_rng(0); nm_ = len(DL)
    PAIRS = np.array(list(itertools.combinations(range(nm_), 2))); PI, PJ = PAIRS[:, 0], PAIRS[:, 1]
    def select(M, idx):
        X = M[idx]; D = X[:, PI]-X[:, PJ]
        p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0); return bh(p) < .05, D.mean(0)
    others = ["IABE"] + ["ATE:"+s for s in KEEP]; names = ["bias_rmse(ceiling)"] + others
    ALL = np.arange(10); halves = [np.array(c) for c in itertools.combinations(range(10), 5)]
    NS = 1000; hs = rng.integers(0, len(halves), NS)
    A = np.zeros((NS, len(UNITS), len(names))); N = np.zeros((NS, len(UNITS)))
    for si, h in enumerate(hs):
        h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
        for ui, u in enumerate(UNITS):
            mask, d1 = select(mats[(*u, "bias_rmse")], h1); N[si, ui] = mask.sum()
            if not mask.sum(): continue
            def dir2(k):
                m = mats[(*u, k)][h2].mean(0); return np.sign(m[PI]-m[PJ])
            A[si, ui, 0] = (dir2("bias_rmse") == np.sign(d1))[mask].sum()
            for oi, k in enumerate(others): A[si, ui, 1+oi] = (dir2(k) == np.sign(d1))[mask].sum()
    uidx = {u: i for i, u in enumerate(UNITS)}
    groups = {"A": [uidx[u] for u in UNITS if u[0]=="A"], "B": [uidx[u] for u in UNITS if u[0]=="B"], "ALL": list(range(len(UNITS)))}
    pooled = lambda ix, c: A[:, ix, c].sum()/max(N[:, ix].sum(), 1)
    rows = []
    for g, gi in groups.items():
        gi = np.array(gi)
        for ci, nm in enumerate(names):
            bs = []
            for _ in range(4000):
                ix = gi[rng.integers(0, len(gi), len(gi))]; bs.append(pooled(ix, ci))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(group=g, measure=nm, pooled_rate=pooled(gi, ci), lo=lo, hi=hi, pairs_per_split=N[:, gi].sum(1).mean()))
    return pd.DataFrame(rows)
def distinguishable(DL):  # 同 rE_stats：unit 內配對 t + BH(55 對 → 這裡 C(n,2))，IABE
    UNITS, mats = build(DL); PAIRS = np.array(list(itertools.combinations(range(len(DL)), 2))); PI, PJ = PAIRS[:,0], PAIRS[:,1]
    k = 0; n = 0; per = {}
    for u in UNITS:
        X = mats[(*u, "IABE")]; D = X[:, PI]-X[:, PJ]; p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0)
        s = (bh(p) < .05).sum(); k += s; n += len(p); per[u] = s
    return k, n, per
def flip(DL):  # 同 tost_backfill.py：regime A 各序列 IABE 最小者，兩個不同 seed 贏家不同的機率，序列平均
    w = w0[w0.method.isin(DL) & (w0.regime == "A")]; pd_ = []; ps = {}
    for s in sorted(w.eval_sequence.unique()):
        piv = w[w.eval_sequence == s].pivot_table(index="seed", columns="method", values="iabe_rot")
        win = piv.idxmin(axis=1); cnt = win.value_counts().values; n = len(win)
        ps[s] = 1-(cnt*(cnt-1)).sum()/(n*(n-1)); pd_.append(ps[s])
    return float(np.mean(pd_)), ps
def spearman_ate_iabe(DL):  # 同 rE_stats：seed 平均後 ATE(raw) 與 IABE 的 Spearman，9 cell 中位數
    UNITS, mats = build(DL); r = {u: stats.spearmanr(mats[(*u, "ATE:a_raw_gyro(paper)")].mean(0), mats[(*u, "IABE")].mean(0))[0] for u in UNITS}
    return float(np.median(list(r.values()))), float(np.median([v for u, v in r.items() if u[0]=="A"])), r
def closed_loop(excl):  # 同 analyze_3.py 前段（rng 20261016、B=5000，依相同 groupby 順序消耗）
    HERE = R+"experiments/closed_loop"; B = 5000; rng = np.random.default_rng(20261016)
    P = pd.read_csv(HERE+"/pre_outage_long.csv"); V = pd.read_csv(HERE+"/variants_long.csv")
    ALL = pd.concat([P, V]); ALL["cfg"] = ALL.cfg.replace({"base":"O_base","q10":"O_q10","q01":"O_q01"})
    if excl: ALL = ALL[~ALL.run_id.str.contains("__mamba__")]
    pred = pd.read_csv(HERE+"/predictors.csv")
    def units(d):
        ctl = d[d.run_id == "CTRL:bref"].set_index(["seq","rep"])["abs"].rename("common")
        m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq","rep"])
        u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max()-g.delta.min(), iqr=g.delta.quantile(.75)-g.delta.quantile(.25))).join(ctl).reset_index()
        u["r_med"] = u.common/u.med; u["r_iqr"] = u.common/u.iqr; u["r_rng"] = u.common/u.rng; return u
    fr = lambda u: np.array([u.r_med.median(), u.r_iqr.median(), u.r_rng.median(), u.common.median(), u.med.median()])
    rows = []
    for (cfg, L), d in ALL.groupby(["cfg","L"]):
        u = units(d); seqs = sorted(u.seq.unique()); pt = fr(u); bs = []
        for _ in range(B):
            s = rng.choice(seqs, len(seqs)); bs.append(fr(pd.concat([u[u.seq == x] for x in s])))
        lo, hi = np.nanpercentile(np.array(bs, float), [2.5, 97.5], 0)
        ps = u.groupby("seq").apply(lambda x: x.common.median()/x.med.median(), include_groups=False)
        row = dict(cfg=cfg, L=L, n_methods=d[~d.run_id.str.startswith("CTRL:")].run_id.str.split("__").str[1].nunique())
        for i, k in enumerate(["ratio_med","ratio_iqr","ratio_rng","common_m","median_delta_m"]):
            row[k] = pt[i]; row[k+"_lo"] = lo[i]; row[k+"_hi"] = hi[i]
        row["per_seq_min"] = ps.min()
        # 序列內 Spearman 中位（C_abs，同 analyze_3）
        m = d[~d.run_id.str.startswith("CTRL:")]
        a = m.groupby(["seq","run_id"]).agg(abs=("abs", lambda x: np.sqrt(np.mean(x**2))), delta=("delta", lambda x: np.sqrt(np.mean(x**2)))).reset_index()
        a = a.merge(pred[["run_id","seq","iabe","bias_rmse"]], on=["run_id","seq"])
        for tgt in ("abs","delta"):
            for pr in ("iabe","bias_rmse"):
                row[f"rho_{tgt}_{pr}"] = float(np.median([stats.spearmanr(x[pr], x[tgt])[0] for _, x in a.groupby("seq")]))
        rows.append(row)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    ag11, ag10 = agreement(DL11), agreement(DL10); ag11["set"] = "orig11"; ag10["set"] = "no_mamba10"
    pd.concat([ag11, ag10]).to_csv(OUT+"/agreement.csv", index=False); print(ag11.round(3).to_string()); print(ag10.round(3).to_string())
    d11, d10 = distinguishable(DL11), distinguishable(DL10); print("distinguishable", d11[:2], d10[:2])
    pd.DataFrame([dict(set="orig11", k=d11[0], n=d11[1]), dict(set="no_mamba10", k=d10[0], n=d10[1])]).to_csv(OUT+"/distinguishable.csv", index=False)
    f11, f10 = flip(DL11), flip(DL10); print("flip", f11, f10)
    pd.DataFrame([dict(set="orig11", mean=f11[0], **f11[1]), dict(set="no_mamba10", mean=f10[0], **f10[1])]).to_csv(OUT+"/flip.csv", index=False)
    s11, s10 = spearman_ate_iabe(DL11), spearman_ate_iabe(DL10); print("rho", s11[:2], s10[:2])
    pd.DataFrame([dict(set="orig11", med9=s11[0], medA=s11[1]), dict(set="no_mamba10", med9=s10[0], medA=s10[1])]).to_csv(OUT+"/spearman_ate_iabe.csv", index=False)
    c11, c10 = closed_loop(False), closed_loop(True); c11["set"] = "all15"; c10["set"] = "no_mamba14"
    pd.concat([c11, c10]).to_csv(OUT+"/closed_loop.csv", index=False)
    print(c11[c11.L.isin([5,10])].round(3).to_string()); print(c10[c10.L.isin([5,10])].round(3).to_string())

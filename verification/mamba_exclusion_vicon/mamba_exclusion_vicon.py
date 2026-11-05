"""Vicon (V1_01..V2_03, regime A) Mamba exclusion sensitivity. Logic copied (not modified originals) from
euroc_full/analysis_vicon/{vcommon,rf_agreement,seed_stability,t3_analyze}.py. CPU only. Reads only; writes to this folder.
Each metric is first recomputed with DL=11 (reproduction check vs full_2026-10-23 CSVs), then DL=10 (no mamba)."""
import itertools, json, os, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), ""); OUT = os.path.dirname(os.path.abspath(__file__))
RES = R+"euroc_full/results_vicon"; FULL = R+"euroc_full/analysis_vicon/full_2026-10-23/"
SEQS = ["V1_01","V1_02","V1_03","V2_01","V2_02","V2_03"]
DL11 = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
DL10 = [m for m in DL11 if m != 'mamba']
KEEP = ["a_raw_gyro(paper)","b_gyro-GT-bias(oracle)","c_static-mean-bias","d_mahony(kp=1,ki=0.05)",
        "e_mahony_gated(kp=1,ki=0.05)","s_mahony(kp=5,ki=0.05)"]
def load_wide():  # vcommon.load_wide copy
    idx = pd.read_csv(RES+"/runs_index.csv"); rows = []
    for _, r in idx.iterrows():
        if str(r["status"]) != "ok": continue
        mp = f"{RES}/runs/{r['run_id']}/metrics.json"
        if not os.path.exists(mp): continue
        for seq, mt in json.load(open(mp)).items():
            d = {"run_id": r["run_id"], "eval_sequence": seq}
            for k, v in (mt or {}).items():
                if v is not None: d[k] = float(v)
            rows.append(d)
    return pd.DataFrame(rows).merge(idx[["run_id","regime","method","seed"]], on="run_id")
W = load_wide(); W = W[W.regime == "A"]
cnt = W[W.method.isin(DL11)].groupby(["method","eval_sequence"]).seed.nunique().unstack(fill_value=0).reindex(index=DL11, columns=SEQS, fill_value=0)
assert [m for m in DL11 if (cnt.loc[m] >= 10).all()] == DL11, "complete-case set != 11"
ATT = pd.read_csv(FULL+"t1_ate_per_run_gated.csv", comment="#"); ATT = ATT[ATT.regime == "A"]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)[::-1]; out = np.empty_like(p); run = 1
    for k, i in enumerate(o): run = min(run, p[i]*n/(n-k)); out[i] = run
    return out
def build(DL):
    w = W[W.method.isin(DL)]; att = ATT[ATT.method.isin(DL)]
    piv = lambda df, c: df.pivot_table(index="seed", columns="method", values=c)[DL]; mats = {}
    for seq in SEQS:
        g = w[w.eval_sequence == seq]
        mats[(seq,"IABE")] = piv(g,"iabe_rot").values; mats[(seq,"bias_rmse")] = piv(g,"bias_rmse_avg").values
        for src, gs in att[att.seq == seq].groupby("source"):
            if src in KEEP: mats[(seq,"ATE:"+src)] = piv(gs,"ate").values
        assert mats[(seq,"IABE")].shape == (10, len(DL))
    return mats
def agreement(DL):  # rf_agreement.py: rng(0), 1000 splits, 4000 seq-bootstrap
    mats = build(DL); rng = np.random.default_rng(0); K = len(DL); UNITS = [("A", s) for s in SEQS]
    PAIRS = np.array(list(itertools.combinations(range(K), 2))); PI, PJ = PAIRS[:,0], PAIRS[:,1]
    others = ["IABE"] + ["ATE:"+s for s in KEEP]; names = ["bias_rmse (ceiling)"] + others
    def select(M, idx):
        X = M[idx]; D = X[:,PI]-X[:,PJ]; p = np.nan_to_num(stats.ttest_1samp(D, 0, axis=0).pvalue, nan=1.0); return bh(p) < .05, D.mean(0)
    ALL = np.arange(10); halves = [np.array(c) for c in itertools.combinations(range(10), 5)]
    NS = 1000; hs = rng.integers(0, len(halves), NS); A = np.zeros((NS, 6, len(names))); N = np.zeros((NS, 6))
    for si, h in enumerate(hs):
        h1 = halves[h]; h2 = np.setdiff1d(ALL, h1)
        for ui, seq in enumerate(SEQS):
            mask, d1 = select(mats[(seq,"bias_rmse")], h1); N[si, ui] = mask.sum()
            if not mask.sum(): continue
            def dir2(k):
                m = mats[(seq,k)][h2].mean(0); return np.sign(m[PI]-m[PJ])
            A[si,ui,0] = (dir2("bias_rmse") == np.sign(d1))[mask].sum()
            for oi, k in enumerate(others): A[si,ui,1+oi] = (dir2(k) == np.sign(d1))[mask].sum()
    pooled = lambda ix, c: A[:,ix,c].sum()/max(N[:,ix].sum(), 1); gi = np.arange(6); rows = []
    for ci, nm in enumerate(names):
        bs, ab = [], []
        for _ in range(4000):
            ix = gi[rng.integers(0, 6, 6)]; bs.append(pooled(ix, ci)); ab.append(pooled(ix, ci)-.5)
        lo, hi = np.percentile(bs, [2.5, 97.5]); ab = np.array(ab)
        rows.append(dict(measure=nm, pooled_rate=pooled(gi, ci), lo=lo, hi=hi, p_le_half=(np.sum(ab <= 0)+1)/(len(ab)+1), pairs_per_split=N.sum(1).mean()))
    per = {nm: [A[:,u,ci].sum()/max(N[:,u].sum(),1) for u in range(6)] for ci, nm in enumerate(names)}
    return pd.DataFrame(rows), pd.DataFrame(per, index=SEQS)
def flip(DL):
    mats = build(DL); rows = []
    for metric in ["IABE","bias_rmse"]:
        v = []
        for seq in SEQS:
            c = np.bincount(mats[(seq,metric)].argmin(1), minlength=len(DL)); n = 10; v.append(1-(c*(c-1)).sum()/(n*(n-1)))
        rows.append(dict(metric=metric, mean_flip=np.mean(v), random_baseline=1-1/len(DL), **dict(zip(SEQS, v))))
    return pd.DataFrame(rows)
def distinguishable(DL):  # seed_stability: BH-t over C(K,2) pairs per seq, or |d|>=2
    mats = build(DL); P = list(itertools.combinations(range(len(DL)), 2)); tot = dict(bh=0, d2=0, either=0, n=0)
    for seq in SEQS:
        M = mats[(seq,"IABE")]; ps, ds = [], []
        for i, j in P:
            d = M[:,i]-M[:,j]; sd = d.std(ddof=1); ps.append(stats.ttest_1samp(d, 0).pvalue); ds.append(abs(d.mean())/sd if sd > 0 else np.inf)
        b = bh(np.array(ps)) < .05; e = np.array(ds) >= 2
        tot["bh"] += b.sum(); tot["d2"] += e.sum(); tot["either"] += (b|e).sum(); tot["n"] += len(P)
    return tot
def closed_loop(excl):  # t3_analyze.py: rng 20261016, B=5000, same groupby order
    B = 5000; rng = np.random.default_rng(20261016)
    P = pd.read_csv(FULL+"t3_pre_outage_long.csv", comment="#"); V = pd.read_csv(FULL+"t3_variants_long.csv", comment="#")
    ALL = pd.concat([P, V]); ALL["cfg"] = ALL.cfg.replace({"base":"O_base","q10":"O_q10","q01":"O_q01"})
    if excl: ALL = ALL[~ALL.run_id.str.contains("__mamba__")]
    def units(d):
        ctl = d[d.run_id == "CTRL:bref"].set_index(["seq","rep"])["abs"].rename("common")
        m = d[~d.run_id.str.startswith("CTRL:")]; g = m.groupby(["seq","rep"])
        u = pd.DataFrame(dict(med=g.delta.median(), rng=g.delta.max()-g.delta.min(), iqr=g.delta.quantile(.75)-g.delta.quantile(.25), sd_abs=g["abs"].std())).join(ctl).reset_index()
        u["r_med"] = u.common/u.med; u["r_rng"] = u.common/u.rng; u["r_iqr"] = u.common/u.iqr; u["sd_rel"] = u.sd_abs/u.common; return u
    fr = lambda u: np.array([u.r_med.median(), u.r_iqr.median(), u.r_rng.median(), u.sd_rel.median(), u.common.median(), u.med.median()])
    rows = []
    for (cfg, L), d in ALL.groupby(["cfg","L"]):
        u = units(d); seqs = sorted(u.seq.unique()); pt = fr(u); bs = []
        for _ in range(B):
            s = rng.choice(seqs, len(seqs)); bs.append(fr(pd.concat([u[u.seq == x] for x in s])))
        lo, hi = np.nanpercentile(np.array(bs, float), [2.5, 97.5], 0); row = dict(cfg=cfg, L=L, n_units=len(u))
        for i, k in enumerate(["ratio_med","ratio_iqr","ratio_rng","sdabs_over_common","common_m","median_delta_m"]):
            row[k] = pt[i]; row[k+"_lo"] = lo[i]; row[k+"_hi"] = hi[i]
        rows.append(row)
    return pd.DataFrame(rows)
def outlier():  # Mamba bias_rmse_avg / constant_perseq bias_rmse_avg, per seq (median over 10 seeds)
    g = W[W.method.isin(DL11 + ["constant_perseq"])]; med = g.groupby(["method","eval_sequence"]).bias_rmse_avg.median().unstack()
    r = med.loc[DL11].div(med.loc["constant_perseq"], axis=1); r.insert(0, "const_perseq_abs", np.nan)
    out = r.drop(columns="const_perseq_abs").T; out["const_abs"] = med.loc["constant_perseq"]
    out["mamba_rank_of_11(1=best)"] = r.rank(axis=0).loc["mamba"]; out["median_other10"] = r.loc[DL10].median(); out["max_other10"] = r.loc[DL10].max()
    return out
if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    a11, p11 = agreement(DL11); a10, p10 = agreement(DL10); a11["set"] = "orig11"; a10["set"] = "no_mamba10"
    pd.concat([a11, a10]).to_csv(OUT+"/agreement.csv", index=False); print(a11.round(3).to_string()); print(a10.round(3).to_string())
    pd.concat([p11.assign(set="orig11"), p10.assign(set="no_mamba10")]).to_csv(OUT+"/agreement_per_seq.csv")
    print(p11.round(2).T.to_string()); print(p10.round(2).T.to_string())
    f = pd.concat([flip(DL11).assign(set="orig11"), flip(DL10).assign(set="no_mamba10")]); f.to_csv(OUT+"/flip.csv", index=False); print(f.round(3).to_string())
    d = pd.DataFrame([dict(set="orig11", **distinguishable(DL11)), dict(set="no_mamba10", **distinguishable(DL10))]); d.to_csv(OUT+"/distinguishable.csv", index=False); print(d)
    o = outlier(); o.to_csv(OUT+"/mamba_outlier_ratio.csv"); print(o.round(2).to_string())
    c11, c10 = closed_loop(False), closed_loop(True); c11["set"] = "all"; c10["set"] = "no_mamba"
    pd.concat([c11, c10]).to_csv(OUT+"/closed_loop.csv", index=False)
    cols = ["cfg","L","n_units","ratio_med","ratio_med_lo","ratio_med_hi","common_m","median_delta_m"]
    print(c11[c11.L.isin([5,10])][cols].round(3).to_string()); print(c10[c10.L.isin([5,10])][cols].round(3).to_string())

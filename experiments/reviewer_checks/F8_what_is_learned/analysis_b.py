"""F8 items 4, 5, 6 and reproduction of 0.554. The pipeline is ../F1_noise_ceiling/common.py (252 splits)."""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
from common import *
M = pd.read_csv(os.path.join(HERE, "stats_enriched.csv"))
units, bias, ate = load()
MAMBA = DL.index("mamba"); mpair = (PI == MAMBA) | (PJ == MAMBA); none = np.zeros(len(PI), bool)
seqs = sorted({u[1] for u in units})
G = {"A": [u for u in units if u[0] == "A"], "B": [u for u in units if u[0] == "B"], "ALL": units}
def mat(col):
    out = {}
    for (reg, seq) in units:
        g = M[(M.regime == reg) & (M.seq == seq) & M.method.isin(DL)]
        p = g.pivot_table(index="seed", columns="method", values=col)[DL]; assert p.shape == (10, 11); out[(reg, seq)] = p.values
    return out
OFF2, FL2, VREF = mat("off2"), mat("fl2"), mat("var_ref")
for u in units: assert np.allclose(np.sqrt(OFF2[u] + FL2[u]), bias[u], rtol=1e-9)
OFFONLY = {u: np.sqrt(OFF2[u] + VREF[u]) for u in units}      # b_hat -> test-window mean
FLONLY = {u: np.sqrt(FL2[u]) for u in units}                  # keep only the fluctuation around the window mean
sel = precompute_selection(units, bias)
from scipy import stats as st

def jk(num, den, g):
    """num, den: dict unit -> value (already summed over the 252 splits). Jackknife with sequences as clusters, t quantile."""
    us = G[g]; cl = sorted({u[1] for u in us}); k = len(cl)
    def stat(excl=None):
        uu = [u for u in us if u[1] != excl]; d = sum(den[u] for u in uu); return sum(num[u] for u in uu) / d if d else np.nan
    th = stat(); loo = np.array([stat(c) for c in cl]); se = np.sqrt((k-1)/k * ((loo - loo.mean())**2).sum()); t = st.t.ppf(.975, k-1)
    return th, th - t*se, th + t*se, k

# ---------- reproduction of raw ATE 0.554 ----------
for src in ("raw", "oracle"):
    A, N = agree_counts(ate[src], units, sel)
    a = {u: A[:, i].sum() for i, u in enumerate(units)}; n = {u: N[:, i].sum() for i, u in enumerate(units)}
    print(f"[repro b] {src} ATE vs bias RMSE agreement: " + "  ".join(f"{g}: {jk(a, n, g)[0]:.3f} [{jk(a,n,g)[1]:.3f},{jk(a,n,g)[2]:.3f}]" for g in G))

# ---------- item 5 ----------
print("\n[item 5] pair selection rule (common.py select()): for each (half-1 seed set, case): D = bias_i - bias_j over the 5 seeds of half 1, one-sample t test (H0: mean difference = 0), 55 pairs, BH(q=.05); the direction of a selected pair = sign(half-1 mean difference); half 2 (the other 5 seeds) takes no part in the selection and is only used to see whether the direction reproduces.")
h1 = HALVES[0]; h2 = np.setdiff1d(ALL, h1); u = ("A", "MH_01")
X = bias[u]; D1 = X[h1][:, PI] - X[h1][:, PJ]; p1 = np.nan_to_num(st.ttest_1samp(D1, 0, axis=0).pvalue, nan=1.0); q1 = bh(p1)
d2 = (X[h2][:, PI] - X[h2][:, PJ]).mean(0); ix = np.where(q1 < .05)[0]
print(f"  example: case={u}, half-1 seed indices={list(h1)} (seed values {[42+i for i in h1]}), half 2={list(h2)}; {len(ix)}/55 pairs selected; first 8 by q:")
for i in ix[np.argsort(q1[ix])][:8]:
    print(f"   {DL[PI[i]]:>13} - {DL[PJ[i]]:<13} half-1 mean difference {D1[:, i].mean():+.2e}  q={q1[i]:.4f}  half-2 mean difference {d2[i]:+.2e}  direction {'reproduced' if np.sign(d2[i]) == np.sign(D1[:, i].mean()) else 'not reproduced'}")
print(f"   selected pairs of this example whose direction reproduces in half 2: {int((np.sign(d2[ix]) == np.sign(D1.mean(0)[ix])).sum())}/{len(ix)}")
# own-retest agreement (ceiling) and permutation control
A_, N_ = agree_counts(bias, units, sel)
a0 = {uu: A_[:, i].sum() for i, uu in enumerate(units)}; n0 = {uu: N_[:, i].sum() for i, uu in enumerate(units)}
selfres = {g: jk(a0, n0, g) for g in G}
print("  bias RMSE own-retest direction agreement (252 splits): " + "  ".join(f"{g}: {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for g, v in selfres.items()))
print("  mean number of selected pairs per (split, case): ", {g: round(sum(n0[uu] for uu in G[g]) / (252*len(G[g])), 2) for g in G})
rng = np.random.default_rng(0); NP = 20; nulla = {uu: 0. for uu in units}; nulln = {uu: 0. for uu in units}
for _ in range(NP):
    pb = {}
    for uu in units:
        Xp = bias[uu].copy()
        for r in range(10): Xp[r] = Xp[r][rng.permutation(11)]    # randomly permute the method labels within each seed
        pb[uu] = Xp
    selp = precompute_selection(units, pb); Ap, Np = agree_counts(pb, units, selp)
    for i, uu in enumerate(units): nulla[uu] += Ap[:, i].sum(); nulln[uu] += Np[:, i].sum()
print(f"  permutation control (method labels shuffled within each seed, {NP} times): mean number of selected pairs per (split, case): ", {g: round(sum(nulln[uu] for uu in G[g]) / (252*NP*len(G[g])), 3) for g in G},
      "; share of selected pairs whose direction reproduces in half 2:", {g: round(sum(nulla[uu] for uu in G[g]) / max(sum(nulln[uu] for uu in G[g]), 1), 3) for g in G})
# pairs distinguishable with all 10 seeds
print("  pairs distinguishable under paired t + BH with all 10 seeds: ", {g: sum(int(select(bias[uu], ALL)[0].sum()) for uu in G[g]) for g in G}, "(55 pairs per case)")

# ---------- item 4 ----------
def decomp(excl_mask):
    acc = {}
    rows = []
    for hi, hh in enumerate(HALVES):
        for uu in units:
            mask, _ = sel[(hi, uu)]; mask = mask & ~excl_mask
            if not mask.any(): continue
            do = (OFF2[uu][hh][:, PI] - OFF2[uu][hh][:, PJ]).mean(0)[mask]; df = (FL2[uu][hh][:, PI] - FL2[uu][hh][:, PJ]).mean(0)[mask]
            dt = do + df; s = np.sign(dt)
            a = acc.setdefault(uu, dict(off=0., fl=0., tot=0., n=0, offdom=0, same=0, offsign=0))
            a["off"] += (s*do).sum(); a["fl"] += (s*df).sum(); a["tot"] += np.abs(dt).sum(); a["n"] += mask.sum()
            a["offdom"] += (np.abs(do) > np.abs(df)).sum(); a["same"] += (np.sign(do) == np.sign(df)).sum(); a["offsign"] += (np.sign(do) == s).sum()
            rows.append(np.where(np.abs(dt) > 0, s*do/np.abs(dt), np.nan))
    return acc, np.concatenate(rows) if rows else np.array([])
res4 = []
print("\n[item 4] split of the MSE difference of the distinguishable pairs (half-1 seed mean; pooled share = sum sign(dMSE)*d(off^2) / sum |dMSE|, dominated by pairs with a large MSE; per-pair count measures are reported too)")
for lab, ex in (("with Mamba", none), ("without Mamba", mpair), ("Mamba pairs only", ~mpair)):
    acc, per = decomp(ex)
    for g in G:
        num = {uu: acc.get(uu, dict(off=0))["off"] for uu in G[g]}; den = {uu: acc.get(uu, dict(tot=0))["tot"] for uu in G[g]}
        th, lo, hi, k = jk(num, den, g)
        n = sum(acc.get(uu, dict(n=0))["n"] for uu in G[g])
        cnt = lambda key: jk({uu: acc.get(uu, {}).get(key, 0) for uu in G[g]}, {uu: acc.get(uu, {}).get("n", 0) for uu in G[g]}, g)
        offdom, same, offsign = cnt("offdom"), cnt("same"), cnt("offsign")
        # per-pair median share (no CI)
        res4.append(dict(subset=lab, group=g, n_pair_instances=n, offset_share=th, lo=lo, hi=hi, frac_pairs_offset_dominant=offdom[0], offdom_lo=offdom[1], offdom_hi=offdom[2],
                         frac_off_fl_same_sign=same[0], frac_offset_sign_matches_total=offsign[0]))
        print(f"  {lab:8s} {g:3s}: n={n:6d} offset share {th:.3f} [{lo:.3f},{hi:.3f}] (clusters={k}); share of pairs with |d(off^2)| > |d(fl^2)| {offdom[0]:.3f} [{offdom[1]:.3f},{offdom[2]:.3f}]; d(off^2) and d(fl^2) of the same sign {same[0]:.3f}; d(off^2) has the sign of the total difference {offsign[0]:.3f}")
pd.DataFrame(res4).to_csv(os.path.join(HERE, "item4_pairs.csv"), index=False)
# which methods enter the selected pairs: selection frequency of each method
cnts = np.zeros(11)
for hi in range(252):
    for uu in units:
        m = sel[(hi, uu)][0]; 
        for i in np.where(m)[0]: cnts[PI[i]] += 1; cnts[PJ[i]] += 1
print("  share of the selected pairs each method appears in: ", {DL[i]: round(cnts[i]/cnts.sum()*2, 3) for i in range(11)})

# ---------- item 6 ----------
def agree(src, ex):
    A = np.zeros((252, len(units))); N = np.zeros_like(A)
    for hi in range(252):
        h2 = np.setdiff1d(ALL, HALVES[hi])
        for ui, uu in enumerate(units):
            mask, s1 = sel[(hi, uu)]; mask = mask & ~ex; N[hi, ui] = mask.sum()
            if mask.any():
                m = src[uu][h2].mean(0); A[hi, ui] = (np.sign(m[PI] - m[PJ]) == s1)[mask].sum()
    return {uu: A[:, i].sum() for i, uu in enumerate(units)}, {uu: N[:, i].sum() for i, uu in enumerate(units)}
res6 = []
print("\n[item 6] pairs selected with the original bias RMSE on half 1; direction agreement of the half-2 method means of a substitute metric (ceiling = own retest of bias RMSE)")
for lab, ex in (("with Mamba", none), ("without Mamba", mpair)):
    for name, src in (("bias RMSE (own retest, ceiling)", bias), ("offset-only (b_hat -> window mean)", OFFONLY), ("fluctuation-only (fl only)", FLONLY)):
        a, n = agree(src, ex)
        for g in G:
            th, lo, hi, k = jk(a, n, g); res6.append(dict(subset=lab, metric=name, group=g, agree=th, lo=lo, hi=hi))
    sub = pd.DataFrame(res6); sub = sub[sub.subset == lab]
    for g in G:
        s = sub[sub.group == g].set_index("metric"); ceil = s.loc["bias RMSE (own retest, ceiling)", "agree"]
        print(f"  {lab:6s} {g:3s}: ceiling {ceil:.3f} | offset-only {s.loc['offset-only (b_hat -> window mean)','agree']:.3f} [{s.loc['offset-only (b_hat -> window mean)','lo']:.3f},{s.loc['offset-only (b_hat -> window mean)','hi']:.3f}] (fraction of ceiling {s.loc['offset-only (b_hat -> window mean)','agree']/ceil:.3f}) | fluct-only {s.loc['fluctuation-only (fl only)','agree']:.3f} [{s.loc['fluctuation-only (fl only)','lo']:.3f},{s.loc['fluctuation-only (fl only)','hi']:.3f}] (fraction of ceiling {s.loc['fluctuation-only (fl only)','agree']/ceil:.3f})")
pd.DataFrame(res6).to_csv(os.path.join(HERE, "item6_agree.csv"), index=False)
# all 10 seeds, noise-free version: direction of the method-mean ranking of the 55 pairs (pairs distinguishable with all seeds under BH)
print("\n[item 6, addendum] among the pairs distinguishable with all 10 seeds (BH), share whose offset-only method-mean ordering has the same direction as the original ordering:")
for lab, ex in (("with Mamba", none), ("without Mamba", mpair)):
    for g in G:
        a = n = 0
        for uu in G[g]:
            m, d = select(bias[uu], ALL); m = m & ~ex; mo = OFFONLY[uu].mean(0)
            a += (np.sign(mo[PI] - mo[PJ]) == np.sign(d))[m].sum(); n += m.sum()
        print(f"  {lab} {g}: {a}/{n} = {a/max(n,1):.3f}")

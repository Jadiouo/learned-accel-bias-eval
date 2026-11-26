"""F4: sequence-cluster bootstrap for the nine-case pool, Mamba share of selected pairs, ATE-vs-IABE agreement.
Read-only on existing csv; pipeline from ../F1_noise_ceiling/common.py (252 splits enumerated)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
from common import *

units, bias, ate = load()
w = pd.read_csv(R + "euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R + "euroc_full/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
w = w[w.method.isin(DL)]
iabe = {}
for reg, seq in units:
    g = w[(w.regime == reg) & (w.eval_sequence == seq)]
    p = g.pivot_table(index="seed", columns="method", values="iabe_rot")[DL]
    assert p.shape == (10, 11); iabe[(reg, seq)] = p.values

G = groups(units); MAMBA = DL.index("mamba")
mpair = (PI == MAMBA) | (PJ == MAMBA)
rows = []

# 1. sequence-cluster vs case-cluster bootstrap, nine-case pool
sel_b = precompute_selection(units, bias)
seqs = sorted({u[1] for u in units}); seq_units = {s: [i for i, u in enumerate(units) if u[1] == s] for s in seqs}
for k in ("raw", "oracle"):
    A, N = agree_counts(ate[k], units, sel_b)
    a, n = A.sum(0), N.sum(0)          # per unit, summed over the 252 splits
    pt = a.sum() / n.sum()
    rng = np.random.default_rng(1)
    bs_case, bs_seq = [], []
    for _ in range(4000):
        ix = rng.integers(0, 9, 9); bs_case.append(a[ix].sum() / n[ix].sum())
        sx = rng.integers(0, len(seqs), len(seqs)); ix = np.concatenate([seq_units[seqs[j]] for j in sx])
        bs_seq.append(a[ix].sum() / n[ix].sum())
    p_case = (np.sum(np.array(bs_case) <= .5) + 1) / 4001; p_seq = (np.sum(np.array(bs_seq) <= .5) + 1) / 4001
    rows.append(dict(item="nine-case pool", source=k, point=pt,
                     case_lo=np.percentile(bs_case, 2.5), case_hi=np.percentile(bs_case, 97.5), p_le_half_case=p_case,
                     seq_lo=np.percentile(bs_seq, 2.5), seq_hi=np.percentile(bs_seq, 97.5), p_le_half_seq=p_seq))

# 2. Mamba share of selected pairs
tot = {g: [0, 0] for g in G}
for hi in range(len(HALVES)):
    for ui, u in enumerate(units):
        mask, _ = sel_b[(hi, u)]
        for g, gi in G.items():
            if ui in gi: tot[g][0] += (mask & mpair).sum(); tot[g][1] += mask.sum()
for g, (m, n) in tot.items():
    rows.append(dict(item="Mamba share of pairs selected by bias RMSE (half seeds, BH)", source=g, point=m / n, n_selected=n))
for g, gi in G.items():
    m = n = 0
    for ui in gi:
        mask, _ = select(iabe[units[ui]], ALL); m += (mask & mpair).sum(); n += mask.sum()
    rows.append(dict(item="Mamba share of pairs resolved by IABE (10 seeds, BH)", source=g, point=m / n, n_selected=n))

# 3. ATE vs IABE direction agreement (select by IABE on half 1, ATE sign on half 2)
sel_i = precompute_selection(units, iabe)
def rate(A, N, gi): return A[:, gi].sum() / N[:, gi].sum()
def agree_excl(ate_by_unit, sel, excl):
    A = np.zeros((len(HALVES), len(units))); N = np.zeros_like(A)
    for hi in range(len(HALVES)):
        h2 = np.setdiff1d(ALL, HALVES[hi])
        for ui, u in enumerate(units):
            mask, s1 = sel[(hi, u)]; mask = mask & ~excl
            N[hi, ui] = mask.sum()
            if not mask.sum(): continue
            m = ate_by_unit[u][h2].mean(0); A[hi, ui] = (np.sign(m[PI] - m[PJ]) == s1)[mask].sum()
    return A, N
none = np.zeros(len(PI), bool)
for lab, ex in (("all 11", none), ("without Mamba", mpair)):
    for k, src in (("IABE own retest", iabe), ("raw ATE", ate["raw"]), ("oracle ATE", ate["oracle"])):
        A, N = agree_excl(src, sel_i, ex)
        for g, gi in G.items():
            rows.append(dict(item=f"agreement with IABE ranking ({lab})", source=f"{k} / {g}", point=rate(A, N, gi)))
    for k, src in (("bias RMSE own retest", bias), ("IABE", iabe), ("raw ATE", ate["raw"]), ("oracle ATE", ate["oracle"])):
        A, N = agree_excl(src, sel_b, ex)
        for g, gi in G.items():
            rows.append(dict(item=f"agreement with bias-RMSE ranking ({lab})", source=f"{k} / {g}", point=rate(A, N, gi)))

P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, "f4_results.csv"), index=False)
pd.set_option("display.width", 250); print(P.round(4).to_string())

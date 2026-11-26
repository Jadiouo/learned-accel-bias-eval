"""F1 noise-ceiling simulation. Usage: python simulate.py [N_SIM] (default 2000; a small value gives a quick trial, N is part of the output file names)."""
import os, sys, time; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
OUT = HERE + "/"
NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
units, bias, ate = load(); G = groups(units); NU = len(units)
# Pair-selection info independent of ATE (252 splits x 55 pairs)
S1, MASK, W2 = {}, {}, np.zeros((len(HALVES), 10))
for hi, h1 in enumerate(HALVES): W2[hi, np.setdiff1d(ALL, h1)] = 1/5
for u in units:
    S1[u] = np.zeros((len(HALVES), 55)); MASK[u] = np.zeros((len(HALVES), 55), bool)
    for hi, h1 in enumerate(HALVES):
        m, d = select(bias[u], h1); MASK[u][hi] = m; S1[u][hi] = np.sign(d)
NPAIR = np.array([MASK[u].sum() for u in units], float)       # selected pairs per case over the 252 splits
def agree_u(u, T):
    H = W2 @ T; D = np.sign(H[:, PI]-H[:, PJ])
    return float(((D == S1[u]) & MASK[u]).sum())
def reliability(u, T):
    H1 = ((1-(W2 > 0))/5) @ T; H2 = W2 @ T
    return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(H1, H2)]))

def make_sim(T, b, variant, rng):
    m = T.mean(0); E = T-m; ms = np.sort(m)
    order = np.argsort(np.argsort(b.mean(0)))          # rank of the mean bias, 0..10
    if variant == "V3":
        z = (b.mean(0)-b.mean(0).mean())/b.mean(0).std(ddof=1); sig = m.mean()+z*m.std(ddof=1)
    elif variant == "C1":                                # negative control: signal order unrelated to bias
        sig = ms[rng.permutation(11)]
    else: sig = ms[order]
    if variant == "C0": Ns = np.zeros_like(E)
    elif variant == "V1":
        Ns = rng.permutation(E.ravel()).reshape(E.shape)
    else:                                                # V2, V3, C1: seeds permuted independently within each method
        Ns = np.stack([rng.permutation(E[:, j]) for j in range(11)], 1)
    return sig[None, :]+Ns

observed = {k: np.array([agree_u(u, ate[k][u]) for u in units]) for k in ("raw", "oracle")}
obs_rel = {k: np.array([reliability(u, ate[k][u]) for u in units]) for k in ("raw", "oracle")}
def rate(a, ix): return a[ix].sum()/NPAIR[ix].sum()
rows, cal = [], []
t0 = time.time()
variants = ["V1", "V2", "V3", "C0", "C1"]
brng = np.random.default_rng(777)
for k in ("raw", "oracle"):
    for vi, var in enumerate(variants):
        rng = np.random.default_rng([20261126, 0 if k == "raw" else 1, vi])
        a = np.zeros((NSIM, NU)); rel = np.zeros((min(NSIM, 100), NU))
        for s in range(NSIM):
            for ui, u in enumerate(units):
                Ts = make_sim(ate[k][u], bias[u], var, rng); a[s, ui] = agree_u(u, Ts)
                if s < rel.shape[0]: rel[s, ui] = reliability(u, Ts)
        cal.append(dict(source=k, variant=var, **{f"{u[0]}/{u[1]}": rel[:, i].mean() for i, u in enumerate(units)},
                        mean_all=rel.mean(), obs_mean_all=obs_rel[k].mean()))
        np.save(OUT+f"sim_agree_{k}_{var}_N{NSIM}.npy", a)
        for g, gi in G.items():
            gi = np.array(gi); fixed = a[:, gi].sum(1)/NPAIR[gi].sum()
            boot = np.array([rate(a[s], gi[brng.integers(0, len(gi), len(gi))]) if len(gi) > 1 else np.nan for s in range(NSIM)])
            obs = observed[k][gi].sum()/NPAIR[gi].sum()
            for ver, d in (("fixed_cases", fixed), ("with_case_bootstrap", boot)):
                if np.isnan(d).any(): continue
                lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
                rows.append(dict(source=k, variant=var, group=g, version=ver, n_sim=NSIM, ceil_median=med, ceil_p2_5=lo, ceil_p97_5=hi,
                                 observed=obs, obs_over_ceil=obs/med, excess_ratio=(obs-.5)/(med-.5) if abs(med-.5) > 1e-9 else np.nan,
                                 obs_percentile_in_ceiling=float((d <= obs).mean())))
        print(k, var, "done %.0fs"%(time.time()-t0), flush=True)
P = pd.DataFrame(rows); P.to_csv(OUT+f"results_ceiling_N{NSIM}.csv", index=False)
pd.DataFrame(cal).to_csv(OUT+f"results_calibration_N{NSIM}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(P[P.version == "fixed_cases"].round(3).drop(columns=["version"]).to_string())

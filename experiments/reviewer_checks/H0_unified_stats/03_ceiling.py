"""H0 step 3: ATE seed-noise ceiling, K=11 (check against F1) and K=10 (no Mamba). Design copied from ../F1_noise_ceiling/simulate.py (V1,V2,V3,C0,C1),
generalised to K methods. usage: 03_ceiling.py <K11|K10> <NSIM>.  K11 uses F1's seeds so that first NSIM rows equal F1's saved arrays (pipeline check).
K10 uses seeds [20261127, src, variant]."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib import *
which, NSIM = sys.argv[1], int(sys.argv[2]); cols = MAMBA_SETS["included" if which == "K11" else "excluded"]; K = len(cols); NP = K*(K-1)//2
units, bias, iabe, seeds = load_mh(); NU = len(units); G = groups_of(units)
att = pd.read_csv(R+"experiments/attitude_controls/ate_per_run_gated.csv"); att = att[att.method.isin(DL)]
ATE = {"raw": ate_from_long(att, units, seeds, "a_raw_gyro(paper)"), "oracle": ate_from_long(att, units, seeds, "b_gyro-GT-bias(oracle)")}
bias = {u: bias[u][:, cols] for u in units}; ATE = {k: {u: v[u][:, cols] for u in units} for k, v in ATE.items()}
ag = Agree(units, bias, K); PI, PJ = ag.PI, ag.PJ
W2 = np.zeros((252, 10))
for hi, h1 in enumerate(HALVES): W2[hi, np.setdiff1d(ALL10, h1)] = 1/5
S1 = {u: np.array([ag.sel[(hi, u)][1] for hi in range(252)]) for u in units}; MASK = {u: np.array([ag.sel[(hi, u)][0] for hi in range(252)]) for u in units}
NPAIR = np.array([MASK[u].sum() for u in units], float)
def agree_u(u, T):
    H = W2 @ T; D = np.sign(H[:, PI]-H[:, PJ]); return float(((D == S1[u]) & MASK[u]).sum())
def make_sim(T, b, variant, rng):
    m = T.mean(0); E = T-m; ms = np.sort(m); order = np.argsort(np.argsort(b.mean(0)))
    if variant == "V3":
        z = (b.mean(0)-b.mean(0).mean())/b.mean(0).std(ddof=1); sig = m.mean()+z*m.std(ddof=1)
    elif variant == "C1": sig = ms[rng.permutation(K)]
    else: sig = ms[order]
    if variant == "C0": Ns = np.zeros_like(E)
    elif variant == "V1": Ns = rng.permutation(E.ravel()).reshape(E.shape)
    else: Ns = np.stack([rng.permutation(E[:, j]) for j in range(K)], 1)
    return sig[None, :]+Ns
observed = {k: np.array([agree_u(u, ATE[k][u]) for u in units]) for k in ("raw", "oracle")}
rows = []; variants = ["V1", "V2", "V3", "C0", "C1"]; t0 = time.time()
for k in ("raw", "oracle"):
    for vi, var in enumerate(variants):
        seed = [20261126, 0 if k == "raw" else 1, vi] if which == "K11" else [20261127, 0 if k == "raw" else 1, vi]
        rng = np.random.default_rng(seed); a = np.zeros((NSIM, NU))
        for s in range(NSIM):
            for ui, u in enumerate(units): a[s, ui] = agree_u(u, make_sim(ATE[k][u], bias[u], var, rng))
        f1p = R+f"experiments/reviewer_checks/F1_noise_ceiling/sim_agree_{k}_{var}_N2000.npy"   # written by F1 simulate.py; not stored in the repository
        if which == "K11" and os.path.exists(f1p):
            f1 = np.load(f1p)[:NSIM]
            print("K11 check vs F1", k, var, "identical:", bool(np.array_equal(a, f1)), flush=True)
        for g, gi in G.items():
            gi = np.array(gi); d = a[:, gi].sum(1)/NPAIR[gi].sum(); obs = observed[k][gi].sum()/NPAIR[gi].sum()
            lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
            rows.append(dict(attitude=k, variant=var, group=g, mamba="included" if which == "K11" else "excluded", n_sim=NSIM, seed=str(seed), ceil_median=med, ceil_p2_5=lo, ceil_p97_5=hi,
                             observed_point=obs, obs_percentile_in_ceiling=float((d <= obs).mean()), n_pairs=NPAIR[gi].sum()/252))
        print(which, k, var, "%.0fs" % (time.time()-t0), flush=True)
pd.DataFrame(rows).to_csv(HERE+f"/ceiling_{which}_N{NSIM}.csv", index=False)

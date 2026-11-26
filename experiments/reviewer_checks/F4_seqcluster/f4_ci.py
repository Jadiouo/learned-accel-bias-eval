"""F4 addendum: sequence-cluster bootstrap CIs for ATE-vs-IABE and IABE-vs-bias agreement (252 splits enumerated)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F1_noise_ceiling"))
from common import *
units, bias, ate = load()
w = pd.read_csv(R + "euroc_full/results/metrics_wide.csv").merge(
    pd.read_csv(R + "euroc_full/results/runs_index.csv")[["run_id", "regime", "method", "seed"]], on="run_id")
w = w[w.method.isin(DL)]
iabe = {u: w[(w.regime == u[0]) & (w.eval_sequence == u[1])].pivot_table(index="seed", columns="method", values="iabe_rot")[DL].values for u in units}
G = groups(units); seqs = sorted({u[1] for u in units})
sel_i = precompute_selection(units, iabe); sel_b = precompute_selection(units, bias)
rows = []
for selname, sel, meas in (("IABE", sel_i, {"raw ATE": ate["raw"], "oracle ATE": ate["oracle"]}),
                           ("bias RMSE", sel_b, {"IABE": iabe})):
    for k, src in meas.items():
        A, N = agree_counts(src, units, sel); a, n = A.sum(0), N.sum(0)
        for g, gi in G.items():
            gi = np.array(gi); rng = np.random.default_rng(2)
            su = {s: [i for i in gi if units[i][1] == s] for s in seqs}; ss = [s for s in seqs if su[s]]
            bs = []
            for _ in range(4000):
                ix = np.concatenate([su[ss[j]] for j in rng.integers(0, len(ss), len(ss))]).astype(int)
                bs.append(a[ix].sum() / n[ix].sum())
            rows.append(dict(selector=selname, measure=k, group=g, point=a[gi].sum() / n[gi].sum(),
                             seq_lo=np.percentile(bs, 2.5), seq_hi=np.percentile(bs, 97.5), n_clusters=len(ss),
                             per_unit=";".join(f"{units[i][1]}:{a[i]/n[i]:.2f}" if n[i] else f"{units[i][1]}:na" for i in gi)))
P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, "f4_ci.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); print(P.round(3).to_string())

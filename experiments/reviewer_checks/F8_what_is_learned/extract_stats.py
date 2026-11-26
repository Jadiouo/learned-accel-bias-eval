"""Read series.npz of every run and store per-run statistics (not the raw series). Output: stats_per_run.csv."""
import os, glob, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "..", "..", "euroc_full", "results")) + "/"   # <repo>/euroc_full/results/
idx = pd.read_csv(R + "runs_index.csv")
rows = []
for r in idx.itertuples():
    p = f"{R}runs/{r.run_id}/series.npz"
    if not os.path.exists(p): continue
    z = np.load(p)
    for k in z.files:
        if not k.startswith("b_hat_"): continue
        seq = k[6:]
        bh = np.asarray(z[k], np.float64); br = np.asarray(z["b_ref_" + seq], np.float64)
        if np.isnan(bh).any() or np.isnan(br).any(): continue
        mh, mr = bh.mean(0), br.mean(0); dh, dr = bh - mh, br - mr
        d = bh - br
        row = dict(run_id=r.run_id, regime=r.regime, method=r.method, seed=r.seed, seq=seq, n=len(bh),
                   rmse_avg=np.sqrt((d**2).mean()), off2=((mh-mr)**2).sum()/3, fl2=((dh-dr)**2).mean(0).sum()/3,
                   var_ref=(dr**2).mean(0).sum()/3, var_hat=(dh**2).mean(0).sum()/3,
                   cov_all=(dh*dr).mean(0).sum()/3,
                   mean_ref_norm=np.linalg.norm(mr), mean_hat_norm=np.linalg.norm(mh))
        for i, a in enumerate("xyz"):
            row.update({f"mr_{a}": mr[i], f"mh_{a}": mh[i], f"vr_{a}": (dr[i]**2).mean(), f"vh_{a}": (dh[i]**2).mean(),
                        f"cov_{a}": (dh[i]*dr[i]).mean()})
        rows.append(row)
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "stats_per_run.csv"), index=False)
print(S.shape, S.groupby(["regime"]).size().to_dict())

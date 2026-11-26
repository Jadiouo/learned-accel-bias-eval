import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
units, bias, ate = load()
for k in ("raw","oracle"):
    print("==", k)
    for u in units:
        T = ate[k][u]; m = T.mean(0); sd = T.std(0, ddof=1); Rz = T-m
        cv = sd/m
        # same-seed residual correlation across methods (mean over pairs)
        C = np.corrcoef(Rz.T); off = C[np.triu_indices(11,1)].mean()
        rho = stats.spearmanr(m, sd).correlation
        # bias ranking vs ranking of the mean ATE per method
        rb = stats.spearmanr(m, bias[u].mean(0)).correlation
        print(u, "mean range %.3g-%.3g"%(m.min(), m.max()), "CV median %.2f (%.2f-%.2f)"%(np.median(cv),cv.min(),cv.max()),
              "spearman(mean,sd)=%.2f"%rho, "mean cross-method resid corr=%.2f"%off, "rank corr(ATEmean,biasmean)=%.2f"%rb)

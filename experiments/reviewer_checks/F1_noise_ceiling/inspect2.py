import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
units, bias, ate = load()
for k in ("raw","oracle"):
    print("==", k)
    for u in units:
        T = ate[k][u]; m = T.mean(0); sd = T.std(0, ddof=1)
        print(u, "sd(method means)=%.3g"%m.std(ddof=1), "median seed sd=%.3g"%np.median(sd), "sd ratio max/min=%.1f"%(sd.max()/sd.min()),
              "signal/noise(5-seed mean)=%.2f"%(m.std(ddof=1)/(np.median(sd)/np.sqrt(5))))

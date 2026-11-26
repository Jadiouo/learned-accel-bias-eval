import os, sys; sys.dont_write_bytecode=True
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); M=pd.read_csv(os.path.join(HERE,"stats_enriched.csv"))
DL=['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
d=M[M.method.isin(DL)].copy(); d["fl_over_sig"]=d.fl/d.sig_ref; d["sdhat"]=np.sqrt(d.var_hat/1)
c=M[M.method.str.startswith("constant")].set_index(["regime","seq"]).rmse_avg
d["ratio_const"]=d.rmse_avg/[c[(r,s)] for r,s in zip(d.regime,d.seq)]
d["off_over_const"]=d.off/[c[(r,s)] for r,s in zip(d.regime,d.seq)]
out=[]
for g,regs in {"A":["A"],"B":["B"],"ALL":["A","B"]}.items():
    x=d[d.regime.isin(regs)]
    print(g,"fl/sigma_ref median %.1f (min %.1f); sigma_hat (RMS) median %.5f vs sigma_ref median %.5f; rmse/constant baseline median %.2f (range %.2f-%.2f); share of |corr| > 0.3 (all axes) %.3f; share of runs with sigma_hat < sigma_ref %.3f"%(x.fl_over_sig.median(),x.fl_over_sig.min(),np.sqrt(x.var_hat/ 1).median(),x.sig_ref.median(),x.ratio_const.median(),x.ratio_const.min(),x.ratio_const.max(),(x.corr_all.abs()>.3).mean(),(x.sd_ratio<1).mean()))
print(d.groupby("method")[["share_fl","ratio_const"]].median().round(3).T.to_string())

import pandas as pd, numpy as np
d=pd.read_csv('ate_per_run.csv'); g=9.81
rs=d.groupby(['regime','seq']).eps_rms.median().reset_index()
s=pd.read_csv('table_sources_by_seq.csv'); e=s.groupby('seq').eps_rms_median.first()
for r,x in rs.groupby('regime'):
    th=np.degrees(x.eps_rms/g)
    print(r,list(x.seq),'eps',x.eps_rms.round(5).tolist(),'deg',th.round(4).tolist(),'median',round(th.median(),4),'range',round(th.min(),4),round(th.max(),4),'oracle0.125/med',round(0.125/th.median(),1))
print(e)

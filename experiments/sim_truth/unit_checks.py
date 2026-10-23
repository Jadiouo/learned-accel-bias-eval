import numpy as np, sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'euroc_full', 'src'))
from sim_truth import *
tr=load_traj("MH_01_easy"); print("N",tr["N"], "|f| mean",np.linalg.norm(tr["f"],axis=1).mean())
# 1 gyro round trip
Rh=tr["R"][0]
for k in range(tr["N"]-1): Rh=Rh@so3_exp(tr["w"][k]*DT)
print("gyro roundtrip err deg",np.rad2deg(np.arccos(np.clip((np.trace(tr["R"][-1].T@Rh)-1)/2,-1,1))))
# 2 closed form constant bias, body frame identity R
eps=np.tile([0.01,0,0],(1,tr["N"],1)); N=tr["N"]
I=np.tile(np.eye(3),(N,1,1)); 
v=iabe_rot(eps,I)[0]; k=np.arange(N); 
cf=np.sqrt(np.mean((DT**2*0.01*k*(k-1)/2)**2)) 
print("iabe const body",v,"closed",cf)
# 3 iabe_rot equals euroc_full metrics on synthetic inputs
try:
    from euroc_full.metrics import MetricInputs, iabe_rot as ir
    rng=np.random.default_rng(0); e=make_eps("lowfreq",0.02,tr,rng)
    ba=np.array([0.1,-0.05,0.02]); fm=tr["f"]+ba
    mi=MetricInputs(a_raw=fm,w_raw=tr["w"],b_hat=ba-e,b_ref=np.tile(ba,(N,1)),R_ref=tr["R"],v0=tr["v0"],p0=tr["p0"],dt=DT,g_vec=GVEC)
    print("metrics.iabe_rot",ir(mi)["value"],"mine",iabe_rot(e[None],tr["R"])[0])
except Exception as ex: print("import fail",ex)
# 4 Mode M sanity: attitude error small w/ eps=0, Truth_B ~ IABE for small eps
rng=np.random.default_rng(1)
eps=np.array([make_eps("const",0.01,tr,rng),make_eps("highfreq",0.01,tr,rng),make_eps("lowfreq",0.01,tr,rng)])
t0=time.time(); d2,ate,att,Ar=run_mode_M(tr,eps,0.01,rng); print("modeM time",time.time()-t0)
print("TruthB",np.sqrt(d2),"IABE",iabe_rot(eps,tr["R"]),"ATE",ate,"att deg",att,"A rms",Ar)
d2,ate,att,Ar=run_mode_I(tr,eps,1.0,rng); print("modeI TruthB",np.sqrt(d2),"ATE",ate,"att",att,"A",Ar)

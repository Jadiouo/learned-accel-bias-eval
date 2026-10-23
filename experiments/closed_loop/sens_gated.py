# POST-HOC sensitivity: looser gate, b_ref only, L=10. Writes table_gated_sens.csv
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, run_gated as g, run_closed_loop as rc, eskf
from eskf import FS
rows = []
for ta, tw, sa, ev in [(0.3,0.2,0.3,20),(1.0,0.5,0.3,20),(1.0,0.5,1.0,20),(2.0,1.0,0.3,20),(0.3,0.2,0.1,5)]:
    g.THR_A, g.THR_W, g.SIG_A, g.TILT_EVERY = ta, tw, sa, ev
    for seq in rc.SEQS:
        D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]; i1 = D["i1"]; ko = i1 - 10 * FS
        b = rc.sources(D)["CTRL:bref"][ko:i1]
        for rep in range(5):
            sig, attS, qS = rc.CFGS["base"]; nz, mn = rc.rep_noise(rep, len(f)//FS+2)
            S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
            Sn = eskf.run_aided(S, f, w, pg, 0, ko, mn, sig, eskf.make_Q(1.0), snaps=[ko])[ko]
            p, fr = g.outage_prop(Sn, f, w, ko, i1, b, True)
            rows.append(dict(thr_a=ta, thr_w=tw, sig_a=sa, every=ev, seq=seq, rep=rep, abs=np.linalg.norm(p-pg[i1]), gate=fr))
T = pd.DataFrame(rows).groupby(["thr_a","thr_w","sig_a","every"]).agg(abs_med=("abs","median"), abs_q25=("abs",lambda x:x.quantile(.25)), abs_q75=("abs",lambda x:x.quantile(.75)), gate=("gate","mean")).reset_index()
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_gated_sens.csv"), index=False); print(T.round(3))

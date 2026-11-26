# -*- coding: utf-8 -*-
"""Verifies the causal shift actually changes what is injected (print differences), boundaries, and effect on O_base end positions."""
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lib, numpy as np, pandas as pd
rc = lib.rc; HD = HERE
O = pd.read_csv(HD + "/out_O.csv")
print("seq  i0  i1  n_win  len | L | mean|b_caus-b_cur| over outage (est mean, m/s2) | mean|b_cur| | |dbref| | frac outage frames where hold-rule differs | max |caus[j]-cur[j-200]| | mean end-pos shift est (m) cur->caus | same bref")
for seq in rc.SEQS:
    D = rc.load_seq(seq); i0, i1 = D["i0"], D["i1"]
    sc, sk = lib.sources(D, "cur"), lib.sources(D, "caus_both")
    est = [n for n in sc if not n.startswith("CTRL:")]
    chk = max(np.abs(sk[n][i0 + 200:i1] - sc[n][i0:i1 - 200]).max() for n in est)
    for L in (5, 10, 20):
        ko = i1 - L * 200
        if ko < i0: continue
        dif = np.mean([np.linalg.norm(sk[n][ko:i1] - sc[n][ko:i1], axis=1).mean() for n in est])
        mag = np.mean([np.linalg.norm(sc[n][ko:i1], axis=1).mean() for n in est])
        dref = np.linalg.norm(sk["CTRL:bref"][ko:i1] - sc["CTRL:bref"][ko:i1], axis=1).mean()
        hold = lib.hold_first_second(sk["CTRL:bref"], ko); fh = np.mean(np.any(hold[ko:i1] != sk["CTRL:bref"][ko:i1], axis=1))
        g = O[(O.seq == seq) & (O.L == L)]
        def shift(sel):
            a = g[(g["align"] == "cur") & sel].set_index(["rep", "run_id"])[["ex", "ey", "ez"]]; b = g[(g["align"] == "caus") & sel].set_index(["rep", "run_id"])[["ex", "ey", "ez"]]
            return np.linalg.norm((a - b).to_numpy(), axis=1).mean()
        print(f"{seq} {i0} {i1} {D['n']} {len(D['f'])} | {L} | {dif:.4f} | {mag:.4f} | {dref:.2e} | {fh:.3f} | {chk:.2e} | {shift(g.run_id != 'CTRL:bref'):.4f} | {shift(g.run_id == 'CTRL:bref'):.4f}")
    print("   boundary: first outage frame (L=20 or 10) ko - (i0+200) =", min(i1 - L * 200 for L in (5, 10, 20) if i1 - L * 200 >= i0) - (i0 + 200), "(>0 means outage starts after first window completes)")

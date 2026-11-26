# -*- coding: utf-8 -*-
"""usage: run.py O|P <out.csv> [stride] [seqs comma]. Writes ONLY to the given out path (in A8 dir).
Rows: end position of the outage for each (seq,cfg,L,rep,run_id,align). align in cur|caus|caush.
O: Q default, no bias injection before outage (== variants_long 'base' -> 'O_base'); all 228 estimators.
P: P_off (Q_ba=0, P_ba=0, bias injected from frame 0 with extend_back), estimator subset every `stride`-th run + bref."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))  # this folder
sys.path.insert(0, HERE)
import lib
import numpy as np, pandas as pd
rc, eskf = lib.rc, lib.eskf
import run_pre_outage as rp
FS = eskf.FS
which, outp = sys.argv[1], sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
seqs = sys.argv[4].split(",") if len(sys.argv) > 4 else rc.SEQS
t0 = time.time(); rows = []

def flush():
    pd.DataFrame(rows).to_csv(outp, index=False)

def row(seq, cfg, L, rep, rid, align, pe, pgt):
    return dict(seq=seq, cfg=cfg, L=L, rep=rep, run_id=rid, align=align, ex=pe[0], ey=pe[1], ez=pe[2],
                abs=float(np.linalg.norm(pe - pgt)))

def aligned(src_cur, src_caus, ko, align):
    if align == "cur": return src_cur
    if align == "caus": return src_caus
    return {n: lib.hold_first_second(b, ko) for n, b in src_caus.items()}

for seq in seqs:
    D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]; i0, i1 = D["i0"], D["i1"]
    sig, attS, qS = rc.CFGS["base"]
    Ls = [L for L in rc.LS if i1 - L * FS >= i0]; kos = {L: i1 - L * FS for L in Ls}
    assert all(k >= i0 + lib.WIN + 1 for k in kos.values())      # boundary: outage start is after first window completes
    s_cur = lib.sources(D, "cur"); s_caus = lib.sources(D, "caus_both")   # estimators: caus; bref: caus (cur bref lives in s_cur)
    keep = [n for n in s_cur if not n.startswith("CTRL:") or n == "CTRL:bref"]
    est = [n for n in keep if n != "CTRL:bref"]
    if which == "P": est = est[::stride]
    names = est + ["CTRL:bref"]
    for rep in range(rc.K_O):
        nz, mn = rc.rep_noise(rep, len(f) // FS + 2)
        if which == "O":
            Q = rp.make_Q_ba(1.0)
            S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
            snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
            for L in Ls:
                ko = kos[L]; Sn = snaps[ko]; Rs = eskf.outage_rotations(Sn, w, ko, i1)
                for align in ("cur", "caus", "caush"):
                    src = aligned(s_cur, s_caus, ko, align)
                    pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, np.stack([src[n][ko:i1] for n in names]))
                    for j, n in enumerate(names): rows.append(row(seq, "O_base", L, rep, n, align, pe[j], pg[i1]))
        else:
            Q = rp.make_Q_ba(1.0, False)
            for align0 in ("cur", "caus"):          # aided phase alignment; caush shares the aided phase with caus
                src0 = s_cur if align0 == "cur" else s_caus
                for n in names:
                    bofs = rp.extend_back(src0[n], i0)
                    S = eskf.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
                    S["P"][12:15, 12:15] = 0.0
                    snaps = eskf.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, bofs=bofs, snaps=list(kos.values()))
                    for L in Ls:
                        ko = kos[L]; Sn = snaps[ko]; Rs = eskf.outage_rotations(Sn, w, ko, i1)
                        for align in ([align0] if align0 == "cur" else ["caus", "caush"]):
                            b = bofs if align != "caush" else lib.hold_first_second(bofs, ko)
                            pe = eskf.outage_end_pos(Sn, f, Rs, ko, i1, b[ko:i1] + Sn["ba"])[0]
                            rows.append(row(seq, "P_off", L, rep, n, align, pe, pg[i1]))
        flush(); print(which, seq, "rep", rep, "%.0fs" % (time.time() - t0), flush=True)
flush(); print("done %.0fs" % (time.time() - t0), flush=True)

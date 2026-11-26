# F3 step 1: reproduce consequences_long.csv (cfg=base, mode=O, L=5/10) for ALL sources, incl. CTRL:bref (offset 0).
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))

def task(args):
    seq, rep = args
    D, packs = snapshots(seq, rep); src = RC.sources(D); names = list(src); rows = []
    for L, pk in packs.items():
        ko, i1, Sn, Rs = pk
        Bm = np.stack([src[n][ko:i1] for n in names] + [np.tile(Sn["ba"], (i1 - ko, 1))])
        ab, pe = end_err(D, pk, Bm)
        ib = names.index("CTRL:bref")
        dl = np.linalg.norm(pe - pe[ib], axis=1)
        for j, n in enumerate(names + ["CTRL:eskf_self"]):
            rows.append(dict(seq=seq, L=L, rep=rep, run_id=n, abs=float(ab[j]), delta=float(dl[j])))
    return rows

if __name__ == "__main__":
    t = time.time()
    with Pool(4) as p: out = p.map(task, [(s, r) for s in SEQS for r in range(K)], chunksize=1)
    mine = pd.DataFrame([r for x in out for r in x])
    ref = pd.read_csv(CL + "/consequences_long.csv")
    ref = ref[(ref.cfg == "base") & (ref["mode"] == "O") & ref.L.isin([5, 10])][["seq", "L", "rep", "run_id", "abs", "delta"]]
    m = mine.merge(ref, on=["seq", "L", "rep", "run_id"], suffixes=("", "_ref"), how="outer", indicator=True)
    print("merge:", m._merge.value_counts().to_dict())
    m = m[m._merge == "both"]
    m["d_abs"] = (m["abs"] - m["abs_ref"]).abs(); m["d_delta"] = (m["delta"] - m["delta_ref"]).abs()
    out_rows = []
    for key, g in [("ALL sources", m), ("CTRL:bref (offset 0)", m[m.run_id == "CTRL:bref"])]:
        for L in (5, 10):
            gg = g[g.L == L]
            out_rows.append(dict(set=key, L=L, n=len(gg), max_abs_diff_Cabs=gg.d_abs.max(), max_abs_diff_Cdelta=gg.d_delta.max(),
                                 max_rel_diff_Cabs=float((gg.d_abs / gg.abs_ref).max())))
    R = pd.DataFrame(out_rows); print(R.to_string()); R.to_csv(HERE + "/repro_check.csv", index=False)
    print("time %.0fs" % (time.time() - t))

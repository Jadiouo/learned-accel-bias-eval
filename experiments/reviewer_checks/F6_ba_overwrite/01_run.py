# F6 step 1: compute all conditions. Writes only into F6_ba_overwrite/.
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f6core import *
import pandas as pd
from multiprocessing import Pool

def task(args):
    seq, rep, qn = args
    D, packs = snapshots(seq, rep, QS[qn]); src = RC.sources(D); names = list(src)
    rc_rows, src_rows, sh_rows, on_rows = [], [], [], []
    for L, (ko, i1, Sn) in packs.items():
        OG = Outage(D, ko, i1); pg = D["p"][i1]; Rg = D["R"][ko]
        ew, R_notilt, R_noyaw = tilt_split(Sn, Rg)
        bref = src["CTRL:bref"][ko:i1]; ba = np.tile(Sn["ba"], (i1 - ko, 1))
        # states
        S_d = state_with(Sn, R=R_notilt)
        S_c = state_with(Sn, R=consistent_tilt_R(Sn, bref[0]))
        fm = D["f"][max(ko - FS, 0):ko].mean(0)          # extra (c2): actual mean specific force of the last 1 s instead of static gravity
        S_c2 = state_with(Sn, R=consistent_tilt_R(Sn, bref[0], fbar=fm))
        S_c1 = state_with(Sn, R=consistent_tilt_R(Sn, bref[0], first_order=True))
        S_f = state_with(Sn, p=D["p"][ko], v=D["v"][ko], R=Rg, bg=D["bw"][ko])
        conds = {"a": (Sn, bref), "b": (Sn, ba), "c": (S_c, bref), "c1": (S_c1, bref), "c2": (S_c2, bref), "d": (S_d, bref),
                 "e": (S_d, ba), "f": (S_f, bref)}
        pend = {}
        for cn, (S, b) in conds.items():
            pe = OG.endpos(S, b[None])[0]; pend[cn] = pe; e = pe - pg
            rc_rows.append(dict(seq=seq, rep=rep, q=qn, L=L, cond=cn, abs=float(np.linalg.norm(e)), ex=e[0], ey=e[1], ez=e[2]))
        # all sources under (a),(d),(b: eskf_self as extra source),(e-like: all sources with GT tilt = d)
        B = np.stack([src[n][ko:i1] for n in names] + [ba])
        for cn, S in (("a", Sn), ("d", S_d)):
            pe = OG.endpos(S, B); ib = names.index("CTRL:bref")
            for j, n in enumerate(names + ["CTRL:eskf_self"]):
                src_rows.append(dict(seq=seq, rep=rep, q=qn, L=L, cond=cn, run_id=n,
                                     abs=float(np.linalg.norm(pe[j] - pg)), delta=float(np.linalg.norm(pe[j] - pe[ib]))))
        # Shapley: b_ref source (a) and filter-b_a source (b)
        for sname, b in (("bref", bref), ("ba", ba)):
            r = shapley(OG, Sn, D, ko, i1, b, ew); r.update(seq=seq, rep=rep, q=qn, L=L, source=sname); sh_rows.append(r)
        # onset diagnostics
        Rf = Sn["R"]; bref0 = bref[0]
        Rc = S_c["R"]; ewc = Rot.from_matrix(Rc @ Rg.T).as_rotvec(); dyaw = Rot.from_matrix(Rc @ Sn["R"].T).as_rotvec()
        # body-frame rotation of the (c) correction
        on_rows.append(dict(seq=seq, rep=rep, q=qn, L=L, ewx=ew[0], ewy=ew[1], ewz=ew[2],
                            tilt_deg=float(np.degrees(np.hypot(ew[0], ew[1]))), yaw_deg=float(np.degrees(abs(ew[2]))),
                            tilt_c_deg=float(np.degrees(np.hypot(ewc[0], ewc[1]))), yaw_c_deg=float(np.degrees(abs(ewc[2]))),
                            corr_c_deg=float(np.degrees(np.linalg.norm(dyaw))), corr_c_z_deg=float(np.degrees(abs(dyaw[2]))),
                            **{f"ba{i}": float(Sn["ba"][i]) for i in range(3)}, **{f"bref{i}": float(bref0[i]) for i in range(3)},
                            **{f"Rba{i}": float((Rf @ Sn["ba"])[i]) for i in range(3)},
                            **{f"Rbd{i}": float((Rf @ (Sn["ba"] - bref0))[i]) for i in range(3)},
                            vel_err=float(np.linalg.norm(Sn["v"] - D["v"][ko])), pos_err=float(np.linalg.norm(Sn["p"] - D["p"][ko])),
                            fz=float(np.linalg.norm(D["f"][ko]))))
    return rc_rows, src_rows, sh_rows, on_rows

if __name__ == "__main__":
    t = time.time(); tasks = [(s, r, q) for q in QS for s in SEQS for r in range(K)]
    with Pool(2) as p: out = p.map(task, tasks, chunksize=1)
    for i, nm in enumerate(("cond_long", "src_long", "shapley_units", "onset_units")):
        pd.DataFrame([r for o in out for r in o[i]]).to_csv(f"{HERE}/{nm}.csv", index=False)
    print("done %.0fs" % (time.time() - t))

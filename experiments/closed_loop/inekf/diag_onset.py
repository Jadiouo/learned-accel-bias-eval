# -*- coding: utf-8 -*-
"""Diagnostic (not part of the pre-registered decision): how different are the ESKF and InEKF O_base onset states / covariances?
For each (seq, rep) at L=5,10: world roll/pitch error, velocity error, position error vs GT for each filter; angle between the two
filters' attitude estimates; and the tilt-velocity cross-correlation of the covariance (InEKF P mapped back to ESKF coordinates). -> diag_onset.csv"""
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import eskf, inekf, run_closed_loop as rc
from eskf import FS
rows = []
for seq in rc.SEQS:
    D = rc.load_seq(seq); f, w, pg = D["f"], D["w"], D["p"]; i1 = D["i1"]; sig, attS, _ = rc.CFGS["base"]; Q = eskf.make_Q(1.0)
    kos = {L: i1 - L * FS for L in (5, 10)}
    for rep in range(rc.K_O):
        nz, mn = rc.rep_noise(rep, len(f) // FS + 2); sn = {}
        for nm, m in (("eskf", eskf), ("inekf", inekf)):
            S = m.init_state(D["R"][0], D["p"][0], D["v"][0], D["bw"][0], nz, attS)
            sn[nm] = m.run_aided(S, f, w, pg, 0, max(kos.values()), mn, sig, Q, snaps=list(kos.values()))
        for L, ko in kos.items():
            r = dict(seq=seq, rep=rep, L=L); Rg = D["R"][ko]
            for nm in sn:
                s = sn[nm][ko]; ew = Rot.from_matrix(s["R"] @ Rg.T).as_rotvec()
                r[nm + "_tilt_deg"] = np.degrees(np.hypot(ew[0], ew[1])); r[nm + "_vel"] = np.linalg.norm(s["v"] - D["v"][ko])
                r[nm + "_pos"] = np.linalg.norm(s["p"] - D["p"][ko])
                Pe = s["P"] if nm == "eskf" else inekf.cov_i2e(s["P"], s["R"], s["v"], s["p"])
                sd = np.sqrt(np.diag(Pe)); C = Pe / np.outer(sd, sd)
                r[nm + "_xcorr_att_vel"] = np.linalg.norm(C[6:9, 3:6]); r[nm + "_sig_tilt_deg"] = np.degrees(np.sqrt(0.5 * (Pe[6, 6] + Pe[7, 7])))
            a, b = sn["eskf"][ko], sn["inekf"][ko]
            r["d_att_deg"] = np.degrees(np.linalg.norm(Rot.from_matrix(a["R"] @ b["R"].T).as_rotvec()))
            r["d_vel"] = np.linalg.norm(a["v"] - b["v"]); r["d_pos"] = np.linalg.norm(a["p"] - b["p"])
            rows.append(r)
d = pd.DataFrame(rows); d.to_csv(HERE + "/diag_onset.csv", index=False)
pd.set_option("display.width", 250)
print(d.groupby("L").median(numeric_only=True).drop(columns="rep").T.round(4).to_string())

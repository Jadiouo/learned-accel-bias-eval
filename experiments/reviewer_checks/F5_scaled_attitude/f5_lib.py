# -*- coding: utf-8 -*-
"""F5 helpers: scale the oracle attitude's error rotation by s and rebuild open-loop ATE.
Reuses ../F2_aligned_ate/f2_lib.py (read-only import). tilt_yaw_deg COPIED from
experiments/attitude_controls/run_controls_gated.py (not executed)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "F2_aligned_ate"))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import f2_lib as L

G = L.G
ORACLE = "b_gyro-GT-bias(oracle)"

def tilt_yaw_deg(Cg, C):   # copied
    q = Rot.from_matrix(np.einsum("nij,nkj->nik", Cg, C)).as_quat()
    tilt = 2 * np.arctan2(np.hypot(q[:, 0], q[:, 1]), np.hypot(q[:, 3], q[:, 2]))
    yaw = 2 * np.arctan2(q[:, 2], q[:, 3]); yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(tilt), np.degrees(yaw)

def scaled_att(Cg, Rref, s, variant="W"):
    """Cg: oracle attitude, Rref: reference attitude, both (N,3,3). Returns attitude with error rotation scaled by s.
    W: world-frame error E=Cg Rref^T, rotvec scaled by s.  B: body-frame error Rref^T Cg.  T: world frame, only the
    horizontal (x,y) rotvec components scaled, z (yaw) kept."""
    if variant == "B":
        rv = Rot.from_matrix(np.einsum("nji,njk->nik", Rref, Cg)).as_rotvec()
        return np.einsum("nij,njk->nik", Rref, Rot.from_rotvec(s * rv).as_matrix())
    rv = Rot.from_matrix(np.einsum("nij,nkj->nik", Cg, Rref)).as_rotvec()
    if variant == "T":
        rs = rv * np.array([s, s, 1.0])
    else:
        rs = s * rv
    return np.einsum("nij,njk->nik", Rot.from_rotvec(rs).as_matrix(), Rref)

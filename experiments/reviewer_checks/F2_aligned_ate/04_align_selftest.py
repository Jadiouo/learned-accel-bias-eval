"""Self-test of alignment code: synthetic rigid transforms are removed exactly; aligned<=unaligned and se3<=4dof on all real runs."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import f2_lib as L
rng = np.random.default_rng(1); Q = np.cumsum(rng.normal(size=(500, 3)), 0)
out = {}
Rz = Rot.from_euler("z", 0.7).as_matrix(); P = Q @ Rz.T + [3, -2, 1]
out["4dof_err_on_yaw+t_transform"] = float(np.abs(L.align_4dof(P, Q) - Q).max())
Rg = Rot.random(random_state=3).as_matrix(); P = Q @ Rg.T + [5, 1, -4]
out["se3_err_on_general_rigid"] = float(np.abs(L.align_se3(P, Q) - Q).max())
out["4dof_err_on_general_rigid(should be >0)"] = float(np.abs(L.align_4dof(P, Q) - Q).max())
d = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/ate_aligned_per_run.csv")
out["frac_4dof_le_none"] = float((d.ate_4dof <= d.ate_none * (1 + 1e-9)).mean())
out["frac_se3_le_4dof"] = float((d.ate_se3 <= d.ate_4dof * (1 + 1e-9)).mean())
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/align_selftest.json", "w"), indent=1); print(json.dumps(out, indent=1))

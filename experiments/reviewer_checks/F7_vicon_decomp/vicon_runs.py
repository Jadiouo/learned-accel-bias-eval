"""F7 step 1: per-run decomposition terms for the Vicon rooms. Reconstruction logic copied from
euroc_full/analysis_vicon/t1_leakage.py and F2_aligned_ate/f2_lib.py and decomposition/f5_decomp.py (only raw + oracle sources; Mahony skipped).
Reads results_vicon (read-only). Writes vicon_decomp_runs.csv here. Usage: python vicon_runs.py"""
import os, sys, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "euroc_full"))
sys.path.insert(0, ROOT + "/src")
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from euroc_full import data_euroc, metrics, common
cfg = common.load_config(ROOT + "/configs/vicon.yaml", data_dir_override=ROOT + "/kaggle_vicon/data_vicon")
SEQS = list(cfg["sequences"]); STRIDE, FS, G = cfg["stride"], cfg["imu_freq"], 9.81; DT = STRIDE / FS
DL = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
RES = ROOT + "/results_vicon"
def integ_full(w, R0, dt):
    inc = Rot.from_rotvec(w[:-1] * dt).as_matrix(); R = np.empty((len(w), 3, 3)); R[0] = R0
    for k in range(len(w) - 1): R[k + 1] = R[k] @ inc[k]
    return R
def rms3(p): return float(np.sqrt(np.mean(np.sum(p ** 2, 1))))
def dint(a): return np.cumsum(np.cumsum(a * DT, 0) * DT, 0)
def world(R, x): return np.einsum("nij,nj->ni", R, x)
SRCN = {"raw": "a_raw_gyro(paper)", "oracle": "b_gyro-GT-bias(oracle)"}
idx = pd.read_csv(RES + "/runs_index.csv"); idx = idx[(idx.status == "ok") & idx.method.isin(DL)]
NROT = 30; ROTS = Rot.random(NROT, random_state=11).as_matrix()   # same rotations as decomposition/f5_decomp.py
recs = []
for seq in SEQS:
    d = data_euroc.prepare_regime_a(seq, cfg); m = d.merged
    tws = np.asarray(d.test_win_starts, np.int64); i0, i1 = tws[0], tws[-1]
    mi = metrics.build_metric_inputs(m, tws, np.zeros((len(tws), 3)), d.y_test, STRIDE, FS, G)
    seg = m.iloc[i0:i1 + 1]; wf = seg[["wx","wy","wz"]].to_numpy(); bw = seg[["bwx","bwy","bwz"]].to_numpy()
    R0 = Rot.from_quat(seg[["qx","qy","qz","qw"]].to_numpy()).as_matrix()[0]; sel = tws - i0
    Cs = {"raw": integ_full(mi.w_raw, mi.R_ref[0], DT), "oracle": integ_full(wf - bw, R0, 1.0 / FS)[sel]}
    Ls = {k: dint(world(C - mi.R_ref, mi.a_raw - mi.b_ref)) for k, C in Cs.items()}
    mets = {}
    for _, r in idx.iterrows():
        mp = f"{RES}/runs/{r.run_id}/metrics.json"
        mt = json.load(open(mp)).get(seq) if os.path.exists(mp) else None
        if mt is None or r.eval_sequences != seq: continue
        bh = np.load(f"{RES}/runs/{r.run_id}/series.npz")[f"b_hat_{seq}"]; assert len(bh) == len(mi.b_ref)
        eps = mi.b_ref - bh; I = dint(world(mi.R_ref, eps))
        for k, C in Cs.items():
            Lk = Ls[k]; Q = dint(world(C - mi.R_ref, eps))
            row = dict(run_id=r.run_id, method=r.method, seed=int(r.seed), seq=seq, src=k,
                       full=rms3(Lk + Q + I), noL=rms3(Q + I), noQ=rms3(Lk + I), I=rms3(I), negL=rms3(-Lk + Q + I),
                       normL=rms3(Lk), normQ=rms3(Q), iabe_rot=float(mt["iabe_rot"]), bias_rmse=float(mt["bias_rmse_avg"]), gyro_ate=float(mt["gyro_ate"]))
            for j in range(NROT): row[f"rot{j}"] = rms3(Lk @ ROTS[j].T + Q + I)
            recs.append(row)
    print(seq, len(recs), flush=True)
D = pd.DataFrame(recs); D.to_csv(HERE + "/vicon_decomp_runs.csv", index=False)
ref = pd.read_csv(ROOT + "/analysis_vicon/full_2026-10-23/t1_ate_per_run_gated.csv", comment="#")
for k, s in SRCN.items():
    x = D[D.src == k].merge(ref[ref.source == s][["run_id", "seq", "ate", "iabe_rot"]], on=["run_id", "seq"], suffixes=("", "_ref"))
    print(k, "n", len(x), "max rel err full vs t1 csv ate: %.2e" % (abs(x.full - x.ate) / x.ate).max(),
          "| I vs iabe_rot(csv): %.2e" % (abs(x.I - x.iabe_rot_ref) / x.iabe_rot_ref).max(),
          "| median |L|/|I| %.3g  |Q|/|I| %.3g" % ((x.normL / x.I).median(), (x.normQ / x.I).median()))

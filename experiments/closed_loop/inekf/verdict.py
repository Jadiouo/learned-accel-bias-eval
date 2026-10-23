# -*- coding: utf-8 -*-
"""Apply the PREREG.md section-3 decision rules to out_inekf/ (and list the ESKF reference). Writes verdict.md. No tuning."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PAR = os.path.dirname(HERE)
R_i = pd.read_csv(HERE + "/out_inekf/table3_ratio.csv"); R_e = pd.read_csv(PAR + "/table3_ratio.csv")
S_i = pd.read_csv(HERE + "/out_inekf/table3_shapley.csv"); S_e = pd.read_csv(PAR + "/table3_shapley.csv")
T_i = pd.read_csv(HERE + "/out_inekf/table3_state.csv"); T_e = pd.read_csv(PAR + "/table3_state.csv")
out = []
def haA(r, lo_thr, hi_thr):
    p, lo = r.ratio_med, r.ratio_med_lo
    return "仍成立" if (p >= hi_thr and lo >= lo_thr) else ("不成立" if p < 2 else "削弱")
out.append("## H-A 主張 2（比值 = 共同誤差 ÷ 方法差中位）\n")
out.append("| 配置 | L | 濾波器 | 比值 [95% CI] | IQR 比值 [CI] | 全距比值 [CI] | 共同誤差 m | 方法差中位 m | 判定（≥4 且下界≥2.5） | 舊門檻（≥5 且下界≥3） |\n|---|---|---|---|---|---|---|---|---|---|")
cells = []
for cfg in ("O_base", "P_off"):
    for L in (5, 10):
        for nm, T in (("ESKF", R_e), ("InEKF", R_i)):
            r = T[(T.cfg == cfg) & (T.L == L)].iloc[0]; v = haA(r, 2.5, 4); vo = haA(r, 3, 5)
            if nm == "InEKF": cells.append((cfg, L, v, vo, r))
            out.append(f"| {cfg} | {L} | {nm} | {r.ratio_med:.1f} [{r.ratio_med_lo:.1f}, {r.ratio_med_hi:.1f}] | {r.ratio_iqr:.1f} [{r.ratio_iqr_lo:.1f}, {r.ratio_iqr_hi:.1f}] | {r.ratio_rng:.1f} [{r.ratio_rng_lo:.1f}, {r.ratio_rng_hi:.1f}] | {r.common_m:.2f} | {r.median_delta_m:.3f} | {v if nm=='InEKF' else haA(r,2.5,4)} | {vo if nm=='InEKF' else haA(r,3,5)} |")
rank = {"仍成立": 0, "削弱": 1, "不成立": 2}
worst = max((c[2] for c in cells), key=lambda x: rank[x]); worst_old = max((c[3] for c in cells), key=lambda x: rank[x])
head = [c for c in cells if c[0] == "O_base" and c[1] == 10][0]
out.append(f"\n**H-A 整體（四 cell 最差者）：{worst}**；頭條 cell（O_base, L=10）：{head[2]}；舊門檻整體：{worst_old}。\n")
C = ["pos", "vel", "tilt", "yaw", "bg"]
def top(row):
    sh = sorted(((row[c], c) for c in C), reverse=True); return sh
out.append("## H-B Shapley（b_ref 臂，份額 = ΣΦ ÷ Σv(∅)）\n")
out.append("| 配置 | L | 濾波器 | tilt [CI] | vel [CI] | pos | yaw | bg | residual | 前兩大 | tilt+vel | 第2、3名 CI 重疊 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
flags = {}
def shap_rows(Ti, Te, cfgname, Lset):
    for L in Lset:
        for nm, T in (("ESKF", Te), ("InEKF", Ti)):
            sub = T[T.L == L]
            if len(sub) == 0: continue
            r = sub.iloc[0]; sh = top(r); t2 = {sh[0][1], sh[1][1]}
            ov = "是" if r[sh[2][1] + "_hi"] >= r[sh[1][1] + "_lo"] else "否"
            if nm == "InEKF": flags[(cfgname, L)] = (t2 == {"tilt", "vel"}, r.tilt + r.vel)
            out.append(f"| {cfgname} | {L} | {nm} | {r.tilt:.3f} [{r.tilt_lo:.3f}, {r.tilt_hi:.3f}] | {r.vel:.3f} [{r.vel_lo:.3f}, {r.vel_hi:.3f}] | {r.pos:.3f} | {r.yaw:.3f} | {r.bg:.3f} | {r.floor:.3f} | {sh[0][1]}, {sh[1][1]} | {r.tilt + r.vel:.3f} | {ov} |")
Sb_i = S_i[S_i.cfg == "base"]; Sb_e = S_e[S_e.cfg == "base"]
shap_rows(Sb_i, Sb_e, "O_base", (5, 10))
n_ok = sum(flags[("O_base", L)][0] for L in (5, 10))
hb = "仍是 tilt 與速度" if n_ok == 2 else ("部分" if n_ok == 1 else "不是")
out.append(f"\n**H-B（O_base）：{hb}**（L=5：{flags[('O_base',5)][0]}，L=10：{flags[('O_base',10)][0]}）\n")
pe = HERE + "/out_inekf/table3_shapley_poff.csv"; pr = HERE + "/verify_eskf_ref/out/table3_shapley_poff.csv"
if os.path.exists(pe) and os.path.exists(pr):
    out.append("P_off（探索性，不進主判定）：\n")
    out.append("| 配置 | L | 濾波器 | tilt [CI] | vel [CI] | pos | yaw | bg | residual | 前兩大 | tilt+vel | 第2、3名 CI 重疊 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    shap_rows(pd.read_csv(pe), pd.read_csv(pr), "P_off", (5, 10))
out.append("\n## H-C 起點 tilt／速度（P_off；r = 單位內方法間差中位 ÷ b_ref 臂誤差，單位中位）\n")
out.append("| L | 濾波器 | r_tilt [CI] | b_ref 臂 tilt (°) | 方法間 tilt 差 (°) | r_vel [CI] | 判定（L=10 主） |\n|---|---|---|---|---|---|---|")
hc = None
for L in (5, 10):
    for nm, T in (("ESKF", T_e), ("InEKF", T_i)):
        r = T[(T.cfg == "P_off") & (T.L == L)].iloc[0]
        v = "仍近似與方法無關" if r.r_tilt < 0.25 else ("明顯隨方法變" if r.r_tilt >= 0.5 else "部分")
        if nm == "InEKF" and L == 10: hc = v
        out.append(f"| {L} | {nm} | {r.r_tilt:.3f} [{r.r_tilt_lo:.3f}, {r.r_tilt_hi:.3f}] | {r.tilt_ref_deg:.3f} | {r.dtilt_deg:.3f} | {r.r_vel:.3f} [{r.r_vel_lo:.3f}, {r.r_vel_hi:.3f}] | {v if (nm=='InEKF' and L==10) else ''} |")
out.append(f"\n**H-C（InEKF，P_off，L=10）：{hc}**\n")
open(HERE + "/verdict.md", "w").write("\n".join(out)); print("\n".join(out))

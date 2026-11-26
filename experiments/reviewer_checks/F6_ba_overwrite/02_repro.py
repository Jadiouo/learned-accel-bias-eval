# F6 step 2: reproduction checks against existing csv. Writes repro_check.txt
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "closed_loop"))
import numpy as np, pandas as pd
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
S = pd.read_csv(HERE + "/src_long.csv"); S = S[(S.q == "base") & (S.cond == "a")]
ref = pd.read_csv(CL + "/consequences_long.csv"); ref = ref[(ref.cfg == "base") & (ref["mode"] == "O") & ref.L.isin([5, 10])]
m = S.merge(ref[["seq", "L", "rep", "run_id", "abs", "delta"]], on=["seq", "L", "rep", "run_id"], suffixes=("", "_ref"), how="outer", indicator=True)
P("1. source-level merge vs consequences_long.csv (base, O):", m._merge.value_counts().to_dict())
m = m[m._merge == "both"]
P("   max |abs - abs_ref| = %.2e, max |delta - delta_ref| = %.2e (n=%d)" % ((m["abs"] - m.abs_ref).abs().max(), (m.delta - m.delta_ref).abs().max(), len(m)))
# Table IV style numbers (paper definition: per (run,seq) RMS over 5 reps, then median over runs/seqs)
def table4(df, L):
    d = df[df.L == L]; a = d.groupby(["run_id", "seq"]).agg(abs=("abs", rms), delta=("delta", rms)).reset_index()
    lm = a[~a.run_id.str.startswith("CTRL:")]
    return dict(bref_abs=a[a.run_id == "CTRL:bref"]["abs"].median(), est_abs=lm["abs"].median(), est_delta=lm.delta.median(),
                zero_delta=a[a.run_id == "CTRL:zero"].delta.median(), self_abs=a[a.run_id == "CTRL:eskf_self"]["abs"].median(),
                self_delta=a[a.run_id == "CTRL:eskf_self"].delta.median())
ref2 = pd.read_csv(CL + "/consequences_long.csv"); ref2 = ref2[(ref2.cfg == "base") & (ref2["mode"] == "O")]
S2 = pd.read_csv(HERE + "/src_long.csv"); S2 = S2[(S2.q == "base") & (S2.cond == "a")]
for L in (5, 10):
    a = table4(S2, L); b = table4(ref2, L)
    P(f"2. Table IV L={L} mine : " + ", ".join(f"{k}={v:.4f}" for k, v in a.items()))
    P(f"   Table IV L={L} csv  : " + ", ".join(f"{k}={v:.4f}" for k, v in b.items()))
# eskf_self (row 10b) is the same as condition (b) of the main output
C = pd.read_csv(HERE + "/cond_long.csv"); Cb = C[(C.q == "base")]
for L in (5, 10):
    x = Cb[(Cb.L == L) & (Cb.cond == "a")]; y = Cb[(Cb.L == L) & (Cb.cond == "b")]
    P(f"   unit-level median C_abs L={L}: (a) {x['abs'].median():.4f}  (b) {y['abs'].median():.4f}; "
      f"5-seq RMS median (a) {x.groupby('seq')['abs'].apply(rms).median():.4f} (b) {y.groupby('seq')['abs'].apply(rms).median():.4f}")
# Shapley vs decomp_units (base, bref, nogate)
U = pd.read_csv(CL + "/decomp_units.csv"); U = U[U.variant == "nogate"]
H = pd.read_csv(HERE + "/shapley_units.csv"); Hb = H[(H.q == "base") & (H.source == "bref")]
mm = Hb.merge(U, on=["seq", "rep", "L"], suffixes=("", "_ref"))
P("3. Shapley (base, b_ref) vs decomp_units.csv: n=%d" % len(mm))
for c in ["total", "floor"] + ["shap_" + x for x in ("pos", "vel", "tilt", "yaw", "bg")] + ["oat_tilt"]:
    P("   max |%s - ref| = %.2e" % (c, (mm[c] - mm[c + "_ref"]).abs().max()))
def share(d, c): return d["shap_" + c].sum() / d["total"].sum()
for L in (5, 10):
    d = Hb[Hb.L == L]; P(f"   tilt Shapley share, L={L}: mine {share(d,'tilt'):.4f}; summary csv:", end=" ") if False else None
    ds = pd.read_csv(CL + "/decomp_summary.csv"); r = ds[(ds.variant == "nogate") & (ds.L == L) & (ds.part == "tilt")].share.iloc[0]
    P(f"   tilt Shapley share, L={L}: mine {share(d,'tilt'):.4f}; decomp_summary.csv {r:.4f}; vel {share(d,'vel'):.4f}; floor {d.floor.sum()/d.total.sum():.4f}")
# Q variants
V = pd.read_csv(CL + "/variants_long.csv"); VD = pd.read_csv(CL + "/variants_decomp_units.csv")
for q in ("q01", "q10"):
    s = pd.read_csv(HERE + "/src_long.csv"); s = s[(s.q == q) & (s.cond == "a")]
    v = V[V.cfg == q]; mq = s.merge(v[["seq", "L", "rep", "run_id", "abs", "delta"]], on=["seq", "L", "rep", "run_id"], suffixes=("", "_ref"))
    P(f"4. {q}: source-level n={len(mq)}  max|abs-ref|={(mq['abs']-mq.abs_ref).abs().max():.2e} max|delta-ref|={(mq.delta-mq.delta_ref).abs().max():.2e}")
    hq = H[(H.q == q) & (H.source == "bref")].merge(VD[VD.cfg == q], on=["seq", "rep", "L"], suffixes=("", "_ref"))
    P(f"   Shapley n={len(hq)} max|shap_tilt-ref|={(hq.shap_tilt-hq.shap_tilt_ref).abs().max():.2e}  max|total-ref|={(hq.total-hq.total_ref).abs().max():.2e}")
open(HERE + "/repro_check.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))

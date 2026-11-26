"""H0 step 6: Fig. 2 rows (nine cases pooled, raw attitude) with and without Mamba, from canonical_agreement.csv."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
P = pd.read_csv(HERE+"/canonical_agreement.csv", keep_default_na=False)
for c in ("point", "lo", "hi"): P[c] = pd.to_numeric(P[c])
rows = [("bias RMSE retest", "bias RMSE own retest", "not applicable"), ("IABE", "IABE", "reference attitude"), ("ATE, leakage removed", "ATE without shared leakage term", "raw"),
        ("ATE, raw attitude", "ATE", "raw"), ("leakage, random dir. (band = range of 30)", "ATE with leakage in random directions", "raw"),
        ("ATE, SE(3)-aligned", "ATE aligned SE(3)", "raw"), ("ATE noise ceiling (band = range of V1-V3 medians)", "ATE noise ceiling band (min-max of V1-V3 medians)", "raw")]
out = []
for lab, m, a in rows:
    r = {"fig_row": lab}
    for mam in ("included", "excluded"):
        x = P[(P.dataset == "Machine Hall") & (P.selector == "bias RMSE") & (P.measure == m) & (P.attitude_source == a) & (P.group == "nine cases pooled") & (P.mamba == mam)].iloc[0]
        r[f"{mam}_point"] = x.point; r[f"{mam}_lo"] = x.lo; r[f"{mam}_hi"] = x.hi; r[f"{mam}_interval_type"] = x.interval_type
    out.append(r)
O = pd.DataFrame(out); O.to_csv(HERE+"/fig2_rows.csv", index=False)
pd.set_option("display.width", 250); print(O.drop(columns=[c for c in O.columns if "interval_type" in c]).round(3).to_string())

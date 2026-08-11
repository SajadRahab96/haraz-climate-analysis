#!/usr/bin/env python3
"""
34_weighting_sensitivity.py
===========================
Sensitivity of the projected signal to the ensemble WEIGHTING SCHEME.

Three schemes are compared on identical downscaled input:
  1. EM-BMA        - Expectation-Maximization posterior weights (this study)
  2. Equal         - w_k = 1/K (no performance weighting at all)
  3. Skill-score   - composite skill scores normalised to sum to one
                     (the ad hoc scheme EM-BMA is contrasted against)

Reported for the long-term window (2061-2100) under both SSPs:
station-averaged mean Tmax, mean annual precipitation, and BMA-weighted
mean discharge at Karesang.

Purpose: demonstrate that the paper's conclusions do not depend on the
particular weighting scheme, pre-empting the standard reviewer objection
that performance weights drive the result.

Output: outputs/bma/sensitivity_weighting_scheme.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BC   = BASE / "outputs" / "bias_corrected"
BIL  = BASE / "outputs" / "bilstm"
OUT  = BASE / "outputs" / "bma"

MODELS   = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR", "CanESM5",
            "UKESM1-0-LL", "GFDL-ESM4", "MRI-ESM2-0"]
STATIONS = ["Amol", "Gharakhil", "Sari"]
SCENARIOS = ["ssp245", "ssp585"]
LT = ("2061-01-01", "2100-12-31")


def _schemes():
    w = pd.read_csv(OUT / "bma_weights.csv").set_index("model")
    em = w["bma_weight"].reindex(MODELS).astype(float)
    em = em / em.sum()
    eq = pd.Series(1.0 / len(MODELS), index=MODELS)
    sk = w["skill_score"].reindex(MODELS).astype(float)
    sk = sk / sk.sum()
    return {"EM-BMA (this study)": em, "Equal weights": eq, "Composite skill weights": sk}


def _model_climate(model, scen):
    """Station-averaged long-term mean Tmax (degC) and mean annual precipitation (mm)."""
    tl, pl = [], []
    for st in STATIONS:
        d = pd.read_csv(BC / f"{st}_{model}_{scen}_bc.csv",
                        parse_dates=["date"], index_col="date").loc[LT[0]:LT[1]]
        tl.append(d["bc_tmax"]); pl.append(d["bc_pr"])
    return (pd.concat(tl, axis=1).mean(axis=1).mean(),
            pd.concat(pl, axis=1).mean(axis=1).mean() * 365.25)


def _model_discharge(scen):
    df = pd.read_csv(BIL / f"future_discharge_{scen}.csv", parse_dates=["date"])
    df = df[(df["date"] >= LT[0]) & (df["date"] <= LT[1])]
    return df.groupby("model")["Q_m3s"].mean()


def main():
    print("=" * 70)
    print("Weighting-scheme sensitivity (long-term 2061-2100)")
    print("=" * 70)
    schemes = _schemes()
    print("\nWeights (%):")
    print(pd.DataFrame({k: (v * 100).round(1) for k, v in schemes.items()}).to_string())

    obs_q = pd.read_csv(BASE / "outputs" / "streamflow" /
                        "karesang_monthly_2000_2016.csv")["Q_m3s"].mean()

    rows = []
    for scen in SCENARIOS:
        t = {m: _model_climate(m, scen)[0] for m in MODELS}
        p = {m: _model_climate(m, scen)[1] for m in MODELS}
        q = _model_discharge(scen)
        for name, w in schemes.items():
            tw = float(sum(w[m] * t[m] for m in MODELS))
            pw = float(sum(w[m] * p[m] for m in MODELS))
            qw = float(sum(w[m] * q[m] for m in MODELS))
            rows.append({"scenario": scen, "scheme": name,
                         "Tmax_degC": round(tw, 2),
                         "AnnualPrecip_mm": round(pw),
                         "Discharge_m3s": round(qw, 2),
                         "Discharge_change_pct": round(100 * (qw - obs_q) / obs_q, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity_weighting_scheme.csv", index=False)
    print("\nProjected long-term signal under each scheme:")
    print(df.to_string(index=False))

    print("\nSpread across weighting schemes (max - min):")
    for scen in SCENARIOS:
        s = df[df.scenario == scen]
        print(f"  {scen}:  Tmax {s.Tmax_degC.max()-s.Tmax_degC.min():.2f} degC | "
              f"precip {s.AnnualPrecip_mm.max()-s.AnnualPrecip_mm.min():.0f} mm | "
              f"discharge change {s.Discharge_change_pct.max()-s.Discharge_change_pct.min():.1f} pp")
    print(f"\nSaved: {OUT / 'sensitivity_weighting_scheme.csv'}")


if __name__ == "__main__":
    main()

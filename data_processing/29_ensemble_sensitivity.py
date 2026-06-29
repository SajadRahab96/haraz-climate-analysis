#!/usr/bin/env python3
"""
29_ensemble_sensitivity.py
==========================
Two ensemble-sensitivity analyses that substantiate the methodological
contributions of the framework:

  A. Two-track inclusion test. Compares the long-term (2061-2100) BMA-weighted
     projection computed from the four LARS-WG 8 library models only against the
     full six-model ensemble (which adds IPSL-CM6A-LR and MPI-ESM1-2-HR, the two
     highest-skill models, downscaled with DQM). Quantifies how much a
     library-bound ensemble would bias the projected Tmax, precipitation and
     discharge.

  B. EM-BMA vs naive skill-score weighting. Contrasts the Expectation-
     Maximization BMA posterior weights with weights obtained by normalising the
     composite skill scores, showing that a genuine BMA reweights (and reranks)
     the members relative to marginal skill.

Inputs : outputs/bma/bma_weights.csv, outputs/bias_corrected/*_bc.csv,
         outputs/bilstm/future_discharge_{scenario}.csv
Outputs: outputs/bma/sensitivity_inclusion.csv
         outputs/bma/sensitivity_weighting.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BC   = BASE / "outputs" / "bias_corrected"
BIL  = BASE / "outputs" / "bilstm"
OUT  = BASE / "outputs" / "bma"

ALL6     = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR", "CanESM5",
            "UKESM1-0-LL", "GFDL-ESM4", "MRI-ESM2-0"]
LIBRARY  = ["CanESM5", "GFDL-ESM4", "MRI-ESM2-0", "UKESM1-0-LL"]
STATIONS = ["Amol", "Gharakhil", "Sari"]
LONG_TERM = ("2061-01-01", "2100-12-31")


def _weights():
    w = pd.read_csv(OUT / "bma_weights.csv")
    return dict(zip(w["model"], w["bma_weight"])), dict(zip(w["model"], w["skill_score"]))


def _renorm(emw, models):
    a = np.array([emw[m] for m in models])
    return dict(zip(models, a / a.sum()))


def _model_climate(model, scenario):
    """Station-averaged long-term mean Tmax (degC) and mean annual precip (mm)."""
    tmax, pr = [], []
    for st in STATIONS:
        d = pd.read_csv(BC / f"{st}_{model}_{scenario}_bc.csv",
                        parse_dates=["date"], index_col="date").loc[LONG_TERM[0]:LONG_TERM[1]]
        tmax.append(d["bc_tmax"]); pr.append(d["bc_pr"])
    return (pd.concat(tmax, axis=1).mean(axis=1).mean(),
            pd.concat(pr, axis=1).mean(axis=1).mean() * 365.25)


def _model_discharge(scenario):
    df = pd.read_csv(BIL / f"future_discharge_{scenario}.csv", parse_dates=["date"])
    df = df[(df["date"] >= LONG_TERM[0]) & (df["date"] <= LONG_TERM[1])]
    return df.groupby("model")["Q_m3s"].mean().to_dict()


def _ens(values, weights):
    return sum(weights[m] * values[m] for m in weights)


def main():
    emw, skill = _weights()

    # ---- Analysis A: two-track inclusion ----
    rows = []
    for scen in ["ssp245", "ssp585"]:
        t = {m: _model_climate(m, scen)[0] for m in ALL6}
        p = {m: _model_climate(m, scen)[1] for m in ALL6}
        q = _model_discharge(scen)
        w6, w4 = _renorm(emw, ALL6), _renorm(emw, LIBRARY)
        for name, vals, dec in [("Tmax_degC", t, 2), ("AnnualPrecip_mm", p, 0), ("Discharge_m3s", q, 2)]:
            full, lib = _ens(vals, w6), _ens(vals, w4)
            rows.append({"scenario": scen, "variable": name,
                         "all6": round(full, dec), "library4": round(lib, dec),
                         "diff": round(full - lib, dec)})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_inclusion.csv", index=False)

    # ---- Analysis B: EM-BMA vs skill weighting ----
    ssum = sum(skill[m] for m in ALL6)
    rows = [{"model": m, "skill_score": round(skill[m], 3),
             "skill_weight_pct": round(100 * skill[m] / ssum, 1),
             "em_bma_weight_pct": round(100 * emw[m], 1)} for m in ALL6]
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_weighting.csv", index=False)

    print("Saved sensitivity_inclusion.csv and sensitivity_weighting.csv")
    print(f"Top model by skill: {max(ALL6, key=lambda m: skill[m])}; "
          f"by EM-BMA weight: {max(ALL6, key=lambda m: emw[m])}")


if __name__ == "__main__":
    main()

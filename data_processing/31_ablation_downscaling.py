#!/usr/bin/env python3
"""
31_ablation_downscaling.py
==========================
Downscaling-route ablation: does the two-track design (LARS-WG 8 for the four
library GCMs + DQM for the two non-library GCMs) bias the ensemble relative to
a methodologically uniform alternative in which DQM is applied to all six GCMs?

Procedure
---------
1. Apply DQM (functions imported from 11_bias_correction_dqm.py) to the four
   LARS-WG library models, writing to outputs/bias_corrected_alldqm/ so the
   production two-track set in outputs/bias_corrected/ is untouched.
2. Build two ensembles with identical EM-BMA weights (weights are estimated on
   raw GCM historical monthlies and are therefore independent of the
   downscaling route):
     - two-track : production _bc.csv set (4 x LARS-WG + 2 x DQM)
     - all-DQM   : DQM series for all six models
3. Compare BMA-weighted period statistics (near-term 2021-2060, long-term
   2061-2100; SSP2-4.5 and SSP5-8.5): station-averaged mean Tmax and mean
   annual precipitation. The inter-model range is reported for context.

Output: outputs/bma/sensitivity_downscaling_track.csv
"""

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BC_TWOTRACK = BASE / "outputs" / "bias_corrected"
BC_ALLDQM   = BASE / "outputs" / "bias_corrected_alldqm"
BC_ALLDQM.mkdir(parents=True, exist_ok=True)
OUT = BASE / "outputs" / "bma"

LIBRARY  = ["CanESM5", "GFDL-ESM4", "MRI-ESM2-0", "UKESM1-0-LL"]
NONLIB   = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR"]
ALL6     = NONLIB + LIBRARY
STATIONS = ["Amol", "Gharakhil", "Sari"]
SCENARIOS = ["ssp245", "ssp585"]
PERIODS  = {"NT": ("2021-01-01", "2060-12-31"), "LT": ("2061-01-01", "2100-12-31")}


def _load_dqm():
    spec = importlib.util.spec_from_file_location(
        "dqm_mod", BASE / "data_processing" / "11_bias_correction_dqm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_dqm_library_models(dqm):
    for stn_key, stn_excel in dqm.STATION_MAP.items():
        obs_df = dqm.load_obs(stn_excel)
        for model in LIBRARY:
            for scen in SCENARIOS:
                out_path = BC_ALLDQM / f"{stn_key}_{model}_{scen}_bc.csv"
                if out_path.exists():
                    print(f"  [exists] {out_path.name}")
                    continue
                res = dqm.process_station_model_scenario(stn_key, model, scen, obs_df)
                if res is not None:
                    res.to_csv(out_path)
                    print(f"  saved {out_path.name} ({len(res):,} days)")


def _period_stats(path, period):
    d = pd.read_csv(path, parse_dates=["date"], index_col="date")
    d = d.loc[PERIODS[period][0]:PERIODS[period][1]]
    return d["bc_tmax"].mean(), d["bc_pr"].mean() * 365.25


def ensemble_stats(source_dirs, weights):
    """source_dirs: dict model -> directory holding its _bc.csv files."""
    rows = {}
    for scen in SCENARIOS:
        for per in PERIODS:
            t_m, p_m = {}, {}
            for m, d in source_dirs.items():
                ts, ps = zip(*[_period_stats(d / f"{s}_{m}_{scen}_bc.csv", per)
                               for s in STATIONS])
                t_m[m], p_m[m] = np.mean(ts), np.mean(ps)
            w = {m: weights[m] for m in source_dirs}
            wsum = sum(w.values())
            w = {m: v / wsum for m, v in w.items()}
            rows[(scen, per)] = (
                sum(w[m] * t_m[m] for m in w), sum(w[m] * p_m[m] for m in w),
                max(t_m.values()) - min(t_m.values()),
            )
    return rows


def main():
    print("=" * 68)
    print("Downscaling-route ablation: two-track vs all-DQM ensembles")
    print("=" * 68)

    dqm = _load_dqm()
    print("\nStep 1: DQM for the four library models -> bias_corrected_alldqm/")
    run_dqm_library_models(dqm)

    wdf = pd.read_csv(OUT / "bma_weights.csv")
    weights = dict(zip(wdf["model"], wdf["bma_weight"]))

    two_track = {m: BC_TWOTRACK for m in ALL6}
    all_dqm = {m: (BC_TWOTRACK if m in NONLIB else BC_ALLDQM) for m in ALL6}

    print("\nStep 2: BMA-weighted period statistics")
    s_tt = ensemble_stats(two_track, weights)
    s_ad = ensemble_stats(all_dqm, weights)

    recs = []
    for (scen, per) in s_tt:
        t1, p1, rng1 = s_tt[(scen, per)]
        t2, p2, _ = s_ad[(scen, per)]
        recs.append({
            "scenario": scen, "period": per,
            "tmax_twotrack": round(t1, 2), "tmax_alldqm": round(t2, 2),
            "tmax_diff": round(t2 - t1, 2),
            "pr_twotrack_mm": round(p1), "pr_alldqm_mm": round(p2),
            "pr_diff_pct": round(100 * (p2 - p1) / p1, 1),
            "intermodel_tmax_range": round(rng1, 2),
        })
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "sensitivity_downscaling_track.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT / 'sensitivity_downscaling_track.csv'}")


if __name__ == "__main__":
    main()

"""
02_gap_filling.py
=================
Multi-source gap-filling for the three Haraz synoptic stations (2000-2020).

Procedure
---------
Gharakhil  : linear temporal interpolation for scattered missing values (<=3 days)
Sari       : structural gap 2000-2005 filled from IRIMO provincial database
             with mean-bias correction (ΔTmax=-0.184°C, ΔTmin=-0.321°C)
             + linear interpolation for scattered missing values
Amol       : structural gaps (year 2000 and 2018-2020) filled via MLR
             (Gharakhil + Sari as predictors; Gharakhil only for year 2000)
             + linear interpolation for scattered missing values

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Usage:
    python data_processing/02_gap_filling.py \
        --irimo   data/IRIMO_Daily_ClimateData_2000_2020.xlsx \
        --mazdb   data/Mazandaran.xlsx \
        --output  data/ClimateData_GapFilled_2000_2020.xlsx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.stats import ols_fit, ols_predict

# -- Constants -----------------------------------------------------------------
START_DATE  = "2000-01-01"
END_DATE    = "2020-12-31"
TRAIN_START = "2006-01-01"
TRAIN_END   = "2017-12-31"

SARI_BIAS = {"Tmax": -0.184, "Tmin": -0.321}   # provincial DB - IRIMO portal

VARIABLES = ["Tmax", "Tmin", "Pr"]


def log(msg: str):
    print(msg, flush=True)


def full_index() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, END_DATE, freq="D")


def read_sheet(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, parse_dates=["date"])
    return df.sort_values("date").set_index("date").reindex(full_index())


def linear_interp(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linear interpolation for gaps <= max_gap days; leaves larger gaps as NaN."""
    filled = series.copy()
    nan_mask = series.isna()
    nan_idx  = np.where(nan_mask)[0]
    if len(nan_idx) == 0:
        return filled
    # group consecutive NaN indices
    groups, group = [], [nan_idx[0]]
    for i in nan_idx[1:]:
        if i == group[-1] + 1:
            group.append(i)
        else:
            groups.append(group)
            group = [i]
    groups.append(group)
    for g in groups:
        if len(g) <= max_gap:
            filled.iloc[g] = np.nan   # let pandas interpolate
    filled = filled.interpolate(method="linear", limit=max_gap, limit_direction="both")
    return filled


def fill_gharakhil(irimo_ghk: pd.DataFrame) -> pd.DataFrame:
    log("  Gharakhil: linear interpolation for scattered missing values ...")
    df = irimo_ghk.copy()
    for v in VARIABLES:
        if v in df.columns:
            before = df[v].isna().sum()
            df[v] = linear_interp(df[v])
            after  = df[v].isna().sum()
            log(f"    {v}: filled {before - after} values")
    df["fill_method"] = "observed_or_interpolated"
    return df


def fill_sari(irimo_sari: pd.DataFrame, mazdb_sari: pd.DataFrame) -> pd.DataFrame:
    log("  Sari: structural gap 2000-2005 from IRIMO provincial DB + bias correction ...")
    df = irimo_sari.copy()
    df["fill_method"] = "observed"

    gap_mask = df.index < "2006-01-01"
    for v in ["Tmax", "Tmin"]:
        if v not in df.columns or v not in mazdb_sari.columns:
            continue
        # apply bias correction to provincial DB values
        filled_vals = mazdb_sari.loc[gap_mask, v] + SARI_BIAS[v]
        df.loc[gap_mask, v]            = filled_vals
        df.loc[gap_mask, "fill_method"] = "db_bias_corrected"
    if "Pr" in df.columns and "Pr" in mazdb_sari.columns:
        df.loc[gap_mask, "Pr"]           = mazdb_sari.loc[gap_mask, "Pr"]
        df.loc[gap_mask, "fill_method"]  = "db_bias_corrected"

    # scattered missing values
    for v in VARIABLES:
        if v in df.columns:
            before = df[v].isna().sum()
            df[v] = linear_interp(df[v])
            after  = df[v].isna().sum()
            if before > after:
                df.loc[df[v].isna() == False, "fill_method"] = df.loc[
                    df[v].isna() == False, "fill_method"].where(
                    df.loc[df[v].isna() == False, "fill_method"] != "observed",
                    "interpolated")
                log(f"    {v}: filled {before - after} scattered values")
    return df


def fill_amol(irimo_amol: pd.DataFrame,
              ghk: pd.DataFrame,
              sari: pd.DataFrame) -> pd.DataFrame:
    log("  Amol: MLR gap-filling for structural gaps ...")
    df = irimo_amol.copy()
    df["fill_method"] = "observed"

    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    gap_2000   = df.index.year == 2000
    gap_future = df.index.year >= 2018

    for v in VARIABLES:
        if v not in df.columns:
            continue
        y_train = df.loc[train_mask, v].values
        g_train = ghk.loc[train_mask, v].values if v in ghk.columns else None
        s_train = sari.loc[train_mask, v].values if v in sari.columns else None

        if g_train is None:
            log(f"    {v}: Gharakhil predictor not available, skipping MLR")
            continue

        # Model A: Gharakhil + Sari (for 2018-2020 gap)
        if s_train is not None:
            mask_ab = np.isfinite(y_train) & np.isfinite(g_train) & np.isfinite(s_train)
            X_a = np.column_stack([g_train[mask_ab], s_train[mask_ab]])
            coef_a, r2_a, rmse_a = ols_fit(X_a, y_train[mask_ab])
            log(f"    {v} Model A (Ghk+Sari): R²={r2_a:.3f}, RMSE={rmse_a:.3f}")
            # fill 2018-2020
            g_fut = ghk.loc[gap_future, v].values if v in ghk.columns else None
            s_fut = sari.loc[gap_future, v].values if v in sari.columns else None
            if g_fut is not None and s_fut is not None:
                X_pred = np.column_stack([g_fut, s_fut])
                df.loc[gap_future, v] = ols_predict(coef_a, X_pred)
                df.loc[gap_future & df.index.isin(df.index[gap_future]), "fill_method"] = "mlr_modelA"

        # Model B: Gharakhil only (for year 2000)
        mask_bb = np.isfinite(y_train) & np.isfinite(g_train)
        X_b = g_train[mask_bb].reshape(-1, 1)
        coef_b, r2_b, rmse_b = ols_fit(X_b, y_train[mask_bb])
        log(f"    {v} Model B (Ghk only): R²={r2_b:.3f}, RMSE={rmse_b:.3f}")
        g_2000 = ghk.loc[gap_2000, v].values if v in ghk.columns else None
        if g_2000 is not None:
            X_pred = g_2000.reshape(-1, 1)
            df.loc[gap_2000, v] = ols_predict(coef_b, X_pred)
            df.loc[gap_2000, "fill_method"] = "mlr_modelB"

    # scattered missing values
    for v in VARIABLES:
        if v in df.columns:
            before = df[v].isna().sum()
            df[v] = linear_interp(df[v])
            after  = df[v].isna().sum()
            if before > after:
                log(f"    {v}: filled {before - after} scattered values")
    return df


def main(irimo_path: str, mazdb_path: str, output_path: str):
    log("Loading IRIMO data ...")
    irimo = {s: read_sheet(irimo_path, s)
             for s in ["Gharakhil", "Amol", "Sari_DashtENaz_Airport"]}

    log("Loading IRIMO provincial database (Mazandaran) ...")
    mazdb = {}
    try:
        mazdb["Sari_DashtENaz_Airport"] = read_sheet(mazdb_path, "Sari")
    except Exception as e:
        log(f"  WARNING: could not load mazdb Sari sheet: {e}")
        mazdb["Sari_DashtENaz_Airport"] = pd.DataFrame(index=full_index())

    log("\n--- Processing stations ---")
    ghk  = fill_gharakhil(irimo["Gharakhil"])
    sari = fill_sari(irimo["Sari_DashtENaz_Airport"],
                     mazdb["Sari_DashtENaz_Airport"])
    amol = fill_amol(irimo["Amol"], ghk, sari)

    log(f"\nSaving filled dataset to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        ghk.reset_index().rename(columns={"index": "date"}).to_excel(
            writer, sheet_name="Gharakhil", index=False)
        amol.reset_index().rename(columns={"index": "date"}).to_excel(
            writer, sheet_name="Amol", index=False)
        sari.reset_index().rename(columns={"index": "date"}).to_excel(
            writer, sheet_name="Sari_DashtENaz_Airport", index=False)

    for name, df in [("Gharakhil", ghk), ("Amol", amol),
                     ("Sari", sari)]:
        remaining = sum(df[v].isna().sum() for v in VARIABLES if v in df.columns)
        log(f"  {name}: {remaining} missing values remaining")
    log("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-source gap-filling for HBCD-2020")
    p.add_argument("--irimo",  required=True)
    p.add_argument("--mazdb",  required=True)
    p.add_argument("--output", default="data/ClimateData_GapFilled_2000_2020.xlsx")
    a = p.parse_args()
    main(a.irimo, a.mazdb, a.output)

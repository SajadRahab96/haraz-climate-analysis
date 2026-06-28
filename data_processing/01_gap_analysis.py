"""
01_gap_analysis.py
==================
Systematic gap detection across all three synoptic stations in 2000-2020.

Identifies three categories of data absence:
  (i)   Structural gaps - consecutive periods with no IRIMO portal records
  (ii)  Scattered missing values - isolated absences (1-3 days)
  (iii) Potentially erroneous values - Tmax < Tmin, Pr < 0

Reference:
    Rahab-Rajaei S., Motiee H. (2026). Hydroclimatic Projections, Haraz Watershed.
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Usage:
    python data_processing/01_gap_analysis.py \
        --input data/IRIMO_Daily_ClimateData_2000_2020.xlsx \
        --output reports/gap_analysis_report.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STATIONS   = ["Gharakhil", "Amol", "Sari_DashtENaz_Airport"]
VARIABLES  = ["Tmax", "Tmin", "Pr"]
START_DATE = "2000-01-01"
END_DATE   = "2020-12-31"

STRUCTURAL_THRESHOLDS = {
    "Gharakhil":              {},
    "Amol":                   {},
    "Sari_DashtENaz_Airport": {},
}


def log(msg: str):
    print(msg, flush=True)


def load_data(path: str) -> dict[str, pd.DataFrame]:
    """Load the IRIMO Excel file; returns a dict of {station: DataFrame}."""
    log(f"Loading: {path}")
    xf = pd.ExcelFile(path)
    sheets = {}
    for sheet in xf.sheet_names:
        df = pd.read_excel(xf, sheet_name=sheet, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        sheets[sheet] = df
    return sheets


def full_date_range() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, END_DATE, freq="D")


def analyze_station(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ref = full_date_range()
    df = df.set_index("date").reindex(ref)

    records = []
    for var in VARIABLES:
        if var not in df.columns:
            log(f"  [{name}] column '{var}' not found - skipping")
            continue
        col = df[var]
        n_total   = len(col)
        n_missing = col.isna().sum()

        # detect consecutive runs of NaN (structural gaps if len > 3)
        is_nan = col.isna()
        structural_days = 0
        scattered_days  = 0
        run_len = 0
        for v in is_nan:
            if v:
                run_len += 1
            else:
                if run_len > 3:
                    structural_days += run_len
                elif run_len > 0:
                    scattered_days  += run_len
                run_len = 0
        if run_len > 3:
            structural_days += run_len
        elif run_len > 0:
            scattered_days  += run_len

        # erroneous values (only meaningful for Tmax/Tmin pair)
        n_erroneous = 0
        if var == "Tmax" and "Tmin" in df.columns:
            n_erroneous = int((col < df["Tmin"]).sum())
        if var == "Pr":
            n_erroneous = int((col < 0).sum())

        records.append({
            "station":         name,
            "variable":        var,
            "n_total":         n_total,
            "n_missing":       int(n_missing),
            "structural_days": structural_days,
            "scattered_days":  scattered_days,
            "n_erroneous":     n_erroneous,
            "pct_missing":     round(100.0 * n_missing / n_total, 2),
        })
    return pd.DataFrame(records)


def main(input_path: str, output_path: str):
    sheets = load_data(input_path)
    results = []
    for name in STATIONS:
        if name not in sheets:
            log(f"  WARNING: sheet '{name}' not found in workbook")
            continue
        log(f"  Analyzing {name} ...")
        res = analyze_station(sheets[name], name)
        results.append(res)

    out = pd.concat(results, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log(f"\nGap analysis report saved to: {output_path}")
    log("\n" + out.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Systematic gap analysis for HBCD-2020")
    p.add_argument("--input",  required=True, help="Path to IRIMO_Daily_ClimateData_2000_2020.xlsx")
    p.add_argument("--output", default="reports/gap_analysis_report.csv")
    a = p.parse_args()
    main(a.input, a.output)

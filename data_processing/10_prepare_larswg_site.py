"""
10_prepare_larswg_site.py
=========================
Converts HBCD-2020 (ClimateData_GapFilled_2000_2020.xlsx) into LARS-WG 8
site input files (.dat and .st) for each of the three synoptic stations.

Outputs (written to <project>/LARS weather generator 8/Data/):
  - Amol.dat / Amol.st
  - Gharakhil.dat / Gharakhil.st
  - Sari.dat / Sari.st

Format: YEAR  MONTH  DAY  TMAX  TMIN  RAIN  (tab-separated, no header)

CO2 concentration for 2000-2020 baseline: mean ~390 ppm (NOAA/Mauna Loa).
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_EXCEL = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR    = BASE_DIR / "LARS weather generator 8" / "Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Station metadata ──────────────────────────────────────────────────────────
STATIONS = {
    "Amol": {
        "name_in_excel": "Amol",
        "lat": 36.466,
        "lon": 52.383,
        "alt": 23.7,
        "file_prefix": "Amol",
    },
    "Gharakhil": {
        "name_in_excel": "Gharakhil",
        "lat": 36.487,
        "lon": 52.108,
        "alt": 14.7,
        "file_prefix": "Gharakhil",
    },
    "Sari": {
        "name_in_excel": "Sari (Dasht-E-Naz Airport)",
        "lat": 36.653,
        "lon": 53.193,
        "alt": 16.7,
        "file_prefix": "Sari",
    },
}

# Mean CO2 concentration for 2000-2020 baseline (ppm, Mauna Loa annual mean)
CO2_BASELINE = 390.769   # same value used in the existing Amol.st

# ── Load HBCD-2020 ────────────────────────────────────────────────────────────
print("Loading HBCD-2020 ...")
df_all = pd.read_excel(DATA_EXCEL, sheet_name="All_Stations", parse_dates=["date"])
print(f"  Rows: {len(df_all):,} | Period: {df_all['date'].min().date()} – {df_all['date'].max().date()}")

# ── Process each station ──────────────────────────────────────────────────────
for key, meta in STATIONS.items():
    print(f"\nProcessing {key} ...")

    # Filter to station
    df = df_all[df_all["station_name"] == meta["name_in_excel"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Use calibration period 2000-2020 for LARS-WG site analysis
    # (LARS-WG recommends ≥20 years)
    print(f"  Rows: {len(df):,} | {df['date'].min().date()} – {df['date'].max().date()}")

    # Check for missing values
    missing = df[["tmax", "tmin", "rrr24"]].isnull().sum()
    if missing.any():
        print(f"  WARNING: missing values: {missing[missing>0].to_dict()}")

    # Round to 1 decimal (LARS-WG standard)
    df["tmax"]  = df["tmax"].round(1)
    df["tmin"]  = df["tmin"].round(1)
    df["rrr24"] = df["rrr24"].round(2).clip(lower=0)

    # Extract year/month/day
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day

    # ── Write .dat file ───────────────────────────────────────────────────────
    dat_path = OUT_DIR / f"{meta['file_prefix']}.dat"
    out_rows = df[["year", "month", "day", "tmax", "tmin", "rrr24"]]
    out_rows.to_csv(dat_path, sep="\t", header=False, index=False)
    print(f"  Saved: {dat_path}  ({len(out_rows):,} rows)")

    # ── Write .st site file ───────────────────────────────────────────────────
    st_content = f"""[SITE]\t\t\t
{meta['file_prefix']}
[LAT, LON and ALT]\t
{meta['lat']}\t{meta['lon']}\t{meta['alt']}
[CO2]
{CO2_BASELINE}
[WEATHER FILES]\t\t\t
{meta['file_prefix']}.dat
[FORMAT]\t\t\t
YEAR MONTH DAY MAX MIN RAIN\t\t\t
[END]\t\t\t
"""
    st_path = OUT_DIR / f"{meta['file_prefix']}.st"
    with open(st_path, "w", encoding="utf-8") as f:
        f.write(st_content)
    print(f"  Saved: {st_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LARS-WG site files ready in:", OUT_DIR)
print("\nNext steps:")
print("  1. Open LARSWG.exe")
print("  2. File > Open Site > select each .st file")
print("  3. Analysis > Site Analysis  (creates .stx, .tst, .wgx in Sitebase)")
print("  4. Scenarios > Generate for SSP2-4.5 and SSP5-8.5")
print("  5. Copy outputs to outputs/larswg/ for script 11")
print("="*60)

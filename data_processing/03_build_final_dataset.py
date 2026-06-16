"""
03_build_final_dataset.py
=========================
Assemble the final Haraz Basin Climate Dataset (HBCD-2020) from the gap-filled
output of 02_gap_filling.py.

Produces a single clean Excel workbook with:
  - Three sheets (Gharakhil, Amol, Sari_DashtENaz_Airport)
  - Columns: date, Tmax_C, Tmin_C, Pr_mm, fill_method
  - Zero remaining missing values
  - Complete daily coverage: 2000-01-01 to 2020-12-31

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Usage:
    python data_processing/03_build_final_dataset.py \
        --filled  data/ClimateData_GapFilled_2000_2020.xlsx \
        --output  data/HBCD_2020_Final.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS  = ["Gharakhil", "Amol", "Sari_DashtENaz_Airport"]
START_DATE = "2000-01-01"
END_DATE   = "2020-12-31"


def log(msg: str):
    print(msg, flush=True)


def full_index() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, END_DATE, freq="D")


def provenance_summary(df: pd.DataFrame, name: str) -> None:
    if "fill_method" not in df.columns:
        return
    counts = df["fill_method"].value_counts()
    total  = len(df)
    log(f"\n  {name} data provenance:")
    for method, cnt in counts.items():
        log(f"    {method:30s}: {cnt:5d} days ({100*cnt/total:.1f}%)")


def main(filled_path: str, output_path: str):
    log(f"Loading gap-filled dataset: {filled_path}")
    xf = pd.ExcelFile(filled_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ref = full_index()

    summary_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name in STATIONS:
            if name not in xf.sheet_names:
                log(f"  WARNING: sheet '{name}' not found - skipping")
                continue
            df = pd.read_excel(xf, sheet_name=name, parse_dates=["date"])
            df = df.sort_values("date").set_index("date").reindex(ref)

            # Rename columns for clarity
            rename = {}
            if "Tmax" in df.columns: rename["Tmax"] = "Tmax_C"
            if "Tmin" in df.columns: rename["Tmin"] = "Tmin_C"
            if "Pr"   in df.columns: rename["Pr"]   = "Pr_mm"
            df = df.rename(columns=rename)

            # Final missing-value check
            for col in ["Tmax_C", "Tmin_C", "Pr_mm"]:
                if col in df.columns:
                    n_miss = df[col].isna().sum()
                    if n_miss > 0:
                        log(f"  WARNING: {name} - {col} still has {n_miss} missing values!")

            provenance_summary(df, name)

            # Write sheet
            df.reset_index().rename(columns={"index": "date"}).to_excel(
                writer, sheet_name=name, index=False)

            # Summary row
            for col in ["Tmax_C", "Tmin_C", "Pr_mm"]:
                if col in df.columns:
                    summary_rows.append({
                        "station":   name,
                        "variable":  col,
                        "n_days":    len(df),
                        "n_missing": int(df[col].isna().sum()),
                        "mean":      round(df[col].mean(), 3),
                        "std":       round(df[col].std(), 3),
                        "min":       round(df[col].min(), 3),
                        "max":       round(df[col].max(), 3),
                    })

        # Write summary sheet
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    log(f"\nHBCD-2020 saved to: {output_path}")
    log("\n" + summary_df.to_string(index=False))
    log("\nAll done. HBCD-2020 is ready for LARS-WG 8 and BiLSTM training.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assemble HBCD-2020 final dataset")
    p.add_argument("--filled",  required=True,
                   help="Output of 02_gap_filling.py")
    p.add_argument("--output", default="data/HBCD_2020_Final.xlsx",
                   help="Path for HBCD-2020 Excel file")
    a = p.parse_args()
    main(a.filled, a.output)

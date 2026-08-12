"""
12_prepare_streamflow_data.py
==============================
Prepares the streamflow dataset by:
  1. Reading Karesang monthly discharge from KareSang.xlsx
  2. Converting Solar Hijri dates to Gregorian
  3. Merging with 2000-2020 monthly climate aggregates
  4. Splitting into train (2000-2012) / validation (2013-2016) sets
  5. Saving clean datasets for BiLSTM training

Solar Hijri -> Gregorian conversion:
  Each Hijri year Y begins ~March 21 of Gregorian year (Y + 621).
  Persian months: Farvardin(4) Ordibehesht(5) Khordad(6) Tir(7)
                  Mordad(8) Shahrivar(9) Mehr(10) Aban(11) Azar(12)
                  Dey(1+1) Bahman(2+1) Esfand(3+1)

Output columns (monthly):
  gregorian_date, year, month, Q_m3s,
  P_mm_month, Tmax_mean, Tmin_mean, Tavg_mean

Outputs:
  outputs/streamflow/karesang_monthly_2000_2016.csv
  outputs/streamflow/train_2000_2012.csv
  outputs/streamflow/val_2013_2016.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
FLOW_XLSX  = BASE_DIR / "KareSang.xlsx"
OBS_XLSX   = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR    = BASE_DIR / "outputs" / "streamflow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Persian month column names in the Excel -> Persian month number -> Gregorian shift
# Persian months 1-6 start in Gregorian months 4-9 (same year)
# Persian months 7-12 start in Gregorian months 10-3 (month 7-9: same Gregorian year, 10-12: next year)
PERSIAN_MONTH_COLS = {
    "far": 1,   # Farvardin  -> ~April (Gregorian year = SH year + 621)
    "ord": 2,   # Ordibehesht
    "khr": 3,   # Khordad
    "tir": 4,   # Tir
    "mor": 5,   # Mordad
    "shr": 6,   # Shahrivar
    "mhr": 7,   # Mehr       -> ~October
    "abn": 8,   # Aban
    "azr": 9,   # Azar
    "dey": 10,  # Dey
    "bah": 11,  # Bahman
    "esf": 12,  # Esfand     -> ~March (next Gregorian year)
}

# Approximate Gregorian month for each Persian month
# and whether it falls in the next Gregorian year
PERSIAN_TO_GREGORIAN_MONTH = {
    1:  (4,  0),   # Farvardin  -> April,     same year
    2:  (5,  0),   # Ordibehesht-> May
    3:  (6,  0),   # Khordad    -> June
    4:  (7,  0),   # Tir        -> July
    5:  (8,  0),   # Mordad     -> August
    6:  (9,  0),   # Shahrivar  -> September
    7:  (10, 0),   # Mehr       -> October
    8:  (11, 0),   # Aban       -> November
    9:  (12, 0),   # Azar       -> December
    10: (1,  1),   # Dey        -> January  +1 year
    11: (2,  1),   # Bahman     -> February +1 year
    12: (3,  1),   # Esfand     -> March    +1 year
}

def sh_to_gregorian_year(sh_year: int) -> int:
    """Convert Solar Hijri year to approximate Gregorian year (start of year)."""
    return sh_year + 621


def expand_karesang(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format Karesang Excel (one row per SH year, columns per month)
    to long-format monthly DataFrame with Gregorian dates.
    """
    records = []
    for _, row in df_raw.iterrows():
        sh_year = int(row["sal"])
        greg_year_start = sh_to_gregorian_year(sh_year)

        for col, persian_month in PERSIAN_MONTH_COLS.items():
            if col not in row.index:
                continue
            q_val = row[col]
            if pd.isna(q_val) or q_val <= 0:
                q_val = np.nan

            greg_month, year_offset = PERSIAN_TO_GREGORIAN_MONTH[persian_month]
            greg_year = greg_year_start + year_offset

            records.append({
                "sh_year":    sh_year,
                "persian_month": persian_month,
                "year":   greg_year,
                "month":  greg_month,
                "Q_m3s":  q_val,
            })

    df = pd.DataFrame(records)
    df["gregorian_date"] = pd.to_datetime(
        df[["year", "month"]].assign(day=1)
    )
    df = df.sort_values("gregorian_date").reset_index(drop=True)
    return df


def aggregate_obs_monthly(obs_excel: Path) -> pd.DataFrame:
    """
    Aggregate 2000-2020 daily data to monthly means/sums for all 3 stations.
    Returns a single DataFrame with basin-average climate variables.
    """
    df = pd.read_excel(obs_excel, sheet_name="All_Stations", parse_dates=["date"])

    # Monthly aggregation per station
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month", "station_name"]).agg(
        P_mm_month=("rrr24", "sum"),
        Tmax_mean=("tmax", "mean"),
        Tmin_mean=("tmin", "mean"),
    ).reset_index()

    # Basin average (mean across 3 stations)
    basin_avg = monthly.groupby(["year", "month"]).agg(
        P_mm_month=("P_mm_month", "mean"),
        Tmax_mean=("Tmax_mean", "mean"),
        Tmin_mean=("Tmin_mean", "mean"),
    ).reset_index()

    basin_avg["Tavg_mean"] = (basin_avg["Tmax_mean"] + basin_avg["Tmin_mean"]) / 2
    basin_avg["gregorian_date"] = pd.to_datetime(
        basin_avg[["year", "month"]].assign(day=1)
    )
    return basin_avg


def main():
    print("=" * 60)
    print("Phase III - Streamflow Data Preparation")
    print("=" * 60)

    # -- Load Karesang flow ----------------------------------------------------
    print("\nLoading Karesang discharge ...")
    df_raw = pd.read_excel(FLOW_XLSX, sheet_name="Sheet1")
    karesang_raw = df_raw[df_raw["station"].str.contains("كره سنگ", na=False)].copy()
    print(f"  Karesang rows (SH years): {len(karesang_raw)}")
    print(f"  SH year range: {karesang_raw['sal'].min()} - {karesang_raw['sal'].max()}")

    flow_monthly = expand_karesang(karesang_raw)
    print(f"  Expanded to {len(flow_monthly):,} monthly records")
    print(f"  Gregorian date range: {flow_monthly['gregorian_date'].min().date()} - "
          f"{flow_monthly['gregorian_date'].max().date()}")

    # -- Load climate observations ---------------------------------------------
    print("\nAggregating 2000-2020 to monthly ...")
    clim_monthly = aggregate_obs_monthly(OBS_XLSX)
    print(f"  Monthly climate records: {len(clim_monthly):,}")

    # -- Merge on year-month ---------------------------------------------------
    merged = pd.merge(
        flow_monthly[["gregorian_date", "year", "month", "Q_m3s"]],
        clim_monthly[["gregorian_date", "P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]],
        on="gregorian_date", how="inner"
    )
    merged = merged.sort_values("gregorian_date").reset_index(drop=True)
    merged = merged.dropna(subset=["Q_m3s"])
    # Source Excel contains duplicated SH-year rows (identical Q values):
    # keep one record per calendar month
    n_dup = merged["gregorian_date"].duplicated().sum()
    if n_dup:
        print(f"  NOTE: removed {n_dup} duplicated monthly records")
        merged = merged.drop_duplicates(subset="gregorian_date", keep="first").reset_index(drop=True)

    print(f"\nMerged dataset: {len(merged):,} months")
    print(f"  Date range: {merged['gregorian_date'].min().date()} - {merged['gregorian_date'].max().date()}")
    print(f"  Missing Q: {merged['Q_m3s'].isna().sum()}")
    print(f"\n  Descriptive statistics:")
    print(merged[["Q_m3s", "P_mm_month", "Tmax_mean", "Tmin_mean"]].describe().round(2))

    # -- Filter to study period 2000-2016 -------------------------------------
    study = merged[
        (merged["gregorian_date"] >= "2000-01-01") &
        (merged["gregorian_date"] <= "2016-12-31")
    ].copy()
    print(f"\nStudy period (2000-2016): {len(study):,} months")
    print(f"  Missing Q in study period: {study['Q_m3s'].isna().sum()}")

    # Fill any remaining Q gaps with linear interpolation
    study["Q_m3s"] = study["Q_m3s"].interpolate(method="linear", limit=3)

    # -- Save full dataset -----------------------------------------------------
    full_path = OUT_DIR / "karesang_monthly_2000_2016.csv"
    study.to_csv(full_path, index=False)
    print(f"\nSaved: {full_path}")

    # -- Train / Validation split ----------------------------------------------
    train = study[study["gregorian_date"] <= "2012-12-31"]
    val   = study[
        (study["gregorian_date"] >= "2013-01-01") &
        (study["gregorian_date"] <= "2016-12-31")
    ]

    train_path = OUT_DIR / "train_2000_2012.csv"
    val_path   = OUT_DIR / "val_2013_2016.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    print(f"Train (2000-2012): {len(train):,} months -> {train_path.name}")
    print(f"Val   (2013-2016): {len(val):,}  months -> {val_path.name}")

    # -- Plot: Q time series ---------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(study["gregorian_date"], study["Q_m3s"], "b-o", ms=3, lw=1.2)
    axes[0].axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31"),
                    alpha=0.15, color="orange", label="Validation")
    axes[0].axvspan(pd.Timestamp("2000-01-01"), pd.Timestamp("2012-12-31"),
                    alpha=0.1, color="green", label="Training")
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].set_title("Haraz-Karesang Monthly Discharge (2000-2016)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(study["gregorian_date"], study["P_mm_month"], color="steelblue",
                width=20, alpha=0.7, label="Basin-avg Precipitation")
    axes[1].set_ylabel("Precipitation (mm/month)")
    axes[1].set_title("Basin-Average Monthly Precipitation (2000-2020)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "karesang_discharge_overview.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {fig_path}")

    # -- Correlation matrix ----------------------------------------------------
    print("\nCorrelation with discharge:")
    corr = study[["Q_m3s", "P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]].corr()
    print(corr["Q_m3s"].round(3))

    print("\nDone.")


if __name__ == "__main__":
    main()

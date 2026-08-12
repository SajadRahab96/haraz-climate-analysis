"""
11_bias_correction_dqm.py
=========================
Applies DQM bias correction
to CMIP6 raw GCM outputs (all 6 models, SSP2-4.5 and SSP5-8.5) using the
2000-2020 observational dataset as the reference.

Method: Detrended Quantile Mapping (Cannon et al., 2015; Li et al., 2010)
  - Calibration period: 2000-2014 (overlapping obs and GCM historical)
  - Variables corrected: pr (precipitation), tasmax (Tmax), tasmin (Tmin)
  - Precipitation: separate wet-day frequency and intensity correction
  - Temperature: additive correction preserving long-term trend

Inputs:
  - data/cmip6_gcs/historical/{model}/{station}_{model}_historical.csv
  - data/cmip6_gcs/ssp245/{model}/{station}_{model}_ssp245.csv
  - data/cmip6_gcs/ssp585/{model}/{station}_{model}_ssp585.csv
  - ClimateData_GapFilled_2000_2020.xlsx (reference observations)

Outputs (in outputs/bias_corrected/):
  - {station}_{model}_{scenario}_bc.csv  for 2015-2100

References:
  Cannon, A.J., Sobie, S.R., Murdock, T.Q. (2015). Bias Correction of GCM
  Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
  Quantiles and Extremes? J. Climate, 28(17), 6938-6959.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# -- Configuration -------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_GCS   = BASE_DIR / "data" / "cmip6_gcs"
OBS_EXCEL  = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR    = BASE_DIR / "outputs" / "bias_corrected"
FIG_DIR    = BASE_DIR / "outputs" / "bias_corrected" / "validation_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCENARIOS = ["ssp245", "ssp585"]

STATION_MAP = {
    "Amol":      "Amol",
    "Gharakhil": "Gharakhil",
    "Sari":      "Sari (Dasht-E-Naz Airport)",
}

# Calibration period (obs and historical GCM overlap)
CALIB_START = "2000-01-01"
CALIB_END   = "2014-12-31"

N_QUANTILES = 100   # quantile levels for mapping (1%, 2%, ..., 100%)
WET_THRESH  = 0.1   # mm/day threshold for wet day (obs)


# -- Core DQM Functions --------------------------------------------------------

def build_quantile_map(obs_calib: np.ndarray, gcm_calib: np.ndarray,
                        n_q: int = N_QUANTILES) -> interp1d:
    """
    Build a quantile mapping transfer function from GCM to Obs space.
    Returns an interp1d function: gcm_value -> bias_corrected_value.
    """
    quantiles = np.linspace(0, 100, n_q + 1)
    q_obs = np.percentile(obs_calib, quantiles)
    q_gcm = np.percentile(gcm_calib, quantiles)
    # Avoid duplicate x values in interpolation
    _, unique_idx = np.unique(q_gcm, return_index=True)
    q_gcm_u = q_gcm[unique_idx]
    q_obs_u = q_obs[unique_idx]
    tf = interp1d(q_gcm_u, q_obs_u,
                  kind="linear",
                  bounds_error=False,
                  fill_value=(q_obs_u[0], q_obs_u[-1]))
    return tf


def dqm_temperature(obs_calib: np.ndarray, gcm_calib: np.ndarray,
                     gcm_future: np.ndarray) -> np.ndarray:
    """
    Detrended Quantile Mapping for temperature (additive).
    Steps:
      1. Compute mean GCM trend (future - calib mean)
      2. Detrend future series
      3. Apply quantile map (calib -> obs space)
      4. Re-add trend
    """
    trend = np.mean(gcm_future) - np.mean(gcm_calib)
    gcm_future_detrended = gcm_future - trend
    tf = build_quantile_map(obs_calib, gcm_calib)
    corrected_detrended = tf(gcm_future_detrended)
    corrected = corrected_detrended + trend
    return corrected


def dqm_precipitation(obs_calib: np.ndarray, gcm_calib: np.ndarray,
                       gcm_future: np.ndarray,
                       wet_thresh: float = WET_THRESH) -> np.ndarray:
    """
    Detrended Quantile Mapping for precipitation (multiplicative for intensity,
    separate wet-day frequency adjustment).
    Steps:
      1. Correct wet-day frequency (scale threshold by ratio)
      2. Apply quantile map on wet-day intensities (multiplicative scaling)
      3. Re-add detrended long-term change signal
    """
    gcm_future = np.clip(gcm_future, 0, None)
    gcm_calib  = np.clip(gcm_calib, 0, None)
    obs_calib  = np.clip(obs_calib, 0, None)

    # Wet-day frequency
    obs_wet_freq  = np.mean(obs_calib > wet_thresh)
    gcm_wet_freq  = np.mean(gcm_calib > wet_thresh)
    fut_wet_freq  = np.mean(gcm_future > wet_thresh)

    # Adjust threshold for future GCM to match obs wet-day frequency ratio
    if fut_wet_freq > 0 and gcm_wet_freq > 0:
        freq_scale = obs_wet_freq / gcm_wet_freq
        fut_thresh = max(wet_thresh, np.percentile(gcm_future, (1 - fut_wet_freq * freq_scale) * 100))
    else:
        fut_thresh = wet_thresh

    # Select wet days
    obs_wet   = obs_calib[obs_calib > wet_thresh]
    gcm_wet   = gcm_calib[gcm_calib > wet_thresh]
    fut_wet_mask = gcm_future > fut_thresh
    gcm_fut_wet  = gcm_future[fut_wet_mask]

    corrected = np.zeros_like(gcm_future)

    if len(obs_wet) > 5 and len(gcm_wet) > 5 and len(gcm_fut_wet) > 0:
        # Multiplicative QM on wet intensities
        tf = build_quantile_map(obs_wet, gcm_wet)
        gcm_wet_corr = tf(gcm_fut_wet)
        # Scale factor (DQM: preserve ratio of future to historical)
        if np.mean(gcm_wet) > 0:
            scale = np.mean(gcm_fut_wet) / np.mean(gcm_wet)
            gcm_wet_corr = gcm_wet_corr * scale
        corrected[fut_wet_mask] = np.clip(gcm_wet_corr, 0, None)

    return corrected


# -- Data Loaders --------------------------------------------------------------

def load_obs(station_excel_name: str) -> pd.DataFrame:
    """Load the gap-filled observational baseline for one station, return daily DataFrame."""
    df = pd.read_excel(OBS_EXCEL, sheet_name="All_Stations", parse_dates=["date"])
    df = df[df["station_name"] == station_excel_name].copy()
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"tmax": "obs_tmax", "tmin": "obs_tmin", "rrr24": "obs_pr"})
    return df[["obs_tmax", "obs_tmin", "obs_pr"]]


def parse_360day_dates(raw_dates: pd.Series, year_start: int, year_end: int) -> pd.DatetimeIndex:
    """
    Convert a 360-day calendar date series to Gregorian by linear interpolation.
    The 360-day calendar has exactly 12 months x 30 days per year.
    Strategy: map each 360-day sequence to the corresponding Gregorian calendar
    day-of-year within each year using linear resampling.
    """
    # Build a new Gregorian DatetimeIndex with the same number of rows
    n = len(raw_dates)
    # Approximate start/end from raw strings
    try:
        y0 = int(str(raw_dates.iloc[0])[:4])
        y1 = int(str(raw_dates.iloc[-1])[:4])
    except Exception:
        y0, y1 = year_start, year_end

    # Generate a regular daily Gregorian index of the same length
    # using a simple linear mapping: n rows -> n Gregorian days from y0-01-01
    start = pd.Timestamp(f"{y0}-01-01")
    # 360-day year has 360 days; each Gregorian year ~ 365.25 days
    # Scale factor: Gregorian_days = n * (365.25/360)
    n_greg = round(n * (365.25 / 360))
    greg_idx = pd.date_range(start=start, periods=n_greg, freq="D")
    # Subsample greg_idx to exactly n points (uniform spacing)
    indices = np.linspace(0, n_greg - 1, n, dtype=int)
    return greg_idx[indices]


def load_gcm_csv(model: str, scenario: str, station_key: str) -> pd.DataFrame:
    """
    Load GCM CSV for given model/scenario/station.
    Handles:
      - Standard calendar models: date column parsed normally
      - UKESM1-0-LL: 360-day calendar -> remapped to Gregorian
      - Missing tasmax_C: derived as 2*tas_C - tasmin_C
    """
    station_file_key = {
        "Amol": "Amol",
        "Gharakhil": "Gharakhil",
        "Sari": "Sari_DashtENaz_Airport",
    }[station_key]

    csv_path = DATA_GCS / scenario / model / f"{station_file_key}_{model}_{scenario}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"GCM file not found: {csv_path}")

    # Read without date parsing first to handle 360-day calendars
    df = pd.read_csv(csv_path)

    # Try standard parse; fall back to 360-day remapping if it fails
    raw_date_strings = df["date"].copy()
    try:
        converted = pd.to_datetime(raw_date_strings)
        df["date"] = converted
        valid_dates = True
    except (ValueError, TypeError):
        df["date"] = raw_date_strings   # restore originals
        valid_dates = False

    if not valid_dates:
        # 360-day calendar - UKESM1-0-LL
        print(f"      NOTE: 360-day calendar detected for {csv_path.name} -- remapping to Gregorian")
        greg_idx = parse_360day_dates(df["date"], 2000, 2100)
        df["date"] = greg_idx
        # Remove any duplicate dates created by resampling
        df = df.drop_duplicates(subset=["date"])

    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "pr_mm":   "gcm_pr",
        "tasmax_C":"gcm_tmax",
        "tasmin_C":"gcm_tmin",
        "tas_C":   "gcm_tas",
    })
    # Derive missing Tmax from Tas and Tmin (Tmax = 2*Tas - Tmin)
    if "gcm_tmax" not in df.columns and "gcm_tas" in df.columns and "gcm_tmin" in df.columns:
        df["gcm_tmax"] = 2 * df["gcm_tas"] - df["gcm_tmin"]
    return df[["gcm_pr", "gcm_tmax", "gcm_tmin"]]


# -- Validation Plot -----------------------------------------------------------

def plot_monthly_comparison(obs_monthly, raw_monthly, bc_monthly,
                             label, station, model, scenario, var):
    """Monthly mean comparison: obs vs raw GCM vs bias-corrected."""
    fig, ax = plt.subplots(figsize=(10, 4))
    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.plot(months, obs_monthly, "k-o", label="Obs (2000-2014)", lw=2)
    ax.plot(months, raw_monthly, "r--s", label=f"Raw GCM", lw=1.5, alpha=0.7)
    ax.plot(months, bc_monthly, "b-^", label=f"Bias-corrected", lw=1.5)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_title(f"{station} | {model} | {var} | Calibration Check (2000-2014)")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"bc_calib_{station}_{model}_{var}.png", dpi=150)
    plt.close(fig)


# -- Main Pipeline -------------------------------------------------------------

def process_station_model_scenario(station_key: str, model: str, scenario: str,
                                    obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run DQM for one station x model x scenario combination.
    Returns bias-corrected daily DataFrame for the future period.
    """
    # Load historical GCM (for calibration)
    try:
        hist_df = load_gcm_csv(model, "historical", station_key)
    except FileNotFoundError as e:
        print(f"    SKIP (missing historical): {e}")
        return None

    # Load future GCM
    try:
        fut_df = load_gcm_csv(model, scenario, station_key)
    except FileNotFoundError as e:
        print(f"    SKIP (missing future): {e}")
        return None

    # Calibration period slices
    obs_cal  = obs_df[CALIB_START:CALIB_END]
    hist_cal = hist_df[CALIB_START:CALIB_END]

    # Align on common dates
    common_idx = obs_cal.index.intersection(hist_cal.index)
    if len(common_idx) < 365:
        print(f"    SKIP: insufficient overlap ({len(common_idx)} days)")
        return None

    obs_cal  = obs_cal.loc[common_idx]
    hist_cal = hist_cal.loc[common_idx]

    # -- Bias correct each variable --------------------------------------------
    results = {"date": fut_df.index}

    # Temperature (additive DQM)
    for var, obs_col, gcm_hist_col, gcm_fut_col in [
        ("tmax", "obs_tmax", "gcm_tmax", "gcm_tmax"),
        ("tmin", "obs_tmin", "gcm_tmin", "gcm_tmin"),
    ]:
        obs_arr  = obs_cal[obs_col].values
        gcm_arr  = hist_cal[gcm_hist_col].values
        fut_arr  = fut_df[gcm_fut_col].values

        # Remove NaN
        valid    = ~(np.isnan(obs_arr) | np.isnan(gcm_arr))
        bc_fut   = dqm_temperature(obs_arr[valid], gcm_arr[valid], fut_arr)
        results[f"bc_{var}"] = bc_fut

    # Precipitation (multiplicative DQM)
    obs_pr  = obs_cal["obs_pr"].values
    gcm_pr  = hist_cal["gcm_pr"].values
    fut_pr  = fut_df["gcm_pr"].values
    valid_pr = ~(np.isnan(obs_pr) | np.isnan(gcm_pr))
    bc_pr   = dqm_precipitation(obs_pr[valid_pr], gcm_pr[valid_pr], fut_pr)
    results["bc_pr"] = bc_pr

    # -- Raw GCM for reference -------------------------------------------------
    results["raw_tmax"] = fut_df["gcm_tmax"].values
    results["raw_tmin"] = fut_df["gcm_tmin"].values
    results["raw_pr"]   = fut_df["gcm_pr"].values

    df_out = pd.DataFrame(results).set_index("date")

    # -- Calibration validation plot (first run only) --------------------------
    # Apply the same correction to the calibration period to check quality
    bc_cal_tmax = dqm_temperature(obs_cal["obs_tmax"].values,
                                   hist_cal["gcm_tmax"].values,
                                   hist_cal["gcm_tmax"].values)
    bc_cal_pr   = dqm_precipitation(obs_cal["obs_pr"].values,
                                     hist_cal["gcm_pr"].values,
                                     hist_cal["gcm_pr"].values)

    calib_check = pd.DataFrame({
        "obs_tmax": obs_cal["obs_tmax"].values,
        "raw_tmax": hist_cal["gcm_tmax"].values,
        "bc_tmax":  bc_cal_tmax,
        "obs_pr":   obs_cal["obs_pr"].values,
        "raw_pr":   hist_cal["gcm_pr"].values,
        "bc_pr":    bc_cal_pr,
    }, index=common_idx)

    calib_check["month"] = calib_check.index.month

    for var, obs_col, raw_col, bc_col, ylabel in [
        ("tmax", "obs_tmax", "raw_tmax", "bc_tmax", "Temperature (°C)"),
        ("pr",   "obs_pr",   "raw_pr",   "bc_pr",   "Precipitation (mm/day)"),
    ]:
        obs_m = calib_check.groupby("month")[obs_col].mean()
        raw_m = calib_check.groupby("month")[raw_col].mean()
        bc_m  = calib_check.groupby("month")[bc_col].mean()
        plot_path = FIG_DIR / f"bc_calib_{station_key}_{model}_{var}.png"
        if not plot_path.exists():
            plot_monthly_comparison(obs_m, raw_m, bc_m, ylabel,
                                    station_key, model, scenario, var)

    return df_out


def main():
    print("=" * 65)
    print("Phase II - Bias Correction (DQM) for CMIP6 Projections")
    print("=" * 65)

    # Load observations once per station
    for station_key, station_excel in STATION_MAP.items():
        print(f"\n{'-'*55}")
        print(f"Station: {station_key}")
        obs_df = load_obs(station_excel)
        print(f"  Obs loaded: {len(obs_df):,} days")

        for model in MODELS:
            for scenario in SCENARIOS:
                out_path = OUT_DIR / f"{station_key}_{model}_{scenario}_bc.csv"
                if out_path.exists():
                    print(f"  [SKIP existing] {model} / {scenario}")
                    continue

                print(f"  Processing: {model} / {scenario} ...", end=" ")
                result = process_station_model_scenario(
                    station_key, model, scenario, obs_df
                )
                if result is not None:
                    result.to_csv(out_path)
                    print(f"saved ({len(result):,} days, {result.index[0].date()}-{result.index[-1].date()})")
                else:
                    print("skipped")

    # -- Summary statistics -----------------------------------------------------
    print(f"\n{'='*65}")
    csv_files = list(OUT_DIR.glob("*_bc.csv"))
    print(f"Bias-corrected files created: {len(csv_files)}")
    print(f"Output directory: {OUT_DIR}")

    # Quick stats table
    records = []
    for f in sorted(csv_files):
        parts = f.stem.split("_")
        station = parts[0]
        scenario = parts[-2]
        model_parts = parts[1:-2]
        model = "_".join(model_parts)
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        records.append({
            "Station": station, "Model": model, "Scenario": scenario,
            "Start": str(df.index[0].date()),
            "End": str(df.index[-1].date()),
            "Days": len(df),
            "Mean_Tmax_BC": round(df["bc_tmax"].mean(), 2),
            "Mean_Pr_BC": round(df["bc_pr"].mean(), 3),
        })

    if records:
        summary_df = pd.DataFrame(records)
        summary_path = OUT_DIR / "bias_correction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

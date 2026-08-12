"""
15_drought_indices.py
=====================
Computes standardized drought indices from bias-corrected BMA ensemble outputs:
  - SPI-3  (3-month Standardized Precipitation Index)
  - SPI-12 (12-month SPI)
  - SPEI-12 (12-month Standardized Precipitation-Evapotranspiration Index)
    using Thornthwaite PET estimation from monthly Tavg

Method:
  SPI:  Fit Gamma distribution to monthly precipitation accumulations
        -> standardize using Normal quantile transform (McKee et al., 1993)
  SPEI: Same as SPI but applied to (P - PET) water balance
        Fit Log-Logistic distribution (Vicente-Serrano et al., 2010)

Baseline for fitting:  2000-2020 (2000-2020 observed)
Application periods:   2021-2060 (near-term) and 2061-2100 (long-term)

Outputs:
  outputs/drought/drought_indices_{station}_{scenario}.csv
  outputs/drought/drought_changes_summary.csv
  outputs/drought/figures/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gamma, norm, fisk   # fisk = log-logistic

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
BMA_DIR   = BASE_DIR / "outputs" / "bma"
OBS_XLSX  = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR   = BASE_DIR / "outputs" / "drought"
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245", "ssp585"]
STATIONS  = ["Amol", "Gharakhil", "Sari"]
SCALES    = [3, 12]   # SPI accumulation scales

PERIODS = {
    "baseline":  ("2000-01-01", "2020-12-31"),
    "near_term": ("2021-01-01", "2060-12-31"),
    "long_term": ("2061-01-01", "2100-12-31"),
}

# LARS-WG period transitions: first 11 months of each new 20-yr block are
# contaminated because the 12-month rolling window crosses the period boundary.
# Mask them to NaN so they don't distort period statistics.
# (Only 33 of 960 future months = 3.4% affected)
LARS_WG_TRANSITION_MASKS = [
    ("2041-01-01", "2041-11-30"),
    ("2061-01-01", "2061-11-30"),
    ("2081-01-01", "2081-11-30"),
]


def mask_period_transitions(index_values: np.ndarray,
                            dates: pd.DatetimeIndex) -> np.ndarray:
    """Set to NaN the first 11 months after each LARS-WG period transition."""
    result = index_values.copy().astype(float)
    for start, end in LARS_WG_TRANSITION_MASKS:
        mask = (dates >= start) & (dates <= end)
        result[mask] = np.nan
    return result

# Drought classification thresholds (SPI/SPEI)
DROUGHT_CLASSES = {
    "Extreme Wet":   (2.0,  np.inf),
    "Severe Wet":    (1.5,  2.0),
    "Moderate Wet":  (1.0,  1.5),
    "Near Normal":   (-1.0, 1.0),
    "Moderate Dry":  (-1.5, -1.0),
    "Severe Dry":    (-2.0, -1.5),
    "Extreme Dry":   (-np.inf, -2.0),
}


# -- PET method selector -------------------------------------------------------
# Hargreaves (FAO-56) is used by default. The Thornthwaite (1948) scheme is
# retained for reference only: being a function of mean temperature alone, it is
# well documented to overestimate the PET response to warming and therefore to
# exaggerate future SPEI-based drought (Sheffield et al., 2012; Beguería et al.,
# 2014). Hargreaves additionally uses the diurnal temperature range and
# extraterrestrial radiation, giving a more physically constrained warming
# response while remaining data-parsimonious.
PET_METHOD = "hargreaves"


def _extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Daily extraterrestrial radiation Ra (MJ m-2 day-1), FAO-56 Eq. 21."""
    lat = np.radians(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))
    return Ra


def hargreaves_pet(tmax_monthly: np.ndarray, tmin_monthly: np.ndarray,
                   tavg_monthly: np.ndarray, dates: pd.DatetimeIndex,
                   lat_deg: float = 36.5) -> np.ndarray:
    """
    Monthly PET (mm) via Hargreaves (FAO-56 Eq. 52):
        ET0 = 0.0023 * Ra_mm * (Tmean + 17.8) * sqrt(Tmax - Tmin)   [mm day-1]
    Ra is evaluated at the mid-month day-of-year; the daily rate is multiplied by
    the number of days in the month.
    """
    s = dates.to_series()
    doy = s.apply(lambda d: d.replace(day=15).timetuple().tm_yday).values.astype(float)
    days_in_month = s.dt.days_in_month.values.astype(float)
    Ra = _extraterrestrial_radiation(doy, lat_deg)         # MJ m-2 day-1
    Ra_mm = 0.408 * Ra                                     # mm day-1 equivalent
    tr = np.clip(tmax_monthly - tmin_monthly, 0.0, None)
    et0_daily = 0.0023 * Ra_mm * (tavg_monthly + 17.8) * np.sqrt(tr)
    return np.clip(et0_daily * days_in_month, 0.0, None)


# -- PET (Thornthwaite - reference only) ----------------------------------------
def thornthwaite_pet(tavg_monthly: np.ndarray, lat_deg: float = 36.5) -> np.ndarray:
    """
    Estimate monthly PET (mm) using Thornthwaite (1948) method.
    tavg_monthly: array of monthly mean temperatures (°C)
    lat_deg: latitude for day-length correction
    """
    # Heat index I
    tavg_pos = np.clip(tavg_monthly, 0.5, None)
    I = np.sum((tavg_pos.reshape(-1, 12) / 5) ** 1.514, axis=1)  # annual
    # Repeat I for each month
    I_monthly = np.repeat(I, 12)[:len(tavg_monthly)]

    # Exponent a
    a = 6.75e-7 * I_monthly**3 - 7.71e-5 * I_monthly**2 + 1.792e-2 * I_monthly + 0.49239

    # Unadjusted PET
    pet_unadj = np.where(
        tavg_monthly < 0, 0,
        np.where(tavg_monthly > 26.5,
                 -415.85 + 32.24 * tavg_monthly - 0.43 * tavg_monthly**2,
                 16 * (10 * tavg_monthly / (I_monthly + 1e-8)) ** a)
    )

    # Day-length correction (approximate, mid-month)
    months_arr = np.tile(np.arange(1, 13), len(tavg_monthly) // 12 + 1)[:len(tavg_monthly)]
    # Monthly correction factors for lat ~36.5°N (interpolated from standard tables)
    correction_36N = np.array([0.87, 0.85, 1.03, 1.10, 1.21, 1.23,
                                1.22, 1.17, 1.05, 0.97, 0.86, 0.85])
    K = correction_36N[(months_arr - 1) % 12]
    pet = pet_unadj * K
    return np.clip(pet, 0, None)


# -- SPI Calculation -----------------------------------------------------------
def compute_spi(pr_monthly: np.ndarray, scale: int,
                 calib_mask: np.ndarray) -> np.ndarray:
    """
    Compute SPI at given accumulation scale.
    Fit Gamma on calibration period; apply to full series.
    Returns SPI array (same length as input, NaN for first scale-1 values).
    """
    n = len(pr_monthly)
    # Rolling sum
    pr_acc = np.array([
        pr_monthly[max(0, i - scale + 1):i + 1].sum() if i >= scale - 1 else np.nan
        for i in range(n)
    ])

    spi = np.full(n, np.nan)

    # Fit Gamma on calibration data (only valid, positive values)
    calib_acc = pr_acc[calib_mask & ~np.isnan(pr_acc)]
    calib_acc_pos = calib_acc[calib_acc > 0]
    if len(calib_acc_pos) < 24:
        return spi

    # Probability of zero precipitation
    p0 = np.mean(calib_acc == 0)

    # Fit Gamma distribution
    try:
        shape, loc, scale_g = gamma.fit(calib_acc_pos, floc=0)
    except Exception:
        return spi

    # Transform to SPI
    for i in range(n):
        if np.isnan(pr_acc[i]):
            continue
        if pr_acc[i] == 0:
            H = p0
        else:
            H = p0 + (1 - p0) * gamma.cdf(pr_acc[i], shape, loc, scale_g)
        H = np.clip(H, 1e-6, 1 - 1e-6)
        spi[i] = norm.ppf(H)

    return spi


# -- SPEI Calculation ----------------------------------------------------------
def compute_spei(pr_monthly: np.ndarray, pet_monthly: np.ndarray,
                  scale: int, calib_mask: np.ndarray) -> np.ndarray:
    """
    Compute SPEI-scale using Log-Logistic distribution (Vicente-Serrano et al., 2010).
    """
    n = len(pr_monthly)
    D = pr_monthly - pet_monthly  # water balance

    # Rolling accumulation
    D_acc = np.array([
        D[max(0, i - scale + 1):i + 1].sum() if i >= scale - 1 else np.nan
        for i in range(n)
    ])

    spei = np.full(n, np.nan)
    calib_D = D_acc[calib_mask & ~np.isnan(D_acc)]
    if len(calib_D) < 24:
        return spei

    # Fit Log-Logistic (fisk in scipy)
    try:
        c, loc, sc = fisk.fit(calib_D)
    except Exception:
        return spei

    for i in range(n):
        if np.isnan(D_acc[i]):
            continue
        p = fisk.cdf(D_acc[i], c, loc, sc)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        spei[i] = norm.ppf(p)

    return spei


# -- Drought class frequency ---------------------------------------------------
def drought_frequency(index_values: np.ndarray) -> dict:
    """Return % time in each drought/wet class."""
    valid = index_values[~np.isnan(index_values)]
    if len(valid) == 0:
        return {}
    result = {}
    for cls, (lo, hi) in DROUGHT_CLASSES.items():
        result[cls] = round(100 * np.mean((valid >= lo) & (valid < hi)), 1)
    return result


# -- Build observed monthly series ---------------------------------------------
def load_obs_monthly(station_excel: str) -> pd.DataFrame:
    df = pd.read_excel(OBS_XLSX, sheet_name="All_Stations", parse_dates=["date"])
    df = df[df["station_name"] == station_excel].copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["year", "month"]).agg(
        pr_monthly=("rrr24", "sum"),
        tmax_monthly=("tmax", "mean"),
        tmin_monthly=("tmin", "mean"),
        tavg_monthly=("tmax", lambda x: (x + df.loc[x.index, "tmin"]).mean() / 2),
    ).reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly.sort_values("date").reset_index(drop=True)


STATION_EXCEL_MAP = {
    "Amol":      "Amol",
    "Gharakhil": "Gharakhil",
    "Sari":      "Sari (Dasht-E-Naz Airport)",
}


# -- Main ----------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Phase IV - Drought Indices (SPI-3, SPI-12, SPEI-12)")
    print("=" * 65)

    all_summaries = []

    for station in STATIONS:
        print(f"\n{'-'*50}")
        print(f"Station: {station}")

        # Load observed monthly (baseline)
        obs_monthly = load_obs_monthly(STATION_EXCEL_MAP[station])
        obs_dates = pd.to_datetime(obs_monthly["date"])

        for scenario in SCENARIOS:
            bma_path = BMA_DIR / f"bma_climate_{station}_{scenario}_monthly.csv"
            if not bma_path.exists():
                print(f"  SKIP: BMA file not found: {bma_path.name}")
                print("  (Run 14_bma_ensemble.py first)")
                continue

            bma_df = pd.read_csv(bma_path, parse_dates=["date"], index_col="date")

            # Build full time series: obs baseline + future BMA
            obs_pr   = obs_monthly["pr_monthly"].values
            obs_tmax = obs_monthly["tmax_monthly"].values
            obs_tmin = obs_monthly["tmin_monthly"].values
            fut_pr   = bma_df["pr_monthly_bma"].values
            fut_tmax = bma_df["tmax_monthly_bma"].values
            fut_tmin = bma_df["tmin_monthly_bma"].values

            full_pr    = np.concatenate([obs_pr, fut_pr])
            full_tmax  = np.concatenate([obs_tmax, fut_tmax])
            full_tmin  = np.concatenate([obs_tmin, fut_tmin])
            full_t     = (full_tmax + full_tmin) / 2.0
            full_dates = pd.DatetimeIndex(
                list(obs_dates) + list(bma_df.index)
            )

            # Calibration mask (2000-2020)
            calib_mask = (full_dates >= PERIODS["baseline"][0]) & \
                         (full_dates <= PERIODS["baseline"][1])

            # PET (Hargreaves by default; see PET_METHOD)
            if PET_METHOD == "hargreaves":
                pet = hargreaves_pet(full_tmax, full_tmin, full_t, full_dates)
            else:
                pet = thornthwaite_pet(full_t)

            # Compute indices
            spi3  = compute_spi(full_pr, 3,  calib_mask)
            spi12 = compute_spi(full_pr, 12, calib_mask)
            spei12 = compute_spei(full_pr, pet, 12, calib_mask)

            # Recenter SPEI on calibration period (log-logistic fit bias)
            calib_spei = spei12[calib_mask & ~np.isnan(spei12)]
            if len(calib_spei) >= 24:
                spei12 = spei12 - np.nanmean(calib_spei)

            # Mask period-transition months (contaminated rolling window)
            spi3   = mask_period_transitions(spi3,   full_dates)
            spi12  = mask_period_transitions(spi12,  full_dates)
            spei12 = mask_period_transitions(spei12, full_dates)

            df_out = pd.DataFrame({
                "date":   full_dates,
                "pr_mm":  full_pr,
                "pet_mm": pet,
                "spi3":   spi3,
                "spi12":  spi12,
                "spei12": spei12,
            })

            out_path = OUT_DIR / f"drought_indices_{station}_{scenario}.csv"
            df_out.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

            # -- Period-wise drought frequency ---------------------------------
            for period_name, (p_start, p_end) in PERIODS.items():
                mask = (df_out["date"] >= p_start) & (df_out["date"] <= p_end)
                period_data = df_out[mask]

                for idx_name in ["spi12", "spei12"]:
                    freq = drought_frequency(period_data[idx_name].values)
                    summary = {
                        "station": station, "scenario": scenario,
                        "period": period_name, "index": idx_name,
                        "mean": round(period_data[idx_name].mean(), 3),
                        "n_extreme_dry": int(np.sum(period_data[idx_name] < -2.0)),
                        "n_severe_dry":  int(np.sum(period_data[idx_name] < -1.5)),
                    }
                    summary.update(freq)
                    all_summaries.append(summary)

            # -- Plot ----------------------------------------------------------
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            colors = {"positive": "steelblue", "negative": "firebrick"}

            for ax, (idx_col, title) in zip(axes, [
                ("spi3",  "SPI-3"),
                ("spi12", "SPI-12"),
                ("spei12","SPEI-12"),
            ]):
                vals = df_out[idx_col].values
                dates = df_out["date"]
                pos = np.where(vals >= 0, vals, 0)
                neg = np.where(vals < 0, vals, 0)
                ax.bar(dates, pos, color=colors["positive"], width=20, alpha=0.7)
                ax.bar(dates, neg, color=colors["negative"], width=20, alpha=0.7)
                ax.axhline(-1.5, color="darkred",  ls="--", lw=0.8, alpha=0.7)
                ax.axhline(-2.0, color="darkred",  ls="-",  lw=0.8, alpha=0.7)
                ax.axhline( 1.5, color="darkblue", ls="--", lw=0.8, alpha=0.7)
                ax.axvline(pd.Timestamp("2021-01-01"), color="gray", ls=":", alpha=0.5)
                ax.axvline(pd.Timestamp("2061-01-01"), color="gray", ls="--", alpha=0.5)
                ax.set_ylabel(title)
                ax.set_ylim(-3.5, 3.5)
                ax.grid(True, alpha=0.2)

            axes[0].set_title(f"Drought Indices - {station} ({scenario.upper()})")
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"drought_{station}_{scenario}.png", dpi=150)
            plt.close(fig)

    # -- Save summary ----------------------------------------------------------
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(OUT_DIR / "drought_changes_summary.csv", index=False)
        print(f"\nSummary saved: {OUT_DIR / 'drought_changes_summary.csv'}")
        print(summary_df[["station","scenario","period","index",
                           "mean","n_extreme_dry"]].to_string(index=False))

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()

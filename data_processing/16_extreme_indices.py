"""
16_extreme_indices.py
=====================
Computes ETCCDI precipitation extreme indices from bias-corrected daily projections.

Indices computed (ETCCDI standard definitions):
  R95p   - Annual total precipitation when daily P > 95th percentile (mm/year)
  R99p   - Annual total precipitation when daily P > 99th percentile (mm/year)
  Rx1day - Maximum 1-day precipitation (mm)
  Rx5day - Maximum consecutive 5-day precipitation (mm)
  CDD    - Maximum consecutive dry days (P < 1 mm)
  CWD    - Maximum consecutive wet days (P >= 1 mm)
  SDII   - Simple daily intensity index (mm/wet day)

Also computes temperature extremes:
  TXx   - Annual maximum of Tmax (°C)
  TNn   - Annual minimum of Tmin (°C)
  TX90p - % days with Tmax > 90th percentile (warm days)
  TN10p - % days with Tmin < 10th percentile (cool nights)
  WSDI  - Warm spell duration index (days)

Baseline for percentile computation: 2000-2014 (calibration period)
Analysis periods: 2000-2020 (obs), 2021-2060 (near), 2061-2100 (long)

Outputs:
  outputs/extremes/annual_extremes_{station}_{model}_{scenario}.csv
  outputs/extremes/bma_extremes_{station}_{scenario}_summary.csv
  outputs/extremes/figures/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
BC_DIR    = BASE_DIR / "outputs" / "bias_corrected"
OBS_XLSX  = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
EVAL_DIR  = BASE_DIR / "outputs" / "evaluation"
OUT_DIR   = BASE_DIR / "outputs" / "extremes"
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCENARIOS = ["ssp245", "ssp585"]
STATIONS  = ["Amol", "Gharakhil", "Sari"]

CALIB_START = "2000-01-01"
CALIB_END   = "2014-12-31"
WET_THRESH  = 1.0   # mm/day for wet day (ETCCDI standard)

STATION_EXCEL_MAP = {
    "Amol":      "Amol",
    "Gharakhil": "Gharakhil",
    "Sari":      "Sari (Dasht-E-Naz Airport)",
}


# -- Index computation functions -----------------------------------------------
def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling sum."""
    result = np.full(len(arr), np.nan)
    cs = np.cumsum(np.concatenate([[0], arr]))
    result[window - 1:] = cs[window:] - cs[:-window]
    return result


def max_consecutive(arr: np.ndarray, condition: np.ndarray) -> float:
    """Maximum length of consecutive True values in condition array."""
    max_run, current = 0, 0
    for v in condition:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return float(max_run)


def calendar_day_threshold(dates: pd.Series, values: np.ndarray,
                           pct: float, window: int = 5) -> np.ndarray:
    """
    ETCCDI calendar-day percentile threshold (Zhang et al., 2005).

    For each calendar day d, pool all base-period values within a centred
    `window`-day window (default 5 days) across all years and take the `pct`
    percentile. Returns an array of length 366 indexed by day-of-year
    (index 0 unused; day 366 reuses day 365).
    """
    doy = dates.dt.dayofyear.values.copy()
    doy[doy > 365] = 365
    half = window // 2
    thr = np.full(366, np.nan)
    for d in range(1, 366):
        offsets = [((d - 1 + k) % 365) + 1 for k in range(-half, half + 1)]
        mask = np.isin(doy, offsets)
        if mask.any():
            thr[d] = np.percentile(values[mask], pct)
    thr[0] = thr[1]
    return thr


def compute_annual_extremes(daily_df: pd.DataFrame,
                             pr_95th: float, pr_99th: float,
                             tmax_90th, tmin_10th) -> pd.DataFrame:
    """
    Compute annual extreme indices from a daily DataFrame.
    Expected columns: date, pr (mm/day), tmax (°C), tmin (°C).

    pr_95th / pr_99th : scalar wet-day percentile thresholds (ETCCDI: percentile
        of precipitation on wet days, RR >= 1 mm, over the base period).
    tmax_90th / tmin_10th : either a scalar, or a length-366 calendar-day
        threshold array (ETCCDI day-of-year percentile with moving window).
    """
    tmax_is_array = np.ndim(tmax_90th) > 0
    tmin_is_array = np.ndim(tmin_10th) > 0

    records = []
    for year, grp in daily_df.groupby(daily_df["date"].dt.year):
        pr   = grp["pr"].values
        tmax = grp["tmax"].values
        tmin = grp["tmin"].values
        n    = len(pr)

        wet_mask = pr >= WET_THRESH
        dry_mask = pr < WET_THRESH

        # Precipitation extremes
        r95p   = float(np.sum(pr[pr > pr_95th]))
        r99p   = float(np.sum(pr[pr > pr_99th]))
        rx1day = float(np.nanmax(pr)) if n > 0 else np.nan
        rs5    = rolling_sum(pr, 5)
        rx5day = float(np.nanmax(rs5[4:])) if len(rs5) > 4 else np.nan
        cdd    = max_consecutive(pr, dry_mask)
        cwd    = max_consecutive(pr, wet_mask)
        sdii   = float(np.mean(pr[wet_mask])) if wet_mask.any() else 0.0

        # Day-of-year thresholds for percentile-based temperature indices
        doy = grp["date"].dt.dayofyear.values.copy()
        doy[doy > 365] = 365
        tmax_thr_day = tmax_90th[doy] if tmax_is_array else tmax_90th
        tmin_thr_day = tmin_10th[doy] if tmin_is_array else tmin_10th

        # Temperature extremes
        txx    = float(np.nanmax(tmax))
        tnn    = float(np.nanmin(tmin))
        tx90p  = 100.0 * float(np.mean(tmax > tmax_thr_day))
        tn10p  = 100.0 * float(np.mean(tmin < tmin_thr_day))

        # WSDI: at least 6 consecutive days with Tmax > calendar-day 90th pct
        wsdi_days = 0
        run = 0
        for v in (tmax > tmax_thr_day):
            if v:
                run += 1
            else:
                if run >= 6:
                    wsdi_days += run
                run = 0
        if run >= 6:
            wsdi_days += run

        records.append({
            "year":   year,
            "R95p":   round(r95p,   1),
            "R99p":   round(r99p,   1),
            "Rx1day": round(rx1day, 1),
            "Rx5day": round(rx5day, 1),
            "CDD":    round(cdd,    0),
            "CWD":    round(cwd,    0),
            "SDII":   round(sdii,   2),
            "TXx":    round(txx,    1),
            "TNn":    round(tnn,    1),
            "TX90p":  round(tx90p,  1),
            "TN10p":  round(tn10p,  1),
            "WSDI":   round(wsdi_days, 0),
        })

    return pd.DataFrame(records)


# -- Load observed daily --------------------------------------------------------
def load_obs_daily(station_excel: str) -> pd.DataFrame:
    df = pd.read_excel(OBS_XLSX, sheet_name="All_Stations", parse_dates=["date"])
    df = df[df["station_name"] == station_excel].copy()
    return df.rename(columns={"rrr24": "pr", "tmax": "tmax", "tmin": "tmin"})[
        ["date", "pr", "tmax", "tmin"]
    ].sort_values("date").reset_index(drop=True)


# -- BMA-weighted index --------------------------------------------------------
def bma_extreme_summary(station: str, scenario: str, weights_dict: dict,
                        obs_pr_95: float, obs_pr_99: float,
                        obs_tmax_90: float, obs_tmin_10: float) -> pd.DataFrame:
    """
    Compute BMA-weighted annual extremes across all models.
    Uses observed calibration percentiles (gap-filled baseline) as reference thresholds,
    since BC future files start from 2015 and have no historical overlap.
    """
    all_annual = {}
    for model in MODELS:
        bc_path = BC_DIR / f"{station}_{model}_{scenario}_bc.csv"
        if not bc_path.exists():
            continue
        bc = pd.read_csv(bc_path, parse_dates=["date"])
        bc = bc.rename(columns={"bc_pr": "pr", "bc_tmax": "tmax", "bc_tmin": "tmin"})

        # Use observed calibration percentiles (ETCCDI standard: baseline = obs period)
        ann = compute_annual_extremes(bc, obs_pr_95, obs_pr_99, obs_tmax_90, obs_tmin_10)
        all_annual[model] = ann.set_index("year")

    if len(all_annual) < 2:
        return None

    idx_cols = ["R95p", "R99p", "Rx1day", "Rx5day", "CDD", "CWD", "SDII",
                "TXx", "TNn", "TX90p", "TN10p", "WSDI"]
    models_avail = list(all_annual.keys())
    wts = np.array([weights_dict.get(m, 0) for m in models_avail])
    wts = wts / wts.sum()

    # Common years
    common_years = sorted(set.intersection(*[set(df.index) for df in all_annual.values()]))
    bma_records = []
    for yr in common_years:
        row = {"year": yr}
        for col in idx_cols:
            vals = np.array([all_annual[m].loc[yr, col] for m in models_avail
                             if col in all_annual[m].columns and yr in all_annual[m].index])
            if len(vals) == len(models_avail):
                row[f"{col}_mean"]  = float(np.average(vals, weights=wts))
                row[f"{col}_lower"] = float(np.percentile(vals, 5))
                row[f"{col}_upper"] = float(np.percentile(vals, 95))
        bma_records.append(row)

    return pd.DataFrame(bma_records)


# -- Period comparison plot ----------------------------------------------------
def plot_period_comparison(bma_df: pd.DataFrame, indices: list,
                            station: str, scenario: str):
    periods = {
        "Baseline\n(2000-2020)":   (2000, 2020),
        "Near-term\n(2021-2060)":  (2021, 2060),
        "Long-term\n(2061-2100)":  (2061, 2100),
    }

    n_idx = len(indices)
    fig, axes = plt.subplots(1, n_idx, figsize=(4 * n_idx, 5))
    if n_idx == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        col = f"{idx}_mean"
        if col not in bma_df.columns:
            continue
        period_means = []
        period_labels = []
        for label, (y0, y1) in periods.items():
            mask = (bma_df["year"] >= y0) & (bma_df["year"] <= y1)
            if mask.any():
                period_means.append(bma_df.loc[mask, col].mean())
                period_labels.append(label)
        bars = ax.bar(period_labels, period_means, color=["gray", "steelblue", "firebrick"],
                      alpha=0.8)
        ax.set_title(idx)
        ax.set_ylabel(idx)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, period_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Extreme Indices - {station} ({scenario.upper()})", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"extremes_periods_{station}_{scenario}.png", dpi=150)
    plt.close(fig)


# -- Main ----------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Phase IV - ETCCDI Extreme Climate Indices")
    print("=" * 65)

    # Load BMA weights
    weights_path = BASE_DIR / "outputs" / "bma" / "bma_weights.csv"
    if weights_path.exists():
        w_df = pd.read_csv(weights_path)
        weights_dict = dict(zip(w_df["model"], w_df["bma_weight"]))
    else:
        # Fallback: equal weights
        weights_dict = {m: 1/len(MODELS) for m in MODELS}
        print("WARNING: BMA weights not found. Using equal weights.")

    all_obs_extremes = []
    all_summaries    = []

    for station in STATIONS:
        print(f"\n{'-'*50}")
        print(f"Station: {station}")

        # -- Observed extremes (baseline) --------------------------------------
        obs = load_obs_daily(STATION_EXCEL_MAP[station])
        obs_calib = obs[(obs["date"] >= CALIB_START) & (obs["date"] <= CALIB_END)]
        base = obs_calib if len(obs_calib) >= 365 else obs
        if len(obs_calib) < 365:
            print(f"  WARNING: insufficient calibration obs for {station}; using full record")

        # ETCCDI precipitation thresholds: percentile of WET-DAY precip (RR >= 1 mm)
        wet = base["pr"].values[base["pr"].values >= WET_THRESH]
        pr_95 = float(np.percentile(wet, 95))
        pr_99 = float(np.percentile(wet, 99))
        # ETCCDI temperature thresholds: calendar-day percentile, 5-day window
        tmax_90 = calendar_day_threshold(base["date"], base["tmax"].values, 90)
        tmin_10 = calendar_day_threshold(base["date"], base["tmin"].values, 10)

        obs_annual = compute_annual_extremes(obs, pr_95, pr_99, tmax_90, tmin_10)
        obs_annual["station"] = station
        obs_annual["source"]  = "observed"
        all_obs_extremes.append(obs_annual)

        print(f"  ETCCDI thresholds: wet-day P95={pr_95:.1f}, P99={pr_99:.1f} mm/day, "
              f"Tmax90(ann.mean)={np.nanmean(tmax_90):.1f}°C, "
              f"Tmin10(ann.mean)={np.nanmean(tmin_10):.1f}°C")

        for scenario in SCENARIOS:
            print(f"\n  Scenario: {scenario}")
            bma_ann = bma_extreme_summary(station, scenario, weights_dict,
                                          pr_95, pr_99, tmax_90, tmin_10)

            if bma_ann is None:
                print(f"    SKIP: insufficient BC data. Run 11_bias_correction_dqm.py first.")
                continue

            out_path = OUT_DIR / f"bma_extremes_{station}_{scenario}_annual.csv"
            bma_ann.to_csv(out_path, index=False)
            print(f"    Saved: {out_path.name} ({len(bma_ann)} years)")

            # Period change summary. Baseline is taken from the OBSERVED extremes
            # (2000-2020); near-/long-term from the BMA future extremes, which only
            # span 2021-2100.
            obs_base = obs_annual[(obs_annual["year"] >= 2000) &
                                  (obs_annual["year"] <= 2020)]
            fut_periods = {"near_term": (2021, 2060), "long_term": (2061, 2100)}
            for idx in ["R95p", "Rx5day", "CDD", "CWD", "TXx", "TX90p"]:
                col = f"{idx}_mean"
                if col not in bma_ann.columns or idx not in obs_base.columns:
                    continue
                row = {"station": station, "scenario": scenario, "index": idx,
                       "baseline": round(obs_base[idx].mean(), 2)}
                for pname, (y0, y1) in fut_periods.items():
                    mask = (bma_ann["year"] >= y0) & (bma_ann["year"] <= y1)
                    if mask.any():
                        row[pname] = round(bma_ann.loc[mask, col].mean(), 2)
                if "long_term" in row and row["baseline"] != 0:
                    row["long_pct_change"] = round(
                        100 * (row["long_term"] - row["baseline"]) / abs(row["baseline"]), 1)
                all_summaries.append(row)

            # Plot
            plot_period_comparison(
                bma_ann, ["R95p", "CDD", "CWD", "TXx", "TX90p"],
                station, scenario
            )

    # -- Save observed extremes ------------------------------------------------
    if all_obs_extremes:
        obs_df = pd.concat(all_obs_extremes, ignore_index=True)
        obs_df.to_csv(OUT_DIR / "observed_extremes_annual.csv", index=False)

    # -- Save change summary ---------------------------------------------------
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(OUT_DIR / "extremes_period_changes.csv", index=False)
        print(f"\nPeriod changes summary:")
        print(summary_df[["station","scenario","index","baseline",
                           "near_term","long_term","long_pct_change"]].to_string(index=False))

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()

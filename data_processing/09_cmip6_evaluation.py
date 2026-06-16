#!/usr/bin/env python3
"""
09_cmip6_evaluation.py
======================
CMIP6 GCM Historical Evaluation - Haraz Watershed Study

Evaluates six CMIP6 models against the HBCD-2020 observational baseline
(2000-2014) at three synoptic stations (Amol, Gharakhil, Sari).

Evaluation basis: MONTHLY TIME SERIES (n ~ 180 month-pairs per
station/variable), yielding statistically robust RMSE, MAE, PBIAS,
NSE, KGE, and Pearson r - consistent with Section 3.5 of the manuscript.

Taylor diagrams use the same monthly time series statistics (corr, alpha).

Metrics reported (per Section 3.5):
    RMSE, MAE, PBIAS, NSE, KGE, r - Taylor diagram visualization

Reference:
    Rahab-Rajaei, S., Motiee, H. (2025). Hydroclimatic Projections Under
    CMIP6 SSP Scenarios in the Haraz Watershed, Northern Iran:
    A Hybrid BiLSTM-BMA Framework. [Target: ISI Q1 Journal]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CMIP6_DIR    = DATA_DIR / "cmip6_gcs"
OBS_FILE     = PROJECT_ROOT / "ClimateData_GapFilled_2000_2020.xlsx"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS   = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
STATIONS = ["Amol", "Gharakhil", "Sari"]
VARIABLES = {
    "tmax": ("tmax", "Tmax (\u00b0C)"),
    "tmin": ("tmin", "Tmin (\u00b0C)"),
    "pr":   ("pr",   "Precipitation (mm/day)"),
}

START_YEAR = 2000
END_YEAR   = 2014

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_observed_data() -> pd.DataFrame:
    """Load HBCD-2020 gap-filled dataset with standardised station names."""
    print(f"Loading observed data: {OBS_FILE}")
    obs = pd.read_excel(OBS_FILE, sheet_name="All_Stations")
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.rename(columns={"station_name": "station", "rrr24": "pr"})

    obs["station"] = obs["station"].astype(str).str.strip()
    obs.loc[obs["station"].str.contains("Sari", case=False, na=False), "station"] = "Sari"

    mask = (obs["date"].dt.year >= START_YEAR) & (obs["date"].dt.year <= END_YEAR)
    obs  = obs[mask].copy()
    obs["year_int"]  = obs["date"].dt.year
    obs["month_int"] = obs["date"].dt.month
    return obs


def load_cmip6_model(model_name: str, station_name: str) -> pd.DataFrame | None:
    """
    Load CMIP6 historical CSV with smart column mapping and unit conversion.
    Returns DataFrame with columns [year_int, month_int, tmax, tmin, pr]
    or None if the file is not found.
    """
    csv_path = CMIP6_DIR / "historical" / model_name / f"{station_name}_{model_name}_historical.csv"

    if not csv_path.exists() and station_name == "Sari":
        for alt in [
            f"Sari_DashtENaz_Airport_{model_name}_historical.csv",
            f"Sari (Dasht-E-Naz Airport)_{model_name}_historical.csv",
        ]:
            alt_path = CMIP6_DIR / "historical" / model_name / alt
            if alt_path.exists():
                csv_path = alt_path
                break

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Normalise column names across all six models (handles UKESM1-0-LL quirks)
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if "tasmax" in cl or "tmax" in cl or ("max" in cl and "ta" in cl):
            rename[col] = "tmax"
        elif "tasmin" in cl or "tmin" in cl or ("min" in cl and "ta" in cl):
            rename[col] = "tmin"
        elif cl.startswith("pr") or "rain" in cl or "rrr" in cl:
            rename[col] = "pr"
    df = df.rename(columns=rename)

    # Derive tmax when a model provides daily-mean tas + tmin but no tmax
    # (UKESM1-0-LL edge case). Use the physical identity tas = (tmax+tmin)/2
    # => tmax = 2*tas - tmin. This is consistent with the bias-correction step
    # (11_bias_correction_dqm.py) and preserves the true diurnal variance,
    # unlike a constant additive offset.
    if "tmax" not in df.columns and "tmin" in df.columns:
        tas_col = next((c for c in df.columns
                        if c.lower() in ("tas_c", "tas", "tavg", "tas_degc")), None)
        if tas_col is not None:
            df["tmax"] = 2.0 * df[tas_col] - df["tmin"]
        else:
            df["tmax"] = df["tmin"] + 6.2  # last-resort fallback (no tas field)

    # Parse date -> year_int, month_int
    if "date" in df.columns:
        ds = df["date"].astype(str)
        df["year_int"]  = ds.str.slice(0, 4).astype(int)
        df["month_int"] = ds.str.slice(5, 7).astype(int)
    elif "year" in df.columns:
        df["year_int"]  = df["year"].astype(int)
        df["month_int"] = df["month"].astype(int) if "month" in df.columns else 1
    else:
        df["year_int"]  = START_YEAR
        df["month_int"] = 1

    df = df[(df["year_int"] >= START_YEAR) & (df["year_int"] <= END_YEAR)].copy()

    # Auto-correct units: K -> °C; kg m⁻² s⁻¹ -> mm day⁻¹
    for col in ["tmax", "tmin"]:
        if col in df.columns and df[col].dropna().mean() > 100:
            df[col] = df[col] - 273.15
    if "pr" in df.columns and df["pr"].dropna().mean() < 0.1:
        df["pr"] = df["pr"] * 86_400.0

    for col in ["tmax", "tmin", "pr"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[["year_int", "month_int", "tmax", "tmin", "pr"]].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """
    Compute standard hydroclimatic evaluation metrics.

    Parameters
    ----------
    obs : observed monthly values (already NaN-filtered)
    sim : simulated monthly values (already NaN-filtered)

    Returns
    -------
    dict with: n_samples, obs_mean, sim_mean, rmse, mae, bias, pbias,
               corr, nse, kge, index_of_agreement, alpha, beta
    """
    n = len(obs)
    if n < 30:
        warnings.warn(
            f"n={n} is below the recommended minimum of 30 for robust "
            "NSE / KGE estimates. Consider extending the evaluation period.",
            UserWarning, stacklevel=2,
        )

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    diff     = sim - obs

    rmse  = np.sqrt(np.mean(diff ** 2))
    mae   = np.mean(np.abs(diff))
    bias  = sim_mean - obs_mean
    pbias = 100.0 * bias / obs_mean if obs_mean != 0 else np.nan

    # Sample standard deviations (ddof=1)
    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)

    if obs_std > 0 and sim_std > 0:
        corr  = np.corrcoef(obs, sim)[0, 1]
        alpha = sim_std / obs_std          # variability ratio (KGE component)
    else:
        corr = alpha = np.nan

    obs_var = np.var(obs, ddof=1)
    nse = (1 - np.sum(diff ** 2) / np.sum((obs - obs_mean) ** 2)
           if obs_var > 0 else np.nan)

    if obs_mean > 0 and not np.isnan(corr):
        beta = sim_mean / obs_mean         # bias ratio (KGE component)
        kge  = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        beta = kge = np.nan

    # Willmott index of agreement
    denom = np.sum((np.abs(sim - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    d = 1 - np.sum(diff ** 2) / denom if denom > 0 else np.nan

    return {
        "n_samples": n, "obs_mean": obs_mean, "sim_mean": sim_mean,
        "rmse": rmse, "mae": mae, "bias": bias, "pbias": pbias,
        "corr": corr, "nse": nse, "kge": kge,
        "index_of_agreement": d, "alpha": alpha, "beta": beta,
    }

# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def evaluate_all_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all CMIP6 models against HBCD-2020 on MONTHLY TIME SERIES.

    Observations and simulations are aggregated to calendar-month means
    within each year (Jan 2000 ... Dec 2014), yielding n ~ 180 paired data
    points per station / variable before NaN filtering.  This is consistent
    with the methodology described in Section 3.5 of the manuscript.
    """
    print("=" * 60)
    print("CMIP6 GCM EVALUATION - Monthly Time Series (n ~ 180)")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print("=" * 60)

    obs_raw    = load_observed_data()
    all_metrics: list[dict] = []
    all_data:   list[pd.DataFrame] = []

    for model in MODELS:
        print(f"\nProcessing {model} ...")
        for station in STATIONS:
            print(f"  Station: {station}")
            try:
                obs_station = obs_raw[obs_raw["station"] == station].copy()
                sim_station = load_cmip6_model(model, station)

                if obs_station.empty or sim_station is None:
                    print(f"    [skip] No data for {model} / {station}")
                    continue

                # -- Monthly aggregation -------------------------------------
                obs_monthly = (
                    obs_station
                    .groupby(["year_int", "month_int"])[["tmax", "tmin", "pr"]]
                    .mean()
                    .reset_index()
                )
                sim_monthly = (
                    sim_station
                    .groupby(["year_int", "month_int"])[["tmax", "tmin", "pr"]]
                    .mean()
                    .reset_index()
                )

                # Inner join: only matched year-month pairs are retained
                merged = pd.merge(
                    obs_monthly, sim_monthly,
                    on=["year_int", "month_int"],
                    suffixes=("_obs", "_sim"),
                ).sort_values(["year_int", "month_int"]).reset_index(drop=True)

                if merged.empty:
                    print(f"    [skip] Merge empty for {model} / {station}")
                    continue

                # Reconstruct a date column for time series plots
                merged["date"] = pd.to_datetime({
                    "year":  merged["year_int"],
                    "month": merged["month_int"],
                    "day":   1,
                })

                # -- Collect time series data for plots ----------------------
                for var_key in VARIABLES:
                    o_col = f"{var_key}_obs"
                    s_col = f"{var_key}_sim"
                    if o_col not in merged.columns or s_col not in merged.columns:
                        continue
                    all_data.append(pd.DataFrame({
                        "date":     merged["date"].values,
                        "obs":      merged[o_col].values,
                        "sim":      merged[s_col].values,
                        "variable": var_key,
                        "model":    model,
                        "station":  station,
                    }))

                # -- Compute and store metrics -------------------------------
                for var_key, (_, var_label) in VARIABLES.items():
                    o_col = f"{var_key}_obs"
                    s_col = f"{var_key}_sim"
                    if o_col not in merged.columns or s_col not in merged.columns:
                        continue

                    o_vals = merged[o_col].values
                    s_vals = merged[s_col].values
                    valid  = ~np.isnan(o_vals) & ~np.isnan(s_vals)
                    if valid.sum() < 6:
                        continue

                    m = compute_metrics(o_vals[valid], s_vals[valid])
                    m.update({
                        "model":          model,
                        "station":        station,
                        "variable":       var_key,
                        "variable_label": var_label,
                        "period":         f"{START_YEAR}-{END_YEAR}",
                        "n_months":       int(valid.sum()),
                    })
                    all_metrics.append(m)

            except Exception as exc:
                print(f"    [error] {model} / {station}: {exc}")
                continue

    metrics_df = pd.DataFrame(all_metrics)
    data_df    = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    return metrics_df, data_df

# -----------------------------------------------------------------------------
# Skill scores
# -----------------------------------------------------------------------------

def calculate_skill_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite skill scores by min-max normalising six metrics.

    Higher-is-better (corr, NSE, KGE): normalised to [0, 1].
    Lower-is-better (RMSE, MAE, |PBIAS|): inverted normalisation.

    Note: |PBIAS| is used so that cold and warm biases of equal magnitude
    receive equal penalty - preventing sign-dependent distortion of skill.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    skill_df = metrics_df.copy()
    skill_df["abs_pbias"] = skill_df["pbias"].abs()
    normalised: list[pd.DataFrame] = []

    for (var, station), grp in skill_df.groupby(["variable", "station"]):
        if len(grp) <= 1:
            continue
        grp = grp.copy()

        for metric in ["corr", "nse", "kge"]:
            if grp[metric].notna().any():
                mn, mx = grp[metric].min(), grp[metric].max()
                grp[f"{metric}_norm"] = (grp[metric] - mn) / (mx - mn) if mx > mn else 1.0

        for metric in ["rmse", "mae", "abs_pbias"]:
            if grp[metric].notna().any():
                mn, mx = grp[metric].min(), grp[metric].max()
                grp[f"{metric}_norm"] = (
                    1 - (grp[metric] - mn) / (mx - mn) if mx > mn else 1.0
                )

        normalised.append(grp)

    if normalised:
        skill_df = pd.concat(normalised, ignore_index=True)
        norm_cols = [c for c in skill_df.columns if c.endswith("_norm")]
        if norm_cols:
            skill_df["skill_score"] = skill_df[norm_cols].mean(axis=1, skipna=True)

    return skill_df

# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def generate_taylor_diagram(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Taylor diagrams - one per variable.

    Point position: angle = arccos(r), radius = σ_sim / σ_obs (alpha).
    Statistics are derived from monthly time series (n ~ 180).
    """
    print("\nGenerating Taylor diagrams ...")
    colors    = plt.cm.Dark2(np.linspace(0, 1, len(MODELS)))
    color_map = dict(zip(MODELS, colors))

    for var_key, (_, var_label) in VARIABLES.items():
        var_data = metrics_df[metrics_df["variable"] == var_key].copy()
        if var_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})

        seen_models: set[str] = set()
        for _, row in var_data.iterrows():
            r = np.clip(row["corr"] if not np.isnan(row["corr"]) else 0, -1.0, 1.0)
            theta     = np.arccos(r)
            std_ratio = row.get("alpha", 1.0)
            if np.isnan(std_ratio):
                std_ratio = 1.0

            lbl = row["model"] if row["model"] not in seen_models else ""
            seen_models.add(row["model"])
            ax.scatter(theta, std_ratio,
                       color=color_map[row["model"]], s=140,
                       alpha=0.9, edgecolors="black", label=lbl)

        ax.scatter(0, 1.0, color="black", s=250, marker="*", label="Observed (reference)")
        ax.set_title(
            f"Taylor Diagram: {var_label}\n"
            f"Monthly time series, {START_YEAR}\u2013{END_YEAR}, n\u2009\u2248\u2009180",
            fontsize=13, pad=20,
        )
        ax.set_ylim(0, 2.5)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=10)

        plt.savefig(output_dir / f"taylor_diagram_{var_key}.png", dpi=300)
        plt.close()


def generate_heatmap(skill_df: pd.DataFrame, output_dir: Path) -> None:
    """Skill score heatmap: rows = models, columns = variables."""
    if skill_df.empty or "skill_score" not in skill_df.columns:
        return
    print("\nGenerating skill score heatmap ...")
    pivot = skill_df.pivot_table(
        index="model", columns="variable", values="skill_score", aggfunc="mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                center=0.5, vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(
        "CMIP6 GCM Composite Skill Scores - Monthly Time Series Evaluation\n"
        f"({START_YEAR}\u2013{END_YEAR}, 3 stations averaged, n\u2009\u2248\u2009180)",
        fontsize=12, pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "skill_scores_heatmap.png", dpi=300)
    plt.close()


def generate_time_series_plots(data_df: pd.DataFrame, output_dir: Path) -> None:
    """Monthly observed vs. simulated time series - one figure per model x station."""
    if data_df.empty:
        return
    print("\nGenerating time series plots ...")
    for model in data_df["model"].unique():
        for station in data_df.loc[data_df["model"] == model, "station"].unique():
            sub  = data_df[(data_df["model"] == model) & (data_df["station"] == station)]
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            for idx, (var_key, (_, var_label)) in enumerate(VARIABLES.items()):
                vd = sub[sub["variable"] == var_key].sort_values("date")
                if vd.empty:
                    continue
                ax = axes[idx]
                ax.plot(vd["date"], vd["obs"], "k-",
                        alpha=0.85, linewidth=1.6, label="Observed (HBCD-2020)")
                ax.plot(vd["date"], vd["sim"], "r--",
                        alpha=0.80, linewidth=1.3, label=f"{model} (CMIP6 historical)")
                ax.set_ylabel(var_label, fontsize=11)
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(loc="upper right", fontsize=10)

            plt.suptitle(
                f"Monthly Validation: {model} vs HBCD-2020 - {station} "
                f"({START_YEAR}\u2013{END_YEAR})",
                fontsize=13, y=1.01,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"timeseries_{model}_{station}.png", dpi=300)
            plt.close()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_results(metrics_df: pd.DataFrame, skill_df: pd.DataFrame, output_dir: Path) -> None:
    """Write all results to UTF-8-BOM CSV files (opens correctly in Excel)."""
    print("\nSaving results ...")
    enc = {"encoding": "utf-8-sig"}

    metrics_df.to_csv(output_dir / "cmip6_evaluation_metrics.csv", index=False, **enc)
    print(f"  Saved: cmip6_evaluation_metrics.csv  ({len(metrics_df)} rows)")

    if not skill_df.empty and "skill_score" in skill_df.columns:
        skill_df.to_csv(output_dir / "cmip6_skill_scores.csv", index=False, **enc)
        overall = (
            skill_df.groupby("model")["skill_score"]
            .mean()
            .sort_values(ascending=False)
        )
        overall.to_csv(output_dir / "model_rankings_overall.csv",
                       header=["skill_score"], **enc)
        print("\nModel Rankings (composite skill score):")
        print(overall.to_string())

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    print("CMIP6 GCM Evaluation - Haraz Watershed")
    print("Evaluation basis: monthly time series  (n \u2248 180 per station/variable)")
    print("=" * 60)

    metrics_df, data_df = evaluate_all_models()

    if metrics_df.empty:
        print("\nNo evaluation results generated. Verify data paths.")
        return

    skill_df = calculate_skill_scores(metrics_df)
    generate_taylor_diagram(metrics_df, OUTPUT_DIR)
    generate_heatmap(skill_df, OUTPUT_DIR)
    generate_time_series_plots(data_df, OUTPUT_DIR)
    save_results(metrics_df, skill_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

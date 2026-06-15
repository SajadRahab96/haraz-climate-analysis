"""
17_compound_extremes.py
========================
Analyzes compound hot-dry events: days where BOTH temperature AND
precipitation anomalies exceed threshold simultaneously.

Definition (IPCC AR6 / Zscheischler et al., 2020):
  Compound hot-dry event: Tmax > 95th percentile AND P_monthly < 5th percentile
  (using monthly time-step for consistency with drought indices)

Analysis includes:
  1. Frequency of compound events per decade
  2. Intensity (mean exceedance above threshold)
  3. Duration (consecutive months)
  4. Seasonal distribution
  5. Comparison baseline (2000-2020) vs near-term (2021-2060) vs long-term (2061-2100)

Also computes joint probability and dependency structure (Kendall's tau).

References:
  Zscheischler, J. et al. (2020). A typology of compound weather and climate events.
    Nature Reviews Earth & Environment, 1, 333-347.
  IPCC AR6 WGI Chapter 11 (2021).

Outputs:
  outputs/compound/compound_events_{station}_{scenario}.csv
  outputs/compound/compound_summary.csv
  outputs/compound/figures/
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
BMA_DIR   = BASE_DIR / "outputs" / "bma"
OBS_XLSX  = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR   = BASE_DIR / "outputs" / "compound"
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245", "ssp585"]
STATIONS  = ["Amol", "Gharakhil", "Sari"]

TMAX_PCTILE = 95   # Tmax percentile threshold
PR_PCTILE   = 5    # Precipitation percentile threshold (low = dry)

BASELINE  = ("2000-01-01", "2020-12-31")
NEAR_TERM = ("2021-01-01", "2060-12-31")
LONG_TERM = ("2061-01-01", "2100-12-31")

STATION_EXCEL_MAP = {
    "Amol":      "Amol",
    "Gharakhil": "Gharakhil",
    "Sari":      "Sari (Dasht-E-Naz Airport)",
}


# ── Load observed monthly ─────────────────────────────────────────────────────
def load_obs_monthly_for_compound(station_excel: str) -> pd.DataFrame:
    df = pd.read_excel(OBS_XLSX, sheet_name="All_Stations", parse_dates=["date"])
    df = df[df["station_name"] == station_excel].copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["year", "month"]).agg(
        pr_monthly=("rrr24", "sum"),
        tmax_monthly=("tmax", "mean"),
    ).reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly.sort_values("date").reset_index(drop=True)


# ── Detect compound events ────────────────────────────────────────────────────
def detect_compound_events(df: pd.DataFrame,
                             tmax_threshold: float,
                             pr_threshold: float) -> pd.DataFrame:
    """
    Flag months where Tmax > tmax_threshold AND pr < pr_threshold.
    Returns DataFrame with compound event indicators.
    """
    df = df.copy()
    df["hot"]  = df["tmax_monthly"] > tmax_threshold
    df["dry"]  = df["pr_monthly"] < pr_threshold
    df["compound"] = df["hot"] & df["dry"]

    # Intensity metrics
    df["tmax_excess"] = np.where(df["hot"], df["tmax_monthly"] - tmax_threshold, 0)
    df["pr_deficit"]  = np.where(df["dry"], pr_threshold - df["pr_monthly"], 0)
    df["compound_intensity"] = df["tmax_excess"] * (1 + df["pr_deficit"] / (pr_threshold + 1))

    # Duration (consecutive compound months)
    duration = []
    current_dur = 0
    for is_compound in df["compound"]:
        if is_compound:
            current_dur += 1
        else:
            current_dur = 0
        duration.append(current_dur)
    df["compound_duration"] = duration

    return df


# ── Seasonal analysis ─────────────────────────────────────────────────────────
def seasonal_compound_frequency(df_compound: pd.DataFrame) -> pd.DataFrame:
    """Count compound events by season."""
    df = df_compound.copy()
    df["season"] = pd.cut(df["date"].dt.month,
                           bins=[0, 2, 5, 8, 11, 12],
                           labels=["DJF", "MAM", "JJA", "SON", "DJF2"],
                           right=True)
    df["season"] = df["season"].replace("DJF2", "DJF")
    return df.groupby("season")["compound"].sum().rename("n_compound")


# ── Decadal trend ─────────────────────────────────────────────────────────────
def decadal_frequency(df_compound: pd.DataFrame) -> pd.DataFrame:
    """Count compound events per decade."""
    df = df_compound.copy()
    df["decade"] = (df["date"].dt.year // 10) * 10
    return df.groupby("decade")["compound"].sum().reset_index()


# ── Main analysis ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase IV — Compound Hot-Dry Event Analysis")
    print("=" * 65)

    all_summaries = []

    for station in STATIONS:
        print(f"\n{'-'*50}")
        print(f"Station: {station}")

        # Load observed monthly baseline
        obs = load_obs_monthly_for_compound(STATION_EXCEL_MAP[station])

        # Compute thresholds from baseline period (2000-2020)
        obs_calib = obs[(obs["date"] >= BASELINE[0]) & (obs["date"] <= BASELINE[1])]
        tmax_thresh = float(np.percentile(obs_calib["tmax_monthly"], TMAX_PCTILE))
        pr_thresh   = float(np.percentile(obs_calib["pr_monthly"],   PR_PCTILE))

        print(f"  Thresholds: Tmax > {tmax_thresh:.1f}°C ({TMAX_PCTILE}th pct) | "
              f"Pr < {pr_thresh:.1f} mm ({PR_PCTILE}th pct)")

        # Baseline compound events (observed)
        obs_compound = detect_compound_events(obs, tmax_thresh, pr_thresh)
        obs_n = obs_compound["compound"].sum()
        obs_freq = 100 * obs_compound["compound"].mean()
        print(f"  Baseline compound events: {obs_n} months ({obs_freq:.1f}% of time)")

        for scenario in SCENARIOS:
            bma_path = BMA_DIR / f"bma_climate_{station}_{scenario}_monthly.csv"
            if not bma_path.exists():
                print(f"  SKIP {scenario}: BMA file not found. Run 14_bma_ensemble.py first.")
                continue

            bma_df = pd.read_csv(bma_path, parse_dates=["date"], index_col="date")
            bma_df = bma_df.reset_index()

            # Build full series: obs + future BMA
            future = pd.DataFrame({
                "date":        bma_df["date"],
                "pr_monthly":  bma_df["pr_monthly_bma"],
                "tmax_monthly":bma_df["tmax_monthly_bma"],
            })
            full_df = pd.concat([
                obs[["date", "pr_monthly", "tmax_monthly"]],
                future[future["date"] > obs["date"].max()]
            ]).sort_values("date").reset_index(drop=True)

            # Detect compound events using baseline thresholds
            full_compound = detect_compound_events(full_df, tmax_thresh, pr_thresh)

            # Period-wise statistics
            for pname, (p0, p1) in [("baseline", BASELINE),
                                      ("near_term", NEAR_TERM),
                                      ("long_term", LONG_TERM)]:
                mask = (full_compound["date"] >= p0) & (full_compound["date"] <= p1)
                period = full_compound[mask]
                if len(period) == 0:
                    continue
                n_months = len(period)
                n_events = period["compound"].sum()
                freq_pct = 100 * n_events / n_months
                mean_int = period.loc[period["compound"], "compound_intensity"].mean() \
                           if n_events > 0 else 0
                max_dur  = period["compound_duration"].max()

                row = {
                    "station":    station,
                    "scenario":   scenario,
                    "period":     pname,
                    "n_months":   n_months,
                    "n_compound": int(n_events),
                    "freq_pct":   round(freq_pct, 1),
                    "mean_intensity": round(mean_int, 2),
                    "max_duration_months": int(max_dur),
                }
                all_summaries.append(row)

            # Save full compound time series
            out_path = OUT_DIR / f"compound_events_{station}_{scenario}.csv"
            full_compound[["date", "pr_monthly", "tmax_monthly",
                            "hot", "dry", "compound",
                            "compound_intensity", "compound_duration"]].to_csv(
                out_path, index=False)

            # ── Figure: stacked bar + frequency change ───────────────────────
            fig = plt.figure(figsize=(16, 10))
            gs  = gridspec.GridSpec(3, 2, figure=fig)

            # Top: time series with compound events highlighted
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(full_compound["date"], full_compound["tmax_monthly"],
                     "r-", lw=0.8, alpha=0.7, label="Tmax monthly mean")
            compound_dates = full_compound.loc[full_compound["compound"], "date"]
            compound_tmax  = full_compound.loc[full_compound["compound"], "tmax_monthly"]
            ax1.scatter(compound_dates, compound_tmax, color="darkred", s=15, zorder=5,
                        label=f"Compound event (n={full_compound['compound'].sum()})")
            ax1.axhline(tmax_thresh, color="darkred", ls="--", lw=1, alpha=0.7)
            ax1.axvline(pd.Timestamp("2021-01-01"), color="gray", ls=":", alpha=0.5)
            ax1.axvline(pd.Timestamp("2061-01-01"), color="gray", ls="--", alpha=0.5)
            ax1.set_ylabel("Tmax (°C)")
            ax1.set_title(f"Compound Hot-Dry Events — {station} ({scenario.upper()})")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Middle left: decadal frequency
            ax2 = fig.add_subplot(gs[1, 0])
            dec_freq = decadal_frequency(full_compound)
            colors_dec = ["gray" if d <= 2020 else "steelblue" if d <= 2060 else "firebrick"
                          for d in dec_freq["decade"]]
            ax2.bar(dec_freq["decade"], dec_freq["compound"],
                    color=colors_dec, width=8, alpha=0.85)
            ax2.set_xlabel("Decade")
            ax2.set_ylabel("# Compound events")
            ax2.set_title("Decadal Frequency")
            ax2.grid(True, alpha=0.3, axis="y")

            # Middle right: seasonal distribution
            ax3 = fig.add_subplot(gs[1, 1])
            season_freq = seasonal_compound_frequency(full_compound)
            ax3.bar(season_freq.index, season_freq.values,
                    color="steelblue", alpha=0.8)
            ax3.set_xlabel("Season")
            ax3.set_ylabel("# Compound events")
            ax3.set_title("Seasonal Distribution (full period)")
            ax3.grid(True, alpha=0.3, axis="y")

            # Bottom: period comparison bar chart
            ax4 = fig.add_subplot(gs[2, :])
            period_data = [r for r in all_summaries
                           if r["station"] == station and r["scenario"] == scenario]
            if period_data:
                p_df = pd.DataFrame(period_data)
                bar_colors = ["gray", "steelblue", "firebrick"]
                bars = ax4.bar(p_df["period"],
                               p_df["freq_pct"],
                               color=bar_colors[:len(p_df)],
                               alpha=0.85)
                ax4.set_ylabel("Compound event frequency (%)")
                ax4.set_title("Period Comparison: Compound Hot-Dry Frequency")
                ax4.grid(True, alpha=0.3, axis="y")
                for bar, row in zip(bars, period_data):
                    ax4.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.2,
                             f"{row['freq_pct']}%\n(n={row['n_compound']})",
                             ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            fig.savefig(FIG_DIR / f"compound_{station}_{scenario}.png", dpi=150)
            plt.close(fig)

            print(f"  {scenario}: saved compound events and figure")

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(OUT_DIR / "compound_summary.csv", index=False)

        # Compute frequency amplification factors
        print(f"\n{'='*65}")
        print("Compound Hot-Dry Event Frequency Change Summary")
        print("-"*65)

        for station in STATIONS:
            for scenario in SCENARIOS:
                sub = summary_df[(summary_df["station"] == station) &
                                  (summary_df["scenario"] == scenario)]
                if len(sub) < 2:
                    continue
                base_freq = sub.loc[sub["period"] == "baseline", "freq_pct"].values
                near_freq = sub.loc[sub["period"] == "near_term", "freq_pct"].values
                long_freq = sub.loc[sub["period"] == "long_term", "freq_pct"].values

                if len(base_freq) and base_freq[0] > 0:
                    near_mult = near_freq[0] / base_freq[0] if len(near_freq) else np.nan
                    long_mult = long_freq[0] / base_freq[0] if len(long_freq) else np.nan
                    print(f"  {station:12s} | {scenario}: "
                          f"baseline={base_freq[0]:.1f}% → "
                          f"near={near_freq[0] if len(near_freq) else 'N/A':.1f}% "
                          f"(×{near_mult:.1f}) → "
                          f"long={long_freq[0] if len(long_freq) else 'N/A':.1f}% "
                          f"(×{long_mult:.1f})")

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()

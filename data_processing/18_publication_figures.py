"""
18_publication_figures.py
==========================
Generate all publication-quality figures at 600 DPI for the manuscript.

Figures produced:
  Fig 1  - Study area map (schematic, from station metadata)
  Fig 2  - GCM evaluation heatmap (skill scores)
  Fig 3  - Bias correction validation (obs vs BC, 3 stations)
  Fig 4  - BMA ensemble temperature & precipitation projections (3 stations x 2 scenarios)
  Fig 5  - BiLSTM validation and future streamflow projections
  Fig 6  - Drought indices (SPI-12, SPEI-12) time series and boxplots
  Fig 7  - Extreme indices bar charts (R95p, Rx5day, CDD, TXx, TX90p)
  Fig 8  - Compound hot-dry event frequency heatmap and time series
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")

# -- Settings ------------------------------------------------------------------
BASE   = Path(__file__).resolve().parent.parent
OUT    = BASE / "outputs" / "figures_600dpi"
OUT.mkdir(parents=True, exist_ok=True)

DPI     = 600
FMT     = "png"
PALETTE = {"ssp245": "#2196F3", "ssp585": "#F44336"}
STATIONS = ["Amol", "Gharakhil", "Sari"]
SCENARIOS = ["ssp245", "ssp585"]
SCEN_LAB  = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       100,
    "savefig.dpi":      DPI,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

def savefig(fig, name):
    path = OUT / f"{name}.{FMT}"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# -----------------------------------------------------------------------------
# FIG 2 - GCM evaluation heatmap
# -----------------------------------------------------------------------------
MODELS_ORDER = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR", "CanESM5",
                "UKESM1-0-LL", "GFDL-ESM4", "MRI-ESM2-0"]


def _taylor_panel(ax, sub, title, model_color):
    """Draw a single quarter Taylor diagram (correlation vs. variance ratio)."""
    rmax = 1.7
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, rmax)

    # Angular gridlines = correlation
    rlines = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ax.set_xticks([np.arccos(r) for r in rlines])
    ax.set_xticklabels([f"{r:g}" for r in rlines])
    ax.text(np.deg2rad(46), rmax * 1.22, "Correlation",
            ha="center", va="center", fontsize=8, rotation=-46)

    # Radial gridlines = standard-deviation ratio
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_rlabel_position(83)
    ax.tick_params(axis="y", labelsize=6)

    # Reference variability circle (sigma ratio = 1) and reference point
    th = np.linspace(0, np.pi / 2, 100)
    ax.plot(th, np.ones_like(th), ls="--", color="0.45", lw=0.7, zorder=1)
    ax.plot(0, 1.0, marker="*", color="k", ms=14, zorder=6)

    # Centered-RMS-difference contours (circles centred on the reference point)
    for E in [0.25, 0.5, 0.75, 1.0, 1.25]:
        phi = np.linspace(0, np.pi, 240)
        x = 1.0 + E * np.cos(phi)
        y = E * np.sin(phi)
        rr = np.sqrt(x ** 2 + y ** 2)
        tt = np.arctan2(y, x)
        keep = (tt >= 0) & (tt <= np.pi / 2) & (rr <= rmax)
        ax.plot(tt[keep], rr[keep], ls=":", color="0.7", lw=0.6, zorder=1)

    # Model markers
    for m in MODELS_ORDER:
        if m not in sub.index:
            continue
        r = float(np.clip(sub.loc[m, "corr"], -1, 1))
        a = float(sub.loc[m, "alpha"])
        ax.plot(np.arccos(r), a, "o", color=model_color[m],
                ms=9, mec="k", mew=0.6, zorder=5)
    ax.set_title(title, fontsize=9, pad=16)


def fig_gcm_evaluation():
    eval_dir = BASE / "outputs" / "evaluation"
    score_file = eval_dir / "cmip6_skill_scores.csv"
    metrics_file = eval_dir / "cmip6_evaluation_metrics.csv"
    if not score_file.exists() or not metrics_file.exists():
        print("  SKIP Fig2: evaluation csv not found (run 09_cmip6_evaluation.py)")
        return

    skill = pd.read_csv(score_file)
    metr = pd.read_csv(metrics_file)
    cmap_models = plt.cm.tab10(np.linspace(0, 1, 10))
    model_color = {m: cmap_models[i] for i, m in enumerate(MODELS_ORDER)}

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 0.95],
                           hspace=0.38, wspace=0.22)
    axa = fig.add_subplot(gs[0, 0], projection="polar")
    axb = fig.add_subplot(gs[0, 1], projection="polar")
    axc = fig.add_subplot(gs[1, :])

    for ax, var, lab in [(axa, "tmax", "(a) Taylor diagram: Tmax"),
                         (axb, "pr", "(b) Taylor diagram: Precipitation")]:
        sub = metr[metr["variable"] == var].groupby("model")[["corr", "alpha"]].mean()
        _taylor_panel(ax, sub, lab, model_color)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=model_color[m],
                      markeredgecolor="k", ms=8, label=m) for m in MODELS_ORDER]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="k",
                          markeredgecolor="k", ms=12, label="Observed (reference)"))
    axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(1.12, -0.14),
               ncol=4, fontsize=7.5, frameon=False, columnspacing=1.2)

    # Panel (c): composite skill-score heatmap (variable x model)
    pivot = (skill.groupby(["variable", "model"])["skill_score"].mean()
             .unstack("model").reindex(index=["tmax", "tmin", "pr"], columns=MODELS_ORDER))
    im = axc.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axc.set_xticks(range(len(MODELS_ORDER)))
    axc.set_xticklabels(MODELS_ORDER, rotation=22, ha="right")
    axc.set_yticks(range(3))
    axc.set_yticklabels(["Tmax", "Tmin", "Precip"])
    for i in range(3):
        for j in range(len(MODELS_ORDER)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                axc.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5)
    for s in axc.spines.values():
        s.set_visible(True)
    cb = plt.colorbar(im, ax=axc, fraction=0.026, pad=0.015)
    cb.set_label("Composite skill score", fontsize=8)
    axc.set_title("(c) Composite skill scores by model and variable", fontsize=9, pad=8)

    fig.suptitle("CMIP6 GCM historical evaluation vs. observed baseline (2000–2014, n ≈ 180)",
                 fontsize=10.5, y=0.965)
    savefig(fig, "Fig2_GCM_evaluation")


# -----------------------------------------------------------------------------
# FIG 3 - Bias correction validation (calibration period 2000-2014)
# -----------------------------------------------------------------------------
def _load_dqm_module():
    import importlib.util
    path = BASE / "data_processing" / "11_bias_correction_dqm.py"
    spec = importlib.util.spec_from_file_location("dqm_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_bias_correction():
    obs_path = BASE / "ClimateData_GapFilled_2000_2020.xlsx"
    stn_map = {"Amol": "Amol", "Gharakhil": "Gharakhil", "Sari": "Sari (Dasht-E-Naz Airport)"}
    # Validate the DQM track only: in the production two-track pipeline DQM is
    # applied to the two non-library models; the four library models are
    # handled by LARS-WG, which is anchored to the observed climatology by
    # construction (applying DQM to all six would show the drizzle inflation
    # quantified by the ablation in 31_ablation_downscaling.py).
    models = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR"]

    try:
        dqm = _load_dqm_module()
    except Exception as exc:
        print(f"  SKIP Fig3: could not load DQM module ({exc})")
        return

    obs_all = pd.read_excel(obs_path, sheet_name="All_Stations", parse_dates=["date"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey="row")
    fig.suptitle(
        "Fig. 3 - Bias Correction Validation, DQM track: Observed vs. Bias-Corrected GCM\n"
        "(Monthly climatology, calibration period 2000-2014; mean of IPSL-CM6A-LR and MPI-ESM1-2-HR)",
        y=1.01,
    )

    months = range(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for col_i, station in enumerate(STATIONS):
        obs_df = dqm.load_obs(stn_map[station])
        obs_cal = obs_df[dqm.CALIB_START:dqm.CALIB_END]
        obs_tmax_m = obs_cal["obs_tmax"].resample("ME").mean().groupby(lambda x: x.month).mean()
        obs_pr_m = obs_cal["obs_pr"].resample("ME").sum().groupby(lambda x: x.month).mean()

        raw_tmax_list, raw_pr_list, bc_tmax_list, bc_pr_list = [], [], [], []
        for model in models:
            try:
                hist_df = dqm.load_gcm_csv(model, "historical", station)
            except FileNotFoundError:
                continue
            hist_cal = hist_df[dqm.CALIB_START:dqm.CALIB_END]
            common = obs_cal.index.intersection(hist_cal.index)
            if len(common) < 365:
                continue
            o = obs_cal.loc[common]
            h = hist_cal.loc[common]

            bc_tmax = dqm.dqm_temperature(o["obs_tmax"].values, h["gcm_tmax"].values, h["gcm_tmax"].values)
            bc_pr = dqm.dqm_precipitation(o["obs_pr"].values, h["gcm_pr"].values, h["gcm_pr"].values)

            raw_tmax_list.append(pd.Series(h["gcm_tmax"].values, index=common))
            raw_pr_list.append(pd.Series(h["gcm_pr"].values, index=common))
            bc_tmax_list.append(pd.Series(bc_tmax, index=common))
            bc_pr_list.append(pd.Series(bc_pr, index=common))

        if not bc_tmax_list:
            print(f"  SKIP Fig3 station {station}: no historical GCM overlap")
            continue

        raw_tmax_m = pd.concat(raw_tmax_list, axis=1).mean(axis=1).resample("ME").mean().groupby(lambda x: x.month).mean()
        raw_pr_m = pd.concat(raw_pr_list, axis=1).mean(axis=1).resample("ME").sum().groupby(lambda x: x.month).mean()
        bc_tmax_m = pd.concat(bc_tmax_list, axis=1).mean(axis=1).resample("ME").mean().groupby(lambda x: x.month).mean()
        bc_pr_m = pd.concat(bc_pr_list, axis=1).mean(axis=1).resample("ME").sum().groupby(lambda x: x.month).mean()

        ax = axes[0, col_i]
        ax.plot(months, obs_tmax_m.reindex(months), "k-o", ms=4, lw=1.5, label="Obs")
        ax.plot(months, raw_tmax_m.reindex(months), "gray", ls=":", ms=3, lw=1.0, label="Raw GCM")
        ax.plot(months, bc_tmax_m.reindex(months), "r--s", ms=3.5, lw=1.2, label="Bias-corrected")
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title(station)
        if col_i == 0:
            ax.set_ylabel("Mean Tmax (deg C)")
        if col_i == 2:
            ax.legend(loc="upper right", fontsize=6)

        ax = axes[1, col_i]
        ax.bar([m - 0.25 for m in months], obs_pr_m.reindex(months), color="#5b9bd5", alpha=0.7, width=0.22, label="Obs")
        ax.bar(months, raw_pr_m.reindex(months), color="#bdbdbd", alpha=0.7, width=0.22, label="Raw GCM")
        ax.bar([m + 0.25 for m in months], bc_pr_m.reindex(months), color="#ed7d31", alpha=0.7, width=0.22, label="Bias-corrected")
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        if col_i == 0:
            ax.set_ylabel("Monthly Precip (mm)")
        if col_i == 2:
            ax.legend(loc="upper right", fontsize=6)

    fig.tight_layout()
    savefig(fig, "Fig3_Bias_Correction_Validation")


# -----------------------------------------------------------------------------
# FIG 4 - BMA ensemble temperature projections
# -----------------------------------------------------------------------------
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _seasonal_changes():
    """Pooled station-year seasonal changes vs the observed 2000-2020 baseline.

    Returns dict[(season, scenario, period)] -> (list dTmax degC, list dP %).
    """
    bma_dir = BASE / "outputs" / "bma"
    obs = pd.read_excel(BASE / "ClimateData_GapFilled_2000_2020.xlsx",
                        sheet_name="All_Stations", parse_dates=["date"])
    obs = obs.rename(columns={"rrr24": "pr"})
    smap = {"Amol": "Amol", "Gharakhil": "Gharakhil",
            "Sari": "Sari (Dasht-E-Naz Airport)"}
    month2season = {m: s for s, ms in SEASONS.items() for m in ms}

    out = {}
    for stn in STATIONS:
        o = obs[obs["station_name"] == smap[stn]].copy()
        o["season"] = o["date"].dt.month.map(month2season)
        base_t = o.groupby("season")["tmax"].mean()
        om = o.set_index("date").resample("MS").agg(pr=("pr", "sum"))
        om["season"] = om.index.month.map(month2season)
        base_p = om.groupby("season")["pr"].mean()

        for scen in SCENARIOS:
            f = bma_dir / f"bma_climate_{stn}_{scen}_monthly.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, parse_dates=["date"], index_col="date")
            df["season"] = df.index.month.map(month2season)
            df["year"] = df.index.year
            g = df.groupby(["season", "year"]).agg(
                t=("tmax_monthly_bma", "mean"), p=("pr_monthly_bma", "mean"))
            for (season, year), r in g.iterrows():
                period = "NT" if year <= 2060 else "LT"
                key = (season, scen, period)
                out.setdefault(key, ([], []))
                out[key][0].append(r["t"] - base_t[season])
                out[key][1].append(100.0 * (r["p"] - base_p[season]) / base_p[season])
    return out


def fig_bma_projections():
    bma_dir = BASE / "outputs" / "bma"
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.95], hspace=0.42, wspace=0.55)
    axes = np.array([[fig.add_subplot(gs[r, 2 * c:2 * c + 2]) for c in range(3)]
                     for r in range(2)])
    fig.suptitle("Fig. 4 - BMA Ensemble Projections 2021-2100 (Monthly Mean Tmax and Precipitation)", y=0.99)

    for col_i, station in enumerate(STATIONS):
        for row_i, var in enumerate(["tmax_monthly_bma", "pr_monthly_bma"]):
            ax = axes[row_i, col_i]
            for scen in SCENARIOS:
                f = bma_dir / f"bma_climate_{station}_{scen}_monthly.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f, parse_dates=["date"], index_col="date")
                if var not in df.columns:
                    continue
                annual = df[var].resample("YE").mean()
                # 10-year rolling mean
                smooth = annual.rolling(10, center=True).mean()
                col = PALETTE[scen]
                ax.plot(annual.index.year, annual.values, color=col, alpha=0.25, lw=0.6)
                ax.plot(smooth.index.year, smooth.values, color=col, lw=1.8,
                        label=SCEN_LAB[scen])
                # Uncertainty band
                lower_col = var.replace("_bma", "_lower90")
                upper_col = var.replace("_bma", "_upper90")
                if lower_col in df.columns and upper_col in df.columns:
                    lower = df[lower_col].resample("YE").mean().rolling(10, center=True).mean()
                    upper = df[upper_col].resample("YE").mean().rolling(10, center=True).mean()
                    ax.fill_between(smooth.index.year, lower.values, upper.values,
                                    color=col, alpha=0.12)
            ax.set_xlim(2021, 2100)
            ax.set_title(station if row_i == 0 else "")
            if col_i == 0:
                ax.set_ylabel("Tmax (deg C)" if row_i == 0 else "Precip (mm/month)")
            if col_i == 2 and row_i == 0:
                ax.legend(fontsize=7)
            ax.axvline(2060, color="gray", lw=0.8, ls="--", alpha=0.6)

    for ax in axes[1]:
        ax.set_xlabel("Year")

    # -- Bottom row: seasonal change boxplots (pooled across stations) ---------
    chg = _seasonal_changes()
    ax_t = fig.add_subplot(gs[2, 0:3])
    ax_p = fig.add_subplot(gs[2, 3:6])
    groups = [("ssp245", "NT"), ("ssp245", "LT"), ("ssp585", "NT"), ("ssp585", "LT")]
    shades = {("ssp245", "NT"): "#90CAF9", ("ssp245", "LT"): "#1565C0",
              ("ssp585", "NT"): "#EF9A9A", ("ssp585", "LT"): "#B71C1C"}
    width, offs = 0.17, [-0.30, -0.10, 0.10, 0.30]
    for ax, vi, ylab in [(ax_t, 0, "dTmax vs baseline (deg C)"),
                         (ax_p, 1, "dPrecip vs baseline (%)")]:
        for gi, (scen, per) in enumerate(groups):
            data = [chg.get((s, scen, per), ([], []))[vi] for s in SEASONS]
            pos = [i + offs[gi] for i in range(len(SEASONS))]
            bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.1))
            for b in bp["boxes"]:
                b.set_facecolor(shades[(scen, per)]); b.set_alpha(0.85)
        ax.set_xticks(range(len(SEASONS)))
        ax.set_xticklabels(list(SEASONS))
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylabel(ylab)
        ax.set_title("(g) Seasonal Tmax change" if vi == 0 else "(h) Seasonal precipitation change",
                     fontsize=9)
    handles = [Patch(facecolor=shades[g], label=f"{SCEN_LAB[g[0]]} {g[1]}") for g in groups]
    ax_p.legend(handles=handles, fontsize=7, ncol=2, loc="upper right", framealpha=0.9)

    savefig(fig, "Fig4_BMA_Projections")


# -----------------------------------------------------------------------------
# FIG 5 - BiLSTM validation + future discharge
# -----------------------------------------------------------------------------
def fig_bilstm():
    bilstm_dir = BASE / "outputs" / "bilstm"
    val_file = bilstm_dir / "validation_results.csv"
    if not val_file.exists():
        val_file = bilstm_dir / "bilstm_validation.csv"
    if not val_file.exists():
        print("  SKIP Fig5: validation results not found")
        return

    val = pd.read_csv(val_file, parse_dates=["date"], index_col="date")

    metrics_file = bilstm_dir / "validation_metrics.csv"
    metrics_txt = "NSE/KGE/r: see Table 6"
    if metrics_file.exists():
        m = pd.read_csv(metrics_file).iloc[0]
        metrics_txt = f"NSE={m['NSE']:.3f}\nKGE={m['KGE']:.3f}\nr={m['r']:.3f}"

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # Top-left: validation scatter
    ax0 = fig.add_subplot(gs[0, 0])
    obs  = val.get("Q_obs",  val.get("obs", val.get("observed", None)))
    pred = val.get("Q_pred", val.get("pred", val.get("predicted", None)))
    if obs is not None and pred is not None:
        ax0.scatter(obs, pred, s=10, alpha=0.5, color="#1976D2", edgecolors="none")
        lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
        ax0.plot(lim, lim, "k--", lw=1)
        ax0.set_xlabel("Observed Q (m3/s)")
        ax0.set_ylabel("Predicted Q (m3/s)")
        ax0.set_title("Validation: Obs vs. Pred")
        ax0.text(0.05, 0.88, metrics_txt,
                 transform=ax0.transAxes, fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Top-right: validation time series
    ax1 = fig.add_subplot(gs[0, 1])
    if obs is not None and pred is not None:
        ax1.plot(val.index, obs, "k-", lw=1.2, label="Observed")
        ax1.plot(val.index, pred, "r--", lw=1.2, label="BiLSTM")
        ax1.set_ylabel("Discharge (m3/s)")
        ax1.set_title("Validation Time Series")
        ax1.legend(fontsize=7)

    # Bottom: future discharge per scenario
    ax2 = fig.add_subplot(gs[1, :])
    for scen in SCENARIOS:
        f = bilstm_dir / f"future_discharge_{scen}.csv"
        if not f.exists():
            continue
        df_fut = pd.read_csv(f, parse_dates=["date"])
        # One row per model per month - pivot to models as columns
        all_series = []
        for model, grp in df_fut.groupby("model"):
            grp = grp.set_index("date")["Q_m3s"].resample("YE").mean()
            all_series.append(grp)
        if all_series:
            ensemble = pd.concat(all_series, axis=1)
            mean_  = ensemble.mean(axis=1)
            lower_ = ensemble.quantile(0.05, axis=1)
            upper_ = ensemble.quantile(0.95, axis=1)
            col = PALETTE[scen]
            ax2.plot(mean_.index.year, mean_.values, color=col, lw=1.8, label=SCEN_LAB[scen])
            ax2.fill_between(mean_.index.year, lower_.values, upper_.values,
                             color=col, alpha=0.15)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annual Mean Discharge (m3/s)")
    ax2.set_title("Future Discharge Projections - Karesang Station (BMA Ensemble)")
    ax2.legend(fontsize=7)
    ax2.set_xlim(2021, 2100)
    ax2.axvline(2060, color="gray", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle("Fig. 5 - BiLSTM Streamflow Model: Validation and Future Projections", y=1.01)
    savefig(fig, "Fig5_BiLSTM_Streamflow")


# -----------------------------------------------------------------------------
# FIG 6 - Drought indices
# -----------------------------------------------------------------------------
def fig_drought():
    drought_dir = BASE / "outputs" / "drought"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    fig.suptitle("Fig. 6 - Projected Drought Indices (SPI-12 and SPEI-12) under SSP2-4.5 and SSP5-8.5", y=1.01)

    index_labels = {"spi12": "SPI-12", "spei12": "SPEI-12"}
    for row_i, idx in enumerate(["spi12", "spei12"]):
        for col_i, station in enumerate(STATIONS):
            ax = axes[row_i, col_i]
            for scen in SCENARIOS:
                f = drought_dir / f"drought_indices_{station}_{scen}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f, parse_dates=["date"], index_col="date")
                if idx not in df.columns:
                    continue
                series = df[idx].resample("YE").mean()
                smooth = series.rolling(10, center=True).mean()
                col = PALETTE[scen]
                ax.plot(series.index.year, series.values, color=col, alpha=0.2, lw=0.5)
                ax.plot(smooth.index.year, smooth.values, color=col, lw=1.8,
                        label=SCEN_LAB[scen])
            ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.axhline(-1, color="orange", lw=0.7, ls=":", alpha=0.7)
            ax.axhline(-2, color="red", lw=0.7, ls=":", alpha=0.7)
            ax.set_xlim(2021, 2100)
            ax.set_title(station if row_i == 0 else "")
            if col_i == 0:
                ax.set_ylabel(index_labels[idx])
            if col_i == 2 and row_i == 0:
                ax.legend(fontsize=7)
            if row_i == 1:
                ax.set_xlabel("Year")

    fig.tight_layout()
    savefig(fig, "Fig6_Drought_Indices")


# -----------------------------------------------------------------------------
# FIG 7 - Extreme indices grouped bar chart
# -----------------------------------------------------------------------------
def fig_extremes():
    ext = pd.read_csv(BASE / "outputs" / "extremes" / "extremes_period_changes.csv")
    obs_file = BASE / "outputs" / "extremes" / "observed_extremes_annual.csv"
    obs = pd.read_csv(obs_file).groupby("station")[["R95p","Rx5day","CDD","TXx","TX90p"]].mean()

    indices = ["R95p", "Rx5day", "CDD", "TXx", "TX90p"]
    units   = {"R95p":"mm","Rx5day":"mm","CDD":"days","TXx":"deg C","TX90p":"%"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes_flat[i]
        x = np.arange(len(STATIONS))
        w = 0.18
        obs_vals = [obs.loc[s, idx] if s in obs.index else np.nan for s in STATIONS]
        ax.bar(x - 1.5*w, obs_vals, w, label="Obs baseline", color="#607D8B", alpha=0.85)
        colors_period = {"near_term": {"ssp245":"#90CAF9","ssp585":"#EF9A9A"},
                          "long_term": {"ssp245":"#1565C0","ssp585":"#B71C1C"}}
        offsets = {"near_term": {"ssp245": -0.5*w, "ssp585": 0.5*w},
                   "long_term": {"ssp245":  1.5*w, "ssp585": 2.5*w}}
        for period in ["near_term","long_term"]:
            for scen in SCENARIOS:
                sub = ext[(ext["scenario"]==scen) & (ext["index"]==idx)]
                vals = [sub[sub["station"]==s][period].values[0]
                        if len(sub[sub["station"]==s]) > 0 else np.nan
                        for s in STATIONS]
                off = offsets[period][scen]
                col = colors_period[period][scen]
                lbl = f"{SCEN_LAB[scen]} {'NT' if period=='near_term' else 'LT'}"
                ax.bar(x + off, vals, w, label=lbl, color=col, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(STATIONS)
        ax.set_title(f"{idx} ({units[idx]})")
        ax.set_ylabel(units[idx])
        if i == 0:
            ax.legend(fontsize=5.5, ncol=2, loc="upper left")

    axes_flat[5].set_visible(False)
    fig.suptitle("Fig. 7 - Projected Changes in ETCCDI Extreme Indices\n(NT=2021-2060, LT=2061-2100 vs Obs 2000-2020)", y=1.01)
    fig.tight_layout()
    savefig(fig, "Fig7_Extreme_Indices")


# -----------------------------------------------------------------------------
# FIG 8 - Compound hot-dry events
# -----------------------------------------------------------------------------
def fig_compound():
    summary_file = BASE / "outputs" / "compound" / "compound_summary.csv"
    if not summary_file.exists():
        print("  SKIP Fig8: compound_summary.csv not found")
        return

    cs = pd.read_csv(summary_file)

    def freq(station, scenario, period):
        row = cs[(cs["station"] == station) & (cs["scenario"] == scenario) & (cs["period"] == period)]
        return float(row["freq_pct"].iloc[0]) if not row.empty else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Fig. 8 - Compound Hot-Dry Event Frequency\n"
        "(% of months; baseline 2000-2020, NT=2021-2060, LT=2061-2100)",
        y=1.02,
    )

    for col_i, station in enumerate(STATIONS):
        ax = axes[col_i]
        categories = [
            "Baseline\n(2000-2020)",
            "SSP2-4.5\nNear-term",
            "SSP2-4.5\nLong-term",
            "SSP5-8.5\nNear-term",
            "SSP5-8.5\nLong-term",
        ]
        values = [
            freq(station, "ssp245", "baseline"),
            freq(station, "ssp245", "near_term"),
            freq(station, "ssp245", "long_term"),
            freq(station, "ssp585", "near_term"),
            freq(station, "ssp585", "long_term"),
        ]
        colors_bar = ["#607D8B", "#90CAF9", "#1565C0", "#EF9A9A", "#B71C1C"]

        bars = ax.bar(range(5), values, color=colors_bar, alpha=0.9, edgecolor="white", lw=0.5)
        ax.axhline(values[0], color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xticks(range(5))
        ax.set_xticklabels(categories, fontsize=6.5)
        ax.set_title(station)
        ax.set_ylabel("Frequency (% months)" if col_i == 0 else "")
        ymax = max(v for v in values if not np.isnan(v))
        ax.set_ylim(0, ymax * 1.35 if ymax > 0 else 1)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    savefig(fig, "Fig8_Compound_Events")


# -----------------------------------------------------------------------------
# FIG 9 - Workflow of the integrated framework
# -----------------------------------------------------------------------------
def fig_workflow():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    C_IN   = "#dbe9f6"   # inputs
    C_EVAL = "#fff2cc"   # evaluation
    C_DS   = "#e2efda"   # downscaling
    C_BMA  = "#fce4d6"   # ensemble
    C_OUT  = "#ede7f6"   # impacts
    EDGE   = "#4d4d4d"

    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.12", fc=fc, ec=EDGE, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=13, lw=1.2,
                     color=EDGE, zorder=1))

    # Row 1: inputs
    box(0.4, 12.5, 4.3, 1.1, "CMIP6 GCMs (n = 6)\nhistorical 2000-2014; SSP2-4.5, SSP5-8.5",
        C_IN, bold=True)
    box(5.3, 12.5, 4.3, 1.1, "Observed baseline (2000-2020)\n3 stations, gap-filled daily series",
        C_IN, bold=True)
    # Row 2: evaluation
    box(2.5, 10.6, 5.0, 1.2, "Historical evaluation (monthly, n = 180)\n"
        "RMSE, MAE, PBIAS, NSE, KGE, r; Taylor diagrams\n"
        "composite skill ranking (Table 5)", C_EVAL)
    arrow(2.55, 12.5, 4.0, 11.8)
    arrow(7.45, 12.5, 6.0, 11.8)
    # Row 3: two-track downscaling
    box(0.4, 8.5, 4.4, 1.4, "Track A: LARS-WG 8\n4 library GCMs (CanESM5, GFDL,\nMRI, UKESM); 20-yr time slices", C_DS)
    box(5.2, 8.5, 4.4, 1.4, "Track B: DQM\n2 non-library, top-skill GCMs\n(IPSL-CM6A-LR, MPI-ESM1-2-HR)", C_DS)
    arrow(4.0, 10.6, 2.8, 9.9)
    arrow(6.0, 10.6, 7.2, 9.9)
    # Row 4: station series
    box(2.5, 6.9, 5.0, 0.9, "Station-scale daily series, 2021-2100\n(3 stations, 6 GCMs, 2 SSPs)", C_DS)
    arrow(2.6, 8.5, 4.2, 7.8)
    arrow(7.4, 8.5, 5.8, 7.8)
    # Row 5: BMA
    box(2.5, 5.2, 5.0, 1.1, "Bayesian Model Averaging (EM)\nposterior weights + 90% predictive intervals\n(Raftery et al., 2005)", C_BMA, bold=True)
    arrow(5.0, 6.9, 5.0, 6.3)
    # Row 6: impacts (two branches)
    box(0.4, 3.2, 4.4, 1.3, "Autoregressive BiLSTM\nmonthly discharge at Karesang\n(one-step and recursive modes)", C_OUT)
    box(5.2, 3.2, 4.4, 1.3, "Drought: SPI-3/12, SPEI-12\nETCCDI extreme indices\ncompound hot-dry events", C_OUT)
    arrow(4.2, 5.2, 2.8, 4.5)
    arrow(5.8, 5.2, 7.2, 4.5)
    # Row 7: assessment
    box(2.5, 1.3, 5.0, 1.1, "Future assessment\nnear-term (2021-2060) vs long-term (2061-2100)\nchange signals + uncertainty", "#f2f2f2", bold=True)
    arrow(2.6, 3.2, 4.2, 2.4)
    arrow(7.4, 3.2, 5.8, 2.4)

    ax.set_title("Workflow of the integrated downscaling-BMA-BiLSTM framework",
                 fontsize=10.5, fontweight="bold", pad=10)
    savefig(fig, "Fig9_Workflow")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"Generating publication figures at {DPI} DPI")
    print(f"Output: {OUT}")
    print("=" * 60)

    print("\nFig 2 - GCM Evaluation:")
    fig_gcm_evaluation()

    print("\nFig 3 - Bias Correction Validation:")
    fig_bias_correction()

    print("\nFig 4 - BMA Projections:")
    fig_bma_projections()

    print("\nFig 5 - BiLSTM Streamflow:")
    fig_bilstm()

    print("\nFig 6 - Drought Indices:")
    fig_drought()

    print("\nFig 7 - Extreme Indices:")
    fig_extremes()

    print("\nFig 8 - Compound Events:")
    fig_compound()

    print("\nFig 9 - Workflow:")
    fig_workflow()

    files = list(OUT.glob("*.png"))
    print(f"\nDone. {len(files)} figures saved in {OUT}")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.0f} KB)")

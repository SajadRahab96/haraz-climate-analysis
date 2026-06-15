"""
14_bma_ensemble.py
==================
Bayesian Model Averaging (BMA) for the CMIP6 multi-model climate and streamflow
projections, following Raftery et al. (2005).

Method
------
For each variable (Tmax, Tmin, monthly precipitation) and station, the BMA
predictive density of the observation y given the K ensemble members f_k is

    p(y | f_1..f_K) = sum_k  w_k * g_k(y | f_k)

with a Gaussian conditional density g_k = N(y ; a_k + b_k f_k , sigma^2).
Each member is first bias-corrected by ordinary least squares (a_k + b_k f_k)
against the observations on the 2000-2014 training period; the weights w_k and
the shared variance sigma^2 are then estimated by Expectation-Maximisation (EM)
that maximises the BMA predictive log-likelihood (Raftery et al., 2005). The
weights are non-negative and sum to one and therefore have a genuine posterior-
probability interpretation, unlike a simple skill-score normalisation.

Predictive uncertainty
-----------------------
For a future ensemble {Y_k(t)} the BMA mean and variance are

    mu(t)   = sum_k w_k Y_k(t)
    var(t)  = sum_k w_k (Y_k(t) - mu(t))^2   +   sigma^2
              (between-member spread)            (within-member)

and the 90% predictive interval is mu(t) +/- 1.645 sqrt(var(t)). This replaces
the earlier code, which reported unweighted percentiles across the members and
ignored the weights entirely.

Assumptions
-----------
* Monthly aggregation makes an approximately Gaussian conditional density
  reasonable for all three variables; for monthly precipitation (strictly
  positive, mildly right-skewed at these humid Caspian stations) the Gaussian
  BMA is used for the WEIGHTS, which are robust to the distributional choice,
  while the precipitation predictive interval is clipped at zero.
* A single weight vector per model (the mean of the per-variable/station EM
  weights, renormalised) is used for the streamflow BMA and the extreme-index
  weighting, so that one consistent, reported weight set propagates downstream.

Outputs:
  outputs/bma/bma_weights.csv                    (per-model BMA weight + skill)
  outputs/bma/bma_weights_by_var_station.csv     (full EM weight table)
  outputs/bma/bma_climate_{station}_{scenario}_monthly.csv
  outputs/bma/bma_discharge_{scenario}_monthly.csv
  outputs/bma/figures/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
EVAL_DIR    = BASE_DIR / "outputs" / "evaluation"
BC_DIR      = BASE_DIR / "outputs" / "bias_corrected"
BILSTM_DIR  = BASE_DIR / "outputs" / "bilstm"
CMIP6_DIR   = BASE_DIR / "data" / "cmip6_gcs"
OBS_XLSX    = BASE_DIR / "ClimateData_GapFilled_2000_2020.xlsx"
OUT_DIR     = BASE_DIR / "outputs" / "bma"
FIG_DIR     = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCENARIOS = ["ssp245", "ssp585"]
STATIONS  = ["Amol", "Gharakhil", "Sari"]
VARIABLES = ["tmax", "tmin", "pr"]

CALIB_START, CALIB_END = "2000-01-01", "2014-12-31"
NEAR_TERM = ("2021-01-01", "2060-12-31")
LONG_TERM = ("2061-01-01", "2100-12-31")
BASELINE  = ("2000-01-01", "2020-12-31")

LARS_WG_MODELS = {"CanESM5", "GFDL-ESM4", "MRI-ESM2-0", "UKESM1-0-LL"}

STATION_EXCEL_MAP = {
    "Amol": "Amol", "Gharakhil": "Gharakhil", "Sari": "Sari (Dasht-E-Naz Airport)",
}
PERIOD_TRANSITIONS = ["2041-01-01", "2041-11-01", "2061-01-01",
                      "2061-11-01", "2081-01-01", "2081-11-01"]


# ── Load monthly obs + GCM historical (2000-2014) for EM training ─────────────
def _load_obs_monthly(station_excel: str) -> pd.DataFrame:
    df = pd.read_excel(OBS_XLSX, sheet_name="All_Stations", parse_dates=["date"])
    df = df[df["station_name"] == station_excel].copy()
    df = df[(df["date"] >= CALIB_START) & (df["date"] <= CALIB_END)]
    df["ym"] = df["date"].dt.to_period("M")
    g = df.groupby("ym").agg(tmax=("tmax", "mean"), tmin=("tmin", "mean"),
                             pr=("rrr24", "sum"))
    return g


def _load_gcm_hist_monthly(model: str, station_key: str) -> pd.DataFrame | None:
    """Monthly historical GCM series (2000-2014), with the UKESM tmax fix."""
    fname = f"{station_key}_{model}_historical.csv"
    path = CMIP6_DIR / "historical" / model / fname
    if not path.exists() and station_key == "Sari":
        for alt in ["Sari_DashtENaz_Airport", "Sari (Dasht-E-Naz Airport)"]:
            p = CMIP6_DIR / "historical" / model / f"{alt}_{model}_historical.csv"
            if p.exists():
                path = p
                break
    if not path.exists():
        return None

    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "tasmax" in cl:
            rename[c] = "tmax"
        elif "tasmin" in cl:
            rename[c] = "tmin"
        elif cl.startswith("pr"):
            rename[c] = "pr"
        elif cl in ("tas_c", "tas"):
            rename[c] = "tas"
    df = df.rename(columns=rename)
    # UKESM: derive tmax = 2*tas - tmin (consistent with 09 and 11)
    if "tmax" not in df.columns and "tas" in df.columns and "tmin" in df.columns:
        df["tmax"] = 2.0 * df["tas"] - df["tmin"]
    # Units
    for c in ("tmax", "tmin"):
        if c in df.columns and df[c].dropna().mean() > 100:
            df[c] = df[c] - 273.15
    if "pr" in df.columns and df["pr"].dropna().mean() < 0.1:
        df["pr"] = df["pr"] * 86400.0

    df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= CALIB_START) & (df["date"] <= CALIB_END)]
    df["ym"] = df["date"].dt.to_period("M")
    agg = {v: ("mean" if v in ("tmax", "tmin") else "sum")
           for v in ("tmax", "tmin", "pr") if v in df.columns}
    return df.groupby("ym").agg(**{v: (v, how) for v, how in agg.items()})


# ── BMA EM (Gaussian, per-member linear bias correction) ──────────────────────
def _ols_bias_correct(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return a + b*f fitted to y (ordinary least squares)."""
    A = np.column_stack([np.ones_like(f), f])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def fit_bma_em(y: np.ndarray, F: np.ndarray,
               max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """
    Gaussian BMA via EM (Raftery et al., 2005).

    y : (n,) observations
    F : (n, K) bias-corrected member forecasts
    Returns (weights[K], sigma2, loglik).
    """
    n, K = F.shape
    w = np.full(K, 1.0 / K)
    resid2 = (y[:, None] - F) ** 2
    sigma2 = max(resid2.mean(), 1e-6)

    def _loglik(w, sigma2):
        comp = w[None, :] * np.exp(-resid2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
        return np.log(comp.sum(axis=1) + 1e-300).sum()

    ll_old = _loglik(w, sigma2)
    for _ in range(max_iter):
        # E-step: responsibilities
        comp = w[None, :] * np.exp(-resid2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
        denom = comp.sum(axis=1, keepdims=True) + 1e-300
        z = comp / denom
        # M-step
        w = z.mean(axis=0)
        sigma2 = max((z * resid2).sum() / n, 1e-6)
        ll = _loglik(w, sigma2)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return w, sigma2, ll_old


def compute_bma_weights() -> tuple[pd.DataFrame, dict, dict]:
    """
    Fit BMA-EM per (variable, station) on 2000-2014; return:
      - per-model aggregate weight DataFrame (model, skill_score, bma_weight)
      - dict {(var,station): {model: weight}}
      - dict {(var,station): sigma2}
    """
    # skill scores (for reference column only)
    skill = {}
    rk = EVAL_DIR / "model_rankings_overall.csv"
    if rk.exists():
        s = pd.read_csv(rk)
        skill = dict(zip(s["model"], s["skill_score"]))

    obs_cache = {st: _load_obs_monthly(STATION_EXCEL_MAP[st]) for st in STATIONS}
    gcm_cache = {}
    for st in STATIONS:
        st_key = "Sari_DashtENaz_Airport" if st == "Sari" else st
        for m in MODELS:
            gcm_cache[(m, st)] = _load_gcm_hist_monthly(m, st_key)

    weights_vs = {}     # (var,station) -> {model: w}
    sigma_vs = {}       # (var,station) -> sigma2
    rows = []
    for var in VARIABLES:
        for st in STATIONS:
            obs = obs_cache[st]
            members, mem_models = [], []
            for m in MODELS:
                g = gcm_cache[(m, st)]
                if g is None or var not in g.columns:
                    continue
                joined = obs[[var]].join(g[[var]], how="inner", lsuffix="_o", rsuffix="_s").dropna()
                if len(joined) < 24:
                    continue
                members.append((m, joined.index, joined[f"{var}_o"].values, joined[f"{var}_s"].values))
                mem_models.append(m)
            if len(mem_models) < 2:
                continue
            # Align on common months
            common = members[0][1]
            for _, idx, _, _ in members[1:]:
                common = common.intersection(idx)
            y = obs.loc[common, var].values
            F = np.column_stack([
                _ols_bias_correct(y, g.loc[common, var].values)
                for g in [gcm_cache[(m, st)] for m in mem_models]
            ])
            w, sigma2, ll = fit_bma_em(y, F)
            weights_vs[(var, st)] = dict(zip(mem_models, w))
            sigma_vs[(var, st)] = sigma2
            for m, wk in zip(mem_models, w):
                rows.append({"variable": var, "station": st, "model": m,
                             "weight": round(float(wk), 4), "sigma2": round(sigma2, 4),
                             "n_train": len(common)})

    by_vs = pd.DataFrame(rows)
    # Aggregate per-model weight = mean across (var,station), renormalised
    agg = by_vs.groupby("model")["weight"].mean()
    agg = (agg / agg.sum()).reindex(MODELS).fillna(0.0)
    weights_df = pd.DataFrame({
        "model": MODELS,
        "skill_score": [round(skill.get(m, np.nan), 4) for m in MODELS],
        "bma_weight": [round(float(agg[m]), 4) for m in MODELS],
    })
    return weights_df, weights_vs, sigma_vs, by_vs


# ── BMA predictive aggregation for a future ensemble ──────────────────────────
def bma_predict(ensemble: np.ndarray, weights: np.ndarray, sigma2: float,
                ci: float = 0.90, clip_zero: bool = False) -> dict:
    """
    ensemble : (K, T) future member series; weights : (K,)
    Returns BMA mean and predictive 90% interval (between + within variance).
    """
    z = 1.6448536269514722  # Phi^{-1}(0.95)
    w = weights / weights.sum()
    mu = np.average(ensemble, axis=0, weights=w)
    between = np.average((ensemble - mu[None, :]) ** 2, axis=0, weights=w)
    sd = np.sqrt(between + sigma2)
    lower, upper = mu - z * sd, mu + z * sd
    if clip_zero:
        mu = np.clip(mu, 0, None)
        lower = np.clip(lower, 0, None)
        upper = np.clip(upper, 0, None)
    return {"mean": mu, "lower": lower, "upper": upper, "std": sd}


# ── Climate BMA ───────────────────────────────────────────────────────────────
def process_climate_bma(station: str, scenario: str,
                        weights_vs: dict, sigma_vs: dict, agg_weights: np.ndarray):
    dfs, avail = {}, []
    for m in MODELS:
        p = BC_DIR / f"{station}_{m}_{scenario}_bc.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["date"], index_col="date")
            dfs[m] = d.resample("MS").agg(pr_monthly=("bc_pr", "sum"),
                                          tmax_monthly=("bc_tmax", "mean"),
                                          tmin_monthly=("bc_tmin", "mean"))
            avail.append(m)
    if len(avail) < 2:
        print(f"  SKIP {station}/{scenario}: {len(avail)} models")
        return None

    common = dfs[avail[0]].index
    for m in avail[1:]:
        common = common.intersection(dfs[m].index)
    common = common.sort_values()

    results = {"date": common}
    var_map = {"pr_monthly": "pr", "tmax_monthly": "tmax", "tmin_monthly": "tmin"}
    for col, var in var_map.items():
        # variable/station-specific EM weights and sigma; fall back to aggregate
        wd = weights_vs.get((var, station), dict(zip(MODELS, agg_weights)))
        sigma2 = sigma_vs.get((var, station), 0.0)
        wts = np.array([wd.get(m, 0.0) for m in avail])
        if wts.sum() == 0:
            wts = np.array([1.0 / len(avail)] * len(avail))
        ens = np.array([dfs[m][col].reindex(common).values for m in avail])
        bma = bma_predict(ens, wts, sigma2, clip_zero=(var == "pr"))
        results[f"{col}_bma"] = bma["mean"]
        results[f"{col}_lower90"] = bma["lower"]
        results[f"{col}_upper90"] = bma["upper"]
        results[f"{col}_std"] = bma["std"]

    out = pd.DataFrame(results).set_index("date")
    # flag LARS-WG period-transition months (mixed stochastic distributions)
    lars_frac = sum(m in LARS_WG_MODELS for m in avail) / len(avail)
    tmask = pd.Series(False, index=out.index)
    for ts in PERIOD_TRANSITIONS[::2]:
        te = PERIOD_TRANSITIONS[PERIOD_TRANSITIONS.index(ts) + 1]
        tmask |= (out.index >= ts) & (out.index <= te)
    out["lars_wg_transition"] = (tmask & (lars_frac > 0)).astype(int)
    return out


# ── Discharge BMA (aggregate weights; no per-member discharge obs) ────────────
def process_discharge_bma(scenario: str, agg_weights: np.ndarray):
    p = BILSTM_DIR / f"future_discharge_{scenario}.csv"
    if not p.exists():
        print(f"  SKIP discharge BMA: {p.name} not found")
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.groupby(["date", "model"], as_index=False)["Q_m3s"].mean()
    piv = df.pivot(index="date", columns="model", values="Q_m3s").dropna()
    avail = [m for m in MODELS if m in piv.columns]
    wts = np.array([agg_weights[MODELS.index(m)] for m in avail])
    ens = np.array([piv[m].values for m in avail])
    # within-member variance for discharge is unknown per member; use 0 so the
    # interval reflects weighted between-model spread (conservative lower bound).
    bma = bma_predict(ens, wts, sigma2=0.0, clip_zero=True)
    return pd.DataFrame({"Q_bma_mean": bma["mean"], "Q_bma_lower90": bma["lower"],
                         "Q_bma_upper90": bma["upper"], "Q_bma_std": bma["std"]},
                        index=piv.index)


def plot_bma_timeseries(df, title, mean_col, lo, hi, ylabel, fname):
    fig, ax = plt.subplots(figsize=(16, 5))
    idx = df.index
    ax.fill_between(idx, df[lo], df[hi], alpha=0.25, color="steelblue", label="90% predictive interval")
    ax.plot(idx, df[mean_col], "steelblue", lw=1.5, label="BMA mean")
    ax.plot(idx, df[mean_col].rolling(120).mean(), "r-", lw=2.5, label="10-yr rolling mean")
    ax.axvline(pd.Timestamp("2061-01-01"), color="gray", ls="--", alpha=0.5)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150); plt.close(fig)


def main():
    print("=" * 65)
    print("Phase IV — Bayesian Model Averaging (EM, Raftery et al. 2005)")
    print("=" * 65)

    weights_df, weights_vs, sigma_vs, by_vs = compute_bma_weights()
    weights_df.to_csv(OUT_DIR / "bma_weights.csv", index=False)
    by_vs.to_csv(OUT_DIR / "bma_weights_by_var_station.csv", index=False)
    agg_weights = weights_df["bma_weight"].values

    print("\nBMA weights (EM, aggregate per model):")
    print(weights_df.to_string(index=False))

    for scenario in SCENARIOS:
        print(f"\n{'-'*50}\nScenario: {scenario.upper()}")
        for station in STATIONS:
            dfb = process_climate_bma(station, scenario, weights_vs, sigma_vs, agg_weights)
            if dfb is not None:
                dfb.to_csv(OUT_DIR / f"bma_climate_{station}_{scenario}_monthly.csv")
                print(f"  {station}: saved ({len(dfb)} months)")
                plot_bma_timeseries(dfb, f"BMA Tmax – {station} ({scenario.upper()})",
                                    "tmax_monthly_bma", "tmax_monthly_lower90",
                                    "tmax_monthly_upper90", "Tmax (°C)",
                                    f"bma_tmax_{station}_{scenario}.png")
        dq = process_discharge_bma(scenario, agg_weights)
        if dq is not None:
            dq.to_csv(OUT_DIR / f"bma_discharge_{scenario}_monthly.csv")
            plot_bma_timeseries(dq, f"BMA Discharge – Karesang ({scenario.upper()})",
                                "Q_bma_mean", "Q_bma_lower90", "Q_bma_upper90",
                                "Discharge (m³/s)", f"bma_discharge_{scenario}.png")
            print(f"  Discharge BMA saved ({scenario}).")

    print(f"\n{'='*65}\nBMA outputs in: {OUT_DIR}\nDone.")


if __name__ == "__main__":
    main()

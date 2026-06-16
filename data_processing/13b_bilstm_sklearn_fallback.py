"""
13b_bilstm_sklearn_fallback.py
================================
Fallback: Multi-Layer Perceptron regression (sklearn) as an approximation
when PyTorch/TensorFlow are not available.

For Q1 paper: use 13_bilstm_streamflow.py (PyTorch) as primary.
This script is ONLY for quick testing of the pipeline.

Outputs same files as 13_bilstm_streamflow.py so downstream scripts work.
"""
import warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
FLOW_DIR  = BASE_DIR / "outputs" / "streamflow"
BC_DIR    = BASE_DIR / "outputs" / "bias_corrected"
OUT_DIR   = BASE_DIR / "outputs" / "bilstm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAG        = 3
FEAT_COLS  = ["P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]
MODELS     = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCENARIOS  = ["ssp245", "ssp585"]
STATIONS   = ["Amol", "Gharakhil", "Sari"]


def nse(obs, sim):
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (np.std(sim)/np.std(obs)-1)**2 + (np.mean(sim)/np.mean(obs)-1)**2)

def pbias(obs, sim):
    return 100 * np.sum(obs - sim) / np.sum(obs)


def make_features_flat(df: pd.DataFrame) -> tuple:
    """Flatten lagged features into a 2D matrix for sklearn."""
    X_list, y_list, dates = [], [], []
    for i in range(LAG, len(df)):
        row = []
        for l in range(LAG, -1, -1):
            for col in FEAT_COLS:
                row.append(df[col].iloc[i - l])
        X_list.append(row)
        y_list.append(df["Q_m3s"].iloc[i])
        dates.append(df["gregorian_date"].iloc[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), dates


def main():
    print("=" * 60)
    print("BiLSTM Fallback: MLP Regressor (sklearn)")
    print("NOTE: Use 13_bilstm_streamflow.py for publication-quality results")
    print("=" * 60)

    train_df = pd.read_csv(FLOW_DIR / "train_2000_2012.csv", parse_dates=["gregorian_date"])
    val_df   = pd.read_csv(FLOW_DIR / "val_2013_2016.csv",   parse_dates=["gregorian_date"])
    full_df  = pd.concat([train_df, val_df]).reset_index(drop=True)

    X_full, y_full, dates_full = make_features_flat(full_df)
    n_train = len(train_df) - LAG
    X_train, y_train = X_full[:n_train], y_full[:n_train]
    X_val,   y_val   = X_full[n_train:], y_full[n_train:]
    dates_val = dates_full[n_train:]

    sc_x = StandardScaler()
    X_train_n = sc_x.fit_transform(X_train)
    X_val_n   = sc_x.transform(X_val)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        learning_rate_init=1e-3,
    )
    mlp.fit(X_train_n, y_train)

    y_pred = mlp.predict(X_val_n)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "NSE":   round(nse(y_val, y_pred),   3),
        "KGE":   round(kge(y_val, y_pred),   3),
        "PBIAS": round(pbias(y_val, y_pred),  2),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 3),
        "r":     round(float(np.corrcoef(y_val, y_pred)[0, 1]),           3),
    }
    print("\nValidation Metrics (MLP fallback):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pd.DataFrame({"date": dates_val, "Q_obs": y_val, "Q_pred": y_pred}
                 ).to_csv(OUT_DIR / "validation_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "validation_metrics.csv", index=False)

    # -- Future projection -----------------------------------------------------
    full_obs = pd.read_csv(FLOW_DIR / "karesang_monthly_2000_2016.csv",
                           parse_dates=["gregorian_date"])

    all_projections = []
    for scenario in SCENARIOS:
        for gcm_model in MODELS:
            bc_list = []
            for stn in STATIONS:
                bc_path = BC_DIR / f"{stn}_{gcm_model}_{scenario}_bc.csv"
                if bc_path.exists():
                    sdf = pd.read_csv(bc_path, parse_dates=["date"], index_col="date")
                    sm = sdf.resample("MS").agg(
                        P_mm_month=("bc_pr", "sum"),
                        Tmax_mean=("bc_tmax", "mean"),
                        Tmin_mean=("bc_tmin", "mean"),
                    )
                    sm["Tavg_mean"] = (sm["Tmax_mean"] + sm["Tmin_mean"]) / 2
                    bc_list.append(sm)
            if not bc_list:
                continue
            basin_bc = pd.concat(bc_list).groupby(level=0).mean()
            basin_bc = basin_bc.loc["2015-01-01":"2100-12-31"].reset_index()
            basin_bc = basin_bc.rename(columns={"date": "gregorian_date"})
            basin_bc["gregorian_date"] = pd.to_datetime(basin_bc["gregorian_date"])

            last_obs = full_obs.tail(LAG)[FEAT_COLS + ["gregorian_date"]].copy()
            future_df = pd.concat([last_obs, basin_bc[["gregorian_date"] + FEAT_COLS]]
                                  ).reset_index(drop=True)

            Q_list, d_list = [], []
            for i in range(LAG, len(future_df)):
                row = []
                for l in range(LAG, -1, -1):
                    for col in FEAT_COLS:
                        row.append(future_df[col].iloc[i - l])
                x_norm = sc_x.transform([row])
                q_pred = max(0.0, float(mlp.predict(x_norm)[0]))
                Q_list.append(q_pred)
                d_list.append(future_df["gregorian_date"].iloc[i])

            proj = pd.DataFrame({"date": d_list, "Q_m3s": Q_list,
                                  "model": gcm_model, "scenario": scenario})
            proj = proj[proj["date"].dt.year >= 2021]
            all_projections.append(proj)
            print(f"  {gcm_model}/{scenario}: {len(proj)} months")

    if all_projections:
        all_df = pd.concat(all_projections, ignore_index=True)
        for scen in SCENARIOS:
            sub = all_df[all_df["scenario"] == scen]
            sub.to_csv(OUT_DIR / f"future_discharge_{scen}.csv", index=False)
        print("  Future discharge projections saved.")

    # Validation plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates_val, y_val,  "k-o",  ms=4, lw=1.5, label="Observed")
    ax.plot(dates_val, y_pred, "r--^", ms=4, lw=1.5,
            label=f"MLP Fallback (NSE={metrics['NSE']}, KGE={metrics['KGE']})")
    ax.set_ylabel("Discharge (m3/s)"); ax.set_title("Haraz-Karesang: MLP Validation (2013-2016)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "validation_plot.png", dpi=150); plt.close(fig)

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

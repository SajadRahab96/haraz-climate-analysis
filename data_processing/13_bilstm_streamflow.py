"""
13_bilstm_streamflow.py
========================
Bidirectional LSTM model for monthly streamflow prediction at
Haraz-Karesang station.

Architecture (v2 - autoregressive):
  Input  -> [P_mm, Tmax, Tmin, Tavg, month_sin, month_cos, Q_prev]
             per timestep (LAG+1 = 4 timesteps)
             Q_prev at timestep t = Q(t-1) (antecedent discharge)
  Layer1 -> Bidirectional LSTM (64 units) + Dropout(0.2)
  Layer2 -> Bidirectional LSTM (32 units) + Dropout(0.2)
  Layer3 -> Linear(16) -> ReLU -> Linear(1) - Q_m3s

Key changes vs v1:
  - Added Q(t-1) as autoregressive feature at each sequence step
  - Added month_sin / month_cos for seasonal encoding
  - Future projection: rolling autoregressive (predicted Q fed back)
  - At LARS-WG period boundaries: Q buffer reset to observed climatology

Training:  2000-01 to 2012-12  (156 months)
Validation: 2013-01 to 2016-12  (48 months)
Metrics:   NSE, KGE, PBIAS, RMSE, r

Framework: PyTorch
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} loaded.")
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available.")

# -- Paths ---------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
FLOW_DIR  = BASE_DIR / "outputs" / "streamflow"
BC_DIR    = BASE_DIR / "outputs" / "bias_corrected"
OUT_DIR   = BASE_DIR / "outputs" / "bilstm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAG        = 3
EPOCHS     = 500
BATCH_SIZE = 16
PATIENCE   = 50
SEED       = 42

LARS_WG_MODELS   = ["CanESM5", "GFDL-ESM4", "MRI-ESM2-0", "UKESM1-0-LL"]
DIRECT_BC_MODELS = ["IPSL-CM6A-LR", "MPI-ESM1-2-HR"]
MODELS    = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
PERIOD_BOUNDARIES = ["2041-01-01", "2061-01-01", "2081-01-01"]
SCENARIOS = ["ssp245", "ssp585"]
STATIONS  = ["Amol", "Gharakhil", "Sari"]

# Climate features (same as before); Q_prev added separately
CLIM_COLS = ["P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]
# Total features per timestep: 4 climate + 2 seasonal + 1 Q_prev = 7
N_FEATURES = len(CLIM_COLS) + 2 + 1   # 7


# -- Metrics -------------------------------------------------------------------
def nse(obs, sim):
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (np.std(sim)/np.std(obs)-1)**2 + (np.mean(sim)/np.mean(obs)-1)**2)

def pbias(obs, sim):
    return 100 * np.sum(obs - sim) / np.sum(obs)

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim)**2))


# -- Feature matrix (autoregressive) ------------------------------------------
def make_features(df: pd.DataFrame) -> tuple:
    """
    For each target month t, build sequence of LAG+1 timesteps.
    Each timestep l includes:
      [P, Tmax, Tmin, Tavg, month_sin, month_cos, Q(l-1)]
    where Q(l-1) is the actual discharge one step before l.
    Requires df to have Q_m3s from index 0 (so Q_prev for the earliest
    timestep comes from index 0 of df).
    """
    # Precompute seasonal features
    months = df["gregorian_date"].dt.month.values
    msin = np.sin(2 * np.pi * months / 12).astype(np.float32)
    mcos = np.cos(2 * np.pi * months / 12).astype(np.float32)
    Q    = df["Q_m3s"].values.astype(np.float32)

    X_list, y_list, dates = [], [], []
    # Start from LAG+1 so Q_prev at earliest sequence step is always available
    for i in range(LAG + 1, len(df)):
        seq = []
        for l in range(LAG, -1, -1):   # l = LAG, LAG-1, ..., 0  ->  timesteps t-LAG .. t
            ti = i - l                   # absolute row index for this timestep
            q_prev = Q[ti - 1]           # Q one month before this timestep
            seq.append([
                df[CLIM_COLS[0]].iloc[ti],
                df[CLIM_COLS[1]].iloc[ti],
                df[CLIM_COLS[2]].iloc[ti],
                df[CLIM_COLS[3]].iloc[ti],
                msin[ti],
                mcos[ti],
                q_prev,
            ])
        X_list.append(seq)
        y_list.append(Q[i])
        dates.append(df["gregorian_date"].iloc[i])

    X = np.array(X_list, dtype=np.float32)   # (n, LAG+1, 7)
    y = np.array(y_list,  dtype=np.float32)
    return X, y, dates


def normalize(X_train, X_val):
    mu  = X_train.mean(axis=(0, 1), keepdims=True)
    sig = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mu) / sig, (X_val - mu) / sig, mu, sig


def normalize_y(y_train, y_val):
    mu  = y_train.mean()
    sig = y_train.std() + 1e-8
    return (y_train - mu) / sig, (y_val - mu) / sig, mu, sig


# -- BiLSTM model --------------------------------------------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(hidden2 * 2, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# -- Training ------------------------------------------------------------------
def train_model(train_csv: Path, val_csv: Path):
    print("\n" + "="*60)
    print("Training BiLSTM Streamflow Model - v2 (autoregressive)")
    print("="*60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df = pd.read_csv(train_csv, parse_dates=["gregorian_date"])
    val_df   = pd.read_csv(val_csv,   parse_dates=["gregorian_date"])
    print(f"  Train: {len(train_df)} months | Val: {len(val_df)} months")

    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_full, y_full, dates_full = make_features(full_df)

    # n_train samples produced from training portion
    n_train = len(train_df) - (LAG + 1)
    X_train, y_train = X_full[:n_train], y_full[:n_train]
    X_val,   y_val   = X_full[n_train:], y_full[n_train:]
    dates_val = dates_full[n_train:]

    X_tr_n, X_va_n, mu_x, sig_x = normalize(X_train, X_val)
    y_tr_n, y_va_n, mu_y, sig_y = normalize_y(y_train, y_val)

    model = BiLSTMModel(n_features=N_FEATURES)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Feature shape: {X_train.shape}  (samples, timesteps={LAG+1}, features={N_FEATURES})")

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-6)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr_n)
    y_tr_t = torch.tensor(y_tr_n)
    X_va_t = torch.tensor(X_va_n)
    y_va_t = torch.tensor(y_va_n)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = np.inf
    patience_ctr  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}: train={epoch_loss:.5f}, val={val_loss:.5f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "model_weights.pt")

    # -- Validation metrics ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_va_t).numpy() * sig_y + mu_y
    y_obs = y_val

    metrics = {
        "NSE":   round(nse(y_obs, y_pred),   3),
        "KGE":   round(kge(y_obs, y_pred),   3),
        "PBIAS": round(pbias(y_obs, y_pred),  2),
        "RMSE":  round(rmse(y_obs, y_pred),   3),
        "r":     round(float(np.corrcoef(y_obs, y_pred)[0, 1]), 3),
    }
    print("\n  Validation Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # -- Recursive validation (projection mode: predicted Q fed back) ---------
    # Mirrors how the model is used for future projection, where observed
    # antecedent discharge is unavailable.
    months_full = full_df["gregorian_date"].dt.month.values
    msin_f = np.sin(2 * np.pi * months_full / 12).astype(np.float32)
    mcos_f = np.cos(2 * np.pi * months_full / 12).astype(np.float32)
    Q_obs_full = full_df["Q_m3s"].values.astype(np.float32)
    n_tr_rows  = len(train_df)

    # Q buffer initialised from last observed training values
    q_sim = Q_obs_full.copy()
    rec_preds, rec_dates = [], []
    for i in range(n_tr_rows, len(full_df)):
        seq = []
        for l in range(LAG, -1, -1):
            ti = i - l
            q_prev = q_sim[ti - 1]
            seq.append([
                full_df[CLIM_COLS[0]].iloc[ti],
                full_df[CLIM_COLS[1]].iloc[ti],
                full_df[CLIM_COLS[2]].iloc[ti],
                full_df[CLIM_COLS[3]].iloc[ti],
                msin_f[ti], mcos_f[ti], q_prev,
            ])
        X1 = np.array(seq, dtype=np.float32).reshape(1, LAG + 1, N_FEATURES)
        X1n = (X1 - mu_x) / sig_x
        with torch.no_grad():
            qp = max(0.0, model(torch.tensor(X1n)).item() * sig_y + mu_y)
        q_sim[i] = qp          # feed back prediction (recursive)
        rec_preds.append(qp)
        rec_dates.append(full_df["gregorian_date"].iloc[i])

    rec_obs  = Q_obs_full[n_tr_rows:]
    rec_pred = np.array(rec_preds, dtype=np.float32)
    rec_metrics = {
        "NSE_recursive":   round(nse(rec_obs, rec_pred),   3),
        "KGE_recursive":   round(kge(rec_obs, rec_pred),   3),
        "PBIAS_recursive": round(pbias(rec_obs, rec_pred),  2),
        "RMSE_recursive":  round(rmse(rec_obs, rec_pred),   3),
        "r_recursive":     round(float(np.corrcoef(rec_obs, rec_pred)[0, 1]), 3),
    }
    print("\n  Recursive (projection-mode) Validation Metrics:")
    for k, v in rec_metrics.items():
        print(f"    {k}: {v}")

    pd.DataFrame({"date": dates_val, "Q_obs": y_obs, "Q_pred": y_pred}
                 ).to_csv(OUT_DIR / "validation_results.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    pd.DataFrame([{**metrics, **rec_metrics}]).to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    np.save(OUT_DIR / "norm_params.npy",
            {"mu_x": mu_x, "sig_x": sig_x, "mu_y": mu_y, "sig_y": sig_y})

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("BiLSTM Training History"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(dates_val, y_obs,  "k-o",  ms=4, lw=1.5, label="Observed")
    axes[1].plot(dates_val, y_pred, "r--^", ms=4, lw=1.5,
                 label=f"BiLSTM (NSE={metrics['NSE']}, KGE={metrics['KGE']})")
    axes[1].set_ylabel("Discharge (m3/s)")
    axes[1].set_title("Haraz-Karesang: BiLSTM Validation (2013-2016)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "validation_plot.png", dpi=150)
    plt.close(fig)

    return model, mu_x, sig_x, mu_y, sig_y, metrics


# -- Future Projection (autoregressive) ---------------------------------------
def project_future(model, mu_x, sig_x, mu_y, sig_y, full_obs_csv: Path):
    print("\n" + "="*60)
    print("Projecting Future Streamflow 2021-2100 (autoregressive)")
    print("="*60)

    model.eval()
    obs_monthly = pd.read_csv(full_obs_csv, parse_dates=["gregorian_date"])
    obs_monthly = obs_monthly.sort_values("gregorian_date").reset_index(drop=True)

    # Monthly Q climatology from observed: used to warm-up Q buffer at boundaries
    obs_q_clim = obs_monthly.groupby(obs_monthly["gregorian_date"].dt.month)["Q_m3s"].mean()

    all_projections = []

    for scenario in SCENARIOS:
        for gcm_model in MODELS:
            bc_monthly_list = []
            for stn in STATIONS:
                bc_path = BC_DIR / f"{stn}_{gcm_model}_{scenario}_bc.csv"
                if bc_path.exists():
                    stn_df = pd.read_csv(bc_path, parse_dates=["date"], index_col="date")
                    stn_monthly = stn_df.resample("MS").agg(
                        P_mm_month=("bc_pr",   "sum"),
                        Tmax_mean= ("bc_tmax",  "mean"),
                        Tmin_mean= ("bc_tmin",  "mean"),
                    )
                    stn_monthly["Tavg_mean"] = (stn_monthly["Tmax_mean"] + stn_monthly["Tmin_mean"]) / 2
                    bc_monthly_list.append(stn_monthly)

            if not bc_monthly_list:
                print(f"  SKIP {gcm_model}/{scenario}: no BC files")
                continue

            basin_bc = pd.concat(bc_monthly_list).groupby(level=0).mean()
            basin_bc = basin_bc.loc["2015-01-01":"2100-12-31"].reset_index()
            basin_bc = basin_bc.rename(columns={"date": "gregorian_date"})
            basin_bc["gregorian_date"] = pd.to_datetime(basin_bc["gregorian_date"])

            is_larswg = gcm_model in LARS_WG_MODELS

            # Prepend last (LAG+1) observed months for initial context + Q_prev
            last_obs = obs_monthly.tail(LAG + 1)[CLIM_COLS + ["gregorian_date", "Q_m3s"]].copy()
            future_df = pd.concat([
                last_obs,
                basin_bc[["gregorian_date"] + CLIM_COLS]
            ]).reset_index(drop=True)

            # For future rows, we don't have observed Q - use rolling Q buffer
            # Initialize Q buffer with last observed values
            q_buffer = list(obs_monthly["Q_m3s"].tail(LAG + 1).values)

            # LARS-WG period boundaries (row indices in future_df)
            boundary_indices = set()
            if is_larswg:
                for bdate in PERIOD_BOUNDARIES:
                    bd = pd.Timestamp(bdate)
                    matches = future_df.index[future_df["gregorian_date"] == bd].tolist()
                    if matches:
                        boundary_indices.add(matches[0])

            Q_pred_list, date_list = [], []

            i = LAG + 1
            while i < len(future_df):
                # At a LARS-WG period boundary: reset the Q memory to climatology
                # (synthetic Q has no cross-period temporal continuity) and drop the
                # first LAG warm-up months of the new period. A while-loop is used so
                # the index can actually advance past the warm-up window (a previous
                # `for i in range(...)`+`i += LAG` was a no-op and produced duplicate
                # dates and NaN rows at boundaries).
                if is_larswg and i in boundary_indices:
                    month_at_boundary = future_df["gregorian_date"].iloc[i].month
                    for skip_lag in range(LAG + 1):
                        m = ((month_at_boundary - 1 - skip_lag) % 12) + 1
                        q_buffer[-(skip_lag + 1)] = float(obs_q_clim.get(m, obs_q_clim.mean()))
                    n_skip = min(LAG, len(future_df) - i)
                    for skip_i in range(i, i + n_skip):
                        Q_pred_list.append(np.nan)
                        date_list.append(future_df["gregorian_date"].iloc[skip_i])
                    i += n_skip
                    continue

                # Build feature sequence [t-LAG .. t]
                seq = []
                for l in range(LAG, -1, -1):
                    ti = i - l
                    month_ti = future_df["gregorian_date"].iloc[ti].month
                    # Q_prev for this step = Q buffer value at position -(l+1)
                    buf_idx = -(l + 1)
                    q_prev = q_buffer[buf_idx] if abs(buf_idx) <= len(q_buffer) else q_buffer[0]
                    seq.append([
                        future_df[CLIM_COLS[0]].iloc[ti],
                        future_df[CLIM_COLS[1]].iloc[ti],
                        future_df[CLIM_COLS[2]].iloc[ti],
                        future_df[CLIM_COLS[3]].iloc[ti],
                        np.sin(2 * np.pi * month_ti / 12),
                        np.cos(2 * np.pi * month_ti / 12),
                        q_prev,
                    ])

                X_single = np.array(seq, dtype=np.float32).reshape(1, LAG + 1, N_FEATURES)
                X_norm   = (X_single - mu_x) / sig_x
                with torch.no_grad():
                    q_pred = max(0.0, model(torch.tensor(X_norm)).item() * sig_y + mu_y)

                Q_pred_list.append(q_pred)
                date_list.append(future_df["gregorian_date"].iloc[i])

                # Update rolling Q buffer (drop oldest, append new prediction)
                q_buffer.append(q_pred)
                if len(q_buffer) > LAG + 1:
                    q_buffer.pop(0)
                i += 1

            proj_df = pd.DataFrame({
                "date":     date_list,
                "Q_m3s":    Q_pred_list,
                "model":    gcm_model,
                "scenario": scenario,
            })
            proj_df = proj_df[proj_df["date"].dt.year >= 2021]
            all_projections.append(proj_df)
            print(f"  {gcm_model}/{scenario}: {len(proj_df)} months projected")

    if all_projections:
        all_df = pd.concat(all_projections, ignore_index=True)
        for scen in SCENARIOS:
            sub = all_df[all_df["scenario"] == scen]
            sub.to_csv(OUT_DIR / f"future_discharge_{scen}.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for ax, scen in zip(axes, SCENARIOS):
            sub = all_df[all_df["scenario"] == scen]
            for gm in MODELS:
                gm_sub = sub[sub["model"] == gm]
                if len(gm_sub):
                    ax.plot(gm_sub["date"], gm_sub["Q_m3s"], lw=0.8, alpha=0.6, label=gm)
            ens = sub.groupby("date")["Q_m3s"].mean()
            ax.plot(ens.index, ens.values, "k-", lw=2, label="Ensemble Mean")
            ax.set_title(f"Future Discharge: {scen.upper()}")
            ax.set_ylabel("Discharge (m3/s)")
            ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "future_discharge_projections.png", dpi=150)
        plt.close(fig)
        print("  Future discharge plots saved.")


# -- Main ----------------------------------------------------------------------
def main():
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required.")
        sys.exit(1)

    train_csv = FLOW_DIR / "train_2000_2012.csv"
    val_csv   = FLOW_DIR / "val_2013_2016.csv"
    full_csv  = FLOW_DIR / "karesang_monthly_2000_2016.csv"

    if not train_csv.exists():
        print("ERROR: Run 12_prepare_streamflow_data.py first.")
        sys.exit(1)

    model, mu_x, sig_x, mu_y, sig_y, metrics = train_model(train_csv, val_csv)

    bc_files = list(BC_DIR.glob("*_bc.csv"))
    if bc_files:
        project_future(model, mu_x, sig_x, mu_y, sig_y, full_csv)
    else:
        print("NOTE: No bias-corrected files found. Run 11_bias_correction_dqm.py first.")

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

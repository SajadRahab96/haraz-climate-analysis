#!/usr/bin/env python3
"""
33_deep_ensemble.py
===================
Structural (multi-architecture) deep ensemble for streamflow projection.

Rationale: the framework treats climate-model structural uncertainty with
EM-based BMA; this script extends the same philosophy to the hydrological
component by combining three recurrent architectures (BiLSTM, unidirectional
LSTM, BiGRU), each trained with three random seeds, into an equal-weight
nine-member deep ensemble. Equal weights are deliberate: no member is selected
or weighted on the validation period, eliminating selection bias.

Steps
-----
1. Train the 9 members on 2000-2012 (identical protocol to Section 4.3).
2. Validate on 2013-2016: metrics of the ensemble-mean series in one-step and
   fully recursive modes, plus the per-member range.
3. Project 2021-2100 monthly discharge for all 6 GCMs x 2 SSPs with each
   member (mirroring 13_bilstm_streamflow.py: three-station mean forcing,
   climatology Q-buffer resets at LARS-WG period boundaries for library
   models), and summarise the BMA-weighted long-term change per architecture.

Outputs (outputs/bilstm/ensemble/):
  ensemble_validation.csv, member_validation.csv,
  future_discharge_members_{scenario}.csv, ensemble_future_summary.csv
Production artifacts from 13_bilstm_streamflow.py are not touched.
"""

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

BASE = Path(__file__).resolve().parent.parent
FLOW = BASE / "outputs" / "streamflow"
BC   = BASE / "outputs" / "bias_corrected"
OUT  = BASE / "outputs" / "bilstm" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

LAG = 3
EPOCHS, BATCH, PATIENCE = 500, 16, 50
CLIM_COLS = ["P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]
N_FEATURES = 7
SEEDS = [42, 123, 2027]
ARCHS = [("BiLSTM", "lstm", True), ("LSTM", "lstm", False), ("BiGRU", "gru", True)]
MODELS = ["CanESM5", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
LARS_WG_MODELS = {"CanESM5", "GFDL-ESM4", "MRI-ESM2-0", "UKESM1-0-LL"}
SCENARIOS = ["ssp245", "ssp585"]
STATIONS = ["Amol", "Gharakhil", "Sari"]
PERIOD_BOUNDARIES = ["2041-01-01", "2061-01-01", "2081-01-01"]


def _load32():
    spec = importlib.util.spec_from_file_location(
        "m32", BASE / "data_processing" / "32_bilstm_robustness.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def train_member(m32, cell, bidir, seed, X_tr, y_tr, X_va, y_va):
    torch.manual_seed(seed); np.random.seed(seed)
    model = m32.SeqModel(cell, bidir)
    mu_x = X_tr.mean(axis=(0, 1), keepdims=True); sig_x = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    mu_y = float(y_tr.mean()); sig_y = float(y_tr.std()) + 1e-8
    Xtn = (X_tr - mu_x) / sig_x; Xvn = (X_va - mu_x) / sig_x
    ytn = (y_tr - mu_y) / sig_y; yvn = (y_va - mu_y) / sig_y
    opt = Adam(model.parameters(), lr=1e-3)
    sch = ReduceLROnPlateau(opt, factor=0.5, patience=20, min_lr=1e-6)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtn), torch.tensor(ytn)),
                        batch_size=BATCH, shuffle=True)
    Xvt, yvt = torch.tensor(Xvn), torch.tensor(yvn)
    best, ctr, state = np.inf, 0, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xvt), yvt).item()
        sch.step(vl)
        if vl < best: best, ctr, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else: ctr += 1
        if ctr >= PATIENCE: break
    model.load_state_dict(state)
    with torch.no_grad():
        onestep = model(Xvt).numpy() * sig_y + mu_y
    return model, (mu_x, sig_x, mu_y, sig_y), onestep


def recursive_series(m32, model, norm, full, n_tr_rows):
    o, p = m32.recursive_eval(model, norm, full, n_tr_rows)
    return o, p


def project_member(model, norm, basin_bc_by, obs_monthly, obs_q_clim, gcm, scen):
    """Mirror 13_bilstm_streamflow.project_future for one member/GCM/scenario."""
    mu_x, sig_x, mu_y, sig_y = norm
    basin_bc = basin_bc_by[(gcm, scen)]
    is_lars = gcm in LARS_WG_MODELS
    last_obs = obs_monthly.tail(LAG + 1)[CLIM_COLS + ["gregorian_date", "Q_m3s"]].copy()
    fut = pd.concat([last_obs, basin_bc[["gregorian_date"] + CLIM_COLS]]).reset_index(drop=True)
    q_buffer = list(obs_monthly["Q_m3s"].tail(LAG + 1).values)
    boundary_idx = set()
    if is_lars:
        for b in PERIOD_BOUNDARIES:
            m = fut.index[fut["gregorian_date"] == pd.Timestamp(b)].tolist()
            if m: boundary_idx.add(m[0])
    months = fut["gregorian_date"].dt.month.values
    preds, dates = [], []
    model.eval()
    i = LAG + 1
    while i < len(fut):
        if is_lars and i in boundary_idx:
            mb = months[i]
            for sl in range(LAG + 1):
                mm = ((mb - 1 - sl) % 12) + 1
                q_buffer[-(sl + 1)] = float(obs_q_clim.get(mm, obs_q_clim.mean()))
            n_skip = min(LAG, len(fut) - i)
            for si in range(i, i + n_skip):
                preds.append(np.nan); dates.append(fut["gregorian_date"].iloc[si])
            i += n_skip
            continue
        seq = []
        for l in range(LAG, -1, -1):
            ti = i - l
            q_prev = q_buffer[-(l + 1)] if (l + 1) <= len(q_buffer) else q_buffer[0]
            seq.append([fut[c].iloc[ti] for c in CLIM_COLS] +
                       [np.sin(2 * np.pi * months[ti] / 12),
                        np.cos(2 * np.pi * months[ti] / 12), q_prev])
        X1 = ((np.array(seq, np.float32).reshape(1, LAG + 1, N_FEATURES)) - mu_x) / sig_x
        with torch.no_grad():
            qp = max(0.0, model(torch.tensor(X1)).item() * sig_y + mu_y)
        preds.append(qp); dates.append(fut["gregorian_date"].iloc[i])
        q_buffer.append(qp)
        if len(q_buffer) > LAG + 1: q_buffer.pop(0)
        i += 1
    df = pd.DataFrame({"date": dates, "Q_m3s": preds})
    return df[df["date"].dt.year >= 2021]


def main():
    m32 = _load32()
    tr = pd.read_csv(FLOW / "train_2000_2012.csv", parse_dates=["gregorian_date"])
    va = pd.read_csv(FLOW / "val_2013_2016.csv", parse_dates=["gregorian_date"])
    full = pd.concat([tr, va]).reset_index(drop=True)
    X_full, y_full = m32.make_features(full)
    n_tr = len(tr) - (LAG + 1)
    X_tr, y_tr = X_full[:n_tr], y_full[:n_tr]
    X_va, y_va = X_full[n_tr:], y_full[n_tr:]

    # ---------- 1-2. train members + validation ----------
    print("=" * 64); print("Training 9 ensemble members (3 architectures x 3 seeds)")
    members, one_preds, rec_preds, mrows = [], [], [], []
    for arch, cell, bidir in ARCHS:
        for seed in SEEDS:
            name = f"{arch}_s{seed}"
            model, norm, onestep = train_member(m32, cell, bidir, seed, X_tr, y_tr, X_va, y_va)
            o_rec, p_rec = recursive_series(m32, model, norm, full, len(tr))
            members.append((name, arch, model, norm))
            one_preds.append(onestep); rec_preds.append(p_rec)
            mrows.append({"member": name, "arch": arch, "seed": seed,
                          "NSE_onestep": round(m32.nse(y_va, onestep), 3),
                          "NSE_recursive": round(m32.nse(o_rec, p_rec), 3)})
            print(f"  {name}: one-step {mrows[-1]['NSE_onestep']}, recursive {mrows[-1]['NSE_recursive']}")
    pd.DataFrame(mrows).to_csv(OUT / "member_validation.csv", index=False)

    ens_one = np.mean(one_preds, axis=0)
    ens_rec = np.mean(rec_preds, axis=0)
    o_rec = y_full[n_tr:]
    vrows = []
    for mode, pred in [("one-step", ens_one), ("recursive", ens_rec)]:
        vrows.append({"mode": mode,
                      "NSE": round(m32.nse(y_va if mode == "one-step" else o_rec, pred), 3),
                      "KGE": round(m32.kge(y_va if mode == "one-step" else o_rec, pred), 3),
                      "r": round(float(np.corrcoef(y_va if mode == "one-step" else o_rec, pred)[0, 1]), 3),
                      "PBIAS_pct": round(m32.pbias(y_va if mode == "one-step" else o_rec, pred), 1),
                      "RMSE_m3s": round(float(np.sqrt(np.mean(((y_va if mode == 'one-step' else o_rec) - pred) ** 2))), 2)})
    dfv = pd.DataFrame(vrows); dfv.to_csv(OUT / "ensemble_validation.csv", index=False)
    print("\nEqual-weight 9-member ensemble validation:")
    print(dfv.to_string(index=False))

    # ---------- 3. future projections ----------
    print("\n" + "=" * 64); print("Future projections 2021-2100 (12 GCM-scenario combos x 9 members)")
    obs_monthly = pd.read_csv(FLOW / "karesang_monthly_2000_2016.csv", parse_dates=["gregorian_date"]).sort_values("gregorian_date").reset_index(drop=True)
    obs_q_clim = obs_monthly.groupby(obs_monthly["gregorian_date"].dt.month)["Q_m3s"].mean()
    obs_q_mean = obs_monthly["Q_m3s"].mean()
    basin_bc_by = {}
    for gcm in MODELS:
        for scen in SCENARIOS:
            parts = []
            for stn in STATIONS:
                p = BC / f"{stn}_{gcm}_{scen}_bc.csv"
                d = pd.read_csv(p, parse_dates=["date"], index_col="date")
                mm = d.resample("MS").agg(P_mm_month=("bc_pr", "sum"),
                                          Tmax_mean=("bc_tmax", "mean"),
                                          Tmin_mean=("bc_tmin", "mean"))
                mm["Tavg_mean"] = (mm["Tmax_mean"] + mm["Tmin_mean"]) / 2
                parts.append(mm)
            bb = pd.concat(parts).groupby(level=0).mean().loc["2015-01-01":"2100-12-31"].reset_index()
            bb = bb.rename(columns={"date": "gregorian_date"})
            basin_bc_by[(gcm, scen)] = bb

    wdf = pd.read_csv(BASE / "outputs" / "bma" / "bma_weights.csv")
    W = dict(zip(wdf["model"], wdf["bma_weight"]))

    srows = []
    for scen in SCENARIOS:
        allrec = []
        for name, arch, model, norm in members:
            for gcm in MODELS:
                dfp = project_member(model, norm, basin_bc_by, obs_monthly, obs_q_clim, gcm, scen)
                dfp["member"] = name; dfp["arch"] = arch; dfp["model"] = gcm
                allrec.append(dfp)
        big = pd.concat(allrec, ignore_index=True)
        big.to_csv(OUT / f"future_discharge_members_{scen}.csv", index=False)
        # BMA-weighted long-term change per architecture and for the full ensemble
        lt = big[(big["date"] >= "2061-01-01") & (big["date"] <= "2100-12-31")]
        def bma_lt(sub):
            g = sub.groupby("model")["Q_m3s"].mean()
            w = np.array([W[m] for m in g.index]); w = w / w.sum()
            return float((g.values * w).sum())
        for arch in [a for a, _, _ in ARCHS]:
            q = bma_lt(lt[lt["arch"] == arch])
            srows.append({"scenario": scen, "set": arch, "Q_LT_m3s": round(q, 2),
                          "pct_change_vs_obs": round(100 * (q - obs_q_mean) / obs_q_mean, 1)})
        q = bma_lt(lt)
        srows.append({"scenario": scen, "set": "ENSEMBLE(9)", "Q_LT_m3s": round(q, 2),
                      "pct_change_vs_obs": round(100 * (q - obs_q_mean) / obs_q_mean, 1)})
        print(f"  {scen}: done ({len(big):,} rows)")
    dfs = pd.DataFrame(srows); dfs.to_csv(OUT / "ensemble_future_summary.csv", index=False)
    print("\nBMA-weighted long-term (2061-2100) discharge change vs observed mean "
          f"({obs_q_mean:.2f} m3/s):")
    print(dfs.to_string(index=False))


if __name__ == "__main__":
    main()

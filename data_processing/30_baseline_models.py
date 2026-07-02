#!/usr/bin/env python3
"""
30_baseline_models.py
=====================
Baseline models for monthly streamflow at Haraz-Karesang, to contextualise the
BiLSTM results (Section 4.3). All baselines use the same training period
(2000-2012), the same hold-out validation window (2013-2016) and, where
applicable, the same predictors as the BiLSTM:

  1. Monthly climatology     Q_hat(t) = mean training Q for calendar month m(t)
  2. Persistence             Q_hat(t) = Q(t-1)                     (one-step)
  3. Multiple linear regression (MLR) on the 7 BiLSTM features    (one-step)
  4. Random forest (RF) on the same features, fixed seed          (one-step)

For MLR and RF a recursive variant is also evaluated, feeding predictions back
as the antecedent-discharge input, mirroring the BiLSTM projection mode.

Outputs: outputs/bilstm/baseline_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent.parent
FLOW = BASE / "outputs" / "streamflow"
OUT  = BASE / "outputs" / "bilstm"
SEED = 42


def nse(o, s):  return 1 - np.sum((o - s) ** 2) / np.sum((o - np.mean(o)) ** 2)
def kge(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2
                       + (np.mean(s) / np.mean(o) - 1) ** 2)
def pbias(o, s): return 100 * np.sum(o - s) / np.sum(o)
def rmse(o, s):  return np.sqrt(np.mean((o - s) ** 2))


def features(df, q_prev):
    """Feature matrix matching the BiLSTM inputs at the current time step."""
    m = df["month"].values
    return np.column_stack([
        df["P_mm_month"].values, df["Tmax_mean"].values,
        df["Tmin_mean"].values,  df["Tavg_mean"].values,
        np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
        q_prev,
    ])


def main():
    tr = pd.read_csv(FLOW / "train_2000_2012.csv", parse_dates=["gregorian_date"])
    va = pd.read_csv(FLOW / "val_2013_2016.csv", parse_dates=["gregorian_date"])
    full = pd.concat([tr, va]).reset_index(drop=True)
    n_tr = len(tr)

    Q = full["Q_m3s"].values.astype(float)
    q_prev_full = np.r_[Q[0], Q[:-1]]                # Q(t-1), first month backfilled

    X_full = features(full, q_prev_full)
    X_tr, y_tr = X_full[:n_tr], Q[:n_tr]
    X_va, y_va = X_full[n_tr:], Q[n_tr:]
    months_va = full["month"].values[n_tr:]

    rows = []

    def add(name, mode, pred):
        rows.append({"model": name, "mode": mode,
                     "NSE": round(nse(y_va, pred), 3), "KGE": round(kge(y_va, pred), 3),
                     "r": round(float(np.corrcoef(y_va, pred)[0, 1]), 3),
                     "PBIAS_pct": round(pbias(y_va, pred), 1),
                     "RMSE_m3s": round(rmse(y_va, pred), 2)})

    # 1. climatology
    clim = tr.groupby("month")["Q_m3s"].mean()
    add("Monthly climatology", "n/a", clim.reindex(months_va).values)

    # 2. persistence (one-step by definition)
    add("Persistence Q(t-1)", "one-step", q_prev_full[n_tr:])

    # 3-4. MLR and RF: one-step and recursive
    models = [("MLR (7 features)", LinearRegression()),
              ("Random forest (7 features)", RandomForestRegressor(
                  n_estimators=500, random_state=SEED, min_samples_leaf=2))]
    for name, mdl in models:
        mdl.fit(X_tr, y_tr)
        add(name, "one-step", np.clip(mdl.predict(X_va), 0, None))
        # recursive: feed predictions back as Q(t-1)
        qp = Q[n_tr - 1]
        rec = []
        for i in range(len(y_va)):
            x = X_va[i].copy(); x[-1] = qp
            qp = max(0.0, float(mdl.predict(x.reshape(1, -1))[0]))
            rec.append(qp)
        add(name, "recursive", np.array(rec))

    # BiLSTM reference rows (from validation_metrics.csv, for one CSV of record)
    vm = pd.read_csv(OUT / "validation_metrics.csv").iloc[0]
    rows.append({"model": "BiLSTM (this study)", "mode": "one-step",
                 "NSE": vm["NSE"], "KGE": vm["KGE"], "r": vm["r"],
                 "PBIAS_pct": vm["PBIAS"], "RMSE_m3s": vm["RMSE"]})
    rows.append({"model": "BiLSTM (this study)", "mode": "recursive",
                 "NSE": vm["NSE_recursive"], "KGE": vm["KGE_recursive"], "r": vm["r_recursive"],
                 "PBIAS_pct": vm["PBIAS_recursive"], "RMSE_m3s": vm["RMSE_recursive"]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
32_bilstm_robustness.py
=======================
Three robustness analyses for the BiLSTM streamflow model (Section 4.3):

  A. Training-length sensitivity: retrain the BiLSTM with training windows
     ending in 2008, 2010, and 2012 (108/132/156 months), identical hold-out
     validation (2013-2016, one-step mode).
  B. Architecture variants: unidirectional LSTM and GRU with the same layout,
     hyperparameters, and training protocol; one-step and recursive metrics.
  C. Moving-block bootstrap (block length 12, 1000 resamples) 95% confidence
     intervals for the one-step and recursive validation metrics of the
     PRODUCTION model (loaded from the saved weights; no retraining).

Nothing in outputs/bilstm/ produced by 13_bilstm_streamflow.py is overwritten;
all outputs go to outputs/bilstm/robustness/.
"""

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
BIL  = BASE / "outputs" / "bilstm"
OUT  = BIL / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

LAG, SEED = 3, 42
EPOCHS, BATCH, PATIENCE = 500, 16, 50
CLIM_COLS = ["P_mm_month", "Tmax_mean", "Tmin_mean", "Tavg_mean"]
N_FEATURES = 7


def nse(o, s):  return 1 - np.sum((o - s) ** 2) / np.sum((o - np.mean(o)) ** 2)
def kge(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2
                       + (np.mean(s) / np.mean(o) - 1) ** 2)
def pbias(o, s): return 100 * np.sum(o - s) / np.sum(o)


def make_features(df):
    months = df["gregorian_date"].dt.month.values
    msin = np.sin(2 * np.pi * months / 12).astype(np.float32)
    mcos = np.cos(2 * np.pi * months / 12).astype(np.float32)
    Q = df["Q_m3s"].values.astype(np.float32)
    X, y = [], []
    for i in range(LAG + 1, len(df)):
        seq = []
        for l in range(LAG, -1, -1):
            ti = i - l
            seq.append([df[c].iloc[ti] for c in CLIM_COLS] +
                       [msin[ti], mcos[ti], Q[ti - 1]])
        X.append(seq); y.append(Q[i])
    return np.array(X, np.float32), np.array(y, np.float32)


class SeqModel(nn.Module):
    """Two stacked recurrent layers + dense head; cell/bidirectionality configurable."""
    def __init__(self, cell="lstm", bidir=True, h1=64, h2=32):
        super().__init__()
        RNN = nn.LSTM if cell == "lstm" else nn.GRU
        f = 2 if bidir else 1
        self.r1 = RNN(N_FEATURES, h1, batch_first=True, bidirectional=bidir)
        self.d1 = nn.Dropout(0.2)
        self.r2 = RNN(h1 * f, h2, batch_first=True, bidirectional=bidir)
        self.d2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(h2 * f, 16); self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        o, _ = self.r1(x); o = self.d1(o)
        o, _ = self.r2(o); o = self.d2(o[:, -1, :])
        return self.fc2(self.relu(self.fc1(o))).squeeze(-1)


def train(model, X_tr, y_tr, X_va, y_va):
    torch.manual_seed(SEED); np.random.seed(SEED)
    mu_x = X_tr.mean(axis=(0, 1), keepdims=True); sig_x = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    mu_y = y_tr.mean(); sig_y = y_tr.std() + 1e-8
    Xtn, Xvn = (X_tr - mu_x) / sig_x, (X_va - mu_x) / sig_x
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
        pred = model(Xvt).numpy() * sig_y + mu_y
    return model, (mu_x, sig_x, mu_y, sig_y), pred


def recursive_eval(model, norm, full_df, n_tr_rows):
    mu_x, sig_x, mu_y, sig_y = norm
    months = full_df["gregorian_date"].dt.month.values
    msin = np.sin(2 * np.pi * months / 12).astype(np.float32)
    mcos = np.cos(2 * np.pi * months / 12).astype(np.float32)
    Q = full_df["Q_m3s"].values.astype(np.float32)
    q_sim = Q.copy(); preds = []
    model.eval()
    for i in range(n_tr_rows, len(full_df)):
        seq = []
        for l in range(LAG, -1, -1):
            ti = i - l
            seq.append([full_df[c].iloc[ti] for c in CLIM_COLS] +
                       [msin[ti], mcos[ti], q_sim[ti - 1]])
        X1 = ((np.array(seq, np.float32).reshape(1, LAG + 1, N_FEATURES)) - mu_x) / sig_x
        with torch.no_grad():
            qp = max(0.0, model(torch.tensor(X1)).item() * sig_y + mu_y)
        q_sim[i] = qp; preds.append(qp)
    return Q[n_tr_rows:], np.array(preds, np.float32)


def block_bootstrap_ci(o, s, n_boot=1000, block=12, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(o); nblocks = int(np.ceil(n / block))
    stats = {"NSE": [], "KGE": [], "r": []}
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(st, st + block) % n
                              for st in rng.integers(0, n, nblocks)])[:n]
        ob, sb = o[idx], s[idx]
        if np.std(ob) == 0 or np.std(sb) == 0:
            continue
        stats["NSE"].append(nse(ob, sb)); stats["KGE"].append(kge(ob, sb))
        stats["r"].append(np.corrcoef(ob, sb)[0, 1])
    return {k: (round(float(np.percentile(v, 2.5)), 3),
                round(float(np.percentile(v, 97.5)), 3)) for k, v in stats.items()}


def main():
    tr = pd.read_csv(FLOW / "train_2000_2012.csv", parse_dates=["gregorian_date"])
    va = pd.read_csv(FLOW / "val_2013_2016.csv", parse_dates=["gregorian_date"])
    full = pd.concat([tr, va]).reset_index(drop=True)
    X_full, y_full = make_features(full)
    n_tr_ref = len(tr) - (LAG + 1)
    X_va, y_va = X_full[n_tr_ref:], y_full[n_tr_ref:]

    # ---------- A. training-length sensitivity (BiLSTM) ----------
    print("=" * 60); print("A. Training-length sensitivity")
    rowsA = []
    for end_year, label in [(2008, "2000-2008 (108 mo)"), (2010, "2000-2010 (132 mo)"),
                            (2012, "2000-2012 (156 mo, reference)")]:
        sub = tr[tr["year"] <= end_year].reset_index(drop=True)
        X_tr, y_tr = make_features(sub)
        m = SeqModel("lstm", True)
        m, norm, pred = train(m, X_tr, y_tr, X_va, y_va)
        rowsA.append({"train_window": label, "n_months": len(sub),
                      "NSE": round(nse(y_va, pred), 3), "KGE": round(kge(y_va, pred), 3),
                      "PBIAS_pct": round(pbias(y_va, pred), 1)})
        print(f"  {label}: NSE={rowsA[-1]['NSE']} KGE={rowsA[-1]['KGE']}")
    pd.DataFrame(rowsA).to_csv(OUT / "training_length.csv", index=False)

    # ---------- B. architecture variants ----------
    print("=" * 60); print("B. Architecture variants (identical protocol)")
    X_tr, y_tr = X_full[:n_tr_ref], y_full[:n_tr_ref]
    rowsB = []
    for cell, bidir, name in [("lstm", True, "BiLSTM (retrained ref.)"),
                              ("lstm", False, "LSTM (unidirectional)"),
                              ("gru", True, "BiGRU")]:
        m = SeqModel(cell, bidir)
        m, norm, pred = train(m, X_tr, y_tr, X_va, y_va)
        o_rec, p_rec = recursive_eval(m, norm, full, len(tr))
        rowsB.append({"model": name,
                      "NSE_onestep": round(nse(y_va, pred), 3),
                      "KGE_onestep": round(kge(y_va, pred), 3),
                      "r_onestep": round(float(np.corrcoef(y_va, pred)[0, 1]), 3),
                      "PBIAS_onestep": round(pbias(y_va, pred), 1),
                      "RMSE_onestep": round(float(np.sqrt(np.mean((y_va - pred) ** 2))), 2),
                      "NSE_recursive": round(nse(o_rec, p_rec), 3),
                      "KGE_recursive": round(kge(o_rec, p_rec), 3),
                      "r_recursive": round(float(np.corrcoef(o_rec, p_rec)[0, 1]), 3),
                      "PBIAS_recursive": round(pbias(o_rec, p_rec), 1),
                      "RMSE_recursive": round(float(np.sqrt(np.mean((o_rec - p_rec) ** 2))), 2)})
        print(f"  {name}: one-step NSE={rowsB[-1]['NSE_onestep']}, recursive NSE={rowsB[-1]['NSE_recursive']}")
    pd.DataFrame(rowsB).to_csv(OUT / "architectures.csv", index=False)

    # ---------- C. bootstrap CIs from the PRODUCTION model ----------
    print("=" * 60); print("C. Moving-block bootstrap CIs (production model)")
    vr = pd.read_csv(BIL / "validation_results.csv")
    ci_one = block_bootstrap_ci(vr["Q_obs"].values.astype(float),
                                vr["Q_pred"].values.astype(float))
    # recursive series regenerated deterministically from saved weights
    prod = SeqModel("lstm", True)
    state = torch.load(BIL / "model_weights.pt", weights_only=True)
    # production checkpoint (13_bilstm_streamflow.BiLSTMModel) names the layers
    # lstm1/lstm2; SeqModel names them r1/r2 (dropout layers carry no weights)
    state = {k.replace("lstm1.", "r1.").replace("lstm2.", "r2."): v
             for k, v in state.items()}
    prod.load_state_dict(state)
    np_norm = np.load(BIL / "norm_params.npy", allow_pickle=True).item()
    norm = (np_norm["mu_x"], np_norm["sig_x"], float(np_norm["mu_y"]), float(np_norm["sig_y"]))
    o_rec, p_rec = recursive_eval(prod, norm, full, len(tr))
    print(f"  sanity: recursive NSE from saved weights = {nse(o_rec, p_rec):.3f} (expected ~0.416)")
    ci_rec = block_bootstrap_ci(o_rec.astype(float), p_rec.astype(float))
    rowsC = []
    for mode, ci in [("one-step", ci_one), ("recursive", ci_rec)]:
        for met, (lo, hi) in ci.items():
            rowsC.append({"mode": mode, "metric": met, "ci_lo": lo, "ci_hi": hi})
    dfC = pd.DataFrame(rowsC)
    dfC.to_csv(OUT / "bootstrap_ci.csv", index=False)
    print(dfC.to_string(index=False))
    print("\nAll robustness outputs in:", OUT)


if __name__ == "__main__":
    main()

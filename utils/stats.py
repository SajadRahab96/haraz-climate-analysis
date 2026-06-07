"""
utils/stats.py
==============
Lightweight OLS regression and hydroclimatic performance metrics.
Uses only NumPy — no scikit-learn dependency required.

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis
"""

import numpy as np


# ── OLS Regression ────────────────────────────────────────────────────────────

def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Ordinary Least Squares regression via NumPy least-squares solver.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Predictor matrix (without intercept column — added internally).
    y : ndarray, shape (n,)
        Target vector.

    Returns
    -------
    coef  : ndarray, shape (p+1,)   — [intercept, slope_1, ..., slope_p]
    r2    : float                   — coefficient of determination
    rmse  : float                   — root mean squared error
    """
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / n)
    return coef, r2, rmse


def ols_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply fitted OLS coefficients to new predictors."""
    n = len(X) if X.ndim == 1 else X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    return X_aug @ coef


# ── Performance Metrics ───────────────────────────────────────────────────────

def pearson_r(obs: np.ndarray, sim: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(np.abs(obs[mask] - sim[mask])))


def bias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean bias (sim − obs)."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(sim[mask] - obs[mask]))


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling–Gupta Efficiency (Gupta et al., 2009)."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    r   = pearson_r(o, s)
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta  = s.mean() / o.mean() if o.mean() != 0 else np.nan
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def summary(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Return a dict with r, RMSE, MAE, Bias, NSE, KGE."""
    return {
        "r":    pearson_r(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE":  mae(obs, sim),
        "Bias": bias(obs, sim),
        "NSE":  nse(obs, sim),
        "KGE":  kge(obs, sim),
    }

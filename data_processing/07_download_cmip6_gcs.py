"""
07_download_cmip6_gcs.py
========================
Extract raw CMIP6 daily data (pr, tasmax, tasmin, tas) for the Haraz Watershed
stations DIRECTLY from the Pangeo Google Cloud Zarr archive.

Why this route
--------------
The ESGF download route (04_download_cmip6.py) depends on ESGF data nodes that
are frequently offline or rate-limited. The Pangeo project mirrors the full
CMIP6 archive on Google Cloud as cloud-optimized Zarr stores. xarray streams
ONLY the requested grid cell, so extracting one station over 2000-2100 daily
records transfers a few MB instead of multi-GB NetCDF files. No login required.

Implementation note
-------------------
This version reads the Pangeo master CSV catalog with pandas and opens each Zarr
store directly with gcsfs + xarray. This deliberately avoids intake-esm, which
on some newer fsspec/aiohttp versions fails to load the remote JSON catalog
(FileNotFoundError on a URL that actually exists). The CSV route is robust and
handles non-standard model calendars (e.g. UKESM1-0-LL 360-day).

tas handling
------------
For each model/experiment, the script first attempts to download tas directly
from the GCS archive. If tas is not available (not all models publish tas in
the "day" table), it is computed as (tasmax + tasmin) / 2 and the column
tas_source is set to "computed". When tas is downloaded directly, tas_source
is set to "downloaded".

Progress reporting
------------------
Every variable download reports:
  - Per-variable status: size downloaded, elapsed time
  - Job-level: variables done / total, data downloaded this job
  - Global: total data downloaded, total elapsed, estimated time remaining

Stations  : Gharakhil, Amol, Sari (Dasht-e-Naz Airport)
Models    : MPI-ESM1-2-HR, GFDL-ESM4, UKESM1-0-LL, IPSL-CM6A-LR, MRI-ESM2-0, CanESM5
Variables : pr (mm/day), tasmax (°C), tasmin (°C), tas (°C)
Periods   : historical 2000-2014  |  ssp245 & ssp585 2015-2100

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Pangeo / ESGF Cloud Data Working Group. CMIP6 Google Cloud Zarr archive.

Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Requirements:
    pip install xarray zarr gcsfs dask netCDF4 pandas numpy

Usage:
    python data_processing/07_download_cmip6_gcs.py --output data/cmip6_gcs/

    # Subset example:
    python data_processing/07_download_cmip6_gcs.py \\
        --output data/cmip6_gcs/ \\
        --models CanESM5 MRI-ESM2-0 \\
        --experiments historical ssp245
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

# ── Configuration ─────────────────────────────────────────────────────────────
STATIONS = {
    "Gharakhil":              {"lat": 36.487, "lon": 52.108},
    "Amol":                   {"lat": 36.470, "lon": 52.350},
    "Sari_DashtENaz_Airport": {"lat": 36.653, "lon": 53.193},
}

MODELS = ["MPI-ESM1-2-HR", "GFDL-ESM4", "UKESM1-0-LL",
          "IPSL-CM6A-LR", "MRI-ESM2-0", "CanESM5"]

# Ensemble members differ by model (UKESM1-0-LL uses the f2 forcing variant).
MEMBERS = {
    "MPI-ESM1-2-HR": "r1i1p1f1",
    "GFDL-ESM4":     "r1i1p1f1",
    "UKESM1-0-LL":   "r1i1p1f2",
    "IPSL-CM6A-LR":  "r1i1p1f1",
    "MRI-ESM2-0":    "r1i1p1f1",
    "CanESM5":       "r1i1p1f1",
}

# Core variables to download; tas is attempted first, computed if unavailable.
VARIABLES_CORE = ["pr", "tasmax", "tasmin"]
VARIABLES_ALL  = ["pr", "tasmax", "tasmin", "tas"]   # tas tried separately

EXPERIMENTS = ["historical", "ssp245", "ssp585"]
TABLE_ID    = "day"

# Time slices per experiment (string-based — works across all CMIP6 calendars).
TIME_RANGE = {
    "historical": ("2000", "2014"),
    "ssp245":     ("2015", "2100"),
    "ssp585":     ("2015", "2100"),
}

GRID_PRIORITY = ["gn", "gr", "gr1", "gr2"]

# Pangeo catalog locations.
CSV_GCS  = "cmip6/pangeo-cmip6.csv"
CSV_HTTP = "https://cmip6.storage.googleapis.com/pangeo-cmip6.csv"   # fallback


# ── Utility helpers ────────────────────────────────────────────────────────────

def _fmt_size(n_bytes: float) -> str:
    """Human-readable byte size."""
    if n_bytes < 1024:
        return f"{n_bytes:.0f} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes/1024**2:.1f} MB"
    return f"{n_bytes/1024**3:.2f} GB"


def _fmt_time(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def log(msg: str):
    """Print immediately (no buffering)."""
    print(msg, flush=True)


def to_360(lon: float) -> float:
    return lon % 360


def nearest_cell(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    lon_vals = ds["lon"].values
    target_lon = to_360(lon) if lon_vals.max() > 180 else lon
    return ds.sel(lat=lat, lon=target_lon, method="nearest")


def convert(values: np.ndarray, var: str) -> np.ndarray:
    if var == "pr":
        return np.clip(values * 86400.0, a_min=0, a_max=None)   # kg m⁻² s⁻¹ → mm/day
    return values - 273.15                                       # K → °C


def pick_store(df: pd.DataFrame, model: str, exp: str, var: str) -> str | None:
    """Return the best zstore URL for one model/experiment/variable, or None."""
    member = MEMBERS.get(model, "r1i1p1f1")
    sub = df[
        (df.source_id    == model)  &
        (df.experiment_id == exp)   &
        (df.variable_id  == var)    &
        (df.table_id     == TABLE_ID) &
        (df.member_id    == member)
    ]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["grid_rank"] = sub["grid_label"].apply(
        lambda g: GRID_PRIORITY.index(g) if g in GRID_PRIORITY else 99)
    sub = sub.sort_values(["grid_rank", "version"], ascending=[True, False])
    return sub.iloc[0]["zstore"]


# ── Core download logic ────────────────────────────────────────────────────────

def download_variable(
    fs: gcsfs.GCSFileSystem,
    catalog_df: pd.DataFrame,
    model: str,
    exp: str,
    var: str,
) -> tuple[dict | None, float]:
    """
    Download one variable for all stations.

    Returns
    -------
    station_data : dict {station_name: (dates_list, values_array)} or None on failure
    approx_bytes : approximate bytes transferred
    """
    zstore = pick_store(catalog_df, model, exp, var)
    if zstore is None:
        return None, 0.0

    try:
        ds = xr.open_zarr(fs.get_mapper(zstore), consolidated=True)
        y0, y1 = TIME_RANGE.get(exp, (None, None))
        if y0 is not None:
            ds = ds.sel(time=slice(y0, y1))

        station_data = {}
        total_vals   = 0
        for st, c in STATIONS.items():
            pt    = nearest_cell(ds, c["lat"], c["lon"])
            dates = [str(t)[:10] for t in pt["time"].values]
            vals  = convert(pt[var].values, var)
            station_data[st] = (dates, vals)
            total_vals      += len(vals)

        ds.close()
        # Approximate bytes: float32 × n_values × 3 stations
        approx_bytes = total_vals * 4.0
        return station_data, approx_bytes

    except Exception as e:
        raise RuntimeError(str(e))


def main(output_base: str, models: list, experiments: list):
    Path(output_base).mkdir(parents=True, exist_ok=True)

    # ── Load catalog ──────────────────────────────────────────────────────────
    log("╔══════════════════════════════════════════════════════════════╗")
    log("║  CMIP6 GCS Downloader — Haraz Watershed                     ║")
    log("╚══════════════════════════════════════════════════════════════╝")
    log("")
    log("Loading Pangeo CMIP6 catalog (may take 1–3 min over VPN) ...")
    fs = gcsfs.GCSFileSystem(token="anon", access="read_only")
    try:
        with fs.open(CSV_GCS) as f:
            catalog_df = pd.read_csv(f)
    except Exception as e:
        log(f"  gcsfs read failed ({str(e)[:60]}); trying HTTP fallback ...")
        catalog_df = pd.read_csv(CSV_HTTP)

    catalog_df["version"] = catalog_df["version"].astype(str)
    log(f"  Catalog loaded: {len(catalog_df):,} Zarr stores.\n")

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs        = [(m, e) for m in models for e in experiments]
    n_jobs      = len(jobs)
    global_t0   = time.time()
    global_mb   = 0.0    # cumulative megabytes
    job_times   = []     # for ETA estimation

    log(f"  Jobs: {n_jobs}  ({len(models)} models × {len(experiments)} experiments)")
    log(f"  Variables per job: {VARIABLES_ALL}")
    log(f"  Stations: {list(STATIONS.keys())}")
    log("")

    # ── Process jobs ──────────────────────────────────────────────────────────
    for job_idx, (model, exp) in enumerate(jobs, 1):
        job_t0  = time.time()
        job_mb  = 0.0

        log("╔══════════════════════════════════════════════════════════════╗")
        log(f"║  Job {job_idx:2d}/{n_jobs}  │  {model:20s} │  {exp}{'':10s}║")
        log("╚══════════════════════════════════════════════════════════════╝")

        frames      = {st: {} for st in STATIONS}   # {station: {col: values}}
        dates_ref   = {st: None for st in STATIONS}
        var_status  = {}   # {var: "ok" | "failed" | "skipped"}
        tas_downloaded = False

        # ── Download core variables (pr, tasmax, tasmin) ───────────────────
        for v_idx, var in enumerate(VARIABLES_ALL, 1):
            bar = _progress_bar(v_idx - 1, len(VARIABLES_ALL))
            log(f"  [{bar}] {v_idx-1}/{len(VARIABLES_ALL)} variables done")
            log(f"  → {var:8s}  ⟳ downloading ...")
            t1 = time.time()
            try:
                station_data, approx_bytes = download_variable(
                    fs, catalog_df, model, exp, var)

                if station_data is None:
                    log(f"  → {var:8s}  ✗ not in catalog  (skipped)")
                    var_status[var] = "not_in_catalog"
                    continue

                col_name = "pr_mm" if var == "pr" else f"{var}_C"
                for st, (dates, vals) in station_data.items():
                    if dates_ref[st] is None:
                        dates_ref[st] = dates
                    frames[st][col_name] = vals

                elapsed_v  = time.time() - t1
                mb_v       = approx_bytes / 1024**2
                job_mb    += mb_v
                global_mb += mb_v

                if var == "tas":
                    tas_downloaded = True

                log(f"  → {var:8s}  ✓  ~{_fmt_size(approx_bytes):>9s}  "
                    f"({_fmt_time(elapsed_v):>6s})  "
                    f"[total this job: {job_mb:.1f} MB]")
                var_status[var] = "downloaded"

            except RuntimeError as e:
                elapsed_v = time.time() - t1
                log(f"  → {var:8s}  ✗ FAILED ({_fmt_time(elapsed_v)}): "
                    f"{str(e)[:80]}")
                var_status[var] = "failed"

        # ── Compute tas if not downloaded ──────────────────────────────────
        if not tas_downloaded:
            log(f"  → {'tas':8s}  ⟳ computing from (tasmax + tasmin) / 2 ...")
            computed_ok = True
            for st in STATIONS:
                tx = frames[st].get("tasmax_C")
                tn = frames[st].get("tasmin_C")
                if tx is not None and tn is not None:
                    frames[st]["tas_C"]      = (tx + tn) / 2.0
                    frames[st]["tas_source"] = np.array(
                        ["computed"] * len(tx), dtype=object)
                else:
                    computed_ok = False
            if computed_ok:
                log(f"  → {'tas':8s}  ✓  computed  [tas_source = 'computed']")
                var_status["tas"] = "computed"
            else:
                log(f"  → {'tas':8s}  ✗ cannot compute (tasmax or tasmin missing)")
                var_status["tas"] = "failed"
        else:
            # mark tas_source as downloaded
            for st in STATIONS:
                if "tas_C" in frames[st]:
                    n = len(frames[st]["tas_C"])
                    frames[st]["tas_source"] = np.array(
                        ["downloaded"] * n, dtype=object)

        # ── Final progress bar for this job ────────────────────────────────
        n_ok  = sum(1 for s in var_status.values() if s in ("downloaded", "computed"))
        bar   = _progress_bar(n_ok, len(VARIABLES_ALL))
        log(f"  [{bar}] {n_ok}/{len(VARIABLES_ALL)} variables done")

        # ── Save CSVs ──────────────────────────────────────────────────────
        any_saved = False
        out_dir   = Path(output_base) / exp / model
        out_dir.mkdir(parents=True, exist_ok=True)

        for st in STATIONS:
            if not dates_ref[st]:
                continue
            df_out = pd.DataFrame({"date": dates_ref[st]})
            for col in ["pr_mm", "tasmax_C", "tasmin_C", "tas_C", "tas_source"]:
                if col in frames[st]:
                    df_out[col] = frames[st][col]
            fpath = out_dir / f"{st}_{model}_{exp}.csv"
            df_out.to_csv(fpath, index=False)
            log(f"  💾 saved {fpath.name}  ({len(df_out):,} days)")
            any_saved = True

        # ── Timing & ETA ──────────────────────────────────────────────────
        job_elapsed = time.time() - job_t0
        job_times.append(job_elapsed)
        total_elapsed = time.time() - global_t0
        avg_job_time  = sum(job_times) / len(job_times)
        remaining     = (n_jobs - job_idx) * avg_job_time

        log("")
        log(f"  ┌─ Job summary ──────────────────────────────────────────")
        log(f"  │  Data this job   : ~{job_mb:.1f} MB")
        log(f"  │  Total downloaded: ~{global_mb:.1f} MB")
        log(f"  │  Job elapsed     : {_fmt_time(job_elapsed)}")
        log(f"  │  Total elapsed   : {_fmt_time(total_elapsed)}")
        log(f"  │  Est. remaining  : ~{_fmt_time(remaining)}")
        log(f"  └────────────────────────────────────────────────────────")
        log("")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - global_t0
    log("╔══════════════════════════════════════════════════════════════╗")
    log("║  DOWNLOAD COMPLETE                                           ║")
    log(f"║  Total jobs      : {n_jobs:3d}                                     ║")
    log(f"║  Total data      : ~{global_mb:6.1f} MB                             ║")
    log(f"║  Total time      : {_fmt_time(total_elapsed):>10s}                          ║")
    log(f"║  Output dir      : {os.path.abspath(output_base)[:38]}  ║")
    log("╚══════════════════════════════════════════════════════════════╝")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stream CMIP6 from Google Cloud (Pangeo Zarr) — Haraz Watershed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", default="data/cmip6_gcs",
        help="Output directory (default: data/cmip6_gcs)")
    p.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Subset of models (default: all 6). Example: --models CanESM5 MRI-ESM2-0")
    p.add_argument(
        "--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS,
        help="Subset of experiments (default: all). Example: --experiments historical ssp245")
    a = p.parse_args()
    main(a.output, a.models, a.experiments)

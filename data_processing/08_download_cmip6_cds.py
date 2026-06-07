"""
08_download_cmip6_cds.py
========================
FALLBACK route: download CMIP6 daily data (pr, tasmax, tasmin, tas) for the
Haraz region from the Copernicus Climate Data Store (CDS), then extract station
series.

Use this when a model/scenario is missing from the Google Cloud archive (07) or
when the CDS extra quality-control layer is preferred. CDS is stable but queued.

One-time setup (NEW CDS infrastructure, since Feb 2025)
-------------------------------------------------------
1. Create a free account at https://cds.climate.copernicus.eu and accept the
   CMIP6 dataset licence on the dataset page.
2. Copy your Personal Access Token from your CDS profile.
3. Create  ~/.cdsapirc  with:
       url: https://cds.climate.copernicus.eu/api
       key: <YOUR-PERSONAL-ACCESS-TOKEN>
4. pip install "cdsapi>=0.7" xarray netCDF4 pandas numpy

tas handling
------------
CDS provides tas directly. If not available, it is computed from
(tasmax + tasmin) / 2 with tas_source = "computed".

Stations  : Gharakhil, Amol, Sari (Dasht-e-Naz Airport)
Models    : MPI-ESM1-2-HR, GFDL-ESM4, UKESM1-0-LL, IPSL-CM6A-LR, MRI-ESM2-0, CanESM5

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Copernicus Climate Change Service (C3S) Climate Data Store, CMIP6 projections.

Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Usage:
    python data_processing/08_download_cmip6_cds.py --output data/cmip6_cds/
"""

import argparse
import glob
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

# ── Configuration ─────────────────────────────────────────────────────────────
STATIONS = {
    "Gharakhil":              {"lat": 36.487, "lon": 52.108},
    "Amol":                   {"lat": 36.470, "lon": 52.350},
    "Sari_DashtENaz_Airport": {"lat": 36.653, "lon": 53.193},
}

# Bounding box around the Haraz basin: [North, West, South, East].
AREA = [37.0, 51.7, 36.0, 53.6]

# CDS uses lowercase + underscores for model names (differs from ESGF/Pangeo).
MODELS_CDS = {
    "MPI-ESM1-2-HR": "mpi_esm1_2_hr",
    "GFDL-ESM4":     "gfdl_esm4",
    "UKESM1-0-LL":   "ukesm1_0_ll",
    "IPSL-CM6A-LR":  "ipsl_cm6a_lr",
    "MRI-ESM2-0":    "mri_esm2_0",
    "CanESM5":       "canesm5",
}

VARIABLES = {
    "pr":     "precipitation",
    "tasmax": "daily_maximum_near_surface_air_temperature",
    "tasmin": "daily_minimum_near_surface_air_temperature",
    "tas":    "near_surface_air_temperature",
}

EXPERIMENTS = {
    "historical": "historical",
    "ssp245":     "ssp2_4_5",
    "ssp585":     "ssp5_8_5",
}

DATE_RANGE = {
    "historical": "2000-01-01/2014-12-31",
    "ssp245":     "2015-01-01/2100-12-31",
    "ssp585":     "2015-01-01/2100-12-31",
}


def log(msg: str):
    print(msg, flush=True)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def download_region(raw_dir: str) -> None:
    """Submit CDS requests for all model × experiment × variable combinations."""
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    total  = len(MODELS_CDS) * len(EXPERIMENTS) * len(VARIABLES)
    done   = 0
    t0     = time.time()

    log(f"Submitting {total} CDS requests ...")
    for model_id, model_cds in MODELS_CDS.items():
        for exp_key, exp_cds in EXPERIMENTS.items():
            for var_key, var_cds in VARIABLES.items():
                done += 1
                target = Path(raw_dir) / f"{var_key}_{model_id}_{exp_key}.zip"
                if target.exists():
                    log(f"  [{done}/{total}] skip (exists): {target.name}")
                    continue
                request = {
                    "temporal_resolution": "daily",
                    "experiment":          exp_cds,
                    "variable":            var_cds,
                    "model":               model_cds,
                    "date":                DATE_RANGE[exp_key],
                    "area":                AREA,
                    "download_format":     "zip",
                    "data_format":         "netcdf",
                }
                try:
                    log(f"  [{done}/{total}] requesting {var_key} {model_id} {exp_key} ...")
                    client.retrieve("projections-cmip6", request, str(target))
                    log(f"    ✓ saved {target.name}  ({_fmt_time(time.time()-t0)} total)")
                except Exception as e:
                    log(f"    ✗ failed: {e}")


def _unzip_all(raw_dir: str) -> None:
    for z in glob.glob(os.path.join(raw_dir, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)


def _nearest(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    lon_vals = ds["lon"].values
    target_lon = (lon % 360) if lon_vals.max() > 180 else lon
    return ds.sel(lat=lat, lon=target_lon, method="nearest")


def _convert(values: np.ndarray, var_key: str) -> np.ndarray:
    if var_key == "pr":
        return np.clip(values * 86400.0, a_min=0, a_max=None)
    return values - 273.15


def extract_stations(raw_dir: str, output_base: str) -> None:
    """Extract station time series from downloaded NetCDF files."""
    log("\nUnzipping downloaded files ...")
    _unzip_all(raw_dir)
    Path(output_base).mkdir(parents=True, exist_ok=True)
    ncfiles = glob.glob(os.path.join(raw_dir, "*.nc"))
    log(f"Found {len(ncfiles)} NetCDF files.")

    for model_id in MODELS_CDS:
        for exp_key in EXPERIMENTS:
            frames      = {st: pd.DataFrame() for st in STATIONS}
            found       = False
            tas_found   = False

            for var_key in VARIABLES:
                match = [f for f in ncfiles
                         if var_key in os.path.basename(f)
                         and model_id in os.path.basename(f)
                         and exp_key in os.path.basename(f)]
                if not match:
                    continue
                found = True
                try:
                    ds    = xr.open_dataset(match[0])
                    cf    = [v for v in ds.data_vars if v in ("pr", "tasmax", "tasmin", "tas")]
                    vname = cf[0] if cf else list(ds.data_vars)[0]

                    col_out = "pr_mm" if var_key == "pr" else f"{var_key}_C"
                    for st, c in STATIONS.items():
                        pt = _nearest(ds, c["lat"], c["lon"])
                        if "date" not in frames[st]:
                            frames[st]["date"] = pd.to_datetime(
                                [str(t)[:10] for t in pt["time"].values], errors="coerce")
                        frames[st][col_out] = _convert(pt[vname].values, var_key)

                    if var_key == "tas":
                        tas_found = True
                    ds.close()
                    log(f"  [{model_id} | {exp_key}] {var_key} extracted.")
                except Exception as e:
                    log(f"  [{model_id} | {exp_key}] {var_key} extraction failed: {e}")

            if not found:
                continue

            # Compute tas if not downloaded
            for st in STATIONS:
                tx = frames[st].get("tasmax_C")
                tn = frames[st].get("tasmin_C")
                if not tas_found and tx is not None and tn is not None:
                    frames[st]["tas_C"]      = (tx + tn) / 2.0
                    frames[st]["tas_source"] = "computed"
                    log(f"  [{model_id} | {exp_key}] tas computed from (tasmax+tasmin)/2")
                elif tas_found and "tas_C" in frames[st]:
                    frames[st]["tas_source"] = "downloaded"

            out_dir = Path(output_base) / exp_key / model_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for st, df in frames.items():
                if df.empty:
                    continue
                fpath = out_dir / f"{st}_{model_id}_{exp_key}.csv"
                df.to_csv(fpath, index=False)
                log(f"  saved {fpath}  ({len(df)} days)")


def main(output_base: str) -> None:
    raw_dir = os.path.join(output_base, "_raw")
    download_region(raw_dir)
    extract_stations(raw_dir, output_base)
    log(f"\nDone. CSVs in: {os.path.abspath(output_base)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download CMIP6 from Copernicus CDS (fallback)")
    p.add_argument("--output", default="data/cmip6_cds",
                   help="Output directory (default: data/cmip6_cds)")
    a = p.parse_args()
    main(a.output)

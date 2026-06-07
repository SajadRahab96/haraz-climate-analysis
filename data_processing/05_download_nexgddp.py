"""
05_download_nexgddp.py
======================
Download NASA NEX-GDDP-CMIP6 data (AWS S3) for a single experiment.

NEX-GDDP is a statistically downscaled product at 0.25° resolution.
Used here for cross-comparison only; the primary downscaling pathway
is LARS-WG 8 applied to raw CMIP6 GCM outputs (see 07_download_cmip6_gcs.py).

Variables : pr, tasmax, tasmin (tas not available in NEX-GDDP)
Resolution: 0.25° × 0.25° (regridded from raw GCMs)

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Thrasher B. et al. (2022). NASA Global Daily Downscaled Projections, CMIP6 (NEX-GDDP-CMIP6).
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Requirements:
    pip install requests xarray netCDF4 pandas numpy

Usage:
    python data_processing/05_download_nexgddp.py \
        --model CanESM5 --experiment ssp245 \
        --output data/nexgddp/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

STATIONS = {
    "Gharakhil":              {"lat": 36.487, "lon": 52.108},
    "Amol":                   {"lat": 36.470, "lon": 52.350},
    "Sari_DashtENaz_Airport": {"lat": 36.653, "lon": 53.193},
}

MODELS = ["MPI-ESM1-2-HR", "GFDL-ESM4", "UKESM1-0-LL",
          "IPSL-CM6A-LR", "MRI-ESM2-0", "CanESM5"]

MEMBERS = {
    "MPI-ESM1-2-HR": "r1i1p1f1",
    "GFDL-ESM4":     "r1i1p1f1",
    "UKESM1-0-LL":   "r1i1p1f2",
    "IPSL-CM6A-LR":  "r1i1p1f1",
    "MRI-ESM2-0":    "r1i1p1f1",
    "CanESM5":       "r1i1p1f1",
}

VARIABLES   = ["pr", "tasmax", "tasmin"]   # tas not in NEX-GDDP
EXPERIMENTS = ["historical", "ssp245", "ssp585"]

YEAR_RANGE = {
    "historical": range(2000, 2015),
    "ssp245":     range(2015, 2101),
    "ssp585":     range(2015, 2101),
}

NEX_BASE = ("https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/"
            "NEX-GDDP-CMIP6/{model}/{experiment}/r1i1p1f1/{variable}/"
            "{variable}_day_{model}_{experiment}_r1i1p1f1_gn_{year}.nc")


def build_url(model, experiment, variable, year, member):
    return NEX_BASE.format(
        model=model, experiment=experiment,
        variable=variable, year=year, member=member)


def convert(values, var):
    if var == "pr":
        return np.clip(values * 86400.0, 0, None)
    return values - 273.15


def download_file(url, dest):
    if dest.exists():
        return True
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ! {url.split('/')[-1]}: {e}", flush=True)
        return False


def main(output_base, model, experiment):
    raw_dir = Path(output_base) / "_raw" / model / experiment
    raw_dir.mkdir(parents=True, exist_ok=True)
    member  = MEMBERS.get(model, "r1i1p1f1")
    frames  = {st: [] for st in STATIONS}

    years = list(YEAR_RANGE.get(experiment, range(2015, 2051)))
    print(f"Model: {model}  |  Experiment: {experiment}  |  {len(years)} years", flush=True)

    for var in VARIABLES:
        print(f"  Variable: {var}", flush=True)
        yearly_dfs = {st: [] for st in STATIONS}
        for year in years:
            url   = build_url(model, experiment, var, year, member)
            fname = raw_dir / f"{var}_{year}.nc"
            ok    = download_file(url, fname)
            if not ok:
                continue
            try:
                ds = xr.open_dataset(fname)
                col = "pr_mm" if var == "pr" else f"{var}_C"
                for st, c in STATIONS.items():
                    lon_vals = ds["lon"].values
                    lon_t = c["lon"] % 360 if lon_vals.max() > 180 else c["lon"]
                    pt    = ds.sel(lat=c["lat"], lon=lon_t, method="nearest")
                    dates = [str(t)[:10] for t in pt["time"].values]
                    tmp   = pd.DataFrame({"date": dates, col: convert(pt[var].values, var)})
                    yearly_dfs[st].append(tmp)
                ds.close()
                print(f"    {year} ✓", flush=True)
            except Exception as e:
                print(f"    {year} ! {e}", flush=True)

        for st in STATIONS:
            if yearly_dfs[st]:
                frames[st].append(pd.concat(yearly_dfs[st], ignore_index=True))

    out_dir = Path(output_base) / experiment / model
    out_dir.mkdir(parents=True, exist_ok=True)
    for st in STATIONS:
        if frames[st]:
            df_all = pd.concat(frames[st], axis=1)
            # deduplicate date column
            df_all = df_all.loc[:, ~df_all.columns.duplicated()]
            fpath  = out_dir / f"{st}_{model}_{experiment}.csv"
            df_all.to_csv(fpath, index=False)
            print(f"  saved {fpath.name}  ({len(df_all)} days)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NEX-GDDP-CMIP6 single model/experiment")
    p.add_argument("--model",      default="CanESM5", choices=MODELS)
    p.add_argument("--experiment", default="ssp245",  choices=EXPERIMENTS)
    p.add_argument("--output",     default="data/nexgddp")
    a = p.parse_args()
    main(a.output, a.model, a.experiment)

"""
06_download_all_cmip6.py
========================
NEX-GDDP-CMIP6 full parallel run - all 6 models x 3 experiments.

Wraps 05_download_nexgddp.py logic with concurrent.futures for speed.
Used for cross-comparison only; primary route is 07_download_cmip6_gcs.py.

Variables : pr, tasmax, tasmin (tas not available in NEX-GDDP)
Resolution: 0.25° x 0.25° (statistically downscaled)

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Usage:
    python data_processing/06_download_all_cmip6.py --output data/nexgddp/ --workers 3
"""

import argparse
import concurrent.futures
import time
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

VARIABLES   = ["pr", "tasmax", "tasmin"]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]

YEAR_RANGE = {
    "historical": range(2000, 2015),
    "ssp245":     range(2015, 2101),
    "ssp585":     range(2015, 2101),
}

NEX_BASE = ("https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/"
            "NEX-GDDP-CMIP6/{model}/{experiment}/{member}/{variable}/"
            "{variable}_day_{model}_{experiment}_{member}_gn_{year}.nc")


def build_url(model, experiment, variable, year, member):
    return NEX_BASE.format(
        model=model, experiment=experiment,
        member=member, variable=variable, year=year)


def convert(values, var):
    if var == "pr":
        return np.clip(values * 86400.0, 0, None)
    return values - 273.15


def download_one(args):
    """Worker function for parallel downloads."""
    model, experiment, output_base = args
    member  = MEMBERS.get(model, "r1i1p1f1")
    raw_dir = Path(output_base) / "_raw" / model / experiment
    raw_dir.mkdir(parents=True, exist_ok=True)

    tag    = f"{model}|{experiment}"
    frames = {st: [] for st in STATIONS}

    for var in VARIABLES:
        for year in YEAR_RANGE.get(experiment, []):
            url   = build_url(model, experiment, var, year, member)
            fname = raw_dir / f"{var}_{year}.nc"
            if not fname.exists():
                try:
                    r = requests.get(url, stream=True, timeout=120)
                    r.raise_for_status()
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                except Exception as e:
                    print(f"  [{tag}] {var} {year}: download failed - {e}", flush=True)
                    continue
            try:
                ds  = xr.open_dataset(fname)
                col = "pr_mm" if var == "pr" else f"{var}_C"
                for st, c in STATIONS.items():
                    lon_vals = ds["lon"].values
                    lon_t = c["lon"] % 360 if lon_vals.max() > 180 else c["lon"]
                    pt    = ds.sel(lat=c["lat"], lon=lon_t, method="nearest")
                    dates = [str(t)[:10] for t in pt["time"].values]
                    frames[st].append(
                        pd.DataFrame({"date": dates, col: convert(pt[var].values, var)}))
                ds.close()
            except Exception as e:
                print(f"  [{tag}] {var} {year}: extract failed - {e}", flush=True)

    out_dir = Path(output_base) / experiment / model
    out_dir.mkdir(parents=True, exist_ok=True)
    for st in STATIONS:
        parts = frames[st]
        if parts:
            df_all = pd.concat(parts, ignore_index=True)
            df_all = df_all.groupby("date").first().reset_index()
            fpath  = out_dir / f"{st}_{model}_{experiment}.csv"
            df_all.to_csv(fpath, index=False)
            print(f"  [{tag}] saved {fpath.name}  ({len(df_all)} days)", flush=True)
    return tag


def main(output_base, max_workers):
    jobs = [(m, e, output_base) for m in MODELS for e in EXPERIMENTS]
    total = len(jobs)
    t0    = time.time()
    print(f"Starting {total} jobs with {max_workers} workers ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, tag in enumerate(ex.map(download_one, jobs), 1):
            elapsed = (time.time() - t0) / 60
            print(f"  [{i}/{total}] {tag} complete  ({elapsed:.1f} min elapsed)", flush=True)
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. Output: {output_base}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NEX-GDDP-CMIP6 full parallel run")
    p.add_argument("--output",  default="data/nexgddp")
    p.add_argument("--workers", type=int, default=3,
                   help="Number of parallel download workers (default: 3)")
    a = p.parse_args()
    main(a.output, a.workers)

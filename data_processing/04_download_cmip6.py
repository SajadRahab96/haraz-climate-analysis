"""
04_download_cmip6.py
====================
CMIP6 retrieval via ESGF (direct) - REFERENCE / LAST-RESORT route.

WARNING: ESGF data nodes are frequently offline or rate-limited.
Use 07_download_cmip6_gcs.py (Google Cloud, recommended) instead.

Variables : pr, tasmax, tasmin, tas
Models    : MPI-ESM1-2-HR, GFDL-ESM4, UKESM1-0-LL, IPSL-CM6A-LR, MRI-ESM2-0, CanESM5
Experiments: historical (2000-2014), ssp245 (2015-2100), ssp585 (2015-2100)

Reference:
    Rahab-Rajaei S., Motiee H. (2025). Hydroclimatic Projections, Haraz Watershed. [ISI Q1]
    Repository: https://github.com/SajadRahab96/haraz-climate-analysis

Requirements:
    pip install requests xarray netCDF4 pandas numpy

Usage:
    python data_processing/04_download_cmip6.py --output data/cmip6/
"""

import argparse
import os
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

VARIABLES   = ["pr", "tasmax", "tasmin", "tas"]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]
TABLE_ID    = "day"

ESGF_SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search"


def search_esgf(model, experiment, variable, member):
    params = {
        "project":       "CMIP6",
        "source_id":     model,
        "experiment_id": experiment,
        "variable_id":   variable,
        "member_id":     member,
        "table_id":      TABLE_ID,
        "type":          "File",
        "format":        "application/solr+json",
        "limit":         10,
        "fields":        "url,title,size",
    }
    try:
        r = requests.get(ESGF_SEARCH_URL, params=params, timeout=30)
        docs = r.json()["response"]["docs"]
        return docs
    except Exception as e:
        print(f"  ESGF search failed: {e}", flush=True)
        return []


def convert(values, var):
    if var == "pr":
        return np.clip(values * 86400.0, 0, None)
    if var in ("tasmax", "tasmin", "tas"):
        return values - 273.15
    return values


def main(output_base):
    Path(output_base).mkdir(parents=True, exist_ok=True)
    print("NOTE: ESGF route - nodes may be unavailable. "
          "Use 07_download_cmip6_gcs.py for reliable access.", flush=True)

    for model in MODELS:
        member = MEMBERS.get(model, "r1i1p1f1")
        for exp in EXPERIMENTS:
            frames = {st: pd.DataFrame() for st in STATIONS}
            found  = False
            for var in VARIABLES:
                docs = search_esgf(model, exp, var, member)
                if not docs:
                    print(f"  [{model} | {exp} | {var}] no files found on ESGF", flush=True)
                    continue
                # Download first matching file
                url_info = docs[0].get("url", [])
                http_urls = [u.split("|")[0] for u in url_info if "HTTPServer" in u]
                if not http_urls:
                    print(f"  [{model} | {exp} | {var}] no HTTP URL available", flush=True)
                    continue
                url   = http_urls[0]
                fname = Path(output_base) / f"{var}_{model}_{exp}.nc"
                print(f"  Downloading {fname.name} ...", flush=True)
                try:
                    with requests.get(url, stream=True, timeout=120) as resp:
                        resp.raise_for_status()
                        with open(fname, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                except Exception as e:
                    print(f"  ! Download failed: {e}", flush=True)
                    continue

                # Extract station series
                try:
                    ds  = xr.open_dataset(fname)
                    col = "pr_mm" if var == "pr" else f"{var}_C"
                    for st, c in STATIONS.items():
                        lon_vals = ds["lon"].values
                        lon_t = c["lon"] % 360 if lon_vals.max() > 180 else c["lon"]
                        pt    = ds.sel(lat=c["lat"], lon=lon_t, method="nearest")
                        dates = [str(t)[:10] for t in pt["time"].values]
                        if "date" not in frames[st]:
                            frames[st]["date"] = dates
                        frames[st][col] = convert(pt[var].values, var)
                    ds.close()
                    found = True
                except Exception as e:
                    print(f"  ! Extraction failed: {e}", flush=True)

            if found:
                out_dir = Path(output_base) / exp / model
                out_dir.mkdir(parents=True, exist_ok=True)
                for st, fr in frames.items():
                    fpath = out_dir / f"{st}_{model}_{exp}.csv"
                    fr.to_csv(fpath, index=False)
                    print(f"  saved {fpath.name}  ({len(fr)} days)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CMIP6 via ESGF (last resort)")
    p.add_argument("--output", default="data/cmip6")
    a = p.parse_args()
    main(a.output)

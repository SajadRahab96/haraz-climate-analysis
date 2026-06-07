# CMIP6 Data Retrieval — Methodology & Route Comparison

**Project:** Hydroclimatic Projections, Haraz Watershed, Northern Iran  
**Repository:** https://github.com/SajadRahab96/haraz-climate-analysis

---

## Overview

This document describes the four data-retrieval routes implemented in this project to obtain
CMIP6 daily climate fields (pr, tasmax, tasmin, tas) for the Haraz watershed study stations.

---

## Target Data Specifications

| Item | Value |
|------|-------|
| Variables | `pr`, `tasmax`, `tasmin`, `tas` |
| Table | `day` |
| Models | MPI-ESM1-2-HR, GFDL-ESM4, UKESM1-0-LL, IPSL-CM6A-LR, MRI-ESM2-0, CanESM5 |
| Experiments | `historical` (2000–2014), `ssp245` (2015–2100), `ssp585` (2015–2100) |
| Stations | Gharakhil (36.487°N, 52.108°E), Amol (36.470°N, 52.350°E), Sari D-e-N (36.653°N, 53.193°E) |
| Output format | CSV per station × model × experiment |

---

## Route 1 — Google Cloud Pangeo Zarr ✅ **Recommended**

**Script:** `data_processing/07_download_cmip6_gcs.py`

The Pangeo/ESGF Cloud Data Working Group mirrors the full CMIP6 archive on Google Cloud
Storage as cloud-optimized Zarr stores. xarray streams **only the requested grid cells**,
so extracting one station over ~100 years of daily data transfers just a few MB per
variable instead of multi-GB NetCDF files.

**Advantages:**
- No login / authentication required (anonymous GCS access)
- Streams only needed pixels — minimal bandwidth
- Handles non-standard model calendars (360-day, noleap)
- Full progress reporting: per-variable size, elapsed, estimated remaining
- `tas` downloaded if available; computed as `(tasmax + tasmin) / 2` otherwise

**Limitations:**
- Requires internet connection; speed depends on VPN / network
- Some older model versions may be missing from the cloud archive

**Usage:**
```bash
python data_processing/07_download_cmip6_gcs.py --output data/cmip6_gcs/

# Subset example:
python data_processing/07_download_cmip6_gcs.py \
    --output data/cmip6_gcs/ \
    --models CanESM5 MRI-ESM2-0 \
    --experiments historical ssp245
```

---

## Route 2 — Copernicus CDS ⚠️ Fallback

**Script:** `data_processing/08_download_cmip6_cds.py`

Downloads a regional bounding box (NetCDF, then extracts station series).
Requires a free CDS account and Personal Access Token.

**Setup:**
```
~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <YOUR-PERSONAL-ACCESS-TOKEN>
```

**Advantages:**
- Stable, well-maintained infrastructure
- Extra QC layer applied by Copernicus

**Limitations:**
- Requires CDS account + API token
- Downloads regional bounding box (larger than needed)
- Requests are queued — can take hours during peak load

---

## Route 3 — NASA NEX-GDDP-CMIP6 (AWS S3)

**Scripts:** `data_processing/05_download_nexgddp.py`, `06_download_all_cmip6.py`

Statistically downscaled product at 0.25° resolution. Useful for cross-comparison
but not the primary source (already downscaled — bypasses LARS-WG step).

---

## Route 4 — ESGF (direct) ⛔ Not recommended

**Script:** `data_processing/04_download_cmip6.py`

Direct access to ESGF data nodes. Frequently unavailable due to node downtime
and rate limiting. Use only as a last resort.

---

## Output Format

Each CSV contains daily records for one station × model × experiment:

```
date,pr_mm,tasmax_C,tasmin_C,tas_C,tas_source
2000-01-01,2.45,14.3,6.1,10.2,downloaded
2000-01-02,0.00,15.7,7.3,11.5,downloaded
...
```

- `tas_source`: `"downloaded"` if `tas` was available in the archive, `"computed"` if derived from `(tasmax + tasmin) / 2`

---

## Route Comparison

| Feature | GCS (07) | CDS (08) | NEX-GDDP | ESGF |
|---------|----------|----------|----------|------|
| Login required | ❌ | ✅ | ❌ | ✅ |
| Downloads only needed pixels | ✅ | ❌ | ✅ | ❌ |
| Stability | High | High | Medium | Low |
| Speed | Fast | Queued | Fast | Slow |
| Calendar handling | ✅ | ✅ | Standard | ✅ |
| Progress display | ✅ | Partial | ❌ | ❌ |
| **Recommended** | **✅ Primary** | Fallback | Cross-check | Last resort |

# haraz-climate-analysis

**Python scripts for CMIP6 climate data processing, quality control, and gap-filling — Haraz Watershed, Northern Iran**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the reproducible data processing pipeline developed for the study:

> **Rahab-Rajaei, S., Motiee, H. (2025).** *Hydroclimatic Projections Under CMIP6 SSP Scenarios in the Haraz Watershed, Northern Iran: A Hybrid BiLSTM–BMA Framework with Compound Extreme Event Analysis.* [Target Journal: Q1 ISI]

The pipeline covers:

1. **Quality Control & Gap Analysis** — systematic identification of structural and scattered missing values in daily climate records
2. **Multi-Source Gap-Filling** — bias-corrected merging of IRIMO portal and provincial database records; MLR-based infilling from adjacent stations
3. **CMIP6 Data Download & Extraction** *(in progress)* — NetCDF processing for 6 GCMs from ESGF / Google Cloud / CDS
4. **LARS-WG 8 Downscaling Support** — preparation of site files and scenario inputs
5. **BiLSTM Streamflow Projection** *(in progress)*
6. **BMA Uncertainty Quantification** *(in progress)*

---

## Study Area

| Station | Lat (°N) | Lon (°E) | Elev. (m) | Period |
|---------|----------|----------|-----------|--------|
| Gharakhil | 36.487 | 52.108 | 14.7 | 2000–2020 |
| Amol | 36.470 | 52.350 | 23.7 | 2000–2020 |
| Sari (Dasht-e-Naz Airport) | 36.653 | 53.193 | 16.7 | 2000–2020 |
| Karesang (Hydrometric) | 36.273 | 52.381 | 375 | 1949–2016 |

---

## Repository Structure

```
haraz-climate-analysis/
│
├── data_processing/
│   ├── 01_gap_analysis.py          # Systematic gap detection across all stations
│   ├── 02_gap_filling.py           # Multi-source gap-filling (bias correction + MLR)
│   ├── 03_build_final_dataset.py   # Assemble HBCD-2020 final dataset
│   ├── 04_download_cmip6.py        # CMIP6 retrieval via ESGF (reference route)
│   ├── 05_download_nexgddp.py      # NEX-GDDP-CMIP6 (AWS S3), single experiment
│   ├── 06_download_all_cmip6.py    # NEX-GDDP-CMIP6 (AWS S3), parallel full run
│   ├── 07_download_cmip6_gcs.py    # CMIP6 via Google Cloud Pangeo Zarr (recommended)
│   └── 08_download_cmip6_cds.py    # CMIP6 via Copernicus CDS (fallback)
│
├── utils/
│   └── stats.py                    # OLS regression and performance metrics (numpy-only)
│
├── docs/
│   └── DATA_RETRIEVAL.md           # CMIP6 data-retrieval methodology (all routes)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/SajadRahab96/haraz-climate-analysis.git
cd haraz-climate-analysis
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Gap Analysis
```bash
python data_processing/01_gap_analysis.py \
    --input data/IRIMO_Daily_ClimateData_2000_2020.xlsx \
    --output reports/gap_analysis_report.csv
```

### Step 2 — Gap Filling
```bash
python data_processing/02_gap_filling.py \
    --irimo   data/IRIMO_Daily_ClimateData_2000_2020.xlsx \
    --mazdb   data/Mazandaran.xlsx \
    --output  data/ClimateData_GapFilled_2000_2020.xlsx
```

### Step 3 — Build Final Dataset
```bash
python data_processing/03_build_final_dataset.py \
    --filled  data/ClimateData_GapFilled_2000_2020.xlsx \
    --output  data/HBCD_2020_Final.xlsx
```

### Step 4 — CMIP6 Retrieval (recommended: Google Cloud)
```bash
# Primary route — streams from the Pangeo Google Cloud Zarr archive (no login)
# Downloads pr, tasmax, tasmin, tas for all 6 GCMs × 3 experiments
python data_processing/07_download_cmip6_gcs.py --output data/cmip6_gcs/

# Fallback route — Copernicus CDS (requires a free CDS account + token)
python data_processing/08_download_cmip6_cds.py --output data/cmip6_cds/
```

See [docs/DATA_RETRIEVAL.md](docs/DATA_RETRIEVAL.md) for the full methodology and a
comparison of all four routes (Google Cloud, NEX-GDDP, CDS, ESGF).

---

## Methods Summary

### Gap-Filling Procedure

| Station | Structural Gap | Method |
|---------|---------------|--------|
| Gharakhil | None | Linear interpolation (≤3 days) |
| Sari | 2000–2005 (2,192 days) | IRIMO Provincial DB + bias correction (ΔTmax=−0.184°C, ΔTmin=−0.321°C) |
| Amol | 2000 + 2018–2020 (1,463 days) | MLR from Gharakhil + Sari (R²=0.986 for Tmax) |

**Validation statistics (overlap period 2006–2017, n=8,764 days):**
- Tmax: r = 0.996, RMSE = 0.802°C
- Tmin: r = 0.996, RMSE = 0.796°C
- Precipitation: r = 0.598, RMSE = 5.752 mm/day

---

## Data Availability

Raw observational data are available from:
- **IRIMO portal**: [http://www.irimo.ir](http://www.irimo.ir)
- **Iran Regional Water Management Organization (IRWMO)**: [https://www.wrm.ir](https://www.wrm.ir)

The processed HBCD-2020 dataset (zero missing values, 2000–2020) is available upon reasonable request.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{rahab_rajaei_2024,
  author       = {Rahab-Rajaei, Sajad},
  title        = {haraz-climate-analysis: Python scripts for CMIP6 climate data 
                  processing and gap-filling, Haraz watershed, northern Iran},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/sajadrahab/haraz-climate-analysis}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

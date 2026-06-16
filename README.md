# haraz-climate-analysis

Python code for the CMIP6-based hydroclimatic projection study of the Haraz watershed, northern Iran.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20713519.svg)](https://doi.org/10.5281/zenodo.20713519)

## Overview

This repository holds the analysis code for the paper:

> Rahab-Rajaei, S. and Motiee, H. (2025). Hydroclimatic Projections Under CMIP6 SSP Scenarios in the Haraz Watershed, Northern Iran: A Hybrid BiLSTM-BMA Framework with Compound Extreme Event Analysis.

The workflow runs from raw daily station records to projected hydroclimatic indices:

1. Quality control and gap analysis of the daily climate records.
2. Multi-source gap-filling (bias-corrected merging of IRIMO portal and provincial data, plus MLR infilling from neighbouring stations), producing a complete daily baseline for 2000-2020.
3. Evaluation of six CMIP6 GCMs (CanESM5, GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, MRI-ESM2-0, UKESM1-0-LL) against the observed baseline with RMSE, MAE, PBIAS, NSE, KGE and r, plus Taylor diagrams.
4. Statistical downscaling: LARS-WG 8 for the four models in its scenario library, and Detrended Quantile Mapping (DQM) applied directly to the two models that are not in the library.
5. Bayesian Model Averaging with weights estimated by Expectation-Maximization (Raftery et al., 2005).
6. Streamflow projection with an autoregressive bidirectional LSTM trained on the Karesang record.
7. Drought (SPI, SPEI with Hargreaves PET), ETCCDI precipitation and temperature extremes, and compound hot-dry events.
8. Manuscript figures at 600 dpi.

## Study area

| Station | Lat (N) | Lon (E) | Elev. (m) | Period | Role |
|---------|---------|---------|-----------|--------|------|
| Gharakhil | 36.487 | 52.108 | 14.7 | 2000-2020 | in/near basin (western) |
| Amol | 36.470 | 52.350 | 23.7 | 2000-2020 | Haraz mainstem |
| Sari (Dasht-e-Naz Airport) | 36.653 | 53.193 | 16.7 | 2000-2020 | eastern-margin regional reference (~75 km east of Amol) |
| Karesang (hydrometric) | 36.273 | 52.381 | 375 | 1949-2016 | streamflow gauge |

## Repository structure

```
haraz-climate-analysis/
  data_processing/
    01_gap_analysis.py             Gap detection across stations
    02_gap_filling.py              Multi-source gap-filling (bias correction + MLR)
    03_build_final_dataset.py      Assemble the gap-filled baseline (2000-2020)
    04_download_cmip6.py           CMIP6 via ESGF (reference route)
    05_download_nexgddp.py         NEX-GDDP-CMIP6 (AWS S3)
    06_download_all_cmip6.py       NEX-GDDP-CMIP6 parallel full run
    07_download_cmip6_gcs.py       CMIP6 via Google Cloud Pangeo Zarr (recommended)
    08_download_cmip6_cds.py       CMIP6 via Copernicus CDS (fallback)
    09_cmip6_evaluation.py         GCM skill scores and Taylor diagrams
    10_prepare_larswg_site.py      LARS-WG 8 site (.st/.dat) inputs
    10b_run_larswg_batch.ps1       LARS-WG 8 batch run (Windows)
    11_bias_correction_dqm.py      DQM bias correction (IPSL, MPI)
    11b_larswg_to_bc.py            LARS-WG output to projection series (4 GCMs)
    12_prepare_streamflow_data.py  Karesang discharge + basin-average climate
    13_bilstm_streamflow.py        Autoregressive BiLSTM (PyTorch)
    13b_bilstm_sklearn_fallback.py MLP fallback (no deep-learning deps)
    14_bma_ensemble.py             EM-based Bayesian Model Averaging
    15_drought_indices.py          SPI / SPEI (Hargreaves PET)
    16_extreme_indices.py          ETCCDI extreme indices
    17_compound_extremes.py        Compound hot-dry events
    18_publication_figures.py      600-dpi manuscript figures
    27_fig1_study_area.py          Study-area map (Figure 1)
  utils/stats.py                   OLS regression and metrics (numpy only)
  docs/DATA_RETRIEVAL.md           CMIP6 retrieval notes
  requirements.txt
  LICENSE
  README.md
```

Scripts are numbered in the intended order of execution (01 to 18). Paths are
resolved relative to the repository root.

## Installation

```bash
git clone https://github.com/SajadRahab96/haraz-climate-analysis.git
cd haraz-climate-analysis
pip install -r requirements.txt
```

The BiLSTM step (`13_bilstm_streamflow.py`) also needs PyTorch (`pip install torch`).
A dependency-free MLP fallback (`13b`) is included for testing the pipeline.

## Data

Large and raw data are not tracked here (see `.gitignore`). Sources:

- IRIMO daily station records: http://www.irimo.ir
- Iran Regional Water Management Organization (IRWMO): streamflow records
- CMIP6: freely available via ESGF or the Pangeo Google Cloud archive (scripts 04-08)

The processed baseline and bias-corrected projections are available on request and
will be archived on Zenodo with a DOI on publication.

## Gap-filling summary

| Station | Structural gap | Method |
|---------|---------------|--------|
| Gharakhil | none | linear interpolation (<= 3 days) |
| Sari | 2000-2005 | provincial DB + mean-bias correction (Tmax -0.184 C, Tmin -0.321 C) |
| Amol | 2000, 2018-2020 | MLR from Gharakhil (and Sari); R2 = 0.98 (Tmax/Tmin), 0.38-0.39 (precip) |

Source-portal cross-validation over 2006-2017 gives r = 0.996 for Tmax and Tmin, and r = 0.60 for daily precipitation.

## Citation

```bibtex
@misc{rahabrajaei_haraz_2025,
  author    = {Rahab-Rajaei, Sajad and Motiee, Homayoun},
  title     = {haraz-climate-analysis: code for CMIP6 hydroclimatic projection of the Haraz watershed, northern Iran},
  year      = {2026},
  doi       = {10.5281/zenodo.20713519},
  publisher = {GitHub},
  url       = {https://github.com/SajadRahab96/haraz-climate-analysis}
}
```

## License

Released under the MIT License (see LICENSE).

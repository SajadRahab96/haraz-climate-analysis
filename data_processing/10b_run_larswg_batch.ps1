# ============================================================================
# 10b_run_larswg_batch.ps1
# PowerShell helper to automate LARS-WG 8 site analysis via command-line.
#
# LARS-WG 8 supports a limited batch mode via command-line arguments.
# Usage: Right-click > "Run with PowerShell" OR run from terminal.
#
# What this does:
#   1. Updates Larswg.ini to point to correct directories
#   2. Runs Site Analysis for each station (writes .stx / .tst / .wgx)
#   3. You still need to run Scenario Generation manually in the GUI
#      (or use the SceExample.sce as template for each station+scenario)
#   4. For reRun you should use your LARSWG directories.
#   5. For understanding LARSWG8, I highly recommend you to read user's manual.
# ============================================================================


$DATA_DIR    = ""
$SITEBASE    = ""
$OUTPUT_DIR  = ""
$SCENARIO_DIR = ""

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path $SITEBASE | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

$stations = @("Amol", "Gharakhil", "Sari")

Write-Host "=========================================="
Write-Host "LARS-WG 8 Batch Site Analysis"
Write-Host "=========================================="

foreach ($station in $stations) {
    $stFile = "$DATA_DIR\$station.st"
    Write-Host ""
    Write-Host "Processing: $station"
    Write-Host "  Site file: $stFile"

    if (-not (Test-Path $stFile)) {
        Write-Host "  ERROR: Site file not found: $stFile" -ForegroundColor Red
        continue
    }

    # LARS-WG batch command: LARSWG.exe /site <path> /analysis
    # This runs Site Analysis and writes .stx/.tst/.wgx to Sitebase
    $proc = Start-Process -FilePath $LARSWG_EXE `
        -ArgumentList "/site `"$stFile`" /sitebase `"$SITEBASE`" /analysis" `
        -Wait -PassThru -NoNewWindow

    if ($proc.ExitCode -eq 0) {
        Write-Host "  Site Analysis complete." -ForegroundColor Green
    } else {
        Write-Host "  LARS-WG returned exit code $($proc.ExitCode)" -ForegroundColor Yellow
        Write-Host "  If batch mode is not supported, run Site Analysis manually in GUI." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Manual steps required in LARS-WG GUI:"
Write-Host "  1. File > Open Site > [station].st"
Write-Host "  2. Analysis > Site Analysis"
Write-Host "  3. Scenarios > Generate Scenarios"
Write-Host "     - Ensemble: CMIP6"
Write-Host "     - Models: CanESM5, GFDL-ESM4, MRI-ESM2-0, UKESM1-0-LL"
Write-Host "     - Scenarios: SSP2-4.5 AND SSP5-8.5"
Write-Host "     - Periods: 2021-2060 and 2061-2100"
Write-Host "     - Output dir: $OUTPUT_DIR"
Write-Host "  4. Copy generated .dat files to:"
Write-Host "     #enter directory"
Write-Host "=========================================="

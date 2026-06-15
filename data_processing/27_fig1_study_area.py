"""
27_fig1_study_area.py
=====================
Figure 1 — Study area location map (pure matplotlib, no GIS deps).
Main panel: Haraz watershed area with stations, Caspian coast, Alborz ridge.
Inset: Iran outline with study area box.

NOTE: This is a publication-grade schematic. For final submission a
DEM-based map (e.g., QGIS + SRTM) is recommended.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "figures_600dpi"
OUT.mkdir(parents=True, exist_ok=True)

# Station data (lon, lat, name, type)
STATIONS = [
    (52.108, 36.487, "Gharakhil", "synoptic"),
    (52.350, 36.470, "Amol", "synoptic"),
    (53.193, 36.653, "Sari (D-e-N)", "synoptic"),
    (52.381, 36.273, "Karesang (hydro.)", "hydro"),
]

# Approximate Haraz watershed boundary polygon (schematic, follows P17 extents)
WSHED = np.array([
    [52.05, 36.62], [52.20, 36.65], [52.35, 36.60], [52.45, 36.50],
    [52.55, 36.35], [52.60, 36.15], [52.50, 35.95], [52.35, 35.88],
    [52.15, 35.90], [52.00, 36.00], [51.90, 36.15], [51.88, 36.35],
    [51.95, 36.52], [52.05, 36.62],
])

# Approximate southern Caspian coastline (schematic polyline)
COAST = np.array([
    [50.5, 37.05], [51.0, 36.85], [51.5, 36.75], [52.0, 36.68],
    [52.4, 36.66], [52.8, 36.72], [53.2, 36.80], [53.7, 36.85], [54.2, 36.95],
])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.08, 0.08, 0.88, 0.86])

# Caspian Sea fill (north of coast)
coast_x = COAST[:, 0]
coast_y = COAST[:, 1]
ax.fill_between(coast_x, coast_y, 37.3, color="#aed4ef", alpha=0.9, zorder=0)
ax.plot(coast_x, coast_y, color="#3b7db8", lw=1.5, zorder=2)
ax.text(52.6, 37.05, "Caspian Sea", fontsize=13, color="#1d5d99",
        style="italic", ha="center", zorder=3)

# Lowland plain band
ax.fill_between(coast_x, coast_y - 0.18, coast_y, color="#dff0d8", alpha=0.8, zorder=0)

# Alborz mountain band (schematic shading south of plain)
xs = np.linspace(50.5, 54.2, 200)
ridge_y = 36.05 + 0.06 * np.sin((xs - 50.5) * 2.2)
ax.fill_between(xs, 35.55, ridge_y, color="#d8c8ae", alpha=0.75, zorder=0)
ax.text(51.1, 35.75, "Alborz Mountains", fontsize=12, color="#7a5c33",
        style="italic", rotation=4)

# Watershed polygon
ax.fill(WSHED[:, 0], WSHED[:, 1], facecolor="#f5e6b8", edgecolor="#b3262a",
        lw=2.2, alpha=0.55, zorder=3)
ax.text(52.18, 36.22, "Haraz\nWatershed", fontsize=12, fontweight="bold",
        color="#7d1f22", ha="center", zorder=4)

# Haraz river (schematic line from mountains to coast near Amol)
river = np.array([
    [52.30, 35.95], [52.28, 36.10], [52.33, 36.25], [52.37, 36.40],
    [52.36, 36.50], [52.38, 36.62], [52.40, 36.66],
])
ax.plot(river[:, 0], river[:, 1], color="#2a6fb0", lw=2.0, zorder=4)
ax.annotate("Haraz River", xy=(52.34, 36.33), xytext=(52.62, 36.36),
            fontsize=10, color="#2a6fb0",
            arrowprops=dict(arrowstyle="-", color="#2a6fb0", lw=0.8))

# Stations
for lon, lat, name, typ in STATIONS:
    if typ == "synoptic":
        ax.plot(lon, lat, "^", ms=13, mfc="#cf2b29", mec="k", mew=1.0, zorder=6)
    else:
        ax.plot(lon, lat, "s", ms=12, mfc="#2456a4", mec="k", mew=1.0, zorder=6)
    dy = 0.045 if name != "Amol" else -0.075
    ax.text(lon + 0.04, lat + dy, name, fontsize=10.5, fontweight="bold", zorder=6)

# Clarify that Sari/Dasht-e-Naz lies on the eastern margin (regional reference station)
ax.annotate("eastern-margin\nregional reference\n(~75 km E of Amol)",
            xy=(53.193, 36.653), xytext=(53.05, 36.40),
            fontsize=8.0, color="#7d1f22", ha="center", style="italic",
            arrowprops=dict(arrowstyle="->", color="#7d1f22", lw=0.8), zorder=6)

# Graticule
ax.set_xticks(np.arange(50.5, 54.3, 0.5))
ax.set_yticks(np.arange(35.6, 37.3, 0.4))
ax.grid(True, ls=":", alpha=0.5)
ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)
ax.set_xlim(50.5, 54.2)
ax.set_ylim(35.55, 37.3)

# North arrow + scale bar
ax.annotate("N", xy=(53.95, 37.12), fontsize=14, fontweight="bold", ha="center")
ax.annotate("", xy=(53.95, 37.10), xytext=(53.95, 36.97),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="k"))
# Scale: 1 deg lon at 36.5N ~ 89.4 km -> 50 km ~ 0.559 deg
ax.plot([50.7, 50.7 + 0.559], [35.65, 35.65], "k-", lw=3)
ax.text(50.7 + 0.28, 35.69, "50 km", ha="center", fontsize=10)

# Legend
legend_items = [
    Line2D([0], [0], marker="^", color="none", mfc="#cf2b29", mec="k", ms=11, label="Synoptic station"),
    Line2D([0], [0], marker="s", color="none", mfc="#2456a4", mec="k", ms=10, label="Hydrometric station"),
    mpatches.Patch(facecolor="#f5e6b8", edgecolor="#b3262a", label="Haraz watershed"),
    Line2D([0], [0], color="#2a6fb0", lw=2, label="Haraz River"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=10, framealpha=0.95)

ax.set_title("Study Area: Haraz Watershed and Adjacent Central Mazandaran Plain, Northern Iran",
             fontsize=12, fontweight="bold")

# ── Inset: Iran outline (simplified polygon) ─────────────────────────────────
iran = np.array([
    [44.0, 39.4], [44.8, 39.7], [46.1, 38.8], [47.0, 39.6], [48.3, 38.9],
    [48.9, 38.4], [49.1, 37.6], [50.1, 37.4], [51.1, 36.8], [52.5, 36.6],
    [54.0, 36.9], [55.0, 37.5], [56.2, 38.1], [57.3, 38.0], [58.4, 37.6],
    [59.3, 37.5], [60.3, 36.6], [61.2, 36.6], [61.3, 35.6], [60.9, 34.3],
    [60.5, 33.5], [60.9, 32.6], [60.8, 31.5], [61.7, 31.0], [61.8, 30.2],
    [60.9, 29.8], [61.5, 29.4], [62.8, 28.3], [62.8, 27.3], [63.2, 27.2],
    [63.3, 26.7], [61.8, 25.8], [61.6, 25.2], [59.5, 25.4], [57.8, 25.7],
    [57.3, 26.8], [56.4, 27.0], [55.5, 26.7], [54.7, 26.5], [53.5, 26.7],
    [52.6, 27.2], [51.5, 27.9], [50.8, 28.8], [50.1, 29.9], [49.6, 30.1],
    [48.9, 30.3], [48.5, 29.9], [48.0, 30.4], [47.7, 30.1], [47.1, 31.0],
    [46.1, 32.6], [45.4, 33.4], [45.4, 34.5], [45.7, 35.0], [46.2, 35.7],
    [45.7, 36.7], [44.8, 37.2], [44.2, 37.9], [44.4, 38.4], [44.0, 39.4],
])
axin = fig.add_axes([0.105, 0.595, 0.26, 0.30])
axin.fill(iran[:, 0], iran[:, 1], facecolor="#e8e8e8", edgecolor="k", lw=1.0)
# Study area box
axin.add_patch(mpatches.Rectangle((51.8, 35.8), 1.7, 1.0, fill=False,
                                   edgecolor="#b3262a", lw=2))
axin.text(48.0, 32.0, "IRAN", fontsize=11, fontweight="bold", ha="center")
axin.annotate("Study\narea", xy=(52.6, 36.3), xytext=(56.8, 33.5), fontsize=8.5,
              color="#7d1f22",
              arrowprops=dict(arrowstyle="->", color="#7d1f22", lw=1.2))
axin.set_xlim(43.5, 64.0)
axin.set_ylim(24.8, 40.2)
axin.set_xticks([]); axin.set_yticks([])
for spine in axin.spines.values():
    spine.set_linewidth(1.2)

fig.savefig(OUT / "Fig1_Study_Area.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("Saved:", OUT / "Fig1_Study_Area.png")

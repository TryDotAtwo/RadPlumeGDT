from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cartopy
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from .config import DomainConfig


REFERENCE_CITIES = {
    "Dimona": (35.033, 31.067),
    "Beersheba": (34.7913, 31.2518),
    "Jerusalem": (35.2137, 31.7683),
    "Tel Aviv": (34.7818, 32.0853),
    "Gaza": (34.4668, 31.5018),
    "Amman": (35.9304, 31.9539),
    "Aqaba": (35.0060, 29.5321),
    "Cairo": (31.2357, 30.0444),
}


COUNTRY_LABELS = {
    "Israel": (35.00, 31.95),
    "Jordan": (36.15, 31.15),
    "Egypt": (31.95, 29.70),
    "Saudi Arabia": (37.55, 29.35),
}


@lru_cache(maxsize=1)
def _load_world_background() -> np.ndarray | None:
    background_path = Path(cartopy.config["repo_data_dir"]) / "raster" / "natural_earth" / "50-natural-earth-1-downsampled.png"
    if not background_path.exists():
        return None
    return plt.imread(background_path)


@lru_cache(maxsize=1)
def _load_vector_layers() -> tuple[list, list, list]:
    try:
        land_path = shpreader.natural_earth("50m", "physical", "land")
        coast_path = shpreader.natural_earth("50m", "physical", "coastline")
        border_path = shpreader.natural_earth("50m", "cultural", "admin_0_boundary_lines_land")
    except Exception:
        return [], [], []

    land_geoms = list(shpreader.Reader(land_path).geometries())
    coast_geoms = list(shpreader.Reader(coast_path).geometries())
    border_geoms = list(shpreader.Reader(border_path).geometries())
    return land_geoms, coast_geoms, border_geoms


def _plot_geometry(ax, geometry, *, edgecolor: str, linewidth: float, facecolor: str | None = None, alpha: float = 1.0, zorder: int = 1) -> None:
    if isinstance(geometry, Polygon):
        x, y = geometry.exterior.xy
        ax.fill(x, y, facecolor=facecolor or "none", edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        return

    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            _plot_geometry(ax, part, edgecolor=edgecolor, linewidth=linewidth, facecolor=facecolor, alpha=alpha, zorder=zorder)
        return

    if isinstance(geometry, LineString):
        x, y = geometry.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        return

    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            _plot_geometry(ax, part, edgecolor=edgecolor, linewidth=linewidth, facecolor=facecolor, alpha=alpha, zorder=zorder)


def _text_with_halo(ax, x: float, y: float, text: str, *, size: int, color: str, ha: str = "left") -> None:
    artist = ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        ha=ha,
        va="center",
        zorder=12,
    )
    artist.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])


def draw_geographic_context(
    ax,
    extent: tuple[float, float, float, float],
    domain: DomainConfig,
    *,
    show_raster: bool = True,
    raster_alpha: float = 0.72,
) -> None:
    west, east, south, north = extent
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_facecolor("#f8fbff")

    world_background = _load_world_background()
    if show_raster and world_background is not None:
        ax.imshow(
            world_background,
            extent=(-180.0, 180.0, -90.0, 90.0),
            origin="upper",
            interpolation="nearest",
            alpha=raster_alpha,
            zorder=0,
            aspect="auto",
        )

    land_geoms, coast_geoms, border_geoms = _load_vector_layers()
    for geometry in land_geoms:
        _plot_geometry(
            ax,
            geometry,
            edgecolor="#cfd6dd",
            linewidth=0.30,
            facecolor="#f5efe2",
            alpha=0.88 if show_raster else 1.0,
            zorder=1,
        )
    for geometry in border_geoms:
        _plot_geometry(ax, geometry, edgecolor="#627284", linewidth=0.55, alpha=0.55, zorder=4)
    for geometry in coast_geoms:
        _plot_geometry(ax, geometry, edgecolor="#274c77", linewidth=0.90, alpha=0.92, zorder=5)

    ax.grid(True, color="#2d3e50", alpha=0.22, linewidth=0.6)
    ax.set_xlabel("Longitude", color="#14202b")
    ax.set_ylabel("Latitude", color="#14202b")
    ax.tick_params(colors="#14202b")

    ax.scatter(
        [domain.source_lon],
        [domain.source_lat],
        s=180,
        marker="*",
        color="#ffca3a",
        edgecolor="black",
        linewidth=0.9,
        zorder=14,
    )
    _text_with_halo(ax, domain.source_lon + 0.08, domain.source_lat + 0.04, "Source", size=10, color="#101820")

    for name, (lon, lat) in REFERENCE_CITIES.items():
        if west <= lon <= east and south <= lat <= north and name != "Dimona":
            ax.scatter([lon], [lat], s=16, color="#0e2433", alpha=0.85, zorder=13)
            _text_with_halo(ax, lon + 0.05, lat + 0.03, name, size=8, color="#162635")

    for label, (lon, lat) in COUNTRY_LABELS.items():
        if west <= lon <= east and south <= lat <= north:
            ax.text(
                lon,
                lat,
                label,
                fontsize=10,
                color="#203243",
                ha="center",
                va="center",
                alpha=0.85,
                zorder=6,
                bbox={"facecolor": "white", "alpha": 0.32, "edgecolor": "none", "pad": 1.6},
            )

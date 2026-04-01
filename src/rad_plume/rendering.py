from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm

from .config import DispersionConfig, DomainConfig, HazardConfig, TimelineConfig, VisualConfig
from .dispersion import DispersionAggregateResult, DispersionResult, DispersionSnapshot, run_dispersion_aggregate
from .geography import draw_geographic_context
from .hazard_analysis import ScenarioHazardResult, project_total_effective_dose_msv
from .meteo import PreparedMeteo, iter_interpolated_time_chunks


PLUME_ZONE_COLORS = ["#fff3b0", "#ffd166", "#fca311", "#f77f00", "#d62828"]
PLUME_VISUAL_FRACTIONS = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def _finite_field_mean(field: np.ndarray) -> float:
    """Mean of finite values; 0.0 if empty or all-NaN (avoids nanmean empty-slice warnings)."""
    if field.size == 0:
        return 0.0
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _ffmpeg_writer(fps: int) -> FFMpegWriter:
    return FFMpegWriter(fps=fps, codec="libx264", extra_args=["-preset", "ultrafast", "-crf", "24"])


def _save_animation_with_progress(animation: FuncAnimation, output_path: Path, fps: int, dpi: int, label: str, total_frames: int) -> None:
    progress = tqdm(total=total_frames, desc=label, unit="frame")

    def _progress_callback(frame_index: int, _total: int) -> None:
        target = frame_index + 1
        if target > progress.n:
            progress.update(target - progress.n)

    try:
        animation.save(
            output_path,
            writer=_ffmpeg_writer(fps),
            dpi=dpi,
            progress_callback=_progress_callback,
        )
    finally:
        if progress.n < total_frames:
            progress.update(total_frames - progress.n)
        progress.close()


def _prepare_plot_fields(frame_meteo: PreparedMeteo, resolution_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    west, east, south, north = frame_meteo.extent
    lon_values = np.arange(west, east + resolution_deg, resolution_deg)
    lat_values = np.arange(south, north + resolution_deg, resolution_deg)
    plot_ds = frame_meteo.ds.interp(longitude=lon_values, latitude=lat_values, method="linear")
    u_frames = plot_ds["u10"].values
    v_frames = plot_ds["v10"].values
    speed_frames = np.hypot(u_frames, v_frames)
    lon2d, lat2d = np.meshgrid(lon_values, lat_values)
    return lon2d, lat2d, u_frames, v_frames, speed_frames


def _height_to_pressure_hpa(height_m: float) -> float:
    capped_height_m = float(np.clip(height_m, 0.0, 11_000.0))
    return float(1013.25 * np.power(max(1.0 - capped_height_m / 44_330.0, 1e-4), 5.255))


def _representative_cloud_height_m(plot_ds: xr.Dataset, domain: DomainConfig, dispersion: DispersionConfig) -> np.ndarray:
    weights = np.asarray(dispersion.release_layer_mass_fractions, dtype=float)
    fractions = np.asarray(dispersion.release_layer_height_fractions, dtype=float)
    mean_fraction = float(np.dot(weights, fractions) / max(weights.sum(), 1e-9))
    if "blh" in plot_ds.data_vars:
        source_blh = plot_ds["blh"].interp(
            longitude=xr.DataArray([domain.source_lon], dims="points"),
            latitude=xr.DataArray([domain.source_lat], dims="points"),
            method="linear",
        ).values[:, 0]
        mixing_height = np.where(np.isfinite(source_blh), source_blh, dispersion.mixing_height_m)
    else:
        mixing_height = np.full(plot_ds.sizes["time"], dispersion.mixing_height_m, dtype=float)
    return np.clip(mixing_height * mean_fraction, dispersion.min_mixing_height_m, dispersion.max_mixing_height_m)


def _prepare_wind_view_fields(
    frame_meteo: PreparedMeteo,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, tuple[float, float, float, float]]:
    west = float(frame_meteo.ds.longitude.min())
    east = float(frame_meteo.ds.longitude.max())
    south = float(frame_meteo.ds.latitude.min())
    north = float(frame_meteo.ds.latitude.max())
    lon_values = np.arange(west, east + resolution_deg, resolution_deg)
    lat_values = np.arange(south, north + resolution_deg, resolution_deg)
    plot_ds = frame_meteo.ds.interp(longitude=lon_values, latitude=lat_values, method="linear")
    lon2d, lat2d = np.meshgrid(lon_values, lat_values)
    field_extent = (west, east, south, north)

    if "u_pl" not in plot_ds.data_vars or "v_pl" not in plot_ds.data_vars or "pressure_level" not in plot_ds.coords:
        u_frames = plot_ds["u10"].values
        v_frames = plot_ds["v10"].values
        return lon2d, lat2d, u_frames, v_frames, np.hypot(u_frames, v_frames), "10 m wind", field_extent

    cloud_heights_m = _representative_cloud_height_m(plot_ds, domain, dispersion)
    u_frames: list[np.ndarray] = []
    v_frames: list[np.ndarray] = []
    pressure_level_note = "mean cloud steering level"
    for frame_index, height_m in enumerate(cloud_heights_m):
        target_pressure = _height_to_pressure_hpa(float(height_m))
        u_frame = plot_ds["u_pl"].isel(time=frame_index).interp(pressure_level=target_pressure, method="linear").values
        v_frame = plot_ds["v_pl"].isel(time=frame_index).interp(pressure_level=target_pressure, method="linear").values
        if np.any(np.isfinite(u_frame)) and np.any(np.isfinite(v_frame)):
            u_frame = np.where(np.isfinite(u_frame), u_frame, plot_ds["u10"].isel(time=frame_index).values)
            v_frame = np.where(np.isfinite(v_frame), v_frame, plot_ds["v10"].isel(time=frame_index).values)
        else:
            u_frame = plot_ds["u10"].isel(time=frame_index).values
            v_frame = plot_ds["v10"].isel(time=frame_index).values
        u_frames.append(u_frame)
        v_frames.append(v_frame)

    u_stack = np.stack(u_frames, axis=0)
    v_stack = np.stack(v_frames, axis=0)
    mean_height = float(np.nanmean(cloud_heights_m))
    pressure_level_note = f"cloud steering level ~{mean_height:.0f} m"
    return lon2d, lat2d, u_stack, v_stack, np.hypot(u_stack, v_stack), pressure_level_note, field_extent


def _fixed_zone_levels(field: np.ndarray, fractions: tuple[float, ...]) -> np.ndarray | None:
    positive = field[field > 0]
    if positive.size == 0:
        return None

    peak = float(np.nanmax(positive))
    levels = np.array([peak * fraction for fraction in fractions if peak * fraction > 0], dtype=float)
    levels = np.unique(np.clip(levels, float(positive.min()), peak))
    if levels.size == 0:
        return None
    if levels[-1] < peak:
        levels = np.append(levels, peak)
    return levels


def _zone_levels_from_peak(peak: float, fractions: tuple[float, ...]) -> np.ndarray:
    levels = np.array([peak * fraction for fraction in fractions if peak * fraction > 0], dtype=float)
    levels = np.unique(levels)
    if levels.size == 0 or levels[-1] < peak:
        levels = np.append(levels, peak)
    return levels


def _dose_display_levels(field_msv: np.ndarray, configured_levels: tuple[float, ...]) -> np.ndarray:
    positive = field_msv[field_msv > 0]
    if positive.size == 0:
        return np.array(configured_levels, dtype=float)
    configured = np.array([level for level in configured_levels if level > 0], dtype=float)
    visible = configured[configured <= float(np.nanmax(positive))]
    if visible.size >= 2:
        return visible
    vmax = float(np.nanmax(positive))
    vmin = max(float(np.nanmin(positive)), min(configured[0], 1e-6)) if configured.size else max(vmax * 1e-3, 1e-6)
    return np.geomspace(vmin, vmax, num=6)


def _add_colorbar(fig, ax, cmap, norm, label: str, *, boundaries=None):
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.84, pad=0.02, boundaries=boundaries)
    colorbar.set_label(label, color="#14202b")
    colorbar.ax.yaxis.set_tick_params(color="#14202b")
    plt.setp(colorbar.ax.get_yticklabels(), color="#14202b")
    return colorbar


def _footer_text(text: str, *, width: int = 150) -> str:
    return fill(" ".join(str(text).split()), width=width)


def _build_plume_discrete_scale(boundaries: np.ndarray) -> tuple[mcolors.ListedColormap, BoundaryNorm]:
    color_list = PLUME_ZONE_COLORS + [PLUME_ZONE_COLORS[-1]]
    cmap = mcolors.ListedColormap(color_list)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def _safe_boundary_norm(boundaries: np.ndarray, cmap: mcolors.Colormap) -> BoundaryNorm:
    # Matplotlib may require extra bins when contour/colorbar uses extensions.
    min_required_colors = max(int(len(boundaries)) + 1, 2)
    ncolors = max(int(cmap.N), min_required_colors)
    return BoundaryNorm(boundaries, ncolors, clip=True)


def _upsample_scalar_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    *,
    scale_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_values = lon2d[0, :]
    lat_values = lat2d[:, 0]
    target_lon = np.linspace(float(lon_values.min()), float(lon_values.max()), len(lon_values) * scale_factor)
    target_lat = np.linspace(float(lat_values.min()), float(lat_values.max()), len(lat_values) * scale_factor)
    field_da = xr.DataArray(field, coords={"latitude": lat_values, "longitude": lon_values}, dims=("latitude", "longitude"))
    upsampled = field_da.interp(latitude=target_lat, longitude=target_lon, method="linear")
    fine_lon2d, fine_lat2d = np.meshgrid(target_lon, target_lat)
    return fine_lon2d, fine_lat2d, upsampled.values


def _spawn_inflow_particles(
    count: int,
    extent: tuple[float, float, float, float],
    mean_u: float,
    mean_v: float,
    rng: np.random.Generator,
    *,
    edge_bias: float,
) -> tuple[np.ndarray, np.ndarray]:
    west, east, south, north = extent
    lon = rng.uniform(west, east, size=count)
    lat = rng.uniform(south, north, size=count)
    if count == 0:
        return lon, lat

    edge_count = int(round(count * edge_bias))
    if edge_count <= 0:
        return lon, lat

    edge_margin_lon = 0.08 * max(east - west, 1e-6)
    edge_margin_lat = 0.08 * max(north - south, 1e-6)
    if abs(mean_u) >= abs(mean_v):
        if mean_u >= 0.0:
            lon[:edge_count] = west + rng.uniform(0.0, edge_margin_lon, size=edge_count)
        else:
            lon[:edge_count] = east - rng.uniform(0.0, edge_margin_lon, size=edge_count)
        lat[:edge_count] = rng.uniform(south, north, size=edge_count)
    else:
        if mean_v >= 0.0:
            lat[:edge_count] = south + rng.uniform(0.0, edge_margin_lat, size=edge_count)
        else:
            lat[:edge_count] = north - rng.uniform(0.0, edge_margin_lat, size=edge_count)
        lon[:edge_count] = rng.uniform(west, east, size=edge_count)
    return lon, lat


def _advect_visual_particles(
    particle_lon: np.ndarray,
    particle_lat: np.ndarray,
    *,
    u_frame: np.ndarray,
    v_frame: np.ndarray,
    plot_lon: np.ndarray,
    plot_lat: np.ndarray,
    field_extent: tuple[float, float, float, float],
    source_lat: float,
    rng: np.random.Generator,
    dt_hours: float,
    refresh_fraction: float,
    inflow_bias: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_field = xr.DataArray(
        u_frame,
        coords={"latitude": plot_lat, "longitude": plot_lon},
        dims=("latitude", "longitude"),
    )
    v_field = xr.DataArray(
        v_frame,
        coords={"latitude": plot_lat, "longitude": plot_lon},
        dims=("latitude", "longitude"),
    )
    lon_points = xr.DataArray(particle_lon, dims="points")
    lat_points = xr.DataArray(particle_lat, dims="points")
    u_values = u_field.interp(longitude=lon_points, latitude=lat_points, method="linear").values
    v_values = v_field.interp(longitude=lon_points, latitude=lat_points, method="linear").values

    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(source_lat))
    particle_lon = particle_lon + np.nan_to_num(u_values, nan=0.0) * dt_hours * 3600.0 / meters_per_deg_lon
    particle_lat = particle_lat + np.nan_to_num(v_values, nan=0.0) * dt_hours * 3600.0 / 111_320.0

    invalid = (
        (particle_lon < field_extent[0])
        | (particle_lon > field_extent[1])
        | (particle_lat < field_extent[2])
        | (particle_lat > field_extent[3])
        | ~np.isfinite(particle_lon)
        | ~np.isfinite(particle_lat)
        | ~np.isfinite(u_values)
        | ~np.isfinite(v_values)
    )
    mean_u = _finite_field_mean(u_frame)
    mean_v = _finite_field_mean(v_frame)
    if invalid.any():
        respawn_lon, respawn_lat = _spawn_inflow_particles(
            int(invalid.sum()),
            field_extent,
            mean_u,
            mean_v,
            rng,
            edge_bias=inflow_bias,
        )
        particle_lon[invalid] = respawn_lon
        particle_lat[invalid] = respawn_lat

    refresh_count = max(1, int(len(particle_lon) * refresh_fraction))
    refresh_indices = rng.choice(len(particle_lon), size=refresh_count, replace=False)
    refresh_lon, refresh_lat = _spawn_inflow_particles(
        refresh_count,
        field_extent,
        mean_u,
        mean_v,
        rng,
        edge_bias=inflow_bias,
    )
    particle_lon[refresh_indices] = refresh_lon
    particle_lat[refresh_indices] = refresh_lat
    return particle_lon, particle_lat


def _source_label(domain: DomainConfig) -> str:
    return f"{domain.source_lat:.3f}N, {domain.source_lon:.3f}E"


def _estimate_wind_speed_vmax(
    raw_meteo: PreparedMeteo,
    domain: DomainConfig,
    timeline: TimelineConfig,
    dispersion: DispersionConfig,
    *,
    start_utc: str,
    end_utc: str,
    chunk_hours: int,
) -> float:
    vmax = 1.0
    for chunk in iter_interpolated_time_chunks(
        prepared=raw_meteo,
        start_utc=start_utc,
        end_utc=end_utc,
        step_minutes=timeline.frame_step_minutes,
        chunk_hours=chunk_hours,
    ):
        chunk_ds = chunk.ds[[name for name in ("u10", "v10", "u_pl", "v_pl", "blh") if name in chunk.ds.data_vars]]
        chunk_prepared = PreparedMeteo(
            ds=chunk_ds,
            extent=chunk.extent,
            native_step_minutes=chunk.native_step_minutes,
            grid_spacing_m=chunk.grid_spacing_m,
            source_summary=chunk.source_summary,
        )
        _, _, _, _, speed_frames, _, _ = _prepare_wind_view_fields(
            chunk_prepared,
            domain,
            dispersion,
            domain.wind_plot_resolution_deg,
        )
        vmax = max(vmax, float(np.nanpercentile(speed_frames, 98)))
    return vmax


def render_wind_animation(
    raw_meteo: PreparedMeteo,
    domain: DomainConfig,
    timeline: TimelineConfig,
    visual: VisualConfig,
    output_path: Path,
    dispersion: DispersionConfig,
    *,
    start_utc: str,
    end_utc: str,
    chunk_hours: int = 48,
) -> None:
    rng = np.random.default_rng(42)
    total_frames = len(pd.date_range(start=pd.Timestamp(start_utc), end=pd.Timestamp(end_utc), freq=f"{timeline.frame_step_minutes}min"))
    speed_vmax = _estimate_wind_speed_vmax(
        raw_meteo,
        domain,
        timeline,
        dispersion,
        start_utc=start_utc,
        end_utc=end_utc,
        chunk_hours=chunk_hours,
    )
    speed_norm = mcolors.Normalize(vmin=0.0, vmax=speed_vmax)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    fig.text(0.5, 0.02, _footer_text(raw_meteo.source_summary), ha="center", va="bottom", fontsize=8, color="#203243")
    draw_geographic_context(ax, raw_meteo.extent, domain)
    speed_image = None
    quiver = None
    particles = None
    title = ax.set_title("", color="#13212e", fontsize=13)
    _add_colorbar(fig, ax, "YlGnBu", speed_norm, "Wind speed, m/s")
    stride = visual.quiver_stride
    particle_lon = np.array([], dtype=float)
    particle_lat = np.array([], dtype=float)
    previous_time = None
    written_frames = 0

    writer = _ffmpeg_writer(timeline.fps)
    progress = tqdm(total=total_frames, desc="Wind frames", unit="frame")
    try:
        with writer.saving(fig, output_path, visual.dpi):
            skip_first_frame = False
            for chunk in iter_interpolated_time_chunks(
                prepared=raw_meteo,
                start_utc=start_utc,
                end_utc=end_utc,
                step_minutes=timeline.frame_step_minutes,
                chunk_hours=chunk_hours,
            ):
                chunk_ds = chunk.ds[[name for name in ("u10", "v10", "u_pl", "v_pl", "blh") if name in chunk.ds.data_vars]]
                chunk_prepared = PreparedMeteo(
                    ds=chunk_ds,
                    extent=chunk.extent,
                    native_step_minutes=chunk.native_step_minutes,
                    grid_spacing_m=chunk.grid_spacing_m,
                    source_summary=chunk.source_summary,
                )
                times = chunk_prepared.ds.time.values
                lon2d, lat2d, u_frames, v_frames, speed_frames, wind_view_label, field_extent = _prepare_wind_view_fields(
                    chunk_prepared,
                    domain,
                    dispersion,
                    domain.wind_plot_resolution_deg,
                )
                plot_lon = lon2d[0, :]
                plot_lat = lat2d[:, 0]

                frame_start_index = 1 if skip_first_frame else 0
                if not skip_first_frame:
                    initial_mean_u = _finite_field_mean(u_frames[0])
                    initial_mean_v = _finite_field_mean(v_frames[0])
                    particle_lon, particle_lat = _spawn_inflow_particles(
                        visual.wind_particle_count,
                        field_extent,
                        initial_mean_u,
                        initial_mean_v,
                        rng,
                        edge_bias=visual.wind_particle_inflow_bias,
                    )
                    for _ in range(8):
                        particle_lon, particle_lat = _advect_visual_particles(
                            particle_lon,
                            particle_lat,
                            u_frame=u_frames[0],
                            v_frame=v_frames[0],
                            plot_lon=plot_lon,
                            plot_lat=plot_lat,
                            field_extent=field_extent,
                            source_lat=domain.source_lat,
                            rng=rng,
                            dt_hours=0.75,
                            refresh_fraction=max(visual.wind_particle_refresh_fraction * 0.3, 0.02),
                            inflow_bias=visual.wind_particle_inflow_bias,
                        )
                    speed_image = ax.imshow(
                        speed_frames[0],
                        extent=(field_extent[0], field_extent[1], field_extent[2], field_extent[3]),
                        origin="lower",
                        cmap="YlGnBu",
                        norm=speed_norm,
                        alpha=0.58,
                        interpolation="bilinear",
                        zorder=2,
                        aspect="auto",
                    )
                    quiver = ax.quiver(
                        lon2d[::stride, ::stride],
                        lat2d[::stride, ::stride],
                        u_frames[0][::stride, ::stride],
                        v_frames[0][::stride, ::stride],
                        color="#17324d",
                        scale=240.0,
                        width=0.0025,
                        alpha=0.88,
                        zorder=8,
                    )
                    particles = ax.scatter(
                        particle_lon,
                        particle_lat,
                        s=15,
                        color="#eef8ff",
                        alpha=0.84,
                        linewidths=0.0,
                        zorder=9,
                    )

                for frame_index in range(frame_start_index, len(times)):
                    current_time = times[frame_index]
                    if previous_time is not None:
                        dt_hours = float((current_time - previous_time) / np.timedelta64(1, "h"))
                        source_index = frame_index - 1 if frame_index > 0 else 0
                        particle_lon, particle_lat = _advect_visual_particles(
                            particle_lon,
                            particle_lat,
                            u_frame=u_frames[source_index],
                            v_frame=v_frames[source_index],
                            plot_lon=plot_lon,
                            plot_lat=plot_lat,
                            field_extent=field_extent,
                            source_lat=domain.source_lat,
                            rng=rng,
                            dt_hours=dt_hours,
                            refresh_fraction=visual.wind_particle_refresh_fraction,
                            inflow_bias=visual.wind_particle_inflow_bias,
                        )

                    speed_image.set_data(speed_frames[frame_index])
                    quiver.set_UVC(
                        u_frames[frame_index][::stride, ::stride],
                        v_frames[frame_index][::stride, ::stride],
                    )
                    particles.set_offsets(np.column_stack((particle_lon, particle_lat)))
                    title.set_text(
                        f"Wind field around source {_source_label(domain)} | {np.datetime_as_string(current_time, unit='m')} UTC | {wind_view_label}",
                    )
                    writer.grab_frame()
                    written_frames += 1
                    if written_frames > progress.n:
                        progress.update(written_frames - progress.n)
                    previous_time = current_time
                skip_first_frame = True
    finally:
        if progress.n < total_frames:
            progress.update(total_frames - progress.n)
        progress.close()
    plt.close(fig)


def render_plume_animation(
    result: DispersionResult,
    frame_meteo: PreparedMeteo,
    domain: DomainConfig,
    timeline: TimelineConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
) -> None:
    times = [snapshot.time for snapshot in result.snapshots]
    extent = result.grid.extent
    lon2d, lat2d, u_frames, v_frames, _ = _prepare_plot_fields(frame_meteo, domain.wind_plot_resolution_deg)

    positive_values = [
        float(np.nanmax(snapshot.cloud_column_bq_m2))
        for snapshot in result.snapshots
        if np.any(snapshot.cloud_column_bq_m2 > 0)
    ]
    concentration_peak = max(positive_values) if positive_values else 1.0
    zone_levels = _zone_levels_from_peak(concentration_peak, PLUME_VISUAL_FRACTIONS)
    plume_boundaries = np.unique(np.concatenate(([max(zone_levels[0] * 0.5, 1e-12)], zone_levels)))
    plume_cmap = plt.get_cmap("YlOrRd", max(len(plume_boundaries) - 1, 2))
    plume_norm = _safe_boundary_norm(plume_boundaries, plume_cmap)
    max_field = np.zeros_like(result.snapshots[0].cloud_column_bq_m2, dtype=np.float32)
    for snapshot in result.snapshots:
        max_field = np.maximum(max_field, snapshot.cloud_column_bq_m2)
    max_positive = max_field[max_field > 0.0]
    footprint_threshold = plume_boundaries[0]
    if max_positive.size:
        footprint_threshold = max(float(np.nanmax(max_positive)) * 1e-4, float(np.nanpercentile(max_positive, 5)))
    footprint = max_field >= footprint_threshold

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    fig.text(0.5, 0.02, _footer_text(result.source_summary), ha="center", va="bottom", fontsize=8, color="#203243")
    _add_colorbar(
        fig,
        ax,
        plume_cmap,
        plume_norm,
        "Cloud column activity, Bq/m^2",
        boundaries=plume_boundaries,
    )

    def update(frame_index: int):
        ax.clear()
        draw_geographic_context(ax, extent, domain)

        snapshot = result.snapshots[frame_index]
        frame_positive = snapshot.cloud_column_bq_m2[snapshot.cloud_column_bq_m2 > 0.0]
        frame_threshold = plume_boundaries[0]
        if frame_positive.size:
            frame_threshold = max(float(np.nanmax(frame_positive)) * 1e-4, plume_boundaries[0])
        concentration = np.ma.masked_less_equal(snapshot.cloud_column_bq_m2, frame_threshold)
        if concentration.count() > 0:
            ax.contourf(
                result.grid.lon2d,
                result.grid.lat2d,
                concentration,
                levels=plume_boundaries,
                cmap=plume_cmap,
                norm=plume_norm,
                alpha=0.50,
                extend="max",
            )
            ax.contour(
                result.grid.lon2d,
                result.grid.lat2d,
                concentration,
                levels=plume_boundaries[1:-1],
                colors="white",
                linewidths=0.55,
                alpha=0.30,
            )
        ax.contour(
            result.grid.lon2d,
            result.grid.lat2d,
            footprint.astype(float),
            levels=[0.5],
            colors="#ffd166",
            linewidths=1.4,
            alpha=0.85,
        )

        near_surface = np.ma.masked_less_equal(snapshot.concentration_bq_m3, 0.0)
        if near_surface.count() > 0:
            surface_peak = float(np.nanmax(snapshot.concentration_bq_m3))
            surface_levels = [
                surface_peak * fraction
                for fraction in (3e-3, 1e-2, 3e-2)
                if surface_peak * fraction > 0.0
            ]
            if len(surface_levels) >= 2:
                ax.contour(
                    result.grid.lon2d,
                    result.grid.lat2d,
                    near_surface,
                    levels=surface_levels,
                    colors=("#7f0000", "#5f0f40", "#370617"),
                    linewidths=(0.8, 1.0, 1.2),
                    alpha=0.60,
                )

        stride = visual.quiver_stride
        ax.quiver(
            lon2d[::stride, ::stride],
            lat2d[::stride, ::stride],
            u_frames[frame_index][::stride, ::stride],
            v_frames[frame_index][::stride, ::stride],
            color="#17324d",
            scale=240.0,
            width=0.0024,
            alpha=0.58,
        )

        if snapshot.puff_lon.size:
            ax.scatter(
                snapshot.puff_lon,
                snapshot.puff_lat,
                s=18,
                color="#9d0208",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.3,
                zorder=10,
            )

        ax.set_title(
            f"Lagrangian puff cloud | {np.datetime_as_string(times[frame_index], unit='m')} UTC | pale field = full cloud body, dark contours = denser near-surface core",
            color="#13212e",
            fontsize=13,
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / timeline.fps, blit=False)
    if visual.save_mp4:
        _save_animation_with_progress(animation, output_path, timeline.fps, visual.dpi, "Plume frames", len(times))
    else:
        plt.show()
    plt.close(fig)


def render_plume_animation_streaming(
    raw_meteo: PreparedMeteo,
    aggregate: DispersionAggregateResult,
    frame_meteo: PreparedMeteo,
    domain: DomainConfig,
    timeline: TimelineConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
    dispersion: DispersionConfig,
) -> None:
    times = frame_meteo.ds.time.values
    extent = aggregate.grid.extent
    lon2d, lat2d, u_frames, v_frames, _ = _prepare_plot_fields(frame_meteo, domain.wind_plot_resolution_deg)

    max_field = aggregate.max_cloud_column_bq_m2
    positive_values = max_field[max_field > 0.0]
    concentration_peak = float(np.nanmax(positive_values)) if positive_values.size else 1.0
    zone_levels = _zone_levels_from_peak(concentration_peak, PLUME_VISUAL_FRACTIONS)
    plume_boundaries = np.unique(np.concatenate(([max(zone_levels[0] * 0.5, 1e-12)], zone_levels)))
    plume_cmap = plt.get_cmap("YlOrRd", max(len(plume_boundaries) - 1, 2))
    plume_norm = _safe_boundary_norm(plume_boundaries, plume_cmap)
    footprint_threshold = plume_boundaries[0]
    if positive_values.size:
        footprint_threshold = max(float(np.nanmax(positive_values)) * 1e-4, float(np.nanpercentile(positive_values, 5)))
    footprint = max_field >= footprint_threshold

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    fig.text(0.5, 0.02, _footer_text(aggregate.source_summary), ha="center", va="bottom", fontsize=8, color="#203243")
    _add_colorbar(
        fig,
        ax,
        plume_cmap,
        plume_norm,
        "Cloud column activity, Bq/m^2",
        boundaries=plume_boundaries,
    )
    writer = _ffmpeg_writer(timeline.fps)
    progress = tqdm(total=len(times), desc="Plume frames", unit="frame")
    frame_index = 0

    def _render_snapshot(snapshot: DispersionSnapshot) -> None:
        nonlocal frame_index
        ax.clear()
        draw_geographic_context(ax, extent, domain)

        frame_positive = snapshot.cloud_column_bq_m2[snapshot.cloud_column_bq_m2 > 0.0]
        frame_threshold = plume_boundaries[0]
        if frame_positive.size:
            frame_threshold = max(float(np.nanmax(frame_positive)) * 1e-4, plume_boundaries[0])
        concentration = np.ma.masked_less_equal(snapshot.cloud_column_bq_m2, frame_threshold)
        if concentration.count() > 0:
            ax.contourf(
                aggregate.grid.lon2d,
                aggregate.grid.lat2d,
                concentration,
                levels=plume_boundaries,
                cmap=plume_cmap,
                norm=plume_norm,
                alpha=0.50,
                extend="max",
            )
            ax.contour(
                aggregate.grid.lon2d,
                aggregate.grid.lat2d,
                concentration,
                levels=plume_boundaries[1:-1],
                colors="white",
                linewidths=0.55,
                alpha=0.30,
            )
        ax.contour(
            aggregate.grid.lon2d,
            aggregate.grid.lat2d,
            footprint.astype(float),
            levels=[0.5],
            colors="#ffd166",
            linewidths=1.4,
            alpha=0.85,
        )

        near_surface = np.ma.masked_less_equal(snapshot.concentration_bq_m3, 0.0)
        if near_surface.count() > 0:
            surface_peak = float(np.nanmax(snapshot.concentration_bq_m3))
            surface_levels = [
                surface_peak * fraction
                for fraction in (3e-3, 1e-2, 3e-2)
                if surface_peak * fraction > 0.0
            ]
            if len(surface_levels) >= 2:
                ax.contour(
                    aggregate.grid.lon2d,
                    aggregate.grid.lat2d,
                    near_surface,
                    levels=surface_levels,
                    colors=("#7f0000", "#5f0f40", "#370617"),
                    linewidths=(0.8, 1.0, 1.2),
                    alpha=0.60,
                )

        stride = visual.quiver_stride
        ax.quiver(
            lon2d[::stride, ::stride],
            lat2d[::stride, ::stride],
            u_frames[frame_index][::stride, ::stride],
            v_frames[frame_index][::stride, ::stride],
            color="#17324d",
            scale=240.0,
            width=0.0024,
            alpha=0.58,
        )

        if snapshot.puff_lon.size:
            ax.scatter(
                snapshot.puff_lon,
                snapshot.puff_lat,
                s=18,
                color="#9d0208",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.3,
                zorder=10,
            )

        ax.set_title(
            f"Lagrangian puff cloud | {np.datetime_as_string(snapshot.time, unit='m')} UTC | pale field = full cloud body, dark contours = denser near-surface core",
            color="#13212e",
            fontsize=13,
        )
        writer.grab_frame()
        frame_index += 1
        if frame_index > progress.n:
            progress.update(frame_index - progress.n)

    try:
        with writer.saving(fig, output_path, visual.dpi):
            run_dispersion_aggregate(
                raw_meteo=raw_meteo,
                frame_meteo=frame_meteo,
                timeline=timeline,
                domain=domain,
                dispersion=dispersion,
                snapshot_sink=_render_snapshot,
            )
    finally:
        if progress.n < len(times):
            progress.update(len(times) - progress.n)
        progress.close()
    plt.close(fig)


def render_summary_map(
    result: DispersionResult | DispersionAggregateResult,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
) -> None:
    if isinstance(result, DispersionAggregateResult):
        max_field = result.max_cloud_column_bq_m2
    else:
        if not result.snapshots:
            raise ValueError("No snapshots available; summary map cannot be created.")

    # --- 1) Собираем max-over-time поле по телу облака ---
    max_field = np.zeros_like(result.snapshots[0].cloud_column_bq_m2, dtype=np.float32)
    for snapshot in result.snapshots:
        max_field = np.maximum(max_field, snapshot.cloud_column_bq_m2)

    positive = max_field[max_field > 0]
    if positive.size == 0:
        raise ValueError("Cloud field is empty; summary map cannot be created.")
    src_lon = result.grid.source_lon
    src_lat = result.grid.source_lat

    # --- 2) Порог footprint: где облако было хоть раз ---
    vmax = float(np.nanmax(positive))
    threshold = max(vmax * 1e-4, float(np.nanpercentile(positive, 5)))
    footprint = max_field >= threshold

    # --- 3) Рендер ---
    # Для heatmap берём лог-шкалу только для max_field
    vmin = threshold
    cmap = plt.get_cmap("inferno")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    draw_geographic_context(ax, result.grid.extent, domain, show_raster=False)

    # Heatmap max-over-time
    im = ax.pcolormesh(
        result.grid.lon2d,
        result.grid.lat2d,
        np.ma.masked_less(max_field, vmin),
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.62,
    )

    # Белые контуры max-over-time
    levels = np.geomspace(vmin, vmax, 6)
    ax.contour(
        result.grid.lon2d,
        result.grid.lat2d,
        max_field,
        levels=levels,
        colors="white",
        linewidths=0.8,
        alpha=0.55,
    )

    # Отдельный footprint-контур: где облако было
    ax.contour(
        result.grid.lon2d,
        result.grid.lat2d,
        footprint.astype(float),
        levels=[0.5],
        colors="#ffd166",
        linewidths=2.0,
        alpha=0.95,
    )

    # Источник
    ax.scatter(
        [src_lon],
        [src_lat],
        marker="*",
        s=380,
        c="#ffd34d",
        edgecolors="black",
        linewidths=1.2,
        zorder=20,
    )
    ax.text(
        src_lon + 0.08,
        src_lat + 0.03,
        "Source",
        fontsize=12,
        color="#13212e",
        weight="bold",
        zorder=21,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
    cbar.set_label("Max cloud column over time, Bq/m^2", color="#14202b")
    cbar.ax.yaxis.set_tick_params(color="#14202b")
    plt.setp(cbar.ax.get_yticklabels(), color="#14202b")

    start = result.window.incident_start.isoformat()
    end = result.window.simulation_end.isoformat()
    ax.set_title(
        "Cloud footprint summary (max-over-time)\n"
        f"{start} UTC -> {end} UTC | yellow contour = where cloud was detected at least once",
        color="#13212e",
        fontsize=13,
    )

    fig.text(
        0.5,
        0.02,
        _footer_text(result.source_summary),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#203243",
    )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    plt.savefig(output_path, dpi=visual.dpi)
    plt.close(fig)


def _ground_dose_from_deposition(deposition_bq_m2: np.ndarray, hazard: HazardConfig) -> np.ndarray:
    exposure_seconds = float(hazard.dose_integration_hours * 3600.0)
    dose_msv = np.zeros_like(deposition_bq_m2, dtype=np.float64)
    for isotope in hazard.isotope_mix:
        decay_constant = np.log(2.0) / (isotope.half_life_hours * 3600.0)
        integrated_seconds = (1.0 - np.exp(-decay_constant * exposure_seconds)) / decay_constant
        dose_msv += (
            deposition_bq_m2
            * isotope.deposition_fraction
            * isotope.ground_surface_dose_coeff_sv_per_bq_s_m2
            * integrated_seconds
            * 1000.0
        )
    return (dose_msv * float(hazard.ground_dose_multiplier)).astype(np.float32)


def render_summary_ground_dose_map(
    result: DispersionResult | DispersionAggregateResult,
    domain: DomainConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
) -> None:
    final_deposition = result.final_deposition_bq_m2
    ground_dose_msv = _ground_dose_from_deposition(final_deposition, hazard)
    positive = ground_dose_msv[ground_dose_msv > 0.0]
    if positive.size == 0:
        raise ValueError("Ground dose field is empty; summary ground dose map cannot be created.")

    vmin = max(float(np.nanpercentile(positive, 5)), 1e-6)
    vmax = float(np.nanmax(positive))
    levels = _dose_display_levels(ground_dose_msv, hazard.dose_zone_levels_msv)
    levels = levels[levels > 0.0]
    if levels.size < 2:
        levels = np.geomspace(vmin, vmax, num=6)
    cmap = plt.get_cmap("magma", max(len(levels) - 1, 2))
    norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    draw_geographic_context(ax, result.grid.extent, domain, show_raster=False)
    ax.contourf(
        result.grid.lon2d,
        result.grid.lat2d,
        np.ma.masked_less_equal(ground_dose_msv, levels[0]),
        levels=levels,
        cmap=cmap,
        norm=norm,
        alpha=0.68,
        extend="max",
    )
    _add_colorbar(fig, ax, cmap, norm, "Ground dose over exposure window, mSv", boundaries=levels)
    ax.set_title(
        "Ground dose summary (final deposition-based)\n"
        f"{result.window.incident_start.isoformat()} -> {result.window.simulation_end.isoformat()}",
        color="#13212e",
        fontsize=13,
    )
    fig.text(0.5, 0.02, _footer_text(result.source_summary), ha="center", va="bottom", fontsize=9, color="#203243")
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    plt.savefig(output_path, dpi=visual.dpi)
    plt.close(fig)


def render_hazard_ground_dose_map(
    hazard_result: ScenarioHazardResult,
    domain: DomainConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
) -> None:
    fine_lon2d, fine_lat2d, fine_probability = _upsample_scalar_field(
        hazard_result.grid.lon2d,
        hazard_result.grid.lat2d,
        hazard_result.dangerous_ground_dose_probability,
        scale_factor=5,
    )
    boundaries = np.array(hazard.scenario_probability_levels, dtype=float)
    cmap = mcolors.LinearSegmentedColormap.from_list("hazard_ground_dose_probability", ["#edf8fb", "#2c7fb8", "#08306b"], N=max(len(boundaries) + 1, 2))
    norm = _safe_boundary_norm(boundaries, cmap)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    draw_geographic_context(ax, hazard_result.grid.extent, domain, show_raster=False)
    ax.contourf(
        fine_lon2d,
        fine_lat2d,
        np.ma.masked_less_equal(fine_probability, boundaries[1]),
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        alpha=0.65,
        extend="max",
    )
    _add_colorbar(
        fig,
        ax,
        cmap,
        norm,
        f"Probability(ground dose >= {hazard.ground_dose_danger_threshold_msv:.2f} mSv)",
        boundaries=boundaries,
    )
    ax.set_title(
        "Hazard map: dangerous ground dose probability",
        color="#13212e",
        fontsize=13,
    )
    fig.text(0.5, 0.02, _footer_text(f"{hazard_result.scenario_count} scenarios; threshold {hazard.ground_dose_danger_threshold_msv:.2f} mSv"), ha="center", va="bottom", fontsize=9, color="#203243")
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    plt.savefig(output_path, dpi=visual.dpi)
    plt.close(fig)



def render_hazard_probability_map(
    hazard_result: ScenarioHazardResult,
    domain: DomainConfig,
    hazard: HazardConfig,
    visual: VisualConfig,
    output_path: Path,
) -> None:
    impact_levels_colors = ["#fff3b0", "#ffd166", "#fca311", "#f77f00", "#d62828"]
    fine_lon2d, fine_lat2d, fine_probability = _upsample_scalar_field(
        hazard_result.grid.lon2d,
        hazard_result.grid.lat2d,
        hazard_result.hazard_probability,
        scale_factor=5,
    )
    fine_env_lon2d, fine_env_lat2d, fine_max_cloud = _upsample_scalar_field(
        hazard_result.grid.lon2d,
        hazard_result.grid.lat2d,
        hazard_result.max_deposition_bq_m2,
        scale_factor=5,
    )
    impact_boundaries = np.array(hazard.scenario_probability_levels, dtype=float)
    n_required_colors = max(len(impact_boundaries) + 1, 2)
    impact_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hazard_probability",
        impact_levels_colors,
        N=n_required_colors,
    )
    impact_norm = _safe_boundary_norm(impact_boundaries, impact_cmap)
    max_positive = fine_max_cloud[fine_max_cloud > 0.0]
    envelope_threshold = None
    if max_positive.size:
        envelope_threshold = max(
            float(np.nanmax(max_positive)) * max(hazard.trace_floor_peak_fraction, 1e-6),
            1e-12,
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8fbff")
    draw_geographic_context(ax, hazard_result.grid.extent, domain, show_raster=False)

    if envelope_threshold is not None:
        envelope_levels = np.array(
            [
                envelope_threshold,
                envelope_threshold * 3.0,
                envelope_threshold * 10.0,
                max(float(np.nanmax(max_positive)), envelope_threshold * 10.0),
            ],
            dtype=float,
        )
        envelope_levels = np.unique(envelope_levels)
        ax.contourf(
            fine_env_lon2d,
            fine_env_lat2d,
            np.ma.masked_less(fine_max_cloud, envelope_threshold),
            levels=envelope_levels,
            colors=["#fff7d6", "#ffe9a8", "#ffd166"],
            alpha=0.22,
            extend="max",
        )
        ax.contour(
            fine_env_lon2d,
            fine_env_lat2d,
            fine_max_cloud,
            levels=[envelope_threshold],
            colors="#b7791f",
            linewidths=1.0,
            alpha=0.65,
        )

    ax.contourf(
        fine_lon2d,
        fine_lat2d,
        np.ma.masked_less_equal(fine_probability, impact_boundaries[1]),
        levels=impact_boundaries,
        cmap=impact_cmap,
        norm=impact_norm,
        alpha=0.62,
        extend="max",
    )
    ax.contour(
        fine_lon2d,
        fine_lat2d,
        fine_probability,
        levels=impact_boundaries[1:-1],
        colors="#18324a",
        linewidths=0.7,
        alpha=0.45,
    )
    _add_colorbar(fig, ax, impact_cmap, impact_norm, "Fraction of scenarios with cloud passage", boundaries=impact_boundaries)

    scenario_start = hazard_result.scenario_start_times[0]
    scenario_end = hazard_result.scenario_start_times[-1]
    ax.set_title(
        "Accumulated hazard footprint over many incident dates\n"
        f"{scenario_start} -> {scenario_end} | pale envelope = any cloud reach, darker zones = more frequent cloud passage",
        color="#13212e",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.02,
        _footer_text(f"{hazard_result.scenario_count} scenarios; {hazard_result.threshold_note}"),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#203243",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    plt.savefig(output_path, dpi=visual.dpi)
    plt.close(fig)

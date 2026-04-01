from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from .config import DispersionConfig, DomainConfig, TimelineConfig
from .meteo import PreparedMeteo


METERS_PER_DEG_LAT = 111_320.0

STABILITY_WIND_PROFILE_EXPONENT = {
    "A": 0.07,
    "B": 0.07,
    "C": 0.10,
    "D": 0.15,
    "E": 0.35,
    "F": 0.55,
}

STABILITY_MIXING_HEIGHT_M = {
    "A": 1600.0,
    "B": 1300.0,
    "C": 950.0,
    "D": 700.0,
    "E": 350.0,
    "F": 220.0,
}

STABILITY_ALONG_DIFFUSIVITY_SCALE = {
    "A": 2.0,
    "B": 1.8,
    "C": 1.5,
    "D": 1.2,
    "E": 1.0,
    "F": 0.8,
}

STABILITY_CROSS_DIFFUSIVITY_SCALE = {
    "A": 6.0,
    "B": 5.0,
    "C": 4.0,
    "D": 2.5,
    "E": 1.3,
    "F": 0.8,
}

STABILITY_ALONG_TURBULENCE_SCALE = {
    "A": 0.55,
    "B": 0.45,
    "C": 0.35,
    "D": 0.28,
    "E": 0.18,
    "F": 0.12,
}

STABILITY_CROSS_TURBULENCE_SCALE = {
    "A": 1.15,
    "B": 0.95,
    "C": 0.75,
    "D": 0.55,
    "E": 0.30,
    "F": 0.18,
}

PRESSURE_SCALE_HEIGHT_M = 8_000.0


@dataclass(frozen=True)
class SimulationWindow:
    incident_start: pd.Timestamp
    release_end: pd.Timestamp
    simulation_end: pd.Timestamp


@dataclass(frozen=True)
class SimulationGrid:
    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    x2d_m: np.ndarray
    y2d_m: np.ndarray
    lon2d: np.ndarray
    lat2d: np.ndarray
    extent: tuple[float, float, float, float]
    cell_area_m2: float
    source_lon: float
    source_lat: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.x2d_m.shape


@dataclass
class GaussianPuff:
    lon: float
    lat: float
    mass_bq: float
    sigma_cross_m: float
    sigma_along_m: float
    mixing_height_m: float
    transport_height_m: float
    layer_fraction: float
    bearing_rad: float = 0.0
    age_s: float = 0.0


@dataclass(frozen=True)
class StabilityState:
    stability_class: str
    mixing_height_m: float
    wind_profile_exponent: float
    along_diffusivity_scale: float
    cross_diffusivity_scale: float
    along_turbulence_scale: float
    cross_turbulence_scale: float


@dataclass(frozen=True)
class DispersionSnapshot:
    time: np.datetime64
    concentration_bq_m3: np.ndarray
    cloud_column_bq_m2: np.ndarray
    deposition_bq_m2: np.ndarray
    puff_lon: np.ndarray
    puff_lat: np.ndarray


@dataclass(frozen=True)
class DispersionResult:
    grid: SimulationGrid
    snapshots: list[DispersionSnapshot]
    integrated_air_concentration_bq_s_m3: np.ndarray
    source_summary: str
    window: SimulationWindow

    @property
    def final_deposition_bq_m2(self) -> np.ndarray:
        return self.snapshots[-1].deposition_bq_m2


@dataclass(frozen=True)
class FinalDepositionResult:
    grid: SimulationGrid
    final_deposition_bq_m2: np.ndarray
    integrated_air_concentration_bq_s_m3: np.ndarray
    source_summary: str
    window: SimulationWindow


@dataclass(frozen=True)
class DispersionAggregateResult:
    grid: SimulationGrid
    max_cloud_column_bq_m2: np.ndarray
    final_deposition_bq_m2: np.ndarray
    integrated_air_concentration_bq_s_m3: np.ndarray
    source_summary: str
    window: SimulationWindow


@dataclass
class DispersionState:
    puffs: list[GaussianPuff]
    deposition_bq_m2: np.ndarray
    integrated_air_concentration_bq_s_m3: np.ndarray
    snapshots: list[DispersionSnapshot]
    rng: np.random.Generator
    max_cloud_column_bq_m2: np.ndarray | None = None
    last_snapshot_time: np.datetime64 | None = None


def build_simulation_window(timeline: TimelineConfig) -> SimulationWindow:
    incident_start = pd.Timestamp(timeline.incident_start_utc)
    release_end = incident_start + pd.Timedelta(hours=timeline.release_duration_hours)
    simulation_hours = timeline.simulation_duration_hours
    if timeline.demo_mode:
        simulation_hours = min(simulation_hours, timeline.demo_hours)
    simulation_end = incident_start + pd.Timedelta(hours=simulation_hours)
    if simulation_end <= incident_start:
        raise ValueError("simulation_end должен быть позже incident_start")
    return SimulationWindow(
        incident_start=incident_start,
        release_end=release_end,
        simulation_end=simulation_end,
    )


def _meters_per_deg_lon(lat_deg: float) -> float:
    return METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat_deg))


def _lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, source_lon: float, source_lat: float) -> tuple[np.ndarray, np.ndarray]:
    x = (lon - source_lon) * _meters_per_deg_lon(source_lat)
    y = (lat - source_lat) * METERS_PER_DEG_LAT
    return x, y


def _xy_to_lonlat(x_m: np.ndarray, y_m: np.ndarray, source_lon: float, source_lat: float) -> tuple[np.ndarray, np.ndarray]:
    lon = source_lon + x_m / _meters_per_deg_lon(source_lat)
    lat = source_lat + y_m / METERS_PER_DEG_LAT
    return lon, lat


def build_simulation_grid(prepared: PreparedMeteo, domain: DomainConfig) -> SimulationGrid:
    west, east, south, north = prepared.extent
    step_m = domain.grid_resolution_km * 1000.0
    x_min, y_min = _lonlat_to_xy(np.array([west]), np.array([south]), domain.source_lon, domain.source_lat)
    x_max, y_max = _lonlat_to_xy(np.array([east]), np.array([north]), domain.source_lon, domain.source_lat)

    x_coords = np.arange(float(x_min[0]), float(x_max[0]) + step_m, step_m)
    y_coords = np.arange(float(y_min[0]), float(y_max[0]) + step_m, step_m)
    x2d_m, y2d_m = np.meshgrid(x_coords, y_coords)
    lon2d, lat2d = _xy_to_lonlat(x2d_m, y2d_m, domain.source_lon, domain.source_lat)
    extent = (float(lon2d.min()), float(lon2d.max()), float(lat2d.min()), float(lat2d.max()))

    return SimulationGrid(
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        x2d_m=x2d_m,
        y2d_m=y2d_m,
        lon2d=lon2d,
        lat2d=lat2d,
        extent=extent,
        cell_area_m2=step_m ** 2,
        source_lon=domain.source_lon,
        source_lat=domain.source_lat,
    )


def _compute_diffusivity(speed_ms: float, dispersion: DispersionConfig) -> float:
    return dispersion.base_horizontal_diffusivity_m2_s + dispersion.wind_diffusivity_factor_m2_s_per_ms * max(speed_ms, 0.0)


def _grid_spread_floor(prepared: PreparedMeteo, dispersion: DispersionConfig) -> tuple[float, float, float, float]:
    native_step_s = max(prepared.native_step_minutes * 60.0, 3600.0)
    min_cross_sigma_m = max(dispersion.initial_sigma_m, dispersion.coarse_grid_crosswind_fraction * prepared.grid_spacing_m)
    min_along_sigma_m = max(dispersion.initial_sigma_m * 1.25, dispersion.coarse_grid_alongwind_fraction * prepared.grid_spacing_m)
    grid_cross_diffusivity = (min_cross_sigma_m ** 2) / (2.0 * native_step_s)
    grid_along_diffusivity = (min_along_sigma_m ** 2) / (2.0 * native_step_s)
    return min_cross_sigma_m, min_along_sigma_m, grid_cross_diffusivity, grid_along_diffusivity


def _solar_elevation_deg(timestamp: pd.Timestamp, lat_deg: float, lon_deg: float) -> float:
    day_of_year = timestamp.day_of_year
    utc_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (utc_hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    true_solar_time_min = (utc_hour * 60.0 + equation_of_time + 4.0 * lon_deg) % 1440.0
    hour_angle_deg = true_solar_time_min / 4.0 - 180.0
    lat_rad = np.deg2rad(lat_deg)
    hour_angle_rad = np.deg2rad(hour_angle_deg)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle_rad)
    cos_zenith = float(np.clip(cos_zenith, -1.0, 1.0))
    return 90.0 - float(np.rad2deg(np.arccos(cos_zenith)))


def _classify_stability(wind_speed_ms: float, solar_elevation_deg: float) -> str:
    if solar_elevation_deg > 50.0:
        if wind_speed_ms < 2.0:
            return "A"
        if wind_speed_ms < 3.0:
            return "B"
        if wind_speed_ms < 5.0:
            return "B"
        if wind_speed_ms < 6.5:
            return "C"
        return "D"
    if solar_elevation_deg > 20.0:
        if wind_speed_ms < 2.0:
            return "B"
        if wind_speed_ms < 3.0:
            return "B"
        if wind_speed_ms < 5.0:
            return "C"
        if wind_speed_ms < 6.5:
            return "C"
        return "D"
    if solar_elevation_deg > 0.0:
        if wind_speed_ms < 2.0:
            return "C"
        if wind_speed_ms < 5.0:
            return "C"
        return "D"
    if wind_speed_ms < 2.0:
        return "F"
    if wind_speed_ms < 3.0:
        return "E"
    if wind_speed_ms < 5.0:
        return "D"
    return "D"


def _clip_mixing_height(mixing_height_m: float, dispersion: DispersionConfig) -> float:
    return float(np.clip(mixing_height_m, dispersion.min_mixing_height_m, dispersion.max_mixing_height_m))


def _stability_state(
    timestamp: pd.Timestamp,
    lat_deg: float,
    lon_deg: float,
    wind_speed_ms: float,
    dispersion: DispersionConfig,
    mixing_height_override_m: float | None = None,
) -> StabilityState:
    solar_elevation = _solar_elevation_deg(timestamp, lat_deg, lon_deg)
    stability_class = _classify_stability(wind_speed_ms, solar_elevation)
    fallback_mixing_height = max(dispersion.mixing_height_m, STABILITY_MIXING_HEIGHT_M[stability_class])
    mixing_height_m = mixing_height_override_m if mixing_height_override_m is not None else fallback_mixing_height
    mixing_height_m = _clip_mixing_height(mixing_height_m, dispersion)
    return StabilityState(
        stability_class=stability_class,
        mixing_height_m=mixing_height_m,
        wind_profile_exponent=STABILITY_WIND_PROFILE_EXPONENT[stability_class],
        along_diffusivity_scale=STABILITY_ALONG_DIFFUSIVITY_SCALE[stability_class],
        cross_diffusivity_scale=STABILITY_CROSS_DIFFUSIVITY_SCALE[stability_class],
        along_turbulence_scale=STABILITY_ALONG_TURBULENCE_SCALE[stability_class],
        cross_turbulence_scale=STABILITY_CROSS_TURBULENCE_SCALE[stability_class],
    )


def _interp_optional(
    ds: xr.Dataset,
    var_name: str,
    current_time64: np.datetime64,
    lon_points: xr.DataArray,
    lat_points: xr.DataArray,
) -> np.ndarray | None:
    if var_name not in ds.data_vars:
        return None
    return ds[var_name].sel(time=current_time64).interp(longitude=lon_points, latitude=lat_points, method="linear").values


def _source_optional_scalar(
    ds: xr.Dataset,
    var_name: str,
    current_time64: np.datetime64,
    lon: float,
    lat: float,
) -> float | None:
    if var_name not in ds.data_vars:
        return None
    value = ds[var_name].sel(time=current_time64).interp(
        longitude=xr.DataArray([lon], dims="points"),
        latitude=xr.DataArray([lat], dims="points"),
        method="linear",
    ).values[0]
    return None if not np.isfinite(value) else float(value)


def _wind_profile_factor(transport_height_m: float, state: StabilityState, dispersion: DispersionConfig) -> float:
    factor = max(transport_height_m, 10.0) / 10.0
    return float(min(factor ** state.wind_profile_exponent, dispersion.max_wind_profile_factor))


def _height_to_pressure_hpa(height_m: float) -> float:
    capped_height_m = float(np.clip(height_m, 0.0, 11_000.0))
    return float(1013.25 * np.power(max(1.0 - capped_height_m / 44_330.0, 1e-4), 5.255))


def _interp_pressure_level_field(
    ds: xr.Dataset,
    var_name: str,
    current_time64: np.datetime64,
    lon_points: xr.DataArray,
    lat_points: xr.DataArray,
    pressure_points_hpa: xr.DataArray,
) -> np.ndarray | None:
    if var_name not in ds.data_vars or "pressure_level" not in ds.coords:
        return None
    field = ds[var_name].sel(time=current_time64)
    return field.interp(
        longitude=lon_points,
        latitude=lat_points,
        pressure_level=pressure_points_hpa,
        method="linear",
    ).values


def _sample_wind_components(
    ds: xr.Dataset,
    current_time64: np.datetime64,
    lon_points: xr.DataArray,
    lat_points: xr.DataArray,
    transport_heights_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface_u = ds["u10"].sel(time=current_time64).interp(longitude=lon_points, latitude=lat_points, method="linear").values
    surface_v = ds["v10"].sel(time=current_time64).interp(longitude=lon_points, latitude=lat_points, method="linear").values
    pressure_points = xr.DataArray(
        np.array([_height_to_pressure_hpa(height) for height in transport_heights_m], dtype=float),
        dims="points",
    )
    level_u = _interp_pressure_level_field(ds, "u_pl", current_time64, lon_points, lat_points, pressure_points)
    level_v = _interp_pressure_level_field(ds, "v_pl", current_time64, lon_points, lat_points, pressure_points)

    u_values = surface_u.copy()
    v_values = surface_v.copy()
    if level_u is not None and level_v is not None:
        level_mask = np.isfinite(level_u) & np.isfinite(level_v)
        u_values[level_mask] = level_u[level_mask]
        v_values[level_mask] = level_v[level_mask]

    return u_values, v_values, surface_u, surface_v


def _sanitize_wind_component(primary: float, surface: float, max_abs_ms: float) -> float:
    value = primary if np.isfinite(primary) else surface
    if not np.isfinite(value):
        value = 0.0
    return float(np.clip(value, -max_abs_ms, max_abs_ms))


def _field_from_puffs(puffs: list[GaussianPuff], grid: SimulationGrid) -> np.ndarray:
    concentration = np.zeros(grid.shape, dtype=np.float32)
    if not puffs:
        return concentration

    for puff in puffs:
        sigma_cross2 = max(puff.sigma_cross_m ** 2, 1.0)
        sigma_along2 = max(puff.sigma_along_m ** 2, 1.0)
        x0, y0 = _lonlat_to_xy(
            np.array([puff.lon]),
            np.array([puff.lat]),
            grid.source_lon,
            grid.source_lat,
        )
        dx = grid.x2d_m - x0[0]
        dy = grid.y2d_m - y0[0]
        cos_theta = np.cos(puff.bearing_rad)
        sin_theta = np.sin(puff.bearing_rad)
        along = dx * cos_theta + dy * sin_theta
        cross = -dx * sin_theta + dy * cos_theta
        exponent = np.exp(-(along * along) / (2.0 * sigma_along2) - (cross * cross) / (2.0 * sigma_cross2))
        coefficient = puff.mass_bq / (2.0 * np.pi * puff.sigma_along_m * puff.sigma_cross_m * max(puff.mixing_height_m, 1.0))
        concentration += (coefficient * exponent).astype(np.float32)

    return concentration


def _column_from_puffs(puffs: list[GaussianPuff], grid: SimulationGrid) -> np.ndarray:
    column_activity = np.zeros(grid.shape, dtype=np.float32)
    if not puffs:
        return column_activity

    for puff in puffs:
        sigma_cross2 = max(puff.sigma_cross_m ** 2, 1.0)
        sigma_along2 = max(puff.sigma_along_m ** 2, 1.0)
        x0, y0 = _lonlat_to_xy(
            np.array([puff.lon]),
            np.array([puff.lat]),
            grid.source_lon,
            grid.source_lat,
        )
        dx = grid.x2d_m - x0[0]
        dy = grid.y2d_m - y0[0]
        cos_theta = np.cos(puff.bearing_rad)
        sin_theta = np.sin(puff.bearing_rad)
        along = dx * cos_theta + dy * sin_theta
        cross = -dx * sin_theta + dy * cos_theta
        exponent = np.exp(-(along * along) / (2.0 * sigma_along2) - (cross * cross) / (2.0 * sigma_cross2))
        coefficient = puff.mass_bq / (2.0 * np.pi * puff.sigma_along_m * puff.sigma_cross_m)
        column_activity += (coefficient * exponent).astype(np.float32)

    return column_activity


def _apply_mass_losses(puffs: list[GaussianPuff], dispersion: DispersionConfig, dt_s: float) -> None:
    if dispersion.radioactive_half_life_hours:
        decay_factor = np.exp(-np.log(2.0) * dt_s / (dispersion.radioactive_half_life_hours * 3600.0))
    else:
        decay_factor = 1.0

    for puff in puffs:
        deposition_factor = np.exp(-dispersion.dry_deposition_velocity_ms * dt_s / max(puff.mixing_height_m, 1.0))
        puff.mass_bq *= deposition_factor * decay_factor


def _emit_layered_puffs(
    puffs: list[GaussianPuff],
    model_meteo: PreparedMeteo,
    current_time64: np.datetime64,
    dt_s: float,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    min_cross_sigma_m: float,
    min_along_sigma_m: float,
) -> None:
    emitted_mass = dispersion.emission_rate_bq_s * dt_s
    source_lon_points = xr.DataArray([domain.source_lon], dims="points")
    source_lat_points = xr.DataArray([domain.source_lat], dims="points")
    u_source = float(
        model_meteo.ds["u10"].sel(time=current_time64).interp(
            longitude=source_lon_points,
            latitude=source_lat_points,
            method="linear",
        ).values[0]
    )
    v_source = float(
        model_meteo.ds["v10"].sel(time=current_time64).interp(
            longitude=source_lon_points,
            latitude=source_lat_points,
            method="linear",
        ).values[0]
    )
    base_speed = float(np.hypot(u_source, v_source))
    blh_source = _source_optional_scalar(model_meteo.ds, "blh", current_time64, domain.source_lon, domain.source_lat)
    state = _stability_state(
        pd.Timestamp(current_time64),
        domain.source_lat,
        domain.source_lon,
        base_speed,
        dispersion,
        mixing_height_override_m=blh_source,
    )
    bearing_rad = float(np.arctan2(v_source, u_source)) if base_speed > 1e-6 else 0.0

    total_mass_fraction = sum(dispersion.release_layer_mass_fractions)
    for height_fraction, mass_fraction in zip(
        dispersion.release_layer_height_fractions,
        dispersion.release_layer_mass_fractions,
        strict=True,
    ):
        transport_height_m = _clip_mixing_height(state.mixing_height_m * height_fraction, dispersion)
        layer_u, layer_v, _, _ = _sample_wind_components(
            model_meteo.ds,
            current_time64,
            source_lon_points,
            source_lat_points,
            np.array([transport_height_m], dtype=float),
        )
        layer_bearing_rad = bearing_rad
        if np.isfinite(layer_u[0]) and np.isfinite(layer_v[0]) and np.hypot(layer_u[0], layer_v[0]) > 1e-6:
            layer_bearing_rad = float(np.arctan2(layer_v[0], layer_u[0]))
        puffs.append(
            GaussianPuff(
                lon=domain.source_lon,
                lat=domain.source_lat,
                mass_bq=emitted_mass * mass_fraction / total_mass_fraction,
                sigma_cross_m=min_cross_sigma_m,
                sigma_along_m=min_along_sigma_m,
                mixing_height_m=state.mixing_height_m,
                transport_height_m=transport_height_m,
                layer_fraction=height_fraction,
                bearing_rad=layer_bearing_rad,
            )
        )


def _advect_puffs(
    puffs: list[GaussianPuff],
    model_meteo: PreparedMeteo,
    current_time64: np.datetime64,
    grid: SimulationGrid,
    dispersion: DispersionConfig,
    rng: np.random.Generator,
    dt_s: float,
    min_cross_sigma_m: float,
    min_along_sigma_m: float,
    grid_cross_diffusivity: float,
    grid_along_diffusivity: float,
) -> list[GaussianPuff]:
    if not puffs:
        return []

    lon_points = xr.DataArray([puff.lon for puff in puffs], dims="points")
    lat_points = xr.DataArray([puff.lat for puff in puffs], dims="points")
    transport_heights_m = np.array([puff.transport_height_m for puff in puffs], dtype=float)
    u_values, v_values, surface_u_values, surface_v_values = _sample_wind_components(
        model_meteo.ds,
        current_time64,
        lon_points,
        lat_points,
        transport_heights_m,
    )
    u_spread_values = _interp_optional(model_meteo.ds, "u_spread", current_time64, lon_points, lat_points)
    v_spread_values = _interp_optional(model_meteo.ds, "v_spread", current_time64, lon_points, lat_points)
    blh_values = _interp_optional(model_meteo.ds, "blh", current_time64, lon_points, lat_points)
    pressure_points = xr.DataArray(
        np.array([_height_to_pressure_hpa(height) for height in transport_heights_m], dtype=float),
        dims="points",
    )
    u_pl_spread_values = _interp_pressure_level_field(
        model_meteo.ds,
        "u_pl_spread",
        current_time64,
        lon_points,
        lat_points,
        pressure_points,
    )
    v_pl_spread_values = _interp_pressure_level_field(
        model_meteo.ds,
        "v_pl_spread",
        current_time64,
        lon_points,
        lat_points,
        pressure_points,
    )
    omega_values = _interp_pressure_level_field(
        model_meteo.ds,
        "omega_pl",
        current_time64,
        lon_points,
        lat_points,
        pressure_points,
    )

    if u_spread_values is None:
        u_spread_values = np.zeros(len(puffs), dtype=float)
    if v_spread_values is None:
        v_spread_values = np.zeros(len(puffs), dtype=float)
    if u_pl_spread_values is not None and v_pl_spread_values is not None:
        level_mask = np.isfinite(u_pl_spread_values) & np.isfinite(v_pl_spread_values)
        u_spread_values = u_spread_values.copy()
        v_spread_values = v_spread_values.copy()
        u_spread_values[level_mask] = u_pl_spread_values[level_mask]
        v_spread_values[level_mask] = v_pl_spread_values[level_mask]

    alive: list[GaussianPuff] = []
    west, east, south, north = grid.extent
    meters_per_deg_lon = _meters_per_deg_lon(grid.source_lat)
    timestamp = pd.Timestamp(current_time64)

    for index, puff in enumerate(puffs):
        max_advect_wind = max(float(dispersion.max_advect_wind_speed_ms), 1.0)
        u_val = _sanitize_wind_component(u_values[index], surface_u_values[index], max_advect_wind)
        v_val = _sanitize_wind_component(v_values[index], surface_v_values[index], max_advect_wind)

        mixing_height_override = None
        if blh_values is not None and np.isfinite(blh_values[index]):
            mixing_height_override = float(blh_values[index])

        base_speed = float(np.hypot(u_val, v_val))
        state = _stability_state(
            timestamp,
            puff.lat,
            puff.lon,
            base_speed,
            dispersion,
            mixing_height_override_m=mixing_height_override,
        )
        puff.mixing_height_m = state.mixing_height_m
        puff.transport_height_m = _clip_mixing_height(state.mixing_height_m * puff.layer_fraction, dispersion)

        has_multilevel_wind = (
            "u_pl" in model_meteo.ds.data_vars
            and "v_pl" in model_meteo.ds.data_vars
            and np.isfinite(u_val)
            and np.isfinite(v_val)
            and (not np.isclose(u_val, surface_u_values[index]) or not np.isclose(v_val, surface_v_values[index]))
        )
        if has_multilevel_wind:
            u_eff = float(u_val)
            v_eff = float(v_val)
        else:
            wind_factor = _wind_profile_factor(puff.transport_height_m, state, dispersion)
            u_eff = float(u_val) * wind_factor
            v_eff = float(v_val) * wind_factor
        speed_eff = float(np.hypot(u_eff, v_eff))
        if speed_eff > 1e-6:
            puff.bearing_rad = float(np.arctan2(v_eff, u_eff))

        if omega_values is not None and np.isfinite(omega_values[index]):
            pressure_pa = max(_height_to_pressure_hpa(puff.transport_height_m) * 100.0, 1.0)
            dz_dt = -PRESSURE_SCALE_HEIGHT_M * float(omega_values[index]) / pressure_pa
            puff.transport_height_m = _clip_mixing_height(puff.transport_height_m + dz_dt * dt_s, dispersion)

        spread_u = float(u_spread_values[index]) if np.isfinite(u_spread_values[index]) else 0.0
        spread_v = float(v_spread_values[index]) if np.isfinite(v_spread_values[index]) else 0.0
        spread_u = float(np.clip(spread_u, -max_advect_wind, max_advect_wind))
        spread_v = float(np.clip(spread_v, -max_advect_wind, max_advect_wind))
        ensemble_spread_ms = float(np.hypot(spread_u, spread_v))
        base_diffusivity = _compute_diffusivity(speed_eff + 0.6 * ensemble_spread_ms, dispersion)
        along_diffusivity = max(base_diffusivity * state.along_diffusivity_scale, grid_along_diffusivity)
        cross_diffusivity = max(base_diffusivity * state.cross_diffusivity_scale, grid_cross_diffusivity)

        puff.sigma_along_m = max(np.sqrt(puff.sigma_along_m ** 2 + 2.0 * along_diffusivity * dt_s), min_along_sigma_m)
        puff.sigma_cross_m = max(np.sqrt(puff.sigma_cross_m ** 2 + 2.0 * cross_diffusivity * dt_s), min_cross_sigma_m)

        along_std = np.sqrt(2.0 * along_diffusivity * dt_s) * state.along_turbulence_scale * dispersion.alongwind_turbulence_scale
        cross_std = np.sqrt(2.0 * cross_diffusivity * dt_s) * state.cross_turbulence_scale * dispersion.crosswind_turbulence_scale
        along_perturbation = rng.normal(0.0, along_std)
        cross_perturbation = rng.normal(0.0, cross_std)
        cos_theta = np.cos(puff.bearing_rad)
        sin_theta = np.sin(puff.bearing_rad)
        dx_m = u_eff * dt_s + along_perturbation * cos_theta - cross_perturbation * sin_theta
        dy_m = v_eff * dt_s + along_perturbation * sin_theta + cross_perturbation * cos_theta

        puff.lon += dx_m / meters_per_deg_lon
        puff.lat += dy_m / METERS_PER_DEG_LAT
        puff.age_s += dt_s

        if west <= puff.lon <= east and south <= puff.lat <= north and puff.mass_bq > 1.0:
            alive.append(puff)

    return alive


def _simulate_dispersion_core(
    model_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    frame_times: set[np.datetime64] | None,
    grid_prepared_from: PreparedMeteo,
) -> tuple[SimulationGrid, SimulationWindow, np.ndarray, list[DispersionSnapshot], list[GaussianPuff], np.ndarray]:
    window = build_simulation_window(timeline)
    grid = build_simulation_grid(grid_prepared_from, domain)
    min_cross_sigma_m, min_along_sigma_m, grid_cross_diffusivity, grid_along_diffusivity = _grid_spread_floor(model_meteo, dispersion)

    model_times = pd.DatetimeIndex(model_meteo.ds.time.values)
    puffs: list[GaussianPuff] = []
    deposition_bq_m2 = np.zeros(grid.shape, dtype=np.float32)
    integrated_air_concentration_bq_s_m3 = np.zeros(grid.shape, dtype=np.float32)
    snapshots: list[DispersionSnapshot] = []
    rng = np.random.default_rng(dispersion.random_seed)

    for index, current_time in enumerate(model_times):
        current_time64 = current_time.to_datetime64()
        if index < len(model_times) - 1:
            next_time = model_times[index + 1]
            dt_s = float((next_time - current_time).total_seconds())
        else:
            dt_s = 0.0

        if current_time < window.release_end and dt_s > 0:
            _emit_layered_puffs(
                puffs,
                model_meteo,
                current_time64,
                dt_s,
                domain,
                dispersion,
                min_cross_sigma_m,
                min_along_sigma_m,
            )

        concentration_bq_m3 = _field_from_puffs(puffs, grid)
        cloud_column_bq_m2 = _column_from_puffs(puffs, grid)

        if frame_times is not None and current_time64 in frame_times:
            snapshots.append(
                DispersionSnapshot(
                    time=current_time64,
                    concentration_bq_m3=concentration_bq_m3.copy(),
                    cloud_column_bq_m2=cloud_column_bq_m2.copy(),
                    deposition_bq_m2=deposition_bq_m2.copy(),
                    puff_lon=np.array([puff.lon for puff in puffs], dtype=float),
                    puff_lat=np.array([puff.lat for puff in puffs], dtype=float),
                )
            )

        if dt_s <= 0:
            continue

        integrated_air_concentration_bq_s_m3 += concentration_bq_m3 * dt_s
        deposition_bq_m2 += concentration_bq_m3 * dispersion.dry_deposition_velocity_ms * dt_s
        _apply_mass_losses(puffs, dispersion, dt_s)
        puffs = _advect_puffs(
            puffs,
            model_meteo,
            current_time64,
            grid,
            dispersion,
            rng,
            dt_s,
            min_cross_sigma_m,
            min_along_sigma_m,
            grid_cross_diffusivity,
            grid_along_diffusivity,
        )

    return grid, window, deposition_bq_m2, snapshots, puffs, integrated_air_concentration_bq_s_m3


def _process_dispersion_chunk(
    model_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    frame_times: set[np.datetime64] | None,
    grid: SimulationGrid,
    window: SimulationWindow,
    state: DispersionState,
    *,
    skip_first_time: bool,
    store_snapshots: bool,
    snapshot_sink: Callable[[DispersionSnapshot], None] | None,
) -> DispersionState:
    min_cross_sigma_m, min_along_sigma_m, grid_cross_diffusivity, grid_along_diffusivity = _grid_spread_floor(model_meteo, dispersion)
    model_times = pd.DatetimeIndex(model_meteo.ds.time.values)

    for index, current_time in enumerate(model_times):
        if skip_first_time and index == 0:
            continue
        current_time64 = current_time.to_datetime64()
        if index < len(model_times) - 1:
            next_time = model_times[index + 1]
            dt_s = float((next_time - current_time).total_seconds())
        else:
            dt_s = 0.0

        if current_time < window.release_end and dt_s > 0:
            _emit_layered_puffs(
                state.puffs,
                model_meteo,
                current_time64,
                dt_s,
                domain,
                dispersion,
                min_cross_sigma_m,
                min_along_sigma_m,
            )

        concentration_bq_m3 = _field_from_puffs(state.puffs, grid)
        cloud_column_bq_m2 = _column_from_puffs(state.puffs, grid)

        if frame_times is not None and current_time64 in frame_times:
            if state.max_cloud_column_bq_m2 is None:
                state.max_cloud_column_bq_m2 = cloud_column_bq_m2.copy()
            else:
                state.max_cloud_column_bq_m2 = np.maximum(state.max_cloud_column_bq_m2, cloud_column_bq_m2)
            snapshot = DispersionSnapshot(
                time=current_time64,
                concentration_bq_m3=concentration_bq_m3.copy(),
                cloud_column_bq_m2=cloud_column_bq_m2.copy(),
                deposition_bq_m2=state.deposition_bq_m2.copy(),
                puff_lon=np.array([puff.lon for puff in state.puffs], dtype=float),
                puff_lat=np.array([puff.lat for puff in state.puffs], dtype=float),
            )
            if store_snapshots:
                state.snapshots.append(snapshot)
            if snapshot_sink is not None:
                snapshot_sink(snapshot)
            state.last_snapshot_time = current_time64

        if dt_s <= 0:
            continue

        state.integrated_air_concentration_bq_s_m3 += concentration_bq_m3 * dt_s
        state.deposition_bq_m2 += concentration_bq_m3 * dispersion.dry_deposition_velocity_ms * dt_s
        _apply_mass_losses(state.puffs, dispersion, dt_s)
        state.puffs = _advect_puffs(
            state.puffs,
            model_meteo,
            current_time64,
            grid,
            dispersion,
            state.rng,
            dt_s,
            min_cross_sigma_m,
            min_along_sigma_m,
            grid_cross_diffusivity,
            grid_along_diffusivity,
        )

    return state


def _run_dispersion_chunked(
    raw_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    *,
    model_step_minutes: int,
    frame_meteo: PreparedMeteo | None,
    chunk_hours: int,
    store_snapshots: bool = True,
    snapshot_sink: Callable[[DispersionSnapshot], None] | None = None,
) -> tuple[SimulationGrid, SimulationWindow, DispersionState]:
    from .meteo import iter_interpolated_time_chunks

    window = build_simulation_window(timeline)
    grid = build_simulation_grid(frame_meteo or raw_meteo, domain)
    state = DispersionState(
        puffs=[],
        deposition_bq_m2=np.zeros(grid.shape, dtype=np.float32),
        integrated_air_concentration_bq_s_m3=np.zeros(grid.shape, dtype=np.float32),
        snapshots=[],
        rng=np.random.default_rng(dispersion.random_seed),
        max_cloud_column_bq_m2=None,
        last_snapshot_time=None,
    )
    frame_times = None
    if frame_meteo is not None:
        frame_times = {timestamp.to_datetime64() for timestamp in pd.DatetimeIndex(frame_meteo.ds.time.values)}

    skip_first_time = False
    for model_chunk in iter_interpolated_time_chunks(
        prepared=raw_meteo,
        start_utc=window.incident_start.isoformat(),
        end_utc=window.simulation_end.isoformat(),
        step_minutes=model_step_minutes,
        chunk_hours=chunk_hours,
    ):
        state = _process_dispersion_chunk(
            model_meteo=model_chunk,
            timeline=timeline,
            domain=domain,
            dispersion=dispersion,
            frame_times=frame_times,
            grid=grid,
            window=window,
            state=state,
            skip_first_time=skip_first_time,
            store_snapshots=store_snapshots,
            snapshot_sink=snapshot_sink,
        )
        skip_first_time = True

    return grid, window, state


def run_dispersion_simulation(
    raw_meteo: PreparedMeteo,
    frame_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
) -> DispersionResult:
    grid, window, state = _run_dispersion_chunked(
        raw_meteo=raw_meteo,
        timeline=timeline,
        domain=domain,
        dispersion=dispersion,
        model_step_minutes=timeline.model_step_minutes,
        frame_meteo=frame_meteo,
        chunk_hours=48,
    )

    if not state.snapshots:
        raise ValueError("Не удалось сохранить ни одного кадра симуляции.")

    if state.snapshots[-1].time != frame_meteo.ds.time.values[-1]:
        concentration_bq_m3 = _field_from_puffs(state.puffs, grid)
        cloud_column_bq_m2 = _column_from_puffs(state.puffs, grid)
        if state.max_cloud_column_bq_m2 is None:
            state.max_cloud_column_bq_m2 = cloud_column_bq_m2.copy()
        else:
            state.max_cloud_column_bq_m2 = np.maximum(state.max_cloud_column_bq_m2, cloud_column_bq_m2)
        state.snapshots.append(
            DispersionSnapshot(
                time=frame_meteo.ds.time.values[-1],
                concentration_bq_m3=concentration_bq_m3.copy(),
                cloud_column_bq_m2=cloud_column_bq_m2.copy(),
                deposition_bq_m2=state.deposition_bq_m2.copy(),
                puff_lon=np.array([puff.lon for puff in state.puffs], dtype=float),
                puff_lat=np.array([puff.lat for puff in state.puffs], dtype=float),
            )
        )

    return DispersionResult(
        grid=grid,
        snapshots=state.snapshots,
        integrated_air_concentration_bq_s_m3=state.integrated_air_concentration_bq_s_m3,
        source_summary=frame_meteo.source_summary,
        window=window,
    )


def run_dispersion_aggregate(
    raw_meteo: PreparedMeteo,
    frame_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
    *,
    snapshot_sink: Callable[[DispersionSnapshot], None] | None = None,
) -> DispersionAggregateResult:
    grid, window, state = _run_dispersion_chunked(
        raw_meteo=raw_meteo,
        timeline=timeline,
        domain=domain,
        dispersion=dispersion,
        model_step_minutes=timeline.model_step_minutes,
        frame_meteo=frame_meteo,
        chunk_hours=48,
        store_snapshots=False,
        snapshot_sink=snapshot_sink,
    )

    final_frame_time = frame_meteo.ds.time.values[-1]
    if not frame_meteo.ds.time.values.size:
        raise ValueError("No frame times available for aggregate dispersion.")

    if state.max_cloud_column_bq_m2 is None or state.last_snapshot_time != final_frame_time:
        concentration_bq_m3 = _field_from_puffs(state.puffs, grid)
        cloud_column_bq_m2 = _column_from_puffs(state.puffs, grid)
        if state.max_cloud_column_bq_m2 is None:
            state.max_cloud_column_bq_m2 = cloud_column_bq_m2.copy()
        else:
            state.max_cloud_column_bq_m2 = np.maximum(state.max_cloud_column_bq_m2, cloud_column_bq_m2)
        if snapshot_sink is not None:
            snapshot_sink(
                DispersionSnapshot(
                    time=final_frame_time,
                    concentration_bq_m3=concentration_bq_m3.copy(),
                    cloud_column_bq_m2=cloud_column_bq_m2.copy(),
                    deposition_bq_m2=state.deposition_bq_m2.copy(),
                    puff_lon=np.array([puff.lon for puff in state.puffs], dtype=float),
                    puff_lat=np.array([puff.lat for puff in state.puffs], dtype=float),
                )
            )
        state.last_snapshot_time = final_frame_time

    if state.max_cloud_column_bq_m2 is None:
        raise ValueError("No cloud frames were accumulated during aggregate dispersion.")

    return DispersionAggregateResult(
        grid=grid,
        max_cloud_column_bq_m2=state.max_cloud_column_bq_m2,
        final_deposition_bq_m2=state.deposition_bq_m2,
        integrated_air_concentration_bq_s_m3=state.integrated_air_concentration_bq_s_m3,
        source_summary=frame_meteo.source_summary,
        window=window,
    )


def run_dispersion_final_deposition(
    raw_meteo: PreparedMeteo,
    timeline: TimelineConfig,
    domain: DomainConfig,
    dispersion: DispersionConfig,
) -> FinalDepositionResult:
    grid, window, state = _run_dispersion_chunked(
        raw_meteo=raw_meteo,
        timeline=timeline,
        domain=domain,
        dispersion=dispersion,
        model_step_minutes=timeline.model_step_minutes,
        frame_meteo=None,
        chunk_hours=48,
    )
    return FinalDepositionResult(
        grid=grid,
        final_deposition_bq_m2=state.deposition_bq_m2,
        integrated_air_concentration_bq_s_m3=state.integrated_air_concentration_bq_s_m3,
        source_summary=raw_meteo.source_summary,
        window=window,
    )

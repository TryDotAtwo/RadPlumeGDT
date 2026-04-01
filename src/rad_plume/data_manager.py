from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import DataConfig, DataInventoryConfig, DomainConfig, ProjectConfig, SETTINGS
from .meteo import (
    PreparedMeteo,
    _build_time_coord,
    _open_dataset_with_fallback,
    _rename_fields,
    _sort_coords,
    _squeeze_singleton_dims,
    open_meteo_dataset,
    slice_and_interpolate_time,
    iter_interpolated_time_chunks,
)

BLEND_COMPONENT_PROFILES = ("historical_actual", "future_medium_range", "future_seasonal")
RUNTIME_BLEND_PROFILE_PREFIX = "__runtime_blend__:"


@dataclass(frozen=True)
class DataStatus:
    profile: str
    path: Path
    source_kind: str
    source_label: str
    exists: bool
    stale: bool
    status_text: str
    coverage_start: str | None = None
    coverage_end: str | None = None
    init_time: str | None = None


def resolve_data_config(settings: ProjectConfig, profile: str | None) -> tuple[str, DataConfig]:
    resolved_profile = profile or settings.inventory.default_source_profile
    if resolved_profile == "best_available_blend":
        raise ValueError("best_available_blend не соответствует одному data_file и должен открываться через blend helper.")
    if resolved_profile == "historical_actual":
        return resolved_profile, settings.inventory.historical_actual
    if resolved_profile == "future_medium_range":
        return resolved_profile, settings.inventory.future_medium_range
    if resolved_profile == "future_seasonal":
        return resolved_profile, settings.inventory.future_seasonal
    raise ValueError(f"Неизвестный source profile: {resolved_profile}")


def _config_for_profile(settings: ProjectConfig, profile: str) -> DataConfig:
    if profile == "historical_actual":
        return settings.inventory.historical_actual
    if profile == "future_medium_range":
        return settings.inventory.future_medium_range
    if profile == "future_seasonal":
        return settings.inventory.future_seasonal
    raise ValueError(f"Неизвестный профиль данных: {profile}")


def _load_raw_dataset(path: Path) -> xr.Dataset:
    ds = _open_dataset_with_fallback(path)
    ds = _rename_fields(ds)
    ds = _squeeze_singleton_dims(ds)
    ds = _build_time_coord(ds)
    ds = _sort_coords(ds)
    return ds


def _align_dataset_for_blend(
    ds: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_pressure_levels: np.ndarray,
) -> xr.Dataset:
    aligned = ds
    if not np.array_equal(aligned.latitude.values, target_lat) or not np.array_equal(aligned.longitude.values, target_lon):
        aligned = aligned.interp(latitude=target_lat, longitude=target_lon, method="linear")

    if "pressure_level" in aligned.coords and any(name in aligned.data_vars for name in ("u_pl", "v_pl", "omega_pl", "z_pl", "t_pl")):
        available = aligned.pressure_level.values.astype(float)
        keep_levels = target_pressure_levels[(target_pressure_levels >= float(np.nanmin(available))) & (target_pressure_levels <= float(np.nanmax(available)))]
        if keep_levels.size == 0:
            keep_levels = available
        if not np.array_equal(available.astype(float), keep_levels.astype(float)):
            aligned = aligned.interp(pressure_level=keep_levels, method="linear")

    return _sort_coords(aligned)


def _annotate_dataset_source(ds: xr.Dataset, profile: str, prepared: PreparedMeteo) -> xr.Dataset:
    time_values = ds.time.values
    source_grid = xr.DataArray(
        np.full(len(time_values), prepared.grid_spacing_m, dtype=np.float32),
        coords={"time": time_values},
        dims=("time",),
    )
    source_step = xr.DataArray(
        np.full(len(time_values), prepared.native_step_minutes, dtype=np.float32),
        coords={"time": time_values},
        dims=("time",),
    )
    source_code = {
        "historical_actual": 1.0,
        "future_medium_range": 2.0,
        "future_seasonal": 3.0,
    }[profile]
    source_profile_code = xr.DataArray(
        np.full(len(time_values), source_code, dtype=np.float32),
        coords={"time": time_values},
        dims=("time",),
    )
    return ds.assign(
        source_grid_spacing_m=source_grid,
        source_native_step_minutes=source_step,
        source_profile_code=source_profile_code,
    )


def open_best_available_blend(settings: ProjectConfig) -> PreparedMeteo:
    prepared_by_profile: list[tuple[str, PreparedMeteo]] = []
    for profile in BLEND_COMPONENT_PROFILES:
        data_config = _config_for_profile(settings, profile)
        if not data_config.data_file.exists():
            continue
        prepared_by_profile.append((profile, open_meteo_dataset(data_config, settings.domain)))

    if not prepared_by_profile:
        raise FileNotFoundError("Не найден ни один источник данных для best_available_blend.")

    target_profile, target_prepared = min(
        prepared_by_profile,
        key=lambda item: (
            item[1].grid_spacing_m,
            BLEND_COMPONENT_PROFILES.index(item[0]),
        ),
    )
    target_lat = target_prepared.ds.latitude.values.astype(float)
    target_lon = target_prepared.ds.longitude.values.astype(float)
    target_pressure_levels = np.asarray(settings.inventory.transport_pressure_levels_hpa, dtype=float)

    aligned_segments: list[xr.Dataset] = []
    summary_parts: list[str] = []
    next_start: pd.Timestamp | None = None
    for profile in BLEND_COMPONENT_PROFILES:
        prepared = next((item[1] for item in prepared_by_profile if item[0] == profile), None)
        if prepared is None:
            continue

        ds = _align_dataset_for_blend(prepared.ds, target_lat, target_lon, target_pressure_levels)
        if next_start is not None:
            ds = ds.sel(time=slice(next_start, None))
        if ds.sizes.get("time", 0) == 0:
            continue

        ds = _annotate_dataset_source(ds, profile, prepared)
        aligned_segments.append(ds)
        next_start = pd.Timestamp(ds.time.values[-1]) + pd.Timedelta(microseconds=1)
        summary_parts.append(f"{profile} [{pd.Timestamp(ds.time.values[0]).isoformat()} -> {pd.Timestamp(ds.time.values[-1]).isoformat()}]")

    if not aligned_segments:
        raise FileNotFoundError("best_available_blend не удалось собрать: после обрезки по времени не осталось ни одного сегмента.")

    if len(aligned_segments) == 1:
        blended_ds = aligned_segments[0]
    else:
        blended_ds = xr.concat(aligned_segments, dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
        _, unique_index = np.unique(blended_ds.time.values, return_index=True)
        blended_ds = blended_ds.isel(time=np.sort(unique_index))

    native_step_minutes = int(
        np.nanmax(blended_ds["source_native_step_minutes"].values)
    ) if "source_native_step_minutes" in blended_ds.data_vars else target_prepared.native_step_minutes
    grid_spacing_m = float(
        np.nanmax(blended_ds["source_grid_spacing_m"].values)
    ) if "source_grid_spacing_m" in blended_ds.data_vars else target_prepared.grid_spacing_m
    pressure_note = ""
    if any(name in blended_ds.data_vars for name in ("u_pl", "v_pl")):
        pressure_note = f"; pressure levels {', '.join(str(level) for level in settings.inventory.transport_pressure_levels_hpa)} hPa when available"
    source_summary = (
        "Best-available blended meteo: "
        + " -> ".join(summary_parts)
        + f"; target grid from {target_profile}"
        + pressure_note
    )
    return PreparedMeteo(
        ds=_sort_coords(blended_ds),
        extent=target_prepared.extent,
        native_step_minutes=native_step_minutes,
        grid_spacing_m=grid_spacing_m,
        source_summary=source_summary,
    )


def materialize_best_available_blend(settings: ProjectConfig, output_path: Path) -> PreparedMeteo:
    prepared = open_best_available_blend(settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _sort_coords(prepared.ds)
    encoding = {
        name: {"dtype": "float32"}
        for name, var in ds.data_vars.items()
        if np.issubdtype(var.dtype, np.floating)
    }
    ds.to_netcdf(output_path, engine="scipy", encoding=encoding)
    return PreparedMeteo(
        ds=ds,
        extent=prepared.extent,
        native_step_minutes=prepared.native_step_minutes,
        grid_spacing_m=prepared.grid_spacing_m,
        source_summary=prepared.source_summary,
    )


def _latest_seasonal_release(now_utc: pd.Timestamp, inventory: DataInventoryConfig) -> pd.Timestamp:
    release_candidate = pd.Timestamp(
        year=now_utc.year,
        month=now_utc.month,
        day=inventory.seasonal_release_day_utc,
        hour=inventory.seasonal_release_hour_utc,
        tz="UTC",
    )
    if now_utc >= release_candidate:
        release_month = pd.Timestamp(year=now_utc.year, month=now_utc.month, day=1, tz="UTC")
    else:
        previous = now_utc - pd.DateOffset(months=1)
        release_month = pd.Timestamp(year=previous.year, month=previous.month, day=1, tz="UTC")
    return release_month


def inspect_data_status(
    data_config: DataConfig,
    profile: str,
    inventory: DataInventoryConfig,
    now_utc: pd.Timestamp | None = None,
) -> DataStatus:
    now_utc = now_utc or pd.Timestamp(datetime.now(timezone.utc))
    if not data_config.data_file.exists():
        return DataStatus(
            profile=profile,
            path=data_config.data_file,
            source_kind=data_config.source_kind,
            source_label=data_config.source_label,
            exists=False,
            stale=True,
            status_text="missing",
        )

    ds_native = _open_dataset_with_fallback(data_config.data_file)
    init_time = None
    if "forecast_reference_time" in ds_native.coords:
        init_time = pd.Timestamp(np.asarray(ds_native["forecast_reference_time"].values).reshape(-1)[0]).isoformat()

    ds = _load_raw_dataset(data_config.data_file)
    coverage_start = pd.Timestamp(ds.time.values[0]).isoformat()
    coverage_end = pd.Timestamp(ds.time.values[-1]).isoformat()

    if data_config.source_kind == "era5_reanalysis":
        freshness_cutoff = now_utc.floor("D") - pd.Timedelta(days=inventory.era5_latency_days)
        stale = pd.Timestamp(ds.time.values[-1]) < freshness_cutoff.tz_localize(None)
        status_text = "stale" if stale else "fresh"
        return DataStatus(
            profile=profile,
            path=data_config.data_file,
            source_kind=data_config.source_kind,
            source_label=data_config.source_label,
            exists=True,
            stale=stale,
            status_text=status_text,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
        )

    if data_config.source_kind == "ecmwf_medium_range":
        init_ts = pd.Timestamp(init_time) if init_time else None
        freshness_age = pd.Timedelta(hours=inventory.medium_range_latency_hours + 24)
        stale = init_ts is None or (now_utc.tz_localize(None) - init_ts) > freshness_age
        status_text = "stale" if stale else "fresh"
        return DataStatus(
            profile=profile,
            path=data_config.data_file,
            source_kind=data_config.source_kind,
            source_label=data_config.source_label,
            exists=True,
            stale=stale,
            status_text=status_text,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            init_time=init_time,
        )

    latest_release = _latest_seasonal_release(now_utc, inventory)
    init_ts = pd.Timestamp(init_time) if init_time else None
    stale = init_ts is None or (init_ts.year, init_ts.month) != (latest_release.year, latest_release.month)
    status_text = "stale" if stale else "fresh"
    return DataStatus(
        profile=profile,
        path=data_config.data_file,
        source_kind=data_config.source_kind,
        source_label=data_config.source_label,
        exists=True,
        stale=stale,
        status_text=status_text,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        init_time=init_time,
    )


def estimate_seasonal_uncertainty(
    seasonal_config: DataConfig,
    domain: DomainConfig,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> str:
    if not seasonal_config.data_file.exists():
        return "Seasonal uncertainty: file missing."

    ds = _load_raw_dataset(seasonal_config.data_file)
    if "number" not in ds.dims:
        return "Seasonal uncertainty: ensemble dimension is missing."

    if start_utc or end_utc:
        ds = ds.sel(time=slice(start_utc, end_utc))
        if ds.sizes.get("time", 0) == 0:
            return "Seasonal uncertainty: no overlap with requested window."

    speed = np.hypot(ds["u10"], ds["v10"])
    spread = speed.std(dim="number")
    median_spread = float(spread.median().item())
    p90_spread = float(spread.quantile(0.9).item())
    return (
        "Seasonal uncertainty: "
        f"median ensemble speed spread {median_spread:.2f} m/s, "
        f"P90 spread {p90_spread:.2f} m/s."
    )


def compare_forecast_against_actual(
    forecast_config: DataConfig,
    actual_config: DataConfig,
    domain: DomainConfig,
) -> str:
    if not forecast_config.data_file.exists() or not actual_config.data_file.exists():
        return "Forecast-vs-actual error: unavailable because one of the files is missing."

    forecast_prepared = open_meteo_dataset(forecast_config, domain)
    actual_prepared = open_meteo_dataset(actual_config, domain)
    overlap_start = max(pd.Timestamp(forecast_prepared.ds.time.values[0]), pd.Timestamp(actual_prepared.ds.time.values[0]))
    overlap_end = min(pd.Timestamp(forecast_prepared.ds.time.values[-1]), pd.Timestamp(actual_prepared.ds.time.values[-1]))

    if overlap_end <= overlap_start:
        return "Forecast-vs-actual error: no overlapping interval."

    chunk_hours = 48
    sum_sq_speed = 0.0
    sum_speed = 0.0
    count_speed = 0
    sum_sq_u = 0.0
    count_u = 0
    sum_sq_v = 0.0
    count_v = 0

    forecast_chunks = iter_interpolated_time_chunks(
        prepared=forecast_prepared,
        start_utc=overlap_start.isoformat(),
        end_utc=overlap_end.isoformat(),
        step_minutes=60,
        chunk_hours=chunk_hours,
        keep_vars=("u10", "v10"),
    )
    actual_chunks = iter_interpolated_time_chunks(
        prepared=actual_prepared,
        start_utc=overlap_start.isoformat(),
        end_utc=overlap_end.isoformat(),
        step_minutes=60,
        chunk_hours=chunk_hours,
        keep_vars=("u10", "v10"),
    )

    for forecast_chunk, actual_chunk in zip(forecast_chunks, actual_chunks, strict=True):
        if (
            not np.array_equal(forecast_chunk.ds.latitude.values, actual_chunk.ds.latitude.values)
            or not np.array_equal(forecast_chunk.ds.longitude.values, actual_chunk.ds.longitude.values)
        ):
            if forecast_chunk.ds.sizes["latitude"] * forecast_chunk.ds.sizes["longitude"] <= actual_chunk.ds.sizes["latitude"] * actual_chunk.ds.sizes["longitude"]:
                target_lat = forecast_chunk.ds.latitude.values
                target_lon = forecast_chunk.ds.longitude.values
            else:
                target_lat = actual_chunk.ds.latitude.values
                target_lon = actual_chunk.ds.longitude.values
            forecast_ds = forecast_chunk.ds.interp(latitude=target_lat, longitude=target_lon, method="linear")
            actual_ds = actual_chunk.ds.interp(latitude=target_lat, longitude=target_lon, method="linear")
        else:
            forecast_ds = forecast_chunk.ds
            actual_ds = actual_chunk.ds

        delta_u = (forecast_ds["u10"] - actual_ds["u10"]).values
        delta_v = (forecast_ds["v10"] - actual_ds["v10"]).values
        forecast_speed = np.hypot(forecast_ds["u10"].values, forecast_ds["v10"].values)
        actual_speed = np.hypot(actual_ds["u10"].values, actual_ds["v10"].values)
        delta_speed = forecast_speed - actual_speed

        valid_speed = np.isfinite(delta_speed)
        if np.any(valid_speed):
            speed_values = delta_speed[valid_speed].astype(np.float64, copy=False)
            sum_sq_speed += float(np.dot(speed_values, speed_values))
            sum_speed += float(speed_values.sum())
            count_speed += int(speed_values.size)

        valid_u = np.isfinite(delta_u)
        if np.any(valid_u):
            u_values = delta_u[valid_u].astype(np.float64, copy=False)
            sum_sq_u += float(np.dot(u_values, u_values))
            count_u += int(u_values.size)

        valid_v = np.isfinite(delta_v)
        if np.any(valid_v):
            v_values = delta_v[valid_v].astype(np.float64, copy=False)
            sum_sq_v += float(np.dot(v_values, v_values))
            count_v += int(v_values.size)

    if count_speed == 0 or count_u == 0 or count_v == 0:
        return "Forecast-vs-actual error: overlapping chunks produced no finite comparisons."

    rmse_speed = float(np.sqrt(sum_sq_speed / count_speed))
    bias_speed = float(sum_speed / count_speed)
    rmse_u = float(np.sqrt(sum_sq_u / count_u))
    rmse_v = float(np.sqrt(sum_sq_v / count_v))
    return (
        "Forecast-vs-actual error over overlap "
        f"{overlap_start.isoformat()} .. {overlap_end.isoformat()}: "
        f"RMSE speed {rmse_speed:.2f} m/s, bias speed {bias_speed:.2f} m/s, "
        f"RMSE u {rmse_u:.2f} m/s, RMSE v {rmse_v:.2f} m/s."
    )


def write_data_quality_report(settings: ProjectConfig, report_path: Path | None = None) -> Path:
    report_path = report_path or settings.inventory.quality_report_file
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    historical_status = inspect_data_status(settings.inventory.historical_actual, "historical_actual", settings.inventory, now_utc)
    medium_range_status = inspect_data_status(settings.inventory.future_medium_range, "future_medium_range", settings.inventory, now_utc)
    future_status = inspect_data_status(settings.inventory.future_seasonal, "future_seasonal", settings.inventory, now_utc)

    future_uncertainty = estimate_seasonal_uncertainty(settings.inventory.future_seasonal, settings.domain)
    medium_range_error = compare_forecast_against_actual(
        forecast_config=settings.inventory.future_medium_range,
        actual_config=settings.inventory.historical_actual,
        domain=settings.domain,
    )
    actual_error = compare_forecast_against_actual(
        forecast_config=replace(settings.inventory.future_seasonal, ensemble_mode="median"),
        actual_config=settings.inventory.historical_actual,
        domain=settings.domain,
    )

    lines = [
        "# Data Quality Report",
        "",
        f"Generated at: {now_utc.isoformat()}",
        "",
        "## historical_actual",
        f"- path: `{historical_status.path}`",
        f"- exists: `{historical_status.exists}`",
        f"- status: `{historical_status.status_text}`",
        f"- coverage: `{historical_status.coverage_start}` -> `{historical_status.coverage_end}`",
        "",
        "## future_medium_range",
        f"- path: `{medium_range_status.path}`",
        f"- exists: `{medium_range_status.exists}`",
        f"- status: `{medium_range_status.status_text}`",
        f"- init_time: `{medium_range_status.init_time}`",
        f"- coverage: `{medium_range_status.coverage_start}` -> `{medium_range_status.coverage_end}`",
        "",
        "## future_seasonal",
        f"- path: `{future_status.path}`",
        f"- exists: `{future_status.exists}`",
        f"- status: `{future_status.status_text}`",
        f"- init_time: `{future_status.init_time}`",
        f"- coverage: `{future_status.coverage_start}` -> `{future_status.coverage_end}`",
        "",
        "## uncertainty",
        f"- {future_uncertainty}",
        "",
        "## factual error",
        f"- Medium-range vs actual: {medium_range_error}",
        f"- Seasonal vs actual: {actual_error}",
        "",
        "## note",
        "- Best available past/current data: ERA5 hourly reanalysis.",
        "- Best available 0-15 day future forecast: ECMWF medium-range forecast.",
        "- Best available six-month future scenario: seasonal ensemble forecast.",
        "- Default runtime profile is best_available_blend = ERA5 -> medium-range -> seasonal when those files exist.",
        "- Vertical steering winds are merged from dense pressure levels when the active source provides them.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def refresh_profile_data(profile: str, settings: ProjectConfig = SETTINGS) -> None:
    if profile == "historical_actual":
        from .download_era5_box import main as refresh_main
    elif profile == "future_medium_range":
        from .download_medium_range_box import main as refresh_main
    elif profile == "future_seasonal":
        from .download_seasonal_box import main as refresh_main
    else:
        raise ValueError(f"Неизвестный профиль для обновления: {profile}")

    refresh_main(settings)

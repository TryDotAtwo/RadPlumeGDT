from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import shutil
import warnings

import cfgrib
import numpy as np
import cdsapi
import pandas as pd
import xarray as xr

from .config import ProjectConfig, SETTINGS, ensure_runtime_dirs
from .download_progress import run_with_tqdm_heartbeat
from .meteo import _build_time_coord, _open_dataset_with_fallback, canonicalize_netcdf_to_scipy


warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

SEASONAL_PRESSURE_LEVELS_PRIMARY = (1000, 925, 850, 700, 500, 400, 300, 200, 100, 50, 30, 10)
SEASONAL_PRESSURE_LEVELS_FALLBACK = (1000, 925, 850, 700, 500, 300, 200, 100, 50, 10)
SEASONAL_PRESSURE_VARIABLES = (
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "geopotential",
    "specific_humidity",
)


def _naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _dataset_init_time(ds: xr.Dataset) -> pd.Timestamp | None:
    init_value = ds.coords.get("forecast_reference_time")
    if init_value is None:
        return None
    return _naive_timestamp(np.asarray(init_value.values).reshape(-1)[0])


def _dedupe_time_keep_last(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        return ds
    time_values = np.asarray(ds.time.values)
    _, reverse_index = np.unique(time_values[::-1], return_index=True)
    keep = np.sort(len(time_values) - 1 - reverse_index)
    return ds.isel(time=keep).sortby("time")


def _dataset_metadata(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists():
        return None, None
    try:
        ds = xr.open_dataset(path, engine="scipy")
        time_values = pd.to_datetime(ds["time"].values)
        init_time = _dataset_init_time(ds)
        ds.close()
    except Exception:
        return None, None
    if len(time_values) == 0 or init_time is None:
        return init_time, None
    return init_time, pd.Timestamp(time_values[-1])


def _existing_dataset_covers(path: Path, init_month: pd.Timestamp, required_end: pd.Timestamp) -> bool:
    init_time, last_time = _dataset_metadata(path)
    return (
        init_time is not None
        and (init_time.year, init_time.month) == (init_month.year, init_month.month)
        and last_time is not None
        and last_time >= required_end - pd.Timedelta(hours=6)
    )


def _append_or_write_dataset(path: Path, new_ds: xr.Dataset) -> None:
    new_ds = _dedupe_time_keep_last(new_ds)
    new_init = _dataset_init_time(new_ds)
    if path.exists():
        existing = xr.open_dataset(path, engine="scipy").load()
        try:
            existing = _dedupe_time_keep_last(existing)
            existing_init = _dataset_init_time(existing)
            if existing_init is not None and new_init is not None and existing_init != new_init:
                combined = new_ds
            else:
                combined = xr.concat([existing, new_ds], dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
                combined = _dedupe_time_keep_last(combined)
        finally:
            existing.close()
    else:
        combined = new_ds
    if new_init is not None:
        combined = combined.assign_coords(forecast_reference_time=new_init.to_datetime64())
    if path.exists():
        path.unlink()
    combined.to_netcdf(path, engine="scipy")


def latest_release_month(now_utc: pd.Timestamp, settings: ProjectConfig = SETTINGS) -> pd.Timestamp:
    release_day = settings.inventory.seasonal_release_day_utc
    release_hour = settings.inventory.seasonal_release_hour_utc
    release_cutoff = pd.Timestamp(
        year=now_utc.year,
        month=now_utc.month,
        day=release_day,
        hour=release_hour,
        tz="UTC",
    )
    if now_utc >= release_cutoff:
        return pd.Timestamp(year=now_utc.year, month=now_utc.month, day=1, tz="UTC")
    previous = now_utc - pd.DateOffset(months=1)
    return pd.Timestamp(year=previous.year, month=previous.month, day=1, tz="UTC")


def _run_quietly(operation):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return operation()


def build_area(settings: ProjectConfig = SETTINGS) -> list[float]:
    half_span_lat_deg = settings.domain.domain_radius_km / 111.0
    lon_scale = max(111.0 * np.cos(np.deg2rad(settings.domain.source_lat)), 1e-6)
    half_span_lon_deg = settings.domain.domain_radius_km / lon_scale
    west = settings.domain.source_lon - half_span_lon_deg
    east = settings.domain.source_lon + half_span_lon_deg
    south = settings.domain.source_lat - half_span_lat_deg
    north = settings.domain.source_lat + half_span_lat_deg
    return [north, west, south, east]


def _resolve_cds_credentials_path() -> Path | None:
    candidates: list[Path] = []
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.append(Path(user_profile) / ".cdsapirc")
    candidates.append(Path.home() / ".cdsapirc")
    system_drive = os.environ.get("SystemDrive", "C:")
    if not system_drive.endswith("\\"):
        system_drive = system_drive + "\\"
    users_root = Path(system_drive) / "Users"
    if users_root.exists():
        try:
            candidates.extend(sorted(users_root.glob("*/.cdsapirc")))
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_cds_credentials() -> None:
    cds_config = _resolve_cds_credentials_path()
    if cds_config is None:
        raise RuntimeError(
            "CDS credentials file is missing in USERPROFILE or HOME as .cdsapirc. "
            "Create it before running seasonal downloads."
        )


def _load_cds_credentials() -> tuple[str, str]:
    cds_config = _resolve_cds_credentials_path()
    if cds_config is None:
        raise RuntimeError(
            "CDS credentials file is missing in USERPROFILE or HOME as .cdsapirc. "
            "Create it before running seasonal downloads."
        )
    values: dict[str, str] = {}
    for line in cds_config.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    url = values.get("url")
    api_key = values.get("key")
    if not url or not api_key:
        raise RuntimeError(f"CDS credentials file is incomplete: {cds_config}")
    return url, api_key


def _collapse_forecast_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        try:
            ds = _build_time_coord(ds)
        except Exception:
            return ds
    for dim_name in ("forecast_reference_time", "leadtime_hour", "forecast_period"):
        if dim_name in ds.dims and ds.sizes.get(dim_name) == 1:
            ds = ds.squeeze(dim_name, drop=True)
    return ds


def _subset_and_convert_single_grib(grib_path: Path, output_file: Path, area: list[float], init_month: pd.Timestamp) -> None:
    merged = _load_surface_grib_dataset(grib_path, area, init_month)
    if output_file.exists():
        output_file.unlink()
    merged.to_netcdf(output_file, engine="scipy")


def _load_surface_grib_dataset(grib_path: Path, area: list[float], init_month: pd.Timestamp) -> xr.Dataset:
    north, west, south, east = area
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(str(grib_path), backend_kwargs={"indexpath": ""})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        merged = xr.merge(datasets, compat="override", join="outer")
    merged = _collapse_forecast_time(merged)
    if "latitude" in merged.coords and "longitude" in merged.coords:
        merged = merged.sortby("longitude").sortby("latitude")
        merged = merged.sel(longitude=slice(west, east), latitude=slice(south, north))
    merged = merged.load()
    merged = merged.assign_coords(forecast_reference_time=init_month.to_datetime64())
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass
    return merged


def _prepare_pressure_netcdf(path: Path, pressure_file: Path, area: list[float], init_month: pd.Timestamp) -> None:
    north, west, south, east = area
    prepared = _open_dataset_with_fallback(path)
    prepared = _collapse_forecast_time(prepared)
    if "isobaricInhPa" in prepared.coords:
        prepared = prepared.rename({"isobaricInhPa": "pressure_level"})
    if "latitude" in prepared.coords and "longitude" in prepared.coords:
        prepared = prepared.sortby("longitude").sortby("latitude")
        prepared = prepared.sel(longitude=slice(west, east), latitude=slice(south, north))
    if "forecast_reference_time" not in prepared.coords:
        prepared = prepared.assign_coords(forecast_reference_time=init_month.to_datetime64())
    if pressure_file.exists():
        pressure_file.unlink()
    prepared.to_netcdf(pressure_file, engine="scipy")


def _load_pressure_chunk(path: Path, area: list[float], init_month: pd.Timestamp) -> xr.Dataset:
    north, west, south, east = area
    prepared = _open_dataset_with_fallback(path)
    prepared = _collapse_forecast_time(prepared)
    if "isobaricInhPa" in prepared.coords:
        prepared = prepared.rename({"isobaricInhPa": "pressure_level"})
    if "latitude" in prepared.coords and "longitude" in prepared.coords:
        prepared = prepared.sortby("longitude").sortby("latitude")
        prepared = prepared.sel(longitude=slice(west, east), latitude=slice(south, north))
    if "forecast_reference_time" not in prepared.coords:
        prepared = prepared.assign_coords(forecast_reference_time=init_month.to_datetime64())
    return prepared


def _chunked(sequence: list[str], chunk_size: int) -> list[list[str]]:
    return [sequence[index : index + chunk_size] for index in range(0, len(sequence), chunk_size)]


def _download_pressure_chunks(
    *,
    client: cdsapi.Client,
    init_month: pd.Timestamp,
    area: list[float],
    pressure_levels: tuple[int, ...],
    leadtime_hours: list[str],
    temp_dir: Path,
) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    leadtime_chunks = _chunked(leadtime_hours, 8)
    total_chunks = len(leadtime_chunks)
    for chunk_index, leadtime_chunk in enumerate(leadtime_chunks, start=1):
        chunk_path = temp_dir / f"seasonal_pressure_{chunk_index:03d}.nc"
        if chunk_path.exists():
            chunk_path.unlink()
        print(
            f"Seasonal pressure chunk {chunk_index}/{total_chunks}: "
            f"{leadtime_chunk[0]}h -> {leadtime_chunk[-1]}h"
        )
        run_with_tqdm_heartbeat(
            f"Seasonal pressure download {chunk_index}/{total_chunks}",
            lambda chunk=leadtime_chunk, target=str(chunk_path): _run_quietly(
                lambda: client.retrieve(
                    "seasonal-original-pressure-levels",
                    {
                        "originating_centre": "ecmwf",
                        "system": "51",
                        "variable": list(SEASONAL_PRESSURE_VARIABLES),
                        "pressure_level": [str(level) for level in pressure_levels],
                        "year": f"{init_month.year:04d}",
                        "month": f"{init_month.month:02d}",
                        "day": "01",
                        "time": "00:00",
                        "leadtime_hour": chunk,
                        "area": area,
                        "data_format": "netcdf",
                    },
                    target,
                )
            ),
        )
        chunk_paths.append(chunk_path)
    return chunk_paths


def _combine_pressure_chunks(chunk_paths: list[Path], pressure_file: Path, area: list[float], init_month: pd.Timestamp) -> None:
    prepared_chunks = [_load_pressure_chunk(path, area, init_month) for path in chunk_paths]
    combined = xr.concat(prepared_chunks, dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
    _append_or_write_dataset(pressure_file, combined)


def _seasonal_supported_pressure_levels(levels: tuple[int, ...]) -> tuple[int, ...]:
    filtered = tuple(level for level in SEASONAL_PRESSURE_LEVELS_PRIMARY if level in set(levels))
    return filtered or SEASONAL_PRESSURE_LEVELS_PRIMARY


def _resolve_horizon_days(settings: ProjectConfig, init_month: pd.Timestamp, demo: bool) -> int:
    if not demo:
        return 180
    incident_end = pd.Timestamp(settings.timeline.incident_start_utc) + pd.Timedelta(hours=settings.timeline.demo_hours)
    required_days = int(np.ceil(max((incident_end - init_month.tz_localize(None)).total_seconds(), 0.0) / 86_400.0))
    return max(21, min(90, required_days + 7))


def main(settings: ProjectConfig = SETTINGS, *, demo: bool = False) -> None:
    ensure_runtime_dirs()
    _ensure_cds_credentials()
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    init_month = latest_release_month(now_utc, settings)
    out_file = settings.inventory.future_seasonal.data_file
    pressure_file = settings.inventory.future_seasonal.pressure_level_file
    surface_grib = out_file.parent / f".{out_file.stem}_{run_tag}.grib"
    area = build_area(settings)
    horizon_days = _resolve_horizon_days(settings, init_month, demo)
    leadtime_hours = [str(hour) for hour in range(0, 24 * horizon_days + 1, 6)]
    pressure_leadtime_hours = [str(hour) for hour in range(0, 24 * horizon_days + 1, 12)]
    pressure_levels = _seasonal_supported_pressure_levels(settings.inventory.transport_pressure_levels_hpa)
    cds_url, cds_key = _load_cds_credentials()
    client = cdsapi.Client(url=cds_url, key=cds_key, progress=False, quiet=True)
    required_end = init_month.tz_localize(None) + pd.Timedelta(days=horizon_days)
    surface_init, surface_end = _dataset_metadata(out_file)
    if surface_init is not None and (surface_init.year, surface_init.month) == (init_month.year, init_month.month) and surface_end is not None:
        leadtime_hours = [
            lead for lead in leadtime_hours
            if init_month.tz_localize(None) + pd.Timedelta(hours=int(lead)) > surface_end + pd.Timedelta(hours=6)
        ]

    if _existing_dataset_covers(out_file, init_month, required_end):
        print("Reusing existing seasonal NetCDF:", out_file)
    else:
        print("Downloading seasonal forecast single levels...")
        print("Initialization:", init_month.date())
        print("Horizon days:", horizon_days)
        print("Area:", area)
        print("Output:", out_file)

        if not leadtime_hours:
            print("No missing seasonal surface leadtimes; keeping existing NetCDF:", out_file)
        else:
            run_with_tqdm_heartbeat(
                "Seasonal single-level download",
                lambda: _run_quietly(
                    lambda: client.retrieve(
                        "seasonal-original-single-levels",
                        {
                            "originating_centre": "ecmwf",
                            "system": "51",
                            "variable": [
                                "10m_u_component_of_wind",
                                "10m_v_component_of_wind",
                                "2m_temperature",
                            ],
                            "year": f"{init_month.year:04d}",
                            "month": f"{init_month.month:02d}",
                            "day": "01",
                            "time": "00:00",
                            "leadtime_hour": leadtime_hours,
                            "area": area,
                            "format": "grib",
                        },
                        str(surface_grib),
                    )
                ),
            )
            print("Converting seasonal single-level GRIB to NetCDF...")
            run_with_tqdm_heartbeat(
                "Seasonal surface conversion",
                lambda: _append_or_write_dataset(out_file, _load_surface_grib_dataset(surface_grib, area, init_month)),
            )
            canonicalize_netcdf_to_scipy(out_file)
            if surface_grib.exists():
                surface_grib.unlink(missing_ok=True)

    if pressure_file is not None:
        pressure_init, pressure_end = _dataset_metadata(pressure_file)
        if pressure_init is not None and (pressure_init.year, pressure_init.month) == (init_month.year, init_month.month) and pressure_end is not None:
            pressure_leadtime_hours = [
                lead for lead in pressure_leadtime_hours
                if init_month.tz_localize(None) + pd.Timedelta(hours=int(lead)) > pressure_end + pd.Timedelta(hours=12)
            ]
        if _existing_dataset_covers(pressure_file, init_month, required_end):
            print("Reusing existing seasonal pressure NetCDF:", pressure_file)
        else:
            print("Downloading seasonal forecast pressure levels...")
            print("Pressure levels:", pressure_levels)
            print("Pressure cadence: 12-hourly")
            print("Output:", pressure_file)
            temp_dir = pressure_file.parent / f".{pressure_file.stem}_chunks_{run_tag}"
            level_candidates = []
            for candidate in (pressure_levels, SEASONAL_PRESSURE_LEVELS_FALLBACK):
                if candidate not in level_candidates:
                    level_candidates.append(candidate)
            last_error: Exception | None = None
            for level_set in level_candidates:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    print("Attempting seasonal pressure-level request with levels:", level_set)
                    chunk_paths = _download_pressure_chunks(
                        client=client,
                        init_month=init_month,
                        area=area,
                        pressure_levels=level_set,
                        leadtime_hours=pressure_leadtime_hours,
                        temp_dir=temp_dir,
                    )
                    print("Preparing seasonal pressure-level NetCDF...")
                    if not pressure_leadtime_hours:
                        print("No missing seasonal pressure leadtimes; keeping existing NetCDF:", pressure_file)
                    else:
                        run_with_tqdm_heartbeat(
                            "Seasonal pressure preparation",
                            lambda: _combine_pressure_chunks(chunk_paths, pressure_file, area, init_month),
                        )
                    canonicalize_netcdf_to_scipy(pressure_file)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print("Seasonal pressure-level request failed for this level set; trying fallback if available.")
                    print(exc)
            if last_error is not None:
                print("Warning: seasonal pressure-level download failed; continuing without it.")
                print(last_error)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    print("Done:", out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download seasonal forecast data for the project domain.")
    parser.add_argument("--demo", action="store_true", help="Short download demo instead of the full seasonal tail.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(demo=args.demo)

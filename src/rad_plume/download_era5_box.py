from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import zipfile
import warnings

import cdsapi
import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

from .config import ProjectConfig, SETTINGS, ensure_runtime_dirs
from .download_progress import run_with_tqdm_heartbeat
from .meteo import _build_time_coord, _rename_fields, _sort_coords, _squeeze_singleton_dims, canonicalize_netcdf_to_scipy


warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")


ERA5_MODEL_LEVEL_PARAMS = "130/131/132/133"


def build_area(settings: ProjectConfig = SETTINGS) -> list[float]:
    half_span_lat_deg = settings.domain.domain_radius_km / 111.0
    lon_scale = max(111.0 * np.cos(np.deg2rad(settings.domain.source_lat)), 1e-6)
    half_span_lon_deg = settings.domain.domain_radius_km / lon_scale
    west = settings.domain.source_lon - half_span_lon_deg
    east = settings.domain.source_lon + half_span_lon_deg
    south = settings.domain.source_lat - half_span_lat_deg
    north = settings.domain.source_lat + half_span_lat_deg
    return [north, west, south, east]


def _run_quietly(operation):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return operation()


def _resolve_cds_credentials_path() -> Path | None:
    candidates: list[Path] = []
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.append(Path(user_profile) / ".cdsapirc")
    candidates.append(Path.home() / ".cdsapirc")
    users_root = Path(os.environ.get("SystemDrive", "C:") + "\\") / "Users"
    if users_root.exists():
        try:
            candidates.extend(sorted(users_root.glob("*/.cdsapirc")))
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_cds_credentials() -> tuple[str, str]:
    cds_config = _resolve_cds_credentials_path()
    if cds_config is None:
        raise RuntimeError("CDS credentials file is missing in USERPROFILE or HOME as .cdsapirc.")
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


def _maybe_unpack_zip(path: Path) -> None:
    if not path.exists():
        return
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
        if signature != b"PK\x03\x04":
            return
        with zipfile.ZipFile(path, "r") as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if not members:
                raise RuntimeError(f"ZIP archive is empty: {path}")
            preferred = max(members, key=lambda name: archive.getinfo(name).file_size)
            payload = archive.read(preferred)
        path.write_bytes(payload)
    except zipfile.BadZipFile:
        return


def _resolve_request_window(settings: ProjectConfig, demo: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_available = (now_utc - pd.Timedelta(days=settings.inventory.era5_latency_days)).tz_localize(None)
    requested_start = pd.Timestamp(settings.timeline.incident_start_utc)
    requested_end = requested_start + pd.Timedelta(hours=settings.timeline.simulation_duration_hours)

    if requested_start > latest_available:
        end = latest_available.floor("D")
        days = 2 if demo else 7
        start = (end - pd.Timedelta(days=days - 1)).floor("D")
        return start, end

    end = min(requested_end, latest_available).floor("D")
    start = requested_start.floor("D")
    if demo:
        end = min(end, start + pd.Timedelta(days=1))
    if end < start:
        end = start
    return start, end


def _resolve_model_level_window(settings: ProjectConfig, demo: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_final = (now_utc - pd.Timedelta(days=settings.inventory.era5_complete_final_lag_days)).tz_localize(None)
    days = 2 if demo else 7
    end = latest_final.floor("D")
    start = (end - pd.Timedelta(days=days - 1)).floor("D")
    return start, end


def _dataset_time_bounds(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists():
        return None, None
    ds = None
    try:
        ds = xr.open_dataset(path, engine="scipy", cache=False)
        ds = _rename_fields(ds)
        ds = _squeeze_singleton_dims(ds)
        ds = _build_time_coord(ds)
        ds = _sort_coords(ds)
        if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
            return None, None
        return pd.Timestamp(ds.time.values[0]), pd.Timestamp(ds.time.values[-1])
    except Exception:
        return None, None
    finally:
        if ds is not None:
            ds.close()


def _dedupe_time_keep_last(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        return ds
    time_values = np.asarray(ds.time.values)
    _, reverse_index = np.unique(time_values[::-1], return_index=True)
    keep = np.sort(len(time_values) - 1 - reverse_index)
    return ds.isel(time=keep).sortby("time")


def _append_or_write_dataset(path: Path, new_ds: xr.Dataset) -> None:
    new_ds = _dedupe_time_keep_last(new_ds)
    if path.exists():
        existing = xr.open_dataset(path, engine="scipy").load()
        try:
            combined = xr.concat([existing, new_ds], dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
            combined = _dedupe_time_keep_last(combined)
        finally:
            existing.close()
    else:
        combined = new_ds
    if path.exists():
        path.unlink()
    combined.to_netcdf(path, engine="scipy")


def _convert_era5_model_grib_to_netcdf(grib_path: Path, output_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(str(grib_path), backend_kwargs={"indexpath": ""})
        merged = xr.merge(datasets, compat="override", join="outer")
    rename_map = {}
    if "hybrid" in merged.coords:
        rename_map["hybrid"] = "model_level"
    if "level" in merged.coords:
        rename_map["level"] = "model_level"
    if rename_map:
        merged = merged.rename(rename_map)
    merged = merged.load()
    if output_path.exists():
        output_path.unlink()
    merged.to_netcdf(output_path, engine="scipy")
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass


def main(settings: ProjectConfig = SETTINGS, *, demo: bool = False) -> None:
    ensure_runtime_dirs()
    cds_url, cds_key = _load_cds_credentials()
    client = cdsapi.Client(url=cds_url, key=cds_key, progress=False, quiet=True)

    request_start, request_end = _resolve_request_window(settings, demo)
    request_days = pd.date_range(start=request_start, end=request_end, freq="D")
    area = build_area(settings)
    out_file = settings.inventory.historical_actual.data_file
    pressure_file = settings.inventory.historical_actual.pressure_level_file
    model_level_file = settings.inventory.historical_actual.model_level_file

    if demo:
        pressure_levels = settings.inventory.transport_pressure_levels_hpa[:12]
        model_levels = settings.inventory.download_demo_model_levels
    else:
        pressure_levels = settings.inventory.transport_pressure_levels_hpa
        model_levels = settings.inventory.transport_model_levels

    _, surface_end = _dataset_time_bounds(out_file)
    if surface_end is not None and surface_end >= request_end - pd.Timedelta(hours=1):
        print("Reusing existing ERA5 single-level NetCDF:", out_file)
    else:
        surface_request_start = request_start if surface_end is None else max(request_start, surface_end.floor("D"))
        surface_request_days = pd.date_range(start=surface_request_start, end=request_end, freq="D")
        temp_surface_file = out_file.with_name(f".{out_file.stem}_refresh.nc")
        if temp_surface_file.exists():
            temp_surface_file.unlink()
        print("Downloading ERA5 hourly single levels...")
        print("Area:", area)
        print("Period:", surface_request_days[0].date(), "->", surface_request_days[-1].date())
        print("Output:", out_file)

        run_with_tqdm_heartbeat(
            "ERA5 single-level download",
            lambda: _run_quietly(
                lambda: client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "2m_temperature",
                            "boundary_layer_height",
                            "total_precipitation",
                        ],
                        "year": sorted({f"{timestamp.year:04d}" for timestamp in surface_request_days}),
                        "month": sorted({f"{timestamp.month:02d}" for timestamp in surface_request_days}),
                        "day": sorted({f"{timestamp.day:02d}" for timestamp in surface_request_days}),
                        "time": [f"{hour:02d}:00" for hour in range(24)],
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                        "area": area,
                    },
                    str(temp_surface_file),
                )
            ),
        )
        _maybe_unpack_zip(temp_surface_file)
        canonicalize_netcdf_to_scipy(temp_surface_file)
        surface_ds = xr.open_dataset(temp_surface_file, engine="scipy").load()
        try:
            _append_or_write_dataset(out_file, surface_ds)
        finally:
            surface_ds.close()
            temp_surface_file.unlink(missing_ok=True)

    if pressure_file is not None:
        _, pressure_end = _dataset_time_bounds(pressure_file)
        if pressure_end is not None and pressure_end >= request_end - pd.Timedelta(hours=1):
            print("Reusing existing ERA5 pressure NetCDF:", pressure_file)
        else:
            pressure_request_start = request_start if pressure_end is None else max(request_start, pressure_end.floor("D"))
            pressure_request_days = pd.date_range(start=pressure_request_start, end=request_end, freq="D")
            temp_pressure_file = pressure_file.with_name(f".{pressure_file.stem}_refresh.nc")
            if temp_pressure_file.exists():
                temp_pressure_file.unlink()
            print("Downloading ERA5 hourly pressure levels...")
            print("Pressure levels:", pressure_levels)
            print("Output:", pressure_file)
            run_with_tqdm_heartbeat(
                "ERA5 pressure-level download",
                lambda: _run_quietly(
                    lambda: client.retrieve(
                        "reanalysis-era5-pressure-levels",
                        {
                            "product_type": "reanalysis",
                            "variable": [
                                "u_component_of_wind",
                                "v_component_of_wind",
                                "vertical_velocity",
                            ],
                            "pressure_level": [str(level) for level in pressure_levels],
                            "year": sorted({f"{timestamp.year:04d}" for timestamp in pressure_request_days}),
                            "month": sorted({f"{timestamp.month:02d}" for timestamp in pressure_request_days}),
                            "day": sorted({f"{timestamp.day:02d}" for timestamp in pressure_request_days}),
                            "time": [f"{hour:02d}:00" for hour in range(24)],
                            "data_format": "netcdf",
                            "download_format": "unarchived",
                            "area": area,
                        },
                        str(temp_pressure_file),
                    )
                ),
            )
            _maybe_unpack_zip(temp_pressure_file)
            canonicalize_netcdf_to_scipy(temp_pressure_file)
            pressure_ds = xr.open_dataset(temp_pressure_file, engine="scipy").load()
            try:
                _append_or_write_dataset(pressure_file, pressure_ds)
            finally:
                pressure_ds.close()
                temp_pressure_file.unlink(missing_ok=True)

    if model_level_file is not None:
        model_start, model_end = _resolve_model_level_window(settings, demo)
        model_days = pd.date_range(start=model_start, end=model_end, freq="D")
        temp_model_grib = model_level_file.with_suffix(".grib")
        _, model_existing_end = _dataset_time_bounds(model_level_file)
        if model_existing_end is not None and model_existing_end >= model_end - pd.Timedelta(hours=1):
            print("Reusing existing ERA5 model-level NetCDF:", model_level_file)
        else:
            print("Downloading ERA5 model levels...")
            print("Model levels:", model_levels[0], "->", model_levels[-1], f"({len(model_levels)} levels)")
            print("Model-level period:", model_days[0].date(), "->", model_days[-1].date())
            print("Output:", model_level_file)
            try:
                run_with_tqdm_heartbeat(
                    "ERA5 model-level download",
                    lambda: _run_quietly(
                        lambda: client.retrieve(
                            "reanalysis-era5-complete",
                            {
                                "class": "ea",
                                "expver": "1",
                                "stream": "oper",
                                "type": "an",
                                "levtype": "ml",
                                "levelist": "/".join(str(level) for level in model_levels),
                                "param": ERA5_MODEL_LEVEL_PARAMS + "/129/152",
                                "date": "/".join(timestamp.strftime("%Y-%m-%d") for timestamp in model_days),
                                "time": "00/to/23/by/1",
                                "area": area,
                                "grid": "0.25/0.25",
                                "format": "grib",
                            },
                            str(temp_model_grib),
                        )
                    ),
                )
                print("Converting ERA5 model-level GRIB to NetCDF...")
                run_with_tqdm_heartbeat(
                    "ERA5 model-level conversion",
                    lambda: _convert_era5_model_grib_to_netcdf(temp_model_grib, model_level_file),
                )
            except Exception as exc:
                print("Warning: ERA5 model-level download failed; continuing without it.")
                print(exc)

    print("Done:", out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 data for the project domain.")
    parser.add_argument("--demo", action="store_true", help="Short download demo instead of the full window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(demo=args.demo)

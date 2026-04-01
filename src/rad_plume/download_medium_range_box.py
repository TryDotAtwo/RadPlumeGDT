from __future__ import annotations

import argparse
import contextlib
import io
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

from .config import ProjectConfig, SETTINGS, ensure_runtime_dirs
from .download_progress import run_with_tqdm_heartbeat


warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")


PARAMETERS = ["10u", "10v", "2t"]
PRESSURE_PARAMETERS = ["u", "v", "w"]
MODEL_LEVEL_PARAMETERS = ["u", "v", "t", "q"]


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


def _existing_dataset_covers(path: Path, cycle: pd.Timestamp, required_end: pd.Timestamp) -> bool:
    init_time, last_time = _dataset_metadata(path)
    return init_time == cycle.tz_localize(None) and last_time is not None and last_time >= required_end - pd.Timedelta(hours=1)


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


def _run_quietly(operation):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return operation()


def build_steps(horizon_days: int) -> list[int]:
    max_hour = horizon_days * 24
    steps = list(range(0, min(max_hour, 144) + 1, 3))
    if max_hour > 144:
        steps.extend(range(150, min(max_hour, 360) + 1, 6))
    return steps


def _resolve_horizon_days(settings: ProjectConfig, cycle: pd.Timestamp, demo: bool) -> int:
    if not demo:
        return settings.inventory.medium_range_horizon_days
    incident_end = pd.Timestamp(settings.timeline.incident_start_utc) + pd.Timedelta(hours=settings.timeline.demo_hours)
    required_days = int(np.ceil(max((incident_end - cycle).total_seconds(), 0.0) / 86_400.0))
    return max(2, min(settings.inventory.medium_range_horizon_days, required_days))


def build_extent(settings: ProjectConfig = SETTINGS) -> tuple[float, float, float, float]:
    half_span_lat_deg = settings.domain.domain_radius_km / 111.0
    lon_scale = max(111.0 * np.cos(np.deg2rad(settings.domain.source_lat)), 1e-6)
    half_span_lon_deg = settings.domain.domain_radius_km / lon_scale
    west = settings.domain.source_lon - half_span_lon_deg
    east = settings.domain.source_lon + half_span_lon_deg
    south = settings.domain.source_lat - half_span_lat_deg
    north = settings.domain.source_lat + half_span_lat_deg
    return west, east, south, north


def choose_cycle(client: Client, now_utc: pd.Timestamp) -> pd.Timestamp:
    preferred_hours = [12, 0] if now_utc.hour >= 12 else [0, 12]
    for cycle_hour in preferred_hours:
        try:
            latest = _run_quietly(
                lambda: client.latest(
                    {
                        "time": cycle_hour,
                        "stream": "oper",
                        "type": "fc",
                        "step": 0,
                        "param": "10u",
                    }
                )
            )
            return pd.Timestamp(latest)
        except Exception:
            continue
    raise RuntimeError("Unable to establish the latest ECMWF medium-range cycle.")


def choose_client_and_cycle(settings: ProjectConfig, now_utc: pd.Timestamp) -> tuple[Client, pd.Timestamp, str]:
    last_error: Exception | None = None
    for source in settings.inventory.medium_range_open_data_sources:
        client = _run_quietly(lambda: Client(source=source, model="ifs", resol="0p25"))
        try:
            cycle = choose_cycle(client, now_utc)
            return client, cycle, source
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Unable to establish the latest ECMWF medium-range cycle from any configured open-data source.") from last_error


def _collapse_forecast_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" not in ds.coords:
        return ds

    valid_time = pd.to_datetime(ds["valid_time"].values)
    valid_dims = ds["valid_time"].dims

    if valid_time.ndim == 1:
        source_dim = valid_dims[0]
        ds = ds.assign_coords(time=(source_dim, valid_time))
        return ds.swap_dims({source_dim: "time"})

    if valid_time.ndim == 2 and valid_dims == ("time", "step") and ds.sizes.get("time", 0) == 1:
        ds = ds.isel(time=0, drop=True)
        ds = ds.assign_coords(time=("step", valid_time.reshape(-1)))
        ds = ds.swap_dims({"step": "time"})
        return ds

    if valid_time.ndim == 2:
        stacked = ds.stack(forecast_point=valid_dims)
        stacked = stacked.assign_coords(valid_flat=("forecast_point", valid_time.reshape(-1)))
        stacked = stacked.swap_dims({"forecast_point": "valid_flat"}).rename({"valid_flat": "time"})
        return stacked

    return ds


def _subset_region(ds: xr.Dataset, extent: tuple[float, float, float, float]) -> xr.Dataset:
    west, east, south, north = extent

    if "longitude" in ds.coords:
        lon_values = ds.longitude.values.astype(float)
        if np.nanmax(lon_values) > 180.0:
            wrapped = ((lon_values + 180.0) % 360.0) - 180.0
            ds = ds.assign_coords(longitude=wrapped)

    ds = ds.sortby("longitude").sortby("latitude")
    ds = ds.sel(longitude=slice(west, east), latitude=slice(south, north))
    return ds


def _open_grib_dataset(path: Path, extent: tuple[float, float, float, float]) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(
            str(path),
            backend_kwargs={"indexpath": ""},
        )
        merged = xr.merge(datasets, compat="override", join="outer")
    merged = _collapse_forecast_time(merged)
    keep = [name for name in ("u10", "v10", "t2m") if name in merged.data_vars]
    if not keep:
        raise RuntimeError("ECMWF medium-range conversion produced no expected variables (u10, v10, t2m).")
    merged = _subset_region(merged[keep], extent).load()
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass
    return merged


def _open_pressure_grib_dataset(path: Path, extent: tuple[float, float, float, float]) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(
            str(path),
            backend_kwargs={"indexpath": ""},
        )
        merged = xr.merge(datasets, compat="override", join="outer")
    merged = _collapse_forecast_time(merged)
    keep = [name for name in ("u", "v", "w") if name in merged.data_vars]
    if not keep:
        raise RuntimeError("ECMWF medium-range pressure conversion produced no expected variables (u, v, w).")
    merged = _subset_region(merged[keep], extent).load()
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass
    return merged


def _open_model_grib_dataset(path: Path, extent: tuple[float, float, float, float]) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(
            str(path),
            backend_kwargs={"indexpath": ""},
        )
        merged = xr.merge(datasets, compat="override", join="outer")
    merged = _collapse_forecast_time(merged)
    keep = [name for name in ("u", "v", "t", "q") if name in merged.data_vars]
    if not keep:
        raise RuntimeError("ECMWF medium-range model-level conversion produced no expected variables (u, v, t, q).")
    merged = _subset_region(merged[keep], extent).load()
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass
    return merged


def main(settings: ProjectConfig = SETTINGS, *, demo: bool = False) -> None:
    ensure_runtime_dirs()
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    out_file = settings.inventory.future_medium_range.data_file
    pressure_file = settings.inventory.future_medium_range.pressure_level_file
    model_level_file = settings.inventory.future_medium_range.model_level_file
    temp_grib = out_file.with_suffix(".grib2")
    temp_pressure_grib = pressure_file.with_suffix(".grib2") if pressure_file is not None else None
    temp_model_grib = model_level_file.with_suffix(".grib2") if model_level_file is not None else None
    out_file.parent.mkdir(parents=True, exist_ok=True)

    client, cycle, selected_source = choose_client_and_cycle(settings, now_utc)
    horizon_days = _resolve_horizon_days(settings, cycle, demo)
    steps = build_steps(horizon_days)
    extent = build_extent(settings)
    pressure_levels = settings.inventory.transport_pressure_levels_hpa[:12] if demo else settings.inventory.transport_pressure_levels_hpa
    model_levels = settings.inventory.download_demo_model_levels if demo else settings.inventory.transport_model_levels
    required_end = cycle.tz_localize(None) + pd.Timedelta(days=horizon_days)

    existing_init, existing_end = _dataset_metadata(out_file)
    surface_missing_steps = steps
    if existing_init == cycle.tz_localize(None) and existing_end is not None:
        surface_missing_steps = [step for step in steps if cycle.tz_localize(None) + pd.Timedelta(hours=step) > existing_end + pd.Timedelta(hours=1)]

    if _existing_dataset_covers(out_file, cycle, required_end):
        print("Reusing existing medium-range NetCDF:", out_file)
    else:
        print("Downloading ECMWF medium-range forecast...")
        print("Open-data source:", selected_source)
        print("Cycle:", cycle.isoformat())
        print("Horizon days:", horizon_days)
        print("Forecast steps:", f"{steps[0]}h -> {steps[-1]}h", f"({len(steps)} steps)")
        print("Output GRIB:", temp_grib)
        print("Output NetCDF:", out_file)

        if not surface_missing_steps:
            print("No missing medium-range surface steps; keeping existing NetCDF:", out_file)
        else:
            run_with_tqdm_heartbeat(
                "Medium-range single-level download",
                lambda: _run_quietly(
                    lambda: client.retrieve(
                        request={
                            "date": cycle.strftime("%Y%m%d"),
                            "time": int(cycle.hour),
                            "stream": "oper",
                            "type": "fc",
                            "step": surface_missing_steps,
                            "param": PARAMETERS,
                        },
                        target=str(temp_grib),
                    )
                ),
            )

            def _convert_surface() -> None:
                def _operation() -> None:
                    ds = _open_grib_dataset(temp_grib, extent)
                    ds = ds.assign_coords(forecast_reference_time=pd.Timestamp(cycle).to_datetime64())
                    _append_or_write_dataset(out_file, ds)

                _run_quietly(_operation)

            print("Converting medium-range single levels to NetCDF...")
            run_with_tqdm_heartbeat("Medium-range surface conversion", _convert_surface)
            temp_grib.unlink(missing_ok=True)

    if pressure_file is not None and temp_pressure_grib is not None:
        pressure_init, pressure_end = _dataset_metadata(pressure_file)
        pressure_missing_steps = steps
        if pressure_init == cycle.tz_localize(None) and pressure_end is not None:
            pressure_missing_steps = [step for step in steps if cycle.tz_localize(None) + pd.Timedelta(hours=step) > pressure_end + pd.Timedelta(hours=1)]

        if _existing_dataset_covers(pressure_file, cycle, required_end):
            print("Reusing existing medium-range pressure NetCDF:", pressure_file)
        else:
            print("Downloading ECMWF medium-range pressure levels...")
            print("Pressure levels:", settings.inventory.transport_pressure_levels_hpa)
            print("Output GRIB:", temp_pressure_grib)
            print("Output NetCDF:", pressure_file)
            if not pressure_missing_steps:
                print("No missing medium-range pressure steps; keeping existing NetCDF:", pressure_file)
            else:
                run_with_tqdm_heartbeat(
                    "Medium-range pressure-level download",
                    lambda: _run_quietly(
                        lambda: client.retrieve(
                            request={
                                "date": cycle.strftime("%Y%m%d"),
                                "time": int(cycle.hour),
                                "stream": "oper",
                                "type": "fc",
                                "step": pressure_missing_steps,
                                "levtype": "pl",
                                "levelist": list(pressure_levels),
                                "param": PRESSURE_PARAMETERS,
                            },
                            target=str(temp_pressure_grib),
                        )
                    ),
                )

                def _convert_pressure() -> None:
                    def _operation() -> None:
                        pressure_ds = _open_pressure_grib_dataset(temp_pressure_grib, extent)
                        pressure_ds = pressure_ds.assign_coords(forecast_reference_time=pd.Timestamp(cycle).to_datetime64())
                        _append_or_write_dataset(pressure_file, pressure_ds)

                    _run_quietly(_operation)

                print("Converting medium-range pressure levels to NetCDF...")
                run_with_tqdm_heartbeat("Medium-range pressure conversion", _convert_pressure)
                temp_pressure_grib.unlink(missing_ok=True)

    if model_level_file is not None and temp_model_grib is not None:
        print("Downloading ECMWF medium-range model levels...")
        print("Model levels:", model_levels[0], "->", model_levels[-1], f"({len(model_levels)} levels)")
        print("Output GRIB:", temp_model_grib)
        print("Output NetCDF:", model_level_file)
        try:
            if not temp_model_grib.exists():
                run_with_tqdm_heartbeat(
                    "Medium-range model-level download",
                    lambda: _run_quietly(
                        lambda: client.retrieve(
                            request={
                                "date": cycle.strftime("%Y%m%d"),
                                "time": int(cycle.hour),
                                "stream": "oper",
                                "type": "fc",
                                "step": steps,
                                "levtype": "ml",
                                "levelist": list(model_levels),
                                "param": MODEL_LEVEL_PARAMETERS,
                            },
                            target=str(temp_model_grib),
                        )
                    ),
                )
            else:
                print("Reusing existing model-level GRIB download:", temp_model_grib)

            def _convert_model() -> None:
                def _operation() -> None:
                    model_ds = _open_model_grib_dataset(temp_model_grib, extent)
                    model_ds = model_ds.assign_coords(forecast_reference_time=pd.Timestamp(cycle).to_datetime64())
                    if model_level_file.exists():
                        model_level_file.unlink()
                    model_ds.to_netcdf(model_level_file, engine="scipy")

                _run_quietly(_operation)

            print("Converting medium-range model levels to NetCDF...")
            run_with_tqdm_heartbeat("Medium-range model-level conversion", _convert_model)
        except Exception as exc:
            print("Note: medium-range model levels are not available from the current open-data index; using dense pressure levels instead.")
            print(exc)

    print("Done:", out_file)
    print("Kept domain:", extent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECMWF medium-range data for the project domain.")
    parser.add_argument("--demo", action="store_true", help="Short download demo instead of the full horizon.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(demo=args.demo)

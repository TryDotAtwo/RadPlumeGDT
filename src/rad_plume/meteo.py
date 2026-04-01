from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .config import DataConfig, DomainConfig


VAR_RENAMES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "boundary_layer_height": "blh",
    "total_precipitation": "tp",
    "u_component_of_wind": "u_pl",
    "v_component_of_wind": "v_pl",
    "vertical_velocity": "omega_pl",
    "geopotential": "z_pl",
    "temperature": "t_pl",
    "specific_humidity": "q_pl",
    "isobaricInhPa": "pressure_level",
    "plev": "pressure_level",
    "lon": "longitude",
    "lat": "latitude",
}

METERS_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class PreparedMeteo:
    ds: xr.Dataset
    extent: tuple[float, float, float, float]
    native_step_minutes: int
    grid_spacing_m: float
    source_summary: str

    @property
    def time_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self.ds.time.values[0], self.ds.time.values[-1]


def _rename_fields(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {src: dst for src, dst in VAR_RENAMES.items() if src in ds.coords or src in ds.data_vars}
    if rename_map:
        ds = ds.rename(rename_map)
    has_pressure_level = "pressure_level" in ds.coords or "pressure_level" in ds.dims
    if has_pressure_level:
        pressure_var_renames = {
            "u": "u_pl",
            "v": "v_pl",
            "w": "omega_pl",
            "z": "z_pl",
            "t": "t_pl",
            "q": "q_pl",
        }
        rename_map = {src: dst for src, dst in pressure_var_renames.items() if src in ds.data_vars}
        if rename_map:
            ds = ds.rename(rename_map)
    return ds


def _open_dataset_with_fallback(path) -> xr.Dataset:
    path_obj = Path(path).resolve()
    short_path = _windows_short_path(path_obj)

    candidates: list[tuple[str, str | None]] = []
    if path_obj.suffix.lower() == ".nc":
        if short_path is not None:
            candidates.append((short_path, "netcdf4"))
        candidates.append((str(path_obj), "netcdf4"))
    if short_path is not None:
        candidates.append((short_path, None))
    candidates.append((str(path_obj), None))
    if path_obj.suffix.lower() == ".nc":
        if short_path is not None:
            candidates.append((short_path, "scipy"))
        candidates.append((str(path_obj), "scipy"))

    last_error: Exception | None = None
    tried: set[tuple[str, str | None]] = set()
    for candidate_path, engine in candidates:
        key = (candidate_path, engine)
        if key in tried:
            continue
        tried.add(key)
        try:
            open_kwargs = {"cache": False}
            if engine is not None:
                open_kwargs["engine"] = engine
            ds = xr.open_dataset(candidate_path, **open_kwargs)
            try:
                loaded = ds.load()
                ds.close()
                return loaded
            except Exception:
                ds.close()
                raise
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path_obj)


def _windows_short_path(path) -> str | None:
    if os.name != "nt":
        return None
    try:
        from ctypes import create_unicode_buffer, windll

        path_str = str(path)
        required = windll.kernel32.GetShortPathNameW(path_str, None, 0)
        if required <= 0:
            return None
        buffer = create_unicode_buffer(required)
        result = windll.kernel32.GetShortPathNameW(path_str, buffer, required)
        if result <= 0:
            return None
        return buffer.value or None
    except Exception:
        return None


def canonicalize_netcdf_to_scipy(path) -> None:
    path_obj = Path(path).resolve()
    if not path_obj.exists() or path_obj.suffix.lower() != ".nc":
        return

    with path_obj.open("rb") as handle:
        signature = handle.read(4)
    if signature.startswith(b"CDF"):
        return

    ds = _open_dataset_with_fallback(path_obj)
    temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    try:
        if temp_path.exists():
            temp_path.unlink()
        ds.to_netcdf(temp_path, engine="scipy")
        path_obj.unlink()
        temp_path.replace(path_obj)
    finally:
        ds.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _squeeze_singleton_dims(ds: xr.Dataset) -> xr.Dataset:
    dims_to_squeeze = [
        name for name, size in ds.sizes.items() if size == 1 and name not in {"latitude", "longitude", "pressure_level"}
    ]
    for dim_name in dims_to_squeeze:
        ds = ds.squeeze(dim_name, drop=True)
    return ds


def _build_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.coords:
        time_coord = ds["time"]
        if time_coord.ndim == 1 and time_coord.dims == ("time",):
            return ds

    if "valid_time" in ds.coords and ds["valid_time"].ndim == 1:
        valid_time = pd.to_datetime(ds["valid_time"].values)
        source_dim = ds["valid_time"].dims[0]
        ds = ds.assign_coords(time=(source_dim, valid_time))
        return ds.swap_dims({source_dim: "time"})

    if "forecast_period" in ds.coords and "forecast_reference_time" in ds.coords:
        reference = pd.to_datetime(np.asarray(ds["forecast_reference_time"].values).reshape(-1)[0])
        lead_raw = ds["forecast_period"].values
        if np.issubdtype(np.asarray(lead_raw).dtype, np.timedelta64):
            lead = pd.to_timedelta(lead_raw)
        else:
            lead = pd.to_timedelta(lead_raw, unit="h")

        time_values = reference + lead
        ds = ds.assign_coords(time=("forecast_period", time_values))
        return ds.swap_dims({"forecast_period": "time"})

    raise ValueError(
        "Не удалось построить координату time. Ожидались time, valid_time "
        "или forecast_reference_time + forecast_period."
    )


def _attach_ensemble_spread(raw_ds: xr.Dataset, collapsed_ds: xr.Dataset) -> xr.Dataset:
    if "number" not in raw_ds.dims:
        return collapsed_ds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        if "u10" in raw_ds.data_vars:
            collapsed_ds["u_spread"] = raw_ds["u10"].std(dim="number")
        if "v10" in raw_ds.data_vars:
            collapsed_ds["v_spread"] = raw_ds["v10"].std(dim="number")
        if "u_pl" in raw_ds.data_vars:
            collapsed_ds["u_pl_spread"] = raw_ds["u_pl"].std(dim="number")
        if "v_pl" in raw_ds.data_vars:
            collapsed_ds["v_pl_spread"] = raw_ds["v_pl"].std(dim="number")
    return collapsed_ds


def _collapse_ensemble(ds: xr.Dataset, data_config: DataConfig) -> tuple[xr.Dataset, str]:
    if "number" not in ds.dims:
        return ds, "deterministic field"

    raw_ds = ds
    mode = data_config.ensemble_mode.lower()
    count = ds.sizes["number"]
    if mode == "member":
        if not 0 <= data_config.ensemble_member < count:
            raise ValueError(
                f"ensemble_member={data_config.ensemble_member} вне диапазона 0..{count - 1}"
            )
        ds = ds.isel(number=data_config.ensemble_member)
        return _attach_ensemble_spread(raw_ds, ds), f"ensemble member #{data_config.ensemble_member}"

    if mode == "mean":
        ds = ds.mean(dim="number")
        return _attach_ensemble_spread(raw_ds, ds), f"ensemble mean ({count} members)"

    if mode == "median":
        ds = ds.median(dim="number")
        return _attach_ensemble_spread(raw_ds, ds), f"ensemble median ({count} members)"

    raise ValueError(f"Неизвестный ensemble_mode: {data_config.ensemble_mode}")


def _native_step_minutes(ds: xr.Dataset) -> int:
    if ds.sizes.get("time", 0) < 2:
        return 0
    diff_minutes = np.diff(ds.time.values).astype("timedelta64[m]").astype(int)
    return int(np.median(diff_minutes))


def _drop_auxiliary_coords(ds: xr.Dataset) -> xr.Dataset:
    keep_coords = {"time", "latitude", "longitude", "pressure_level", "forecast_reference_time"}
    drop_names = [name for name in ds.coords if name not in keep_coords and name not in ds.dims]
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")
    return ds


def _sort_coords(ds: xr.Dataset) -> xr.Dataset:
    for coord_name in ("time", "latitude", "longitude"):
        if coord_name in ds.coords and ds[coord_name].ndim == 1 and ds[coord_name].dims == (coord_name,):
            values = np.asarray(ds[coord_name].values)
            if values.size < 2:
                continue
            if values[0] <= values[-1] and np.all(values[:-1] <= values[1:]):
                continue
            if values[0] >= values[-1] and np.all(values[:-1] >= values[1:]):
                ds = ds.isel({coord_name: slice(None, None, -1)})
                continue
            order = np.argsort(values, kind="stable")
            ds = ds.isel({coord_name: order})
    return ds


def _drop_duplicate_coords(ds: xr.Dataset) -> xr.Dataset:
    for coord_name in ("time", "latitude", "longitude", "pressure_level"):
        if coord_name not in ds.coords:
            continue
        coord = ds[coord_name]
        if coord.ndim != 1 or coord.dims != (coord_name,):
            continue
        values = np.asarray(coord.values)
        if values.size < 2:
            continue
        _, unique_index = np.unique(values, return_index=True)
        if unique_index.size != values.size:
            ds = ds.isel({coord_name: np.sort(unique_index)})
    return ds


def _grid_spacing_m(ds: xr.Dataset, domain: DomainConfig) -> float:
    lat_values = ds.latitude.values.astype(float)
    lon_values = ds.longitude.values.astype(float)
    lat_spacing_deg = float(np.median(np.abs(np.diff(lat_values)))) if len(lat_values) > 1 else domain.grid_resolution_km / 111.32
    lon_spacing_deg = float(np.median(np.abs(np.diff(lon_values)))) if len(lon_values) > 1 else domain.grid_resolution_km / (111.32 * max(np.cos(np.deg2rad(domain.source_lat)), 1e-6))
    lat_spacing_m = lat_spacing_deg * METERS_PER_DEG_LAT
    lon_spacing_m = lon_spacing_deg * METERS_PER_DEG_LAT * max(np.cos(np.deg2rad(domain.source_lat)), 1e-6)
    return float(max(lat_spacing_m, lon_spacing_m))


def _compute_extent(ds: xr.Dataset, domain: DomainConfig) -> tuple[float, float, float, float]:
    lon_values = ds.longitude.values.astype(float)
    lat_values = ds.latitude.values.astype(float)
    radius_lat_deg = domain.domain_radius_km / 111.32
    radius_lon_deg = domain.domain_radius_km / (111.32 * max(np.cos(np.deg2rad(domain.source_lat)), 1e-6))
    requested_west = domain.source_lon - radius_lon_deg
    requested_east = domain.source_lon + radius_lon_deg
    requested_south = domain.source_lat - radius_lat_deg
    requested_north = domain.source_lat + radius_lat_deg

    west = max(float(lon_values.min()) - domain.map_padding_deg, requested_west)
    east = min(float(lon_values.max()) + domain.map_padding_deg, requested_east)
    south = max(float(lat_values.min()) - domain.map_padding_deg, requested_south)
    north = min(float(lat_values.max()) + domain.map_padding_deg, requested_north)
    return west, east, south, north


def _prepare_raw_dataset(path) -> xr.Dataset:
    ds = _open_dataset_with_fallback(path)
    ds = _rename_fields(ds)
    ds = _squeeze_singleton_dims(ds)
    ds = _build_time_coord(ds)
    ds = _sort_coords(ds)
    ds = _drop_duplicate_coords(ds)
    ds = _sort_coords(ds)
    ds = _drop_auxiliary_coords(ds)
    if "pressure_level" in ds.coords:
        ds = ds.sortby("pressure_level")
    return ds


def _merge_optional_pressure_levels(surface_ds: xr.Dataset, data_config: DataConfig) -> xr.Dataset:
    pressure_path = data_config.pressure_level_file
    if pressure_path is None or not pressure_path.exists():
        return surface_ds

    pressure_ds = _prepare_raw_dataset(pressure_path)
    merged = xr.merge([surface_ds, pressure_ds], compat="override", join="outer")
    if "time" in merged.coords and "u10" in merged.data_vars and "v10" in merged.data_vars:
        finite_u = np.isfinite(merged["u10"]).any(dim=("latitude", "longitude"))
        finite_v = np.isfinite(merged["v10"]).any(dim=("latitude", "longitude"))
        keep_time = finite_u | finite_v
        if keep_time.ndim == 1 and bool(keep_time.any()):
            merged = merged.sel(time=keep_time)
    return _sort_coords(merged)


def open_meteo_dataset(data_config: DataConfig, domain: DomainConfig) -> PreparedMeteo:
    if not data_config.data_file.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_config.data_file}")

    ds = _prepare_raw_dataset(data_config.data_file)
    ds = _merge_optional_pressure_levels(ds, data_config)
    ds, ensemble_summary = _collapse_ensemble(ds, data_config)
    ds = _sort_coords(ds)

    required = {"u10", "v10"}
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise ValueError(f"В датасете не хватает обязательных полей: {', '.join(missing)}")

    native_step = _native_step_minutes(ds)
    grid_spacing_m = _grid_spacing_m(ds, domain)
    extent = _compute_extent(ds, domain)
    pressure_note = ""
    if "pressure_level" in ds.coords and any(name in ds.data_vars for name in ("u_pl", "v_pl")):
        pressure_values = ds["pressure_level"].values.astype(float)
        pressure_note = (
            f"; pressure levels {int(np.nanmin(pressure_values))}..{int(np.nanmax(pressure_values))} hPa "
            f"({len(pressure_values)} levels)"
        )
    source_summary = (
        f"{data_config.source_label}; {ensemble_summary}; native step {native_step} min; "
        f"grid {ds.sizes['latitude']}x{ds.sizes['longitude']}{pressure_note}"
    )
    return PreparedMeteo(
        ds=ds,
        extent=extent,
        native_step_minutes=native_step,
        grid_spacing_m=grid_spacing_m,
        source_summary=source_summary,
    )


def _time_interp_chunk_size(ds: xr.Dataset) -> int:
    max_points_per_step = 1
    for var in ds.data_vars.values():
        time_size = var.sizes.get("time")
        if time_size is None or time_size <= 0:
            continue
        max_points_per_step = max(max_points_per_step, max(int(var.size // time_size), 1))
    target_bytes = 48 * 1024 * 1024
    chunk_size = int(target_bytes / max(max_points_per_step * 8, 1))
    return max(24, min(720, chunk_size))


def slice_and_interpolate_time(
    prepared: PreparedMeteo,
    start_utc: str,
    end_utc: str,
    step_minutes: int,
    keep_vars: tuple[str, ...] | None = None,
) -> PreparedMeteo:
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    available_start = pd.Timestamp(prepared.ds.time.values[0])
    available_end = pd.Timestamp(prepared.ds.time.values[-1])

    if start_ts < available_start or end_ts > available_end:
        raise ValueError(
            "Запрошенный интервал выходит за пределы данных: "
            f"{start_ts} .. {end_ts}, доступно {available_start} .. {available_end}"
        )

    ds = prepared.ds.sel(time=slice(start_ts, end_ts))
    if keep_vars is not None:
        selected_vars = [name for name in keep_vars if name in ds.data_vars]
        if not selected_vars:
            raise ValueError(f"Requested keep_vars are missing in dataset: {keep_vars}")
        ds = ds[selected_vars]
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"После фильтрации по интервалу {start_ts} .. {end_ts} нет ни одного шага.")

    if step_minutes <= 0:
        raise ValueError("step_minutes должен быть > 0")

    full_time_index = pd.date_range(start=start_ts, end=end_ts, freq=f"{step_minutes}min")
    if len(full_time_index) == 0:
        raise ValueError("Пустой временной индекс после интерполяции.")

    if len(full_time_index) == ds.sizes["time"] and np.array_equal(full_time_index.values, ds.time.values):
        interpolated = ds
    else:
        # Keep scipy interpolation bounded by the largest per-timestep variable slab.
        chunk_size = _time_interp_chunk_size(ds)
        if len(full_time_index) <= chunk_size:
            interpolated = ds.interp(time=full_time_index, method="linear", assume_sorted=True)
        else:
            chunks: list[xr.Dataset] = []
            for start in range(0, len(full_time_index), chunk_size):
                stop = min(start + chunk_size, len(full_time_index))
                chunk_index = full_time_index[start:stop]
                chunk_ds = ds.interp(time=chunk_index, method="linear", assume_sorted=True)
                chunks.append(chunk_ds)
            interpolated = xr.concat(chunks, dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
            _, unique_index = np.unique(interpolated.time.values, return_index=True)
            interpolated = interpolated.isel(time=np.sort(unique_index))

    float_cast_map = {
        name: np.float32
        for name, var in interpolated.data_vars.items()
        if np.issubdtype(var.dtype, np.floating) and var.dtype != np.float32
    }
    if float_cast_map:
        interpolated = interpolated.astype(float_cast_map)

    native_step = prepared.native_step_minutes
    if "source_native_step_minutes" in interpolated.data_vars:
        values = interpolated["source_native_step_minutes"].values
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            native_step = int(np.nanmax(finite))

    grid_spacing_m = prepared.grid_spacing_m
    if "source_grid_spacing_m" in interpolated.data_vars:
        values = interpolated["source_grid_spacing_m"].values
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            grid_spacing_m = float(np.nanmax(finite))

    summary = prepared.source_summary + f"; interpolated to {step_minutes} min"
    return PreparedMeteo(
        ds=interpolated,
        extent=prepared.extent,
        native_step_minutes=native_step,
        grid_spacing_m=grid_spacing_m,
        source_summary=summary,
    )


def iter_interpolated_time_chunks(
    prepared: PreparedMeteo,
    start_utc: str,
    end_utc: str,
    step_minutes: int,
    chunk_hours: int,
    keep_vars: tuple[str, ...] | None = None,
):
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    if chunk_hours <= 0:
        raise ValueError("chunk_hours должен быть > 0")

    full_time_index = pd.date_range(start=start_ts, end=end_ts, freq=f"{step_minutes}min")
    if len(full_time_index) == 0:
        raise ValueError("Пустой временной индекс для чанков интерполяции.")

    points_per_chunk = max(int((chunk_hours * 60) / step_minutes), 2)
    available_start = pd.Timestamp(prepared.ds.time.values[0])
    available_end = pd.Timestamp(prepared.ds.time.values[-1])
    start_index = 0
    while start_index < len(full_time_index):
        end_index = min(start_index + points_per_chunk, len(full_time_index))
        if end_index - start_index < 2 and start_index > 0:
            start_index = max(start_index - 1, 0)
            end_index = len(full_time_index)
        chunk_start = full_time_index[start_index]
        chunk_end = full_time_index[end_index - 1]
        if chunk_start > available_end:
            break
        if chunk_end < available_start:
            start_index = end_index - 1 if end_index < len(full_time_index) else end_index
            continue
        chunk_start = max(chunk_start, available_start)
        chunk_end = min(chunk_end, available_end)
        if chunk_start >= chunk_end:
            break
        try:
            yield slice_and_interpolate_time(
                prepared=prepared,
                start_utc=chunk_start.isoformat(),
                end_utc=chunk_end.isoformat(),
                step_minutes=step_minutes,
                keep_vars=keep_vars,
            )
        except ValueError as exc:
            if "нет ни одного шага" in str(exc):
                start_index = end_index - 1 if end_index < len(full_time_index) else end_index
                continue
            raise
        if end_index >= len(full_time_index):
            break
        start_index = end_index - 1

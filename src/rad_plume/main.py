from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import threading
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import MPLCONFIG_DIR, OUTPUT_DIR, SETTINGS, ensure_runtime_dirs
from .download_era5_box import main as download_era5_main
from .download_medium_range_box import main as download_medium_range_main
from .download_seasonal_box import main as download_seasonal_main


ensure_runtime_dirs()
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    module=r"scipy\.interpolate\._interpolate",
)

from .data_manager import (
    BLEND_COMPONENT_PROFILES,
    inspect_data_status,
    materialize_best_available_blend,
    open_best_available_blend,
    refresh_profile_data,
    resolve_data_config,
    RUNTIME_BLEND_PROFILE_PREFIX,
    write_data_quality_report,
)
from .dispersion import build_simulation_window, run_dispersion_aggregate
from .hazard_analysis import compute_scenario_hazard_map
from .meteo import open_meteo_dataset, slice_and_interpolate_time
from .rendering import (
    render_hazard_ground_dose_map,
    render_hazard_probability_map,
    render_plume_animation,
    render_plume_animation_streaming,
    render_summary_ground_dose_map,
    render_summary_map,
    render_wind_animation,
)


FULL_RESEARCH_HORIZON_DAYS = 180
FULL_HAZARD_HORIZON_DAYS = 45
DEMO_HAZARD_HORIZON_DAYS = 5


def _analysis_anchor_utc() -> pd.Timestamp:
    local_now = datetime.now().astimezone()
    return pd.Timestamp(year=local_now.year, month=local_now.month, day=local_now.day)


def _naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _required_window_end(settings) -> pd.Timestamp:
    anchor = pd.Timestamp(settings.timeline.incident_start_utc)
    if settings.timeline.demo_mode:
        return anchor + pd.Timedelta(hours=settings.timeline.demo_hours)
    return anchor + pd.Timedelta(days=FULL_RESEARCH_HORIZON_DAYS)


def _latest_expected_seasonal_release(settings) -> pd.Timestamp:
    now_utc = pd.Timestamp.now(tz="UTC")
    release_candidate = pd.Timestamp(
        year=now_utc.year,
        month=now_utc.month,
        day=settings.inventory.seasonal_release_day_utc,
        hour=settings.inventory.seasonal_release_hour_utc,
        tz="UTC",
    )
    if now_utc >= release_candidate:
        release_month = pd.Timestamp(year=now_utc.year, month=now_utc.month, day=1, tz="UTC")
    else:
        previous = now_utc - pd.DateOffset(months=1)
        release_month = pd.Timestamp(year=previous.year, month=previous.month, day=1, tz="UTC")
    return release_month.tz_localize(None)


def _required_profile_end(settings, profile: str, status=None) -> pd.Timestamp | None:
    anchor = pd.Timestamp(settings.timeline.incident_start_utc)
    required_end = _required_window_end(settings)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)

    if profile == "historical_actual":
        latest_daily_cutoff = (now_utc - pd.Timedelta(days=settings.inventory.era5_latency_days)).floor("D")
        return min(anchor, latest_daily_cutoff)
    if profile == "future_medium_range":
        forecast_cap_end = anchor + pd.Timedelta(days=settings.inventory.medium_range_horizon_days)
        if status is not None and getattr(status, "init_time", None):
            forecast_cap_end = _naive_timestamp(status.init_time) + pd.Timedelta(days=settings.inventory.medium_range_horizon_days)
        return min(required_end, forecast_cap_end)
    if profile == "future_seasonal":
        medium_end = anchor + pd.Timedelta(days=settings.inventory.medium_range_horizon_days)
        if required_end <= medium_end:
            return None
        seasonal_cap_end = _latest_expected_seasonal_release(settings) + pd.Timedelta(days=180)
        return min(required_end, seasonal_cap_end)
    return None


def _needs_refresh_for_window(status, profile: str, settings) -> bool:
    required_end = _required_profile_end(settings, profile, status)
    if not status.exists or status.coverage_end is None:
        return True
    if required_end is None:
        return bool(status.stale)
    coverage_end = pd.Timestamp(status.coverage_end)
    if coverage_end >= required_end - pd.Timedelta(hours=1):
        return bool(status.stale)
    return True


def build_runtime_settings(args: argparse.Namespace):
    settings = SETTINGS
    anchor = _analysis_anchor_utc()
    timeline = replace(
        settings.timeline,
        incident_start_utc=anchor.isoformat(),
    )
    hazard = replace(
        settings.hazard,
        scenario_start_utc=anchor.isoformat(),
        scenario_end_utc=(anchor + pd.Timedelta(days=FULL_HAZARD_HORIZON_DAYS)).isoformat(),
    )
    settings = replace(settings, timeline=timeline, hazard=hazard)
    if args.lat is not None or args.lon is not None or args.radius_km is not None:
        domain = replace(
            settings.domain,
            source_lat=args.lat if args.lat is not None else settings.domain.source_lat,
            source_lon=args.lon if args.lon is not None else settings.domain.source_lon,
            domain_radius_km=args.radius_km if args.radius_km is not None else settings.domain.domain_radius_km,
        )
        settings = replace(settings, domain=domain)
    if args.demo:
        timeline = replace(
            settings.timeline,
            demo_mode=True,
            demo_hours=144,
            simulation_duration_hours=144,
            frame_step_minutes=settings.timeline.demo_frame_step_minutes,
        )
        hazard = replace(
            settings.hazard,
            scenario_start_utc=anchor.isoformat(),
            scenario_end_utc=(anchor + pd.Timedelta(days=DEMO_HAZARD_HORIZON_DAYS)).isoformat(),
        )
        settings = replace(settings, timeline=timeline, hazard=hazard)
    return settings


def create_run_output_dir() -> Path:
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = 2
    while True:
        candidate = OUTPUT_DIR / run_stamp if suffix == 2 else OUTPUT_DIR / f"{run_stamp}_{suffix - 1}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def _prepare_source(settings, source_profile: str | None, output_dir: Path):
    resolved_profile = source_profile or settings.inventory.default_source_profile
    report_path = write_data_quality_report(settings, output_dir / settings.inventory.quality_report_file.name)
    if resolved_profile.startswith(RUNTIME_BLEND_PROFILE_PREFIX):
        runtime_path = Path(resolved_profile.removeprefix(RUNTIME_BLEND_PROFILE_PREFIX))
        runtime_config = replace(
            settings.inventory.historical_actual,
            source_kind="runtime_blend",
            data_file=runtime_path,
            pressure_level_file=None,
            model_level_file=None,
            ensemble_mode="mean",
            source_label="Materialized best-available blend",
        )
        raw_meteo = open_meteo_dataset(runtime_config, settings.domain)
        return raw_meteo, "best_available_blend", report_path
    if resolved_profile == "best_available_blend":
        if settings.inventory.auto_refresh_if_stale:
            for component_profile in BLEND_COMPONENT_PROFILES:
                data_config = getattr(settings.inventory, component_profile)
                status = inspect_data_status(data_config, component_profile, settings.inventory)
                if not _needs_refresh_for_window(status, component_profile, settings):
                    continue
                print(f"Warning: blend component '{component_profile}' needs refresh -> {data_config.data_file}")
                print(f"Attempting refresh for '{component_profile}'...")
                try:
                    refresh_profile_data(component_profile, settings)
                except Exception as exc:
                    print(f"Refresh skipped for '{component_profile}'. Check credentials and network access.")
                    print(exc)
            report_path = write_data_quality_report(settings, output_dir / settings.inventory.quality_report_file.name)
        raw_meteo = open_best_available_blend(settings)
        return raw_meteo, resolved_profile, report_path

    resolved_profile, data_config = resolve_data_config(settings, source_profile)
    status = inspect_data_status(data_config, resolved_profile, settings.inventory)
    if _needs_refresh_for_window(status, resolved_profile, settings):
        print(f"Warning: active dataset '{resolved_profile}' needs refresh -> {data_config.data_file}")
        if settings.inventory.auto_refresh_if_stale:
            print(f"Attempting refresh for '{resolved_profile}'...")
            try:
                refresh_profile_data(resolved_profile, settings)
            except Exception as exc:
                print(exc)
                raise RuntimeError(
                    f"Не удалось автоматически обновить профиль '{resolved_profile}'. "
                    "Проверьте CDS credentials и сетевой доступ."
                ) from exc
            report_path = write_data_quality_report(settings, output_dir / settings.inventory.quality_report_file.name)
            status = inspect_data_status(data_config, resolved_profile, settings.inventory)

    raw_meteo = open_meteo_dataset(data_config, settings.domain)
    return raw_meteo, resolved_profile, report_path


def _prepare_meteo(settings, source_profile: str | None, output_dir: Path):
    raw_meteo, resolved_profile, report_path = _prepare_source(settings, source_profile, output_dir)
    timeline = settings.timeline
    if not timeline.demo_mode:
        available_end = pd.Timestamp(raw_meteo.ds.time.values[-1])
        incident_start = pd.Timestamp(timeline.incident_start_utc)
        simulation_hours = max(int((available_end - incident_start) / pd.Timedelta(hours=1)), 1)
        timeline = replace(timeline, simulation_duration_hours=simulation_hours)

    window = build_simulation_window(timeline)
    frame_meteo = slice_and_interpolate_time(
        raw_meteo,
        window.incident_start.isoformat(),
        window.simulation_end.isoformat(),
        timeline.frame_step_minutes,
        keep_vars=("u10", "v10"),
    )
    model_frame_count = len(pd.date_range(start=window.incident_start, end=window.simulation_end, freq=f"{timeline.model_step_minutes}min"))
    return raw_meteo, frame_meteo, window, resolved_profile, report_path, timeline, model_frame_count


def _prepare_wind_meteo(settings, source_profile: str | None, output_dir: Path):
    raw_meteo, resolved_profile, report_path = _prepare_source(settings, source_profile, output_dir)
    raw_start, raw_end = raw_meteo.time_range
    wind_start = max(pd.Timestamp(settings.timeline.incident_start_utc), pd.Timestamp(raw_start))
    if settings.timeline.demo_mode:
        wind_end = min(pd.Timestamp(raw_end), wind_start + pd.Timedelta(hours=settings.timeline.demo_hours))
    else:
        wind_end = pd.Timestamp(raw_end)
    if wind_end <= wind_start:
        wind_start = pd.Timestamp(raw_start)
        wind_end = pd.Timestamp(raw_end)
    frame_count = len(pd.date_range(start=wind_start, end=wind_end, freq=f"{settings.timeline.frame_step_minutes}min"))
    return raw_meteo, wind_start, wind_end, frame_count, resolved_profile, report_path


def _print_runtime_summary(raw_meteo, model_frame_count: int, frame_meteo, window, source_profile: str, report_path: Path, settings) -> None:
    raw_start, raw_end = raw_meteo.time_range
    print("Source profile:", source_profile)
    print("Data source:", raw_meteo.source_summary)
    print("Available time range:", raw_start, "->", raw_end)
    print("Incident window:", window.incident_start, "->", window.release_end)
    print("Simulation end:", window.simulation_end)
    print("Model step:", settings.timeline.model_step_minutes, "min")
    print("Frame step:", settings.timeline.frame_step_minutes, "min")
    print("Interpolated model frames:", model_frame_count)
    print("Animation frames:", frame_meteo.ds.sizes["time"])
    print("Data report:", report_path)


def run_wind_animation(settings, source_profile: str | None, output_dir: Path) -> Path:
    raw_meteo, wind_start, wind_end, frame_count, resolved_profile, report_path = _prepare_wind_meteo(settings, source_profile, output_dir)
    print("Source profile:", resolved_profile)
    print("Data source:", raw_meteo.source_summary)
    print("Available time range:", raw_meteo.time_range[0], "->", raw_meteo.time_range[-1])
    print("Wind animation window:", wind_start, "->", wind_end)
    print("Frame step:", settings.timeline.frame_step_minutes, "min")
    print("Animation frames:", frame_count)
    print("Data report:", report_path)
    output_path = output_dir / settings.visual.wind_output_mp4
    print("Rendering wind animation:", output_path)
    render_wind_animation(
        raw_meteo,
        settings.domain,
        settings.timeline,
        settings.visual,
        output_path,
        settings.dispersion,
        start_utc=wind_start.isoformat(),
        end_utc=wind_end.isoformat(),
    )
    return output_path


def _build_dispersion_aggregate(settings, source_profile: str | None, output_dir: Path):
    raw_meteo, frame_meteo, window, resolved_profile, report_path, timeline, model_frame_count = _prepare_meteo(settings, source_profile, output_dir)
    effective_settings = replace(settings, timeline=timeline)
    _print_runtime_summary(raw_meteo, model_frame_count, frame_meteo, window, resolved_profile, report_path, effective_settings)
    print("Running puff dispersion model...")
    aggregate = run_dispersion_aggregate(
        raw_meteo=raw_meteo,
        frame_meteo=frame_meteo,
        timeline=timeline,
        domain=settings.domain,
        dispersion=settings.dispersion,
    )
    return raw_meteo, frame_meteo, aggregate, effective_settings


def run_plume_animation_only(settings, source_profile: str | None, output_dir: Path) -> Path:
    raw_meteo, frame_meteo, aggregate, effective_settings = _build_dispersion_aggregate(settings, source_profile, output_dir)
    plume_path = output_dir / settings.visual.plume_output_mp4
    print("Rendering plume animation:", plume_path)
    render_plume_animation_streaming(
        raw_meteo,
        aggregate,
        frame_meteo,
        settings.domain,
        effective_settings.timeline,
        settings.hazard,
        settings.visual,
        plume_path,
        settings.dispersion,
    )
    return plume_path


def run_plume_products(settings, source_profile: str | None, output_dir: Path) -> tuple[Path, Path, Path]:
    raw_meteo, frame_meteo, aggregate, effective_settings = _build_dispersion_aggregate(settings, source_profile, output_dir)
    plume_path = output_dir / settings.visual.plume_output_mp4
    summary_path = output_dir / settings.visual.summary_output_png
    summary_ground_dose_path = output_dir / settings.visual.summary_ground_dose_output_png
    print("Rendering plume animation:", plume_path)
    render_plume_animation_streaming(
        raw_meteo,
        aggregate,
        frame_meteo,
        settings.domain,
        effective_settings.timeline,
        settings.hazard,
        settings.visual,
        plume_path,
        settings.dispersion,
    )
    print("Rendering summary map:", summary_path)
    render_summary_map(aggregate, settings.domain, settings.dispersion, settings.hazard, settings.visual, summary_path)
    print("Rendering summary ground dose map:", summary_ground_dose_path)
    render_summary_ground_dose_map(aggregate, settings.domain, settings.hazard, settings.visual, summary_ground_dose_path)
    return plume_path, summary_path, summary_ground_dose_path


def run_summary_map(settings, source_profile: str | None, output_dir: Path) -> Path:
    _, _, aggregate, _ = _build_dispersion_aggregate(settings, source_profile, output_dir)
    summary_path = output_dir / settings.visual.summary_output_png
    print("Rendering summary map:", summary_path)
    render_summary_map(aggregate, settings.domain, settings.dispersion, settings.hazard, settings.visual, summary_path)
    return summary_path


def run_hazard_map(settings, source_profile: str | None, output_dir: Path, external_jobs_running=None) -> tuple[Path, Path]:
    raw_meteo, resolved_profile, report_path = _prepare_source(settings, source_profile, output_dir)
    print("Source profile:", resolved_profile)
    print("Data source:", raw_meteo.source_summary)
    print("Available time range:", raw_meteo.time_range[0], "->", raw_meteo.time_range[-1])
    print("Hazard sweep window:", settings.hazard.scenario_start_utc, "->", settings.hazard.scenario_end_utc)
    print("Hazard model step:", settings.hazard.model_step_minutes, "min")
    print("Scenario spacing:", settings.hazard.scenario_step_hours, "h")
    print("Data report:", report_path)
    print("Running multi-date hazard sweep...")
    hazard_result = compute_scenario_hazard_map(raw_meteo, settings, external_jobs_running=external_jobs_running)
    hazard_path = output_dir / settings.visual.hazard_output_png
    hazard_ground_dose_path = output_dir / settings.visual.hazard_ground_dose_output_png
    print("Rendering hazard probability map:", hazard_path)
    render_hazard_probability_map(hazard_result, settings.domain, settings.hazard, settings.visual, hazard_path)
    print("Rendering hazard ground dose map:", hazard_ground_dose_path)
    render_hazard_ground_dose_map(hazard_result, settings.domain, settings.hazard, settings.visual, hazard_ground_dose_path)
    return hazard_path, hazard_ground_dose_path


def _run_target_worker(target: str, settings, source_profile: str | None, output_dir_str: str) -> str:
    output_dir = Path(output_dir_str)
    if target == "wind":
        return str(run_wind_animation(settings, source_profile, output_dir))
    if target == "plume":
        return str(run_plume_animation_only(settings, source_profile, output_dir))
    if target == "plume_products":
        plume_path, summary_path, summary_ground_dose_path = run_plume_products(settings, source_profile, output_dir)
        return f"{plume_path}\n{summary_path}\n{summary_ground_dose_path}"
    if target == "summary":
        return str(run_summary_map(settings, source_profile, output_dir))
    if target == "hazard":
        hazard_path, _ = run_hazard_map(settings, source_profile, output_dir)
        return str(hazard_path)
    raise ValueError(f"Unknown target for worker: {target}")


def _prepare_parallel_settings(settings):
    inventory = replace(settings.inventory, auto_refresh_if_stale=False)
    return replace(settings, inventory=inventory)


def run_all(settings, source_profile: str | None, output_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    print("Preflight source check before parallel rendering...")
    worker_source_profile = source_profile
    if (source_profile or settings.inventory.default_source_profile) == "best_available_blend":
        _prepare_source(settings, source_profile, output_dir)
        runtime_blend_path = output_dir / "best_available_blend_runtime.nc"
        print("Materializing best-available blend once for parallel workers:", runtime_blend_path)
        prepared = materialize_best_available_blend(settings, runtime_blend_path)
        print("Blend materialized:", prepared.time_range[0], "->", prepared.time_range[-1])
        worker_source_profile = f"{RUNTIME_BLEND_PROFILE_PREFIX}{runtime_blend_path}"
    else:
        _prepare_source(settings, source_profile, output_dir)
    worker_settings = _prepare_parallel_settings(settings)
    targets = ("wind", "plume_products")
    results: dict[str, Path] = {}
    max_workers = min(settings.visual.parallel_render_processes, len(targets))
    target_count = len(targets)
    completed_lock = threading.Lock()
    completed_targets = 0

    def _external_jobs_running() -> bool:
        with completed_lock:
            return completed_targets < target_count

    hazard_exception: Exception | None = None
    hazard_path: Path | None = None
    hazard_ground_dose_path: Path | None = None

    def _hazard_runner() -> None:
        nonlocal hazard_exception, hazard_path, hazard_ground_dose_path
        try:
            hazard_path, hazard_ground_dose_path = run_hazard_map(
                worker_settings,
                worker_source_profile,
                output_dir,
                external_jobs_running=_external_jobs_running,
            )
        except Exception as exc:  # pragma: no cover - propagated below
            hazard_exception = exc

    hazard_thread = threading.Thread(target=_hazard_runner, name="hazard-runner", daemon=True)
    hazard_thread.start()

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_target_worker, target, worker_settings, worker_source_profile, str(output_dir)): target
            for target in targets
        }
        for future in cf.as_completed(future_map):
            target = future_map[future]
            raw_result = future.result()
            if target == "plume_products":
                plume_raw, summary_raw, summary_ground_dose_raw = raw_result.splitlines()
                results["plume"] = Path(plume_raw)
                results["summary"] = Path(summary_raw)
                results["summary_ground_dose"] = Path(summary_ground_dose_raw)
                print("Completed plume:", results["plume"])
                print("Completed summary:", results["summary"])
                print("Completed summary ground dose:", results["summary_ground_dose"])
            else:
                output_path = Path(raw_result)
                results[target] = output_path
                print(f"Completed {target}:", output_path)
            with completed_lock:
                completed_targets += 1

    hazard_thread.join()
    if hazard_exception is not None:
        raise hazard_exception
    if hazard_path is None or hazard_ground_dose_path is None:
        raise RuntimeError("Hazard task did not produce an output path.")
    results["hazard"] = hazard_path
    results["hazard_ground_dose"] = hazard_ground_dose_path
    print("Completed hazard:", results["hazard"])
    print("Completed hazard ground dose:", results["hazard_ground_dose"])
    return (
        results["wind"],
        results["plume"],
        results["summary"],
        results["summary_ground_dose"],
        results["hazard"],
        results["hazard_ground_dose"],
    )


def run_download_bundle(settings) -> None:
    print("Downloading historical actual data...")
    download_era5_main(settings, demo=settings.timeline.demo_mode)
    print("Downloading medium-range forecast...")
    download_medium_range_main(settings, demo=settings.timeline.demo_mode)
    print("Downloading seasonal forecast...")
    download_seasonal_main(settings, demo=settings.timeline.demo_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather and plume visualization pipeline.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=("wind", "plume", "summary", "hazard", "all", "report", "download"),
        help="Что рендерить.",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=("best_available_blend", "historical_actual", "future_medium_range", "future_seasonal"),
        help="Какой профиль данных использовать.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Демо-режим: 6 суток расчета и кадровый шаг 1 час.",
    )
    parser.add_argument("--lat", type=float, default=None, help="Source latitude override.")
    parser.add_argument("--lon", type=float, default=None, help="Source longitude override.")
    parser.add_argument("--radius-km", type=float, default=None, help="Domain radius override in km.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_runtime_settings(args)
    output_dir = create_run_output_dir()
    print("Run output directory:", output_dir)

    if args.target == "report":
        report_path = write_data_quality_report(settings, output_dir / settings.inventory.quality_report_file.name)
        print("Done:", report_path)
        return

    if args.target == "download":
        run_download_bundle(settings)
        print("Done: downloads refreshed")
        return

    if args.target == "wind":
        run_wind_animation(settings, args.source, output_dir)
        return

    if args.target == "plume":
        plume_path, _, _ = run_plume_products(settings, args.source, output_dir)
        print("Done:", plume_path)
        return

    if args.target == "summary":
        summary_path = run_summary_map(settings, args.source, output_dir)
        print("Done:", summary_path)
        return

    if args.target == "hazard":
        hazard_path, hazard_ground_dose_path = run_hazard_map(settings, args.source, output_dir)
        print("Done:", hazard_path)
        print("Done:", hazard_ground_dose_path)
        return

    wind_path, plume_path, summary_path, summary_ground_dose_path, hazard_path, hazard_ground_dose_path = run_all(settings, args.source, output_dir)
    print("Done:", wind_path)
    print("Done:", plume_path)
    print("Done:", summary_path)
    print("Done:", summary_ground_dose_path)
    print("Done:", hazard_path)
    print("Done:", hazard_ground_dose_path)


if __name__ == "__main__":
    main()

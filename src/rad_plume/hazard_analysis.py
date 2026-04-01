from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, replace
import ctypes
import os
from pathlib import Path
import tempfile
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import HazardConfig, ProjectConfig
from .dispersion import FinalDepositionResult, SimulationGrid, build_simulation_window, run_dispersion_final_deposition
from .meteo import PreparedMeteo

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    module=r"scipy\.interpolate\._interpolate",
)


@dataclass(frozen=True)
class ScenarioHazardResult:
    grid: SimulationGrid
    hazard_probability: np.ndarray
    aggregate_relative_impact: np.ndarray
    safety_index: np.ndarray
    mean_deposition_bq_m2: np.ndarray
    max_deposition_bq_m2: np.ndarray
    mean_equivalent_dose_msv: np.ndarray
    max_equivalent_dose_msv: np.ndarray
    mean_ground_dose_msv: np.ndarray
    max_ground_dose_msv: np.ndarray
    dangerous_ground_dose_probability: np.ndarray
    hit_count: np.ndarray
    scenario_count: int
    threshold_note: str
    scenario_start_times: tuple[str, ...]


def _scenario_times(settings: ProjectConfig) -> list[pd.Timestamp]:
    start = pd.Timestamp(settings.hazard.scenario_start_utc)
    end = pd.Timestamp(settings.hazard.scenario_end_utc)
    if end < start:
        raise ValueError("hazard.scenario_end_utc должен быть не раньше hazard.scenario_start_utc")
    return list(pd.date_range(start=start, end=end, freq=f"{settings.hazard.scenario_step_hours}h"))


def _project_equivalent_dose_msv(field_bq_m2: np.ndarray, hazard: HazardConfig) -> np.ndarray:
    exposure_seconds = float(hazard.dose_integration_hours * 3600.0)
    dose_msv = np.zeros_like(field_bq_m2, dtype=np.float64)
    for isotope in hazard.isotope_mix:
        decay_constant = np.log(2.0) / (isotope.half_life_hours * 3600.0)
        integrated_seconds = (1.0 - np.exp(-decay_constant * exposure_seconds)) / decay_constant
        dose_msv += (
            field_bq_m2
            * isotope.deposition_fraction
            * isotope.ground_surface_dose_coeff_sv_per_bq_s_m2
            * integrated_seconds
            * 1000.0
        )
    return (dose_msv * float(hazard.ground_dose_multiplier)).astype(np.float32)


def project_total_effective_dose_msv(
    deposition_bq_m2: np.ndarray,
    integrated_air_bq_s_m3: np.ndarray,
    hazard: HazardConfig,
) -> np.ndarray:
    exposure_seconds = float(hazard.dose_integration_hours * 3600.0)
    dose_msv = np.zeros_like(deposition_bq_m2, dtype=np.float64)
    inhaled_air_m3 = hazard.breathing_rate_m3_s * integrated_air_bq_s_m3
    for isotope in hazard.isotope_mix:
        decay_constant = np.log(2.0) / (isotope.half_life_hours * 3600.0)
        integrated_seconds = (1.0 - np.exp(-decay_constant * exposure_seconds)) / decay_constant
        isotope_weight = isotope.deposition_fraction
        ground_dose = (
            deposition_bq_m2
            * isotope_weight
            * isotope.ground_surface_dose_coeff_sv_per_bq_s_m2
            * integrated_seconds
        )
        cloud_dose = (
            integrated_air_bq_s_m3
            * isotope_weight
            * isotope.cloud_immersion_dose_coeff_sv_per_bq_s_m3
        )
        inhalation_dose = (
            inhaled_air_m3
            * isotope_weight
            * isotope.inhalation_dose_coeff_sv_per_bq
        )
        dose_msv += (ground_dose + cloud_dose + inhalation_dose) * 1000.0
    return (dose_msv * float(hazard.ground_dose_multiplier)).astype(np.float32)


def _resolve_parallelism(settings: ProjectConfig) -> tuple[int, int]:
    cpu_total = max(1, os.cpu_count() or 1)
    auto_base = max(4, min(12, cpu_total))
    auto_boost = max(auto_base, cpu_total * 2)
    configured_base = settings.hazard.scenario_parallel_workers
    configured_boost = settings.hazard.scenario_parallel_workers_boost
    base_workers = auto_base if configured_base is None else max(1, int(configured_base))
    boost_workers = auto_boost if configured_boost is None else max(base_workers, int(configured_boost))
    memory_cap = _memory_safe_worker_cap(
        per_worker_gb=float(settings.hazard.scenario_memory_per_worker_gb),
        reserve_gb=float(settings.hazard.scenario_memory_reserve_gb),
    )
    base_workers = min(base_workers, memory_cap)
    boost_workers = min(boost_workers, memory_cap)
    return max(1, base_workers), max(1, boost_workers)


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def _memory_safe_worker_cap(per_worker_gb: float, reserve_gb: float) -> int:
    available_bytes = _available_memory_bytes()
    if available_bytes is None:
        return max(1, os.cpu_count() or 1)
    per_worker_bytes = max(int(per_worker_gb * (1024 ** 3)), 256 * 1024 * 1024)
    reserve_bytes = max(int(reserve_gb * (1024 ** 3)), 0)
    budget = max(0, available_bytes - reserve_bytes)
    if budget <= 0:
        return 1
    return max(1, budget // per_worker_bytes)


def _run_single_scenario(raw_meteo: PreparedMeteo, settings: ProjectConfig, incident_start: pd.Timestamp) -> FinalDepositionResult:
    available_end = pd.Timestamp(raw_meteo.ds.time.values[-1])
    max_available_hours = int((available_end - incident_start) / pd.Timedelta(hours=1))
    if settings.timeline.demo_mode:
        requested_hours = int(settings.timeline.simulation_duration_hours)
    else:
        requested_hours = max_available_hours
    simulation_hours = max(1, min(requested_hours, max_available_hours))
    scenario_timeline = replace(
        settings.timeline,
        incident_start_utc=incident_start.isoformat(),
        simulation_duration_hours=simulation_hours,
    )
    return run_dispersion_final_deposition(
        raw_meteo=raw_meteo,
        timeline=scenario_timeline,
        domain=settings.domain,
        dispersion=settings.dispersion,
    )


def _accumulate_scenario(
    final_result: FinalDepositionResult,
    settings: ProjectConfig,
    *,
    hit_count: np.ndarray | None,
    aggregate_relative_impact: np.ndarray | None,
    mean_deposition: np.ndarray | None,
    max_deposition: np.ndarray | None,
    mean_dose: np.ndarray | None,
    max_dose: np.ndarray | None,
    mean_ground_dose: np.ndarray | None,
    max_ground_dose: np.ndarray | None,
    dangerous_ground_dose_hit_count: np.ndarray | None,
):
    deposition_field = final_result.final_deposition_bq_m2.astype(np.float32)
    dose_msv = project_total_effective_dose_msv(
        final_result.final_deposition_bq_m2,
        final_result.integrated_air_concentration_bq_s_m3,
        settings.hazard,
    )
    ground_dose_msv = _project_equivalent_dose_msv(final_result.final_deposition_bq_m2, settings.hazard).astype(np.float32)
    dangerous_threshold = max(float(settings.hazard.ground_dose_danger_threshold_msv), 1e-12)
    # Use a soft exceedance score to preserve plume-like tails in hazard maps.
    dangerous_hit = np.clip(ground_dose_msv / dangerous_threshold, 0.0, 1.0).astype(np.float32)
    scenario_peak = float(np.nanmax(deposition_field)) if np.any(deposition_field > 0) else 0.0
    if scenario_peak > 0.0:
        relative_impact = deposition_field / scenario_peak
    else:
        relative_impact = np.zeros_like(deposition_field, dtype=np.float32)
    trace_floor = max(
        scenario_peak * settings.hazard.trace_floor_peak_fraction,
        1e-12,
    )
    scenario_hit = (deposition_field >= trace_floor).astype(np.float32)

    if hit_count is None:
        hit_count = np.zeros_like(deposition_field, dtype=np.float32)
        aggregate_relative_impact = np.zeros_like(deposition_field, dtype=np.float32)
        mean_deposition = np.zeros_like(deposition_field, dtype=np.float32)
        max_deposition = np.zeros_like(deposition_field, dtype=np.float32)
        mean_dose = np.zeros_like(deposition_field, dtype=np.float32)
        max_dose = np.zeros_like(deposition_field, dtype=np.float32)
        mean_ground_dose = np.zeros_like(deposition_field, dtype=np.float32)
        max_ground_dose = np.zeros_like(deposition_field, dtype=np.float32)
        dangerous_ground_dose_hit_count = np.zeros_like(deposition_field, dtype=np.float32)

    hit_count += scenario_hit
    aggregate_relative_impact += relative_impact
    mean_deposition += deposition_field
    max_deposition = np.maximum(max_deposition, deposition_field)
    mean_dose += dose_msv.astype(np.float32)
    max_dose = np.maximum(max_dose, dose_msv.astype(np.float32))
    mean_ground_dose += ground_dose_msv
    max_ground_dose = np.maximum(max_ground_dose, ground_dose_msv)
    dangerous_ground_dose_hit_count += dangerous_hit
    return (
        hit_count,
        aggregate_relative_impact,
        mean_deposition,
        max_deposition,
        mean_dose,
        max_dose,
        mean_ground_dose,
        max_ground_dose,
        dangerous_ground_dose_hit_count,
    )


def _run_scenario_batch(
    meteo_path: str,
    meteo_extent: tuple[float, float, float, float],
    meteo_native_step_minutes: int,
    meteo_grid_spacing_m: float,
    meteo_source_summary: str,
    settings: ProjectConfig,
    incident_starts_iso: tuple[str, ...],
):
    ds = xr.open_dataset(meteo_path, engine="scipy", cache=False)
    try:
        raw_meteo = PreparedMeteo(
            ds=ds,
            extent=meteo_extent,
            native_step_minutes=meteo_native_step_minutes,
            grid_spacing_m=meteo_grid_spacing_m,
            source_summary=meteo_source_summary,
        )
        grid = None
        hit_count = None
        aggregate_relative_impact = None
        mean_deposition = None
        max_deposition = None
        mean_dose = None
        max_dose = None
        mean_ground_dose = None
        max_ground_dose = None
        dangerous_ground_dose_hit_count = None
        scenario_count = 0
        scenario_labels: list[str] = []
        for incident_start_iso in incident_starts_iso:
            final_result = _run_single_scenario(raw_meteo, settings, pd.Timestamp(incident_start_iso))
            if grid is None:
                grid = final_result.grid
            (
                hit_count,
                aggregate_relative_impact,
                mean_deposition,
                max_deposition,
                mean_dose,
                max_dose,
                mean_ground_dose,
                max_ground_dose,
                dangerous_ground_dose_hit_count,
            ) = _accumulate_scenario(
                final_result,
                settings,
                hit_count=hit_count,
                aggregate_relative_impact=aggregate_relative_impact,
                mean_deposition=mean_deposition,
                max_deposition=max_deposition,
                mean_dose=mean_dose,
                max_dose=max_dose,
                mean_ground_dose=mean_ground_dose,
                max_ground_dose=max_ground_dose,
                dangerous_ground_dose_hit_count=dangerous_ground_dose_hit_count,
            )
            scenario_count += 1
            scenario_labels.append(incident_start_iso)
        if grid is None or hit_count is None:
            raise ValueError("No scenarios computed in worker batch.")
        return (
            grid,
            hit_count,
            aggregate_relative_impact,
            mean_deposition,
            max_deposition,
            mean_dose,
            max_dose,
            mean_ground_dose,
            max_ground_dose,
            dangerous_ground_dose_hit_count,
            scenario_count,
            tuple(scenario_labels),
        )
    finally:
        ds.close()


def compute_scenario_hazard_map(
    raw_meteo: PreparedMeteo,
    settings: ProjectConfig,
    external_jobs_running: Callable[[], bool] | None = None,
) -> ScenarioHazardResult:
    scenario_times = _scenario_times(settings)
    hit_count = None
    aggregate_relative_impact = None
    mean_deposition = None
    max_deposition = None
    mean_dose = None
    max_dose = None
    mean_ground_dose = None
    max_ground_dose = None
    dangerous_ground_dose_hit_count = None
    scenario_labels: list[str] = []
    threshold_notes: list[str] = []
    grid = None
    scenario_count = 0

    available_start = pd.Timestamp(raw_meteo.ds.time.values[0])
    available_end = pd.Timestamp(raw_meteo.ds.time.values[-1])

    eligible_scenarios: list[pd.Timestamp] = []
    for incident_start in scenario_times:
        scenario_timeline = replace(settings.timeline, incident_start_utc=incident_start.isoformat())
        window = build_simulation_window(scenario_timeline)
        if window.incident_start >= available_start and window.simulation_end <= available_end:
            eligible_scenarios.append(incident_start)

    if not eligible_scenarios:
        raise ValueError("Не удалось посчитать ни одного сценария для hazard map.")

    base_workers, boost_workers = _resolve_parallelism(settings)
    batch_size = 1
    remaining = [timestamp.isoformat() for timestamp in eligible_scenarios]

    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(prefix="hazard_runtime_", suffix=".nc", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        raw_meteo.ds.to_netcdf(temp_path, engine="scipy")
        progress = tqdm(total=len(eligible_scenarios), desc="Hazard scenarios", unit="scenario")
        try:
            with cf.ProcessPoolExecutor(max_workers=boost_workers) as executor:
                in_flight: dict[cf.Future, tuple[str, ...]] = {}
                while remaining or in_flight:
                    busy = external_jobs_running is not None and external_jobs_running()
                    target_parallelism = base_workers if busy else boost_workers
                    while remaining and len(in_flight) < target_parallelism:
                        incident_start_iso = remaining.pop(0)
                        batch = (incident_start_iso,)
                        future = executor.submit(
                            _run_scenario_batch,
                            str(temp_path),
                            raw_meteo.extent,
                            raw_meteo.native_step_minutes,
                            raw_meteo.grid_spacing_m,
                            raw_meteo.source_summary,
                            settings,
                            batch,
                        )
                        in_flight[future] = batch

                    if not in_flight:
                        continue

                    done, _ = cf.wait(in_flight.keys(), return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        in_flight.pop(future, None)
                        (
                            batch_grid,
                            batch_hit,
                            batch_rel,
                            batch_mean_dep,
                            batch_max_dep,
                            batch_mean_dose,
                            batch_max_dose,
                            batch_mean_ground_dose,
                            batch_max_ground_dose,
                            batch_dangerous_ground_dose_hit_count,
                            batch_count,
                            batch_labels,
                        ) = future.result()
                        if grid is None:
                            grid = batch_grid
                            hit_count = np.zeros_like(batch_hit, dtype=np.float32)
                            aggregate_relative_impact = np.zeros_like(batch_rel, dtype=np.float32)
                            mean_deposition = np.zeros_like(batch_mean_dep, dtype=np.float32)
                            max_deposition = np.zeros_like(batch_max_dep, dtype=np.float32)
                            mean_dose = np.zeros_like(batch_mean_dose, dtype=np.float32)
                            max_dose = np.zeros_like(batch_max_dose, dtype=np.float32)
                            mean_ground_dose = np.zeros_like(batch_mean_ground_dose, dtype=np.float32)
                            max_ground_dose = np.zeros_like(batch_max_ground_dose, dtype=np.float32)
                            dangerous_ground_dose_hit_count = np.zeros_like(
                                batch_dangerous_ground_dose_hit_count, dtype=np.float32
                            )

                        hit_count += batch_hit
                        aggregate_relative_impact += batch_rel
                        mean_deposition += batch_mean_dep
                        max_deposition = np.maximum(max_deposition, batch_max_dep)
                        mean_dose += batch_mean_dose
                        max_dose = np.maximum(max_dose, batch_max_dose)
                        mean_ground_dose += batch_mean_ground_dose
                        max_ground_dose = np.maximum(max_ground_dose, batch_max_ground_dose)
                        dangerous_ground_dose_hit_count += batch_dangerous_ground_dose_hit_count
                        scenario_count += batch_count
                        scenario_labels.extend(batch_labels)
                        threshold_notes.extend(
                            ["cloud-footprint frequency; pale envelope = any scenario above trace floor"] * batch_count
                        )
                        progress.update(batch_count)
        finally:
            progress.close()
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    if (
        scenario_count == 0
        or grid is None
        or hit_count is None
        or aggregate_relative_impact is None
        or mean_deposition is None
        or max_deposition is None
        or mean_dose is None
        or max_dose is None
        or mean_ground_dose is None
        or max_ground_dose is None
        or dangerous_ground_dose_hit_count is None
    ):
        raise ValueError("Не удалось посчитать ни одного сценария для hazard map.")

    probability = hit_count / float(scenario_count)
    aggregate_relative_impact /= float(scenario_count)
    if float(np.nanmax(aggregate_relative_impact)) > 0.0:
        safety_index = 1.0 - (aggregate_relative_impact / float(np.nanmax(aggregate_relative_impact)))
    else:
        safety_index = np.ones_like(aggregate_relative_impact, dtype=np.float32)
    mean_deposition /= float(scenario_count)
    mean_dose /= float(scenario_count)
    mean_ground_dose /= float(scenario_count)
    dangerous_ground_dose_probability = dangerous_ground_dose_hit_count / float(scenario_count)
    threshold_note = threshold_notes[0]
    if any(note != threshold_note for note in threshold_notes[1:]):
        threshold_note = "mixed aggregation logic across scenarios"

    return ScenarioHazardResult(
        grid=grid,
        hazard_probability=probability,
        aggregate_relative_impact=aggregate_relative_impact,
        safety_index=safety_index,
        mean_deposition_bq_m2=mean_deposition,
        max_deposition_bq_m2=max_deposition,
        mean_equivalent_dose_msv=mean_dose,
        max_equivalent_dose_msv=max_dose,
        mean_ground_dose_msv=mean_ground_dose,
        max_ground_dose_msv=max_ground_dose,
        dangerous_ground_dose_probability=dangerous_ground_dose_probability,
        hit_count=hit_count,
        scenario_count=scenario_count,
        threshold_note=threshold_note,
        scenario_start_times=tuple(scenario_labels),
    )

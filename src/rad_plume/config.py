from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"


@dataclass(frozen=True)
class DataConfig:
    source_kind: str = "seasonal_forecast"
    data_file: Path = DATA_DIR / "seasonal_seas5_6hourly.nc"
    pressure_level_file: Path | None = None
    model_level_file: Path | None = None
    ensemble_mode: str = "median"
    ensemble_member: int = 0
    source_label: str = "ECMWF SEAS5 seasonal forecast"


@dataclass(frozen=True)
class DataInventoryConfig:
    historical_actual: DataConfig = field(
        default_factory=lambda: DataConfig(
            source_kind="era5_reanalysis",
            data_file=DATA_DIR / "era5_actual_hourly.nc",
            pressure_level_file=DATA_DIR / "era5_actual_pressure_levels.nc",
            model_level_file=DATA_DIR / "era5_actual_model_levels.nc",
            ensemble_mode="mean",
            source_label="ERA5 hourly reanalysis",
        )
    )
    future_medium_range: DataConfig = field(
        default_factory=lambda: DataConfig(
            source_kind="ecmwf_medium_range",
            data_file=DATA_DIR / "ecmwf_medium_range_hourly.nc",
            pressure_level_file=DATA_DIR / "ecmwf_medium_range_pressure_levels.nc",
            model_level_file=DATA_DIR / "ecmwf_medium_range_model_levels.nc",
            ensemble_mode="mean",
            source_label="ECMWF medium-range forecast",
        )
    )
    future_seasonal: DataConfig = field(
        default_factory=lambda: DataConfig(
            source_kind="seasonal_forecast",
            data_file=DATA_DIR / "seasonal_seas5_6hourly.nc",
            pressure_level_file=DATA_DIR / "seasonal_seas5_pressure_levels.nc",
            ensemble_mode="median",
            source_label="ECMWF SEAS5 seasonal forecast",
        )
    )
    default_source_profile: str = "best_available_blend"
    auto_refresh_if_stale: bool = True
    era5_latency_days: int = 5
    era5_complete_final_lag_days: int = 70
    medium_range_latency_hours: int = 12
    medium_range_horizon_days: int = 15
    medium_range_open_data_sources: tuple[str, ...] = ("ecmwf", "azure", "aws", "google")
    seasonal_release_day_utc: int = 6
    seasonal_release_hour_utc: int = 12
    transport_model_levels: tuple[int, ...] = tuple(range(90, 138))
    download_demo_model_levels: tuple[int, ...] = tuple(range(120, 138))
    transport_pressure_levels_hpa: tuple[int, ...] = (
        1000,
        975,
        950,
        925,
        900,
        875,
        850,
        825,
        800,
        775,
        750,
        700,
        650,
        600,
        550,
        500,
        450,
        400,
        350,
        300,
        250,
        225,
        200,
        175,
        150,
        125,
        100,
        70,
        50,
        30,
        20,
        10,
    )
    quality_report_file: Path = OUTPUT_DIR / "data_quality_report.md"


@dataclass(frozen=True)
class DomainConfig:
    source_lon: float = 35.033
    source_lat: float = 31.067
    grid_resolution_km: float = 5.0
    wind_plot_resolution_deg: float = 0.10
    map_padding_deg: float = 0.50
    domain_radius_km: float = 400.0
    coastline_context_km: float = 30.0


@dataclass(frozen=True)
class TimelineConfig:
    incident_start_utc: str = "2026-04-15T00:00:00"
    release_duration_hours: int = 1
    simulation_duration_hours: int = 144
    model_step_minutes: int = 15
    frame_step_minutes: int = 60
    fps: int = 1
    demo_mode: bool = False
    demo_hours: int = 1
    demo_frame_step_minutes: int = 60


@dataclass(frozen=True)
class IsotopeDoseConfig:
    name: str
    half_life_hours: float
    deposition_fraction: float
    ground_surface_dose_coeff_sv_per_bq_s_m2: float
    cloud_immersion_dose_coeff_sv_per_bq_s_m3: float
    inhalation_dose_coeff_sv_per_bq: float


@dataclass(frozen=True)
class HazardConfig:
    scenario_start_utc: str = "2026-04-01T00:00:00"
    scenario_end_utc: str = "2026-05-31T00:00:00"
    scenario_step_hours: int = 1
    model_step_minutes: int = 30
    dose_integration_hours: int = 168
    any_hit_threshold_msv: float = 0.0
    trace_floor_peak_fraction: float = 0.001
    trace_floor_min_msv: float = 1e-9
    threshold_mode: str = "relative_peak_fraction"
    relative_peak_fraction: float = 0.02
    absolute_deposition_threshold_bq_m2: float | None = None
    fixed_incident_zone_fractions: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 0.60)
    scenario_probability_levels: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
    scenario_parallel_workers: int | None = None
    scenario_parallel_workers_boost: int | None = None
    scenario_memory_per_worker_gb: float = 1.5
    scenario_memory_reserve_gb: float = 4.0
    ground_dose_danger_threshold_msv: float = 100.0
    ground_dose_multiplier: float = 2.4e5
    dose_zone_levels_msv: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0)
    breathing_rate_m3_s: float = 3.3e-4
    isotope_mix: tuple[IsotopeDoseConfig, ...] = (
        IsotopeDoseConfig(
            name="I-131",
            half_life_hours=8.02 * 24.0,
            deposition_fraction=0.1324,
            ground_surface_dose_coeff_sv_per_bq_s_m2=2.43e-16,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=6.0e-14,
            inhalation_dose_coeff_sv_per_bq=2.2e-8,
        ),
        IsotopeDoseConfig(
            name="I-132",
            half_life_hours=2.295,
            deposition_fraction=0.1420,
            ground_surface_dose_coeff_sv_per_bq_s_m2=1.47e-15,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=7.5e-14,
            inhalation_dose_coeff_sv_per_bq=5.0e-9,
        ),
        IsotopeDoseConfig(
            name="I-133",
            half_life_hours=20.8,
            deposition_fraction=0.3091,
            ground_surface_dose_coeff_sv_per_bq_s_m2=4.27e-16,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=3.2e-14,
            inhalation_dose_coeff_sv_per_bq=6.0e-9,
        ),
        IsotopeDoseConfig(
            name="I-134",
            half_life_hours=0.875,
            deposition_fraction=0.1490,
            ground_surface_dose_coeff_sv_per_bq_s_m2=1.69e-15,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=7.5e-14,
            inhalation_dose_coeff_sv_per_bq=3.0e-9,
        ),
        IsotopeDoseConfig(
            name="I-135",
            half_life_hours=6.57,
            deposition_fraction=0.2675,
            ground_surface_dose_coeff_sv_per_bq_s_m2=9.95e-16,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=5.2e-14,
            inhalation_dose_coeff_sv_per_bq=2.0e-9,
        ),
        IsotopeDoseConfig(
            name="Cs-134",
            half_life_hours=2.06 * 365.25 * 24.0,
            deposition_fraction=3.5e-7,
            ground_surface_dose_coeff_sv_per_bq_s_m2=9.87e-16,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=1.8e-14,
            inhalation_dose_coeff_sv_per_bq=2.0e-8,
        ),
        IsotopeDoseConfig(
            name="Cs-137",
            half_life_hours=30.17 * 365.25 * 24.0,
            deposition_fraction=1.1e-6,
            ground_surface_dose_coeff_sv_per_bq_s_m2=3.01e-18,
            cloud_immersion_dose_coeff_sv_per_bq_s_m3=8.0e-15,
            inhalation_dose_coeff_sv_per_bq=1.3e-8,
        ),
    )


@dataclass(frozen=True)
class DispersionConfig:
    emission_rate_bq_s: float = 6e10
    dry_deposition_velocity_ms: float = 0.004
    mixing_height_m: float = 400.0
    min_mixing_height_m: float = 220.0
    max_mixing_height_m: float = 1800.0
    initial_sigma_m: float = 1500.0
    base_horizontal_diffusivity_m2_s: float = 25.0
    wind_diffusivity_factor_m2_s_per_ms: float = 12.0
    coarse_grid_crosswind_fraction: float = 0.18
    coarse_grid_alongwind_fraction: float = 0.08
    release_layer_height_fractions: tuple[float, ...] = (0.18, 0.45, 0.85)
    release_layer_mass_fractions: tuple[float, ...] = (0.30, 0.45, 0.25)
    crosswind_turbulence_scale: float = 0.75
    alongwind_turbulence_scale: float = 0.35
    max_wind_profile_factor: float = 2.5
    max_advect_wind_speed_ms: float = 80.0
    random_seed: int = 42
    radioactive_half_life_hours: float | None = None
    dose_conversion_factor_msv_per_bq_m2: float | None = 1.0


@dataclass(frozen=True)
class VisualConfig:
    save_mp4: bool = True
    wind_output_mp4: str = "wind_animation.mp4"
    plume_output_mp4: str = "plume_animation.mp4"
    summary_output_png: str = "summary_contamination.png"
    hazard_output_png: str = "scenario_hazard_probability.png"
    summary_ground_dose_output_png: str = "summary_ground_dose_msv.png"
    hazard_ground_dose_output_png: str = "scenario_hazard_ground_dose_danger_probability.png"
    parallel_render_processes: int = 4
    wind_particle_count: int = 1600
    wind_particle_refresh_fraction: float = 0.12
    wind_particle_inflow_bias: float = 0.60
    quiver_stride: int = 4
    plume_levels: int = 18
    wind_speed_levels: int = 18
    dpi: int = 140


@dataclass(frozen=True)
class ProjectConfig:
    inventory: DataInventoryConfig = field(default_factory=DataInventoryConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    dispersion: DispersionConfig = field(default_factory=DispersionConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)


SETTINGS = ProjectConfig()


def ensure_runtime_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MPLCONFIG_DIR.mkdir(exist_ok=True)

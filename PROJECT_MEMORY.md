# Project Memory

## Current Goal

Keep one repository that can:

- download meteorology for a chosen source point,
- blend past/current, medium-range, and seasonal data,
- run one fixed plume scenario,
- run a multi-start hazard sweep,
- render two videos and two maps,
- keep a per-run data quality report.

## Runtime Semantics

- `wind`
  renders wind for the selected window; in demo mode this is 6 days.
- `plume`
  runs one incident and animates the cloud.
- `summary`
  renders the final fixed-incident deposition or dose map.
- `hazard`
  sweeps many incident start times and renders hit frequency.
- `all`
  runs `wind`, `plume`, `summary`, and `hazard` in parallel worker processes.
- `download`
  refreshes ERA5, medium-range, and seasonal meteo in one command; `--demo` uses short windows.

## Current Data Stack

- default source profile: `best_available_blend`
- blend order:
  - `historical_actual`
  - `future_medium_range`
  - `future_seasonal`

Current practical status on this machine:

- ERA5 actual: present
- ERA5 archived model levels: present for demo/stable historical window
- medium-range: present, including dense pressure-level winds
- seasonal surface file: present from live CDS GRIB -> local NetCDF conversion
- seasonal pressure levels: still blocked by CDS request-combination validation

## Vertical Levels

Dense pressure levels now used where available:

`1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10 hPa`

Reason:

- dense near-ground structure matters most for plume steering,
- expected cloud height usually lives in the lower troposphere,
- some higher levels are still useful for stronger lofting and shear.

## Physics Snapshot

Current model:

- Lagrangian Gaussian puff
- layered release
- pressure-level steering wind when available
- surface-wind fallback otherwise
- dry deposition
- day/night stability surrogate
- spread inflation from ensemble uncertainty when available

Still missing:

- wet deposition
- full dynamic PBL
- terrain/building flow
- isotope-resolved transport
- inhalation and cloud-submersion dose

## Important Windows Notes

- some `NetCDF4/HDF5` files fail to open directly through `netCDF4` when the path contains Cyrillic characters
- loader now falls back to Windows short path before giving up
- datasets are eagerly loaded and file handles closed immediately to avoid native Windows crashes

## Wind animation: “looped” perception

- Rendering does **not** repeat the same time index: frames follow `pd.date_range(start, end, frame_step)`, and `iter_interpolated_time_chunks` overlaps chunks by one step only so the **first** frame of each subsequent chunk is skipped — no duplicate times in the MP4.
- If motion looks periodic, usual causes: **coarse native model timestep** (e.g. 360 min in blend summary) plus linear interpolation between times; **seasonal/climatological** tail of the blend; **fixed RNG seed (42)** for wind particles so inflow/refresh patterns look similar across long runs.
- `nanmean` on all-NaN or empty wind slabs produced `RuntimeWarning: Mean of empty slice` and `nan` means; use `_finite_field_mean` so inflow biasing falls back to 0.0.

## UX Notes

- every run writes to a new timestamped output directory
- download scripts show progress instead of silent waiting
- medium-range conversion subsets the domain before loading, which made conversion much faster
- wind particles now respawn from inflow boundaries instead of slowly dying out
- `all` now materializes `best_available_blend` once into the run output dir and workers reopen that file instead of rebuilding the blend in parallel
- `wind` now renders the configured incident window instead of the full blended time span
- correction: non-demo wind must still cover `current local anchor -> end of available data`; only execution is chunked to stay within RAM
- new rule from user: all heavy time-domain passes should be batch/chunk based by default, not just the biggest known hotspot
- plume products no longer rely on retaining the whole snapshot history in RAM; summaries are built from aggregates and mp4 is rendered from a second streaming pass
- frame-time interpolation can now request only the vars needed by the renderer, which keeps fixed-incident animation RAM bounded
- ERA5 appends only missing trailing days; medium-range and seasonal append only missing lead times for the same init and replace files when a newer init arrives

## Git Notes

- repository is now marked as a safe Git directory on this machine
- large meteo files and outputs stay outside version control via `.gitignore`

## Change History

- 2026-03-29: fixed `all` memory blow-up on Windows. Root cause: each process rebuilt `best_available_blend`, so xarray concat/reindex multiplied RAM pressure. Fix: precompute one runtime NetCDF blend and pass that path to workers.
- 2026-03-29: fixed second `all` RAM amplifier. Root cause: `plume` and `summary` each rebuilt the same full-horizon fixed-incident interpolation independently. Fix: `all` now runs `run_plume_products()` once and reuses that single heavy pass for both outputs.
- 2026-03-30: switched dispersion integration to overlapping 48-hour meteo batches. Root cause: full-horizon 15/30-minute interpolation of multilevel fields produced GiB-scale arrays. Fix: keep the full incident-to-end-of-data horizon, but interpolate one time chunk at a time and carry puff state plus accumulated fields across chunks.
- 2026-04-01: fixed false-positive auto-refresh for seasonal tail. Root cause: refresh criterion demanded more horizon than the current monthly seasonal release can physically cover, so `future_seasonal` retried every run. Fix: clamp required seasonal coverage to the latest expected release month plus its actual 180-day tail.
- 2026-04-01: fixed stale medium-range metadata loop. Root cause: when a newer ECMWF cycle arrived, merge logic kept the old `forecast_reference_time`, so the file looked stale forever and re-downloaded every run. Fix: same-init downloads append only missing lead times, but a newer init now replaces the file cleanly.
- 2026-04-01: fixed another RAM amplifier in animation prep. Root cause: `wind` tried to interpolate the entire blend span, and fixed-incident frame prep carried multilevel vars it never rendered. Fix: `wind` now uses the configured incident window, frame interpolation accepts `keep_vars`, and time chunk size is computed from per-step slab size instead of a fixed 720-step batch.
- 2026-04-01: clarified non-demo semantics after user correction. The full `anchor -> data_end` wind horizon is still required; the fix is streaming/chunked rendering, not shrinking the physics or animation window.
- 2026-04-01: expanded batching policy after user clarification. Remaining full-overlap quality-report forecast-vs-actual comparison was switched to chunked accumulation too, so heavy time-axis work follows one consistent rule across the codebase.
- 2026-04-01: removed plume snapshot retention as the last big time-axis RAM sink. Fixed-incident summary now uses an aggregate result (`max cloud over frames` + `final deposition` + `integrated air`), and plume animation is rendered from a separate streaming pass that writes each frame immediately instead of keeping all snapshots.
- 2026-04-01: fixed ERA5 refresh churn. Root cause: the downloader always fetched the same trailing daily window and overwrote the file. Fix: it now reuses covered data, appends only the missing tail, and reuses model levels when that archival window is already present.
- 2026-04-01: wind particle `nanmean` warnings and possible `nan` drift bias. Root cause: some frames had no finite wind values in the plotted slab. Fix: `_finite_field_mean()` returns 0.0 when empty or all-NaN; documented that wind MP4 is not time-looped in code.

## Prompt History

- 2026-03-29: user reported `numpy._core._exceptions._ArrayMemoryError` during `py .\main.py` after medium-range refresh and hazard sweep startup in parallel mode.
- 2026-03-30: user proposed simple batching: compute one time fraction, load new data, continue, without changing the total horizon.
- 2026-04-01: user reported that `py .\main.py` re-downloaded data on every run, asked to keep current files and fetch only fresh tail data, and pointed to `numpy._core._exceptions._ArrayMemoryError` in `slice_and_interpolate_time()` during the `wind` worker.
- 2026-04-01: user clarified that without demo everything must render from the current PC date to the end of available data; only the implementation should be batched for memory and responsiveness.
- 2026-04-01: user requested batching everywhere: all heavy computations should run chunked/batched, not only the worst offender.
- 2026-04-01: user explicitly requested finishing the batching work everywhere, including plume rendering/products.
- 2026-04-01: user suspected wind animation is “just looped”; asked for short explanation and fix for `Mean of empty slice` on wind particle means.

## Next Useful Upgrades

1. Add proper CDS credentials and fetch ERA5 + seasonal pressure levels directly into `data/`.
2. Crack the valid seasonal pressure-level CDS request and merge it into the blended source.
3. Improve plume physics with wet deposition and stronger vertical mixing logic.
4. Add isotope-resolved transport instead of dose-only postprocessing.
5. Add optional comparison mode against HYSPLIT or FLEXPART class output.

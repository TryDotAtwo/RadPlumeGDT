# Rad Plume — radioactive cloud & fallout toy model

<!-- Suggested GitHub repository description (copy-paste into repo settings):
EN: Because the world is tense: a playful pipeline to estimate radioactive plume spread and ground deposition after a nuclear-site accident — not your civil emergency service, but better than a news ticker.
RU: В связи с напряжённостью в мире — с оговорками и юмором: код для оценки облака и осадков при гипотетической аварии на ЯО; не для МЧС, но лучше, чем гадать по ТВ.
-->

## English

**The world is… not boring.** In that spirit, here is code to **play through** how a radioactive plume might drift and what could end up on the ground (dry/wet-style deposition in a simplified sense) if something went wrong at a nuclear facility. It blends weather, runs a Lagrangian puff sketch, draws videos and maps, and writes a small data-quality report each run.

**Disclaimer, but make it cheeky:** this is **not** a regulator-grade emergency tool, **not** HYSPLIT, and **not** a reason to skip the real authorities. It **is** a sandbox for engineers and the curious who want numbers and pictures instead of pure vibes.

---

## Русский

**Мир сейчас не слишком расслабляющий.** На этой почве — **с шуткой и оговорками**: код, который оценивает, **как могло бы разойтись** радиоактивное облако и **что из этого могло бы оказаться на поверхности** (в упрощённой постановке) в случае аварии на ядерном объекте. Подтягивает метео, гоняет лагранжевы «пуфы», рисует ролики и карты, пишет отчёт о данных.

**Оговорка по-честному:** это **не** государственная модель, **не** замена ГСЧС/МЧС и **не** повод игнорировать официальные предупреждения. Это **инженерная игрушка** для тех, кому нужны картинки и порядок величин, а не только бегущая строка на экране.

---

## What it does (for real)

- Download / blend meteorology (ERA5, ECMWF medium-range, seasonal tail).
- Run a **Lagrangian Gaussian puff** transport model with layered release, steering-level wind, dry deposition, simple stability.
- Render **wind** and **plume** animations, **summary** and **hazard** maps (plus dose-style outputs when configured).
- Every run writes a timestamped folder under `outputs/`.

Default demo scenario is framed near **Dimona** (30 MW class point source) — **change coordinates** in config; the pipeline is not tied to one place.

## Quick start

```powershell
py -3 -m pip install -r requirements.txt
py main.py --demo
```

Full pipeline (longer, needs data / CDS where applicable):

```powershell
py main.py
```

Separate targets: `wind`, `plume`, `summary`, `hazard`, `report`, `download`. See sections below.

## What each run produces

Inside `outputs/YYYY-MM-DD_HH-MM-SS/`:

- `wind_animation.mp4`
- `plume_animation.mp4`
- `summary_contamination.png`
- `scenario_hazard_probability.png`
- `data_quality_report.md`
- (when enabled) ground-dose hazard / summary maps as configured

## What `main.py` does

`py main.py` runs wind, plume, summary, and hazard in **parallel** workers where possible.

```powershell
py main.py wind
py main.py plume
py main.py summary
py main.py hazard
py main.py report
py main.py download
```

Demo mode:

```powershell
py main.py --demo
py main.py wind --demo
```

Coordinate override:

```powershell
py main.py --demo --lat 31.067 --lon 35.033 --radius-km 400
```

Without `--demo`, non-demo wind covers the configured incident anchor through **end of available data**, chunked so RAM stays sane.

## Data strategy

Default source: **`best_available_blend`**.

1. `historical_actual` — ERA5 where facts exist  
2. `future_medium_range` — ECMWF open data, ~15 days  
3. `future_seasonal` — long tail, lower resolution  

### Vertical wind stack

Dense lower-troposphere pressure levels (see `config.py` / `PROJECT_MEMORY.md`) for steering and shear; animation uses a representative cloud steering level when pressure-level winds exist, else 10 m wind.

## Physics (honest limits)

- Lagrangian puffs, layered release, advection, spread, dry deposition, day/night surrogate, optional ensemble spread inflation.  
- **Not** full wet deposition, terrain CFD, or isotope-resolved chains in the transport core yet.  
- Dose / hazard layers are **post-process** style — useful for visualization and rough orders of magnitude, not licensing.

## Downloaders

```powershell
py scripts/download_era5_box.py
py scripts/download_medium_range_box.py
py scripts/download_seasonal_box.py
py main.py download
py main.py download --demo
```

CDS downloads need `~/.cdsapirc` where applicable.

## Repo layout

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
|-- PROJECT_MEMORY.md
|-- scripts/
`-- src/rad_plume/
```

Raw meteo: `data/` (gitignored). Outputs: `outputs/` (gitignored).

## Publishing to GitHub (first time)

From this repo root, after creating an **empty** repository on GitHub (no README/license there if you already have them here):

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

If the default branch should stay `master`, omit `branch -M` and push `master` instead.

## References

- [ERA5 pressure levels (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)
- [ECMWF open data](https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%253A+real-time+forecasts+from+IFS+and+AIFS)
- [NOAA HYSPLIT](https://www.ready.noaa.gov/hysplitusersguide/S000.htm) — for comparison mindset, not bundled here

## License

MIT — see `LICENSE`. Software is provided **as is**, without warranty; not for operational emergency response.

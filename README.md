# **RadPlumeGDT**

### *Radioactive Plume Generative Dispersion Model*

**Feed in a source → get consequences**

---

## English

**The world is… not boring.**
So instead of doomscrolling — simulate it.

This repo is a **slightly chaotic engineering sandbox** that answers a very simple question:

> *“If something goes wrong… where does it go?”*

It pulls real meteorology, runs a **Lagrangian Gaussian puff model**, and turns that into:

* maps
* animations
* and outputs that look suspiciously like something from a report

---

### ⚠️ Disclaimer (but stylish)

This is **not** HYSPLIT.
This is **not** an emergency-response tool.
This is **not** something you should use instead of actual authorities.

This **is**:

* a physics-flavored “what if” engine
* a weather-driven uncertainty machine
* a way to replace vibes with plots

---

## Русский

**Мир сейчас не слишком расслабляющий.**
Поэтому вместо паники — моделируем.

Этот репозиторий — **слегка хаотичная инженерная песочница**, которая отвечает на вопрос:

> *«Если что-то пойдёт не так… куда это всё полетит?»*

Он подтягивает реальные метеоданные, гоняет **лагранжеву модель гауссовых облаков** и превращает это в:

* карты
* анимации
* и результаты, подозрительно похожие на что-то из отчёта

---

### ⚠️ Оговорка (но красиво)

Это **не** HYSPLIT.
Это **не** инструмент реагирования.
Это **не** замена официальным источникам.

Это:

* физика с элементами «а что если»
* машина неопределённости на погоде
* способ заменить тревожность на графики

---

## 🚀 What it does (actually)

* Blends meteorology: **ERA5 + ECMWF + seasonal tail**
* Runs **Lagrangian Gaussian puff transport**
* Simulates:

  * advection
  * spread
  * dry deposition
  * simple stability
* Produces:

  * 🎞 plume animations
  * 🌬 wind animations
  * 🗺 contamination maps
  * ⚠️ hazard-style outputs
* Writes a **data-quality report** every run (because we pretend to be responsible)

🗺 Contamination map

Where stuff actually ends up.

<img width="840" height="1120" alt="summary_contamination" src="https://github.com/user-attachments/assets/8811a3ca-4fbf-43aa-9520-0a6c177071b7" />

Ground-level contamination estimate from puff transport + deposition.
Colors = “how much landed here”.
Yes, the wind had opinions.

⚠️ Ground dose (mSv-style)

Turning maps into something that feels serious.

<img width="840" height="1120" alt="summary_ground_dose_msv" src="https://github.com/user-attachments/assets/cf8d2fc4-1b02-4838-b79e-a690c5b449ad" />

Post-processed “dose-like” field derived from deposition.
Not regulatory. Not precise.
But finally — numbers.

☁️ Plume snapshot

The cloud, mid-flight.

<img width="1378" height="884" alt="image" src="https://github.com/user-attachments/assets/c8dd665d-27f5-4552-9d76-b9c1103e42f1" />

Instant view of the plume structure in the atmosphere.
Advection + spread doing their thing.
Basically: physics unfolding in real time.

🌬 Wind field

The invisible boss fight.

<img width="1078" height="837" alt="image" src="https://github.com/user-attachments/assets/4db98e83-7bf0-4a27-a293-e175c3348190" />

Driving force behind everything.
If this changes — everything changes.
The plume is just along for the ride.



---

## 🎯 Philosophy

> Not accurate enough to trust.
> Not dumb enough to ignore.

---

## 🧪 Demo scenario

Default demo is near **Dimona** (≈30 MW class point source).
Change coordinates — pipeline works anywhere.

---
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

## Publishing to GitHub

**Option A — GitHub CLI (recommended)**  
Install: `winget install GitHub.cli`. One-time login (browser):

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" auth login --web
```

Then from the repo root, create `rad-plume` on your account and push `main`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\push_to_github.ps1
```

Optional repo name: `.\scripts\push_to_github.ps1 -RepoName my-rad-plume`

Alternatively, non-interactive automation: set env var `GH_TOKEN` (classic PAT with `repo` scope), then run the same script.

**Option B — manual**  
Create an **empty** repo on GitHub (no README if you already have one here), then:

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## References

- [ERA5 pressure levels (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)
- [ECMWF open data](https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%253A+real-time+forecasts+from+IFS+and+AIFS)
- [NOAA HYSPLIT](https://www.ready.noaa.gov/hysplitusersguide/S000.htm) — for comparison mindset, not bundled here

## License

MIT — see `LICENSE`. Software is provided **as is**, without warranty; not for operational emergency response.

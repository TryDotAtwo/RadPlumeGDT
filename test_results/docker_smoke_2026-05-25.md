# Docker Smoke Test - 2026-05-25

## Scope

- Workspace: `D:\Кодинг\Погода\Погода`
- Base image: `python:3.11-slim-bookworm`
- Built image: `rad-plume:latest`
- Docker compose project name: `rad-plume`
- Fresh meteo download: not run

## Commands

```powershell
docker compose build rad-plume
py -m compileall main.py src
docker compose run --rm rad-plume report
docker run --rm --network none -v "C:\tmp\rad-plume-empty-data:/app/data" -v "C:\tmp\rad-plume-empty-outputs:/app/outputs" rad-plume:latest report
```

## Results

- `docker compose build rad-plume`: passed; image `rad-plume:latest` built from public base `python:3.11-slim-bookworm`.
- `py -m compileall main.py src`: passed.
- `docker compose run --rm rad-plume report`: passed; report written to `/app/outputs/2026-05-25_10-49-24/data_quality_report.md`.
- `docker run ... rad-plume:latest report` with empty mounted `data/` and `--network none`: passed; report written to `/app/outputs/2026-05-25_11-20-48/data_quality_report.md`.

## Notes

- Dockerfile installs only `ffmpeg` through apt; Python packages are installed from `requirements.txt`.
- `report` target can run after a fresh clone with empty `data/`; map/video targets still need meteo files in `data/` or an explicit `download` run.
- No explicit fresh meteo download command was executed.

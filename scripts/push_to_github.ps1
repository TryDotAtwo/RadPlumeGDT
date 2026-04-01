# Push this repo to a new GitHub repository (requires: gh auth login once).
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\push_to_github.ps1
# Optional: .\scripts\push_to_github.ps1 -RepoName my-rad-plume

param(
    [string] $RepoName = "rad-plume"
)

$ErrorActionPreference = "Stop"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) {
    Write-Host "GitHub CLI not found at $gh. Install: winget install GitHub.cli"
    exit 1
}

Push-Location (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

& $gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Not logged in to GitHub. Run once in this terminal:"
    Write-Host "  & `"$gh`" auth login --web"
    Write-Host "Then open the URL, paste the device code, authorize."
    Write-Host ""
    exit 2
}

$desc = "Radioactive plume / fallout scenario tool (meteo blend + puff model). EN/RU README. Not for emergency ops."
if (git remote get-url origin 2>$null) {
    Write-Host "Remote 'origin' already exists. Pushing..."
    git push -u origin main
} else {
    Write-Host "Creating github.com/$(& $gh api user -q .login)/$RepoName and pushing..."
    & $gh repo create $RepoName --public --description $desc --source=. --remote=origin --push
}

Pop-Location
Write-Host "Done."

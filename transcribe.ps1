#!/usr/bin/env pwsh
# aegis-yt-transcriber — one-liner wrapper (Windows PowerShell / pwsh).
#   .\transcribe.ps1 "https://www.youtube.com/watch?v=VIDEO_ID" [--timestamps] [--model NAME] [--lang en]
#
# Prefers `uv` (https://docs.astral.sh/uv/) — it reads pyproject.toml and manages the
# environment automatically. If uv is not installed, falls back to a plain venv + pip.
$ErrorActionPreference = "Stop"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($args.Count -eq 0) {
    Write-Error "usage: .\transcribe.ps1 ""<youtube-url>"" [--timestamps] [--model NAME] [--lang en]"
    exit 2
}

# Fast path: uv handles the env from pyproject.toml.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv run --quiet --project $DIR (Join-Path $DIR "yt2txt.py") @args
    exit $LASTEXITCODE
}

# Fallback: plain venv + pip (no uv on this machine).
$VENV = Join-Path $DIR ".venv"
$PY = Join-Path $VENV "Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Host ">> first run: creating .venv and installing deps (one-time)..."
    Write-Host "   (tip: install uv for a faster setup — https://docs.astral.sh/uv/)"
    python -m venv $VENV
    & $PY -m pip install --quiet --upgrade pip
    & $PY -m pip install --quiet $DIR
}

& $PY (Join-Path $DIR "yt2txt.py") @args
exit $LASTEXITCODE

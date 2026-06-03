# aegis-yt-transcriber

![tests](https://github.com/BinHsu/aegis-yt-transcriber/actions/workflows/test.yml/badge.svg)

Turn a YouTube URL into a text transcript — **including videos that have captions disabled.**

When a video has captions you don't need this (use YouTube's own *Show transcript*, or
[`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/)). This tool is
for the case where captions are off: it downloads the audio and transcribes it **locally**
with Whisper. No third-party transcription service — the audio never leaves your machine.

Runs on **macOS, Linux, and Windows.** On Apple Silicon it uses the Metal-accelerated
`mlx-whisper`; everywhere else it uses the cross-platform `faster-whisper`. The choice is
automatic.

## Quick start

**macOS / Linux:**

```bash
git clone https://github.com/BinHsu/aegis-yt-transcriber.git
cd aegis-yt-transcriber
./transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --timestamps
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/BinHsu/aegis-yt-transcriber.git
cd aegis-yt-transcriber
.\transcribe.ps1 "https://www.youtube.com/watch?v=VIDEO_ID" --timestamps
```

The wrapper prefers [`uv`](https://docs.astral.sh/uv/) — it reads `pyproject.toml` and sets up
the environment automatically on first run. If you don't have `uv`, it falls back to a plain
`python -m venv` + `pip`. Either way the first run installs dependencies, and later runs reuse
them. The transcript lands in `transcripts/<date>_<title>_<id>.txt` (plus a `.timestamps.txt`
with `[mm:ss]` markers if you pass `--timestamps`).

## Requirements

- **`ffmpeg`** on your PATH — `brew install ffmpeg` (macOS), `apt install ffmpeg` (Debian/Ubuntu),
  `winget install ffmpeg` or `choco install ffmpeg` (Windows). Needed to extract audio.
- **Python 3.10+**.
- **`uv`** (recommended, optional) — [install](https://docs.astral.sh/uv/getting-started/installation/).
  Without it the wrapper uses `venv` + `pip` instead.

The Whisper backend is selected for you:

| Platform | Backend | Notes |
|---|---|---|
| macOS (Apple Silicon) | `mlx-whisper` | Metal-accelerated; fastest on a Mac. Installed automatically. |
| Windows / Linux / Intel-Mac | `faster-whisper` | CTranslate2; runs on CPU, or CUDA GPU if present. |

`mlx-whisper` is declared with an environment marker, so `pip`/`uv` installs it **only** on
Apple-Silicon macOS and skips it everywhere else. Force a backend with `--backend mlx|faster`.

## Usage

```bash
./transcribe "<url>"                 # plain transcript
./transcribe "<url>" --timestamps    # also write a [mm:ss] segmented file
./transcribe "<url>" --lang en       # force a language (default: auto-detect)
./transcribe "<url>" --model small   # faster/lighter model
./transcribe "<url>" --backend faster   # override auto-selection
```

Default model per backend:

- **mlx**: `mlx-community/whisper-large-v3-turbo` (~1.6 GB first run, then cached in
  `~/.cache/huggingface`). Lighter: `mlx-community/whisper-small`, `mlx-community/whisper-medium`.
- **faster**: `large-v3` (downloaded from Hugging Face, then cached). Lighter: `small`, `medium`,
  `distil-large-v3`.

### Manual setup (instead of the wrapper)

```bash
# with uv:
uv run yt2txt.py "<url>" --timestamps

# or plain venv + pip:
python3 -m venv .venv
.venv/bin/python -m pip install .          # Windows: .venv\Scripts\python -m pip install .
.venv/bin/python yt2txt.py "<url>" --timestamps
```

## How it works

```
yt-dlp (download best audio) → ffmpeg (extract mp3) → Whisper (mlx | faster) → text
```

The transcription step is pluggable; only the Whisper backend differs by platform, the rest of
the pipeline is identical everywhere.

## Tests

Pure logic (filename slugging, `[mm:ss]` formatting, backend selection) is covered by unit
tests with **Boundary Value Analysis** on the numeric edges. The heavy dependencies are
lazy-imported, so the tests need only `pytest` and run in milliseconds with no model downloads:

```bash
uv run --with pytest --no-project pytest -q     # or: pip install pytest && pytest
```

CI runs them on **Ubuntu, Windows, and macOS** on every push (see the badge above) — that
matrix is what backs the cross-platform claim.

## Notes

- Downloading audio is for **personal transcription** — respect each video's terms and copyright,
  and don't redistribute the audio or transcript without permission.
- The Whisper model and the downloaded audio live **outside** the repo (`~/.cache/huggingface`
  and the git-ignored `audio/` folder), so nothing large is ever committed.
- Whisper occasionally mishears proper nouns (e.g. "Claude" → "Cloud"); a quick find-and-replace
  cleans those up.

## License

See [LICENSE](LICENSE).

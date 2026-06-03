# aegis-yt-transcriber

Turn a YouTube URL into a text transcript — **including videos that have captions disabled**.

When a video has captions, you don't need this (use YouTube's own *Show transcript*, or
`youtube-transcript-api`). This tool is for the case where captions are off: it downloads
the audio and transcribes it locally with Whisper.

## Pipeline
`yt-dlp` (download best audio) → `ffmpeg` (extract mp3) → `mlx-whisper` (Apple-Silicon
Whisper, Metal-accelerated) → clean text.

Everything runs **locally** — no third-party transcription service, no audio leaves the machine.

## Requirements
- macOS on Apple Silicon (mlx-whisper is Metal-accelerated).
- `ffmpeg` on PATH (`brew install ffmpeg`).
- Python deps in a local venv (see below).

## Setup
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage
```bash
.venv/bin/python yt2txt.py "https://www.youtube.com/watch?v=VIDEO_ID" --timestamps
```
Output goes to `transcripts/<title>__<id>.txt` (and a `.timestamps.txt` with `[mm:ss]` markers).

Pick a model with `--model` (default: `mlx-community/whisper-large-v3-turbo`, ~1.6 GB on
first run, then cached). Smaller/faster options:
```bash
--model mlx-community/whisper-small      # fast, lighter
--model mlx-community/whisper-medium     # middle ground
```
Force a language with `--lang en` (default: auto-detect).

## Notes
- Downloading audio is for **personal transcription**; respect each video's terms and copyright.
- The first run downloads the Whisper model from Hugging Face; later runs reuse the cache.

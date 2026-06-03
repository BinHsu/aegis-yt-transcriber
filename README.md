# aegis-yt-transcriber

Turn a YouTube URL into a text transcript — **including videos that have captions disabled.**

When a video has captions you don't need this (use YouTube's own *Show transcript*, or
[`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/)). This tool is
for the case where captions are off: it downloads the audio and transcribes it **locally**
with Whisper. No third-party transcription service — the audio never leaves your machine.

## Quick start

```bash
git clone https://github.com/BinHsu/aegis-yt-transcriber.git
cd aegis-yt-transcriber
./transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --timestamps
```

The first run creates a local `.venv` and installs the dependencies automatically; later runs
reuse it. The transcript lands in `transcripts/<title>__<id>.txt` (plus a `.timestamps.txt`
with `[mm:ss]` markers if you pass `--timestamps`).

## Requirements

- **macOS on Apple Silicon** — `mlx-whisper` is Metal-accelerated. (On other platforms, swap
  the transcriber for `faster-whisper`/`openai-whisper`; the rest of the pipeline is the same.)
- **`ffmpeg`** on your PATH — `brew install ffmpeg`.
- **Python 3** (3.10+).

## Usage

```bash
./transcribe "<url>"                 # plain transcript
./transcribe "<url>" --timestamps    # also write a [mm:ss] segmented file
./transcribe "<url>" --lang en       # force a language (default: auto-detect)
./transcribe "<url>" --model mlx-community/whisper-small   # faster/lighter model
```

Default model: `mlx-community/whisper-large-v3-turbo` (best speed/accuracy on Apple Silicon;
~1.6 GB on first run, then cached in `~/.cache/huggingface`). Lighter options:
`mlx-community/whisper-small`, `mlx-community/whisper-medium`.

### Manual setup (instead of the wrapper)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python yt2txt.py "<url>" --timestamps
```

## How it works

```
yt-dlp (download best audio) → ffmpeg (extract mp3) → mlx-whisper (local Whisper) → text
```

## Notes

- Downloading audio is for **personal transcription** — respect each video's terms and copyright,
  and don't redistribute the audio or transcript without permission.
- The Whisper model and the downloaded audio live **outside** the repo (`~/.cache/huggingface`
  and the git-ignored `audio/` folder), so nothing large is ever committed.
- Whisper occasionally mishears proper nouns (e.g. "Claude" → "Cloud"); a quick find-and-replace
  cleans those up.

## License

See [LICENSE](LICENSE).

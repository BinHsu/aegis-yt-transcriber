#!/usr/bin/env python3
"""aegis-yt-transcriber — YouTube URL -> text transcript (works on videos with NO captions).

Pipeline: yt-dlp (download best audio) -> ffmpeg (extract mp3) -> mlx-whisper
(Apple-Silicon Whisper, Metal-accelerated) -> clean text.

Usage:
    python yt2txt.py <youtube-url> [--model HF_REPO] [--lang en] [--timestamps]

Default model: mlx-community/whisper-large-v3-turbo (first run downloads ~1.6 GB from
Hugging Face, then it's cached). Use a smaller model for speed, e.g.
  --model mlx-community/whisper-small   (or whisper-medium / distil-whisper-large-v3)
"""
from __future__ import annotations
import argparse, re
from datetime import date
from pathlib import Path
import yt_dlp
import mlx_whisper

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
OUT = ROOT / "transcripts"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:80] or "video"


def download_audio(url: str):
    AUDIO.mkdir(exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return AUDIO / f"{info['id']}.mp3", info["id"], info.get("title", info["id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="YouTube URL -> transcript (no-caption friendly)")
    ap.add_argument("url")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lang", default=None, help="force language code, e.g. en (default: auto-detect)")
    ap.add_argument("--timestamps", action="store_true", help="also write a [mm:ss] segmented file")
    a = ap.parse_args()

    print(f"[1/2] downloading audio: {a.url}", flush=True)
    audio, vid, title = download_audio(a.url)
    print(f"      got: {audio.name}  ({title})", flush=True)

    print(f"[2/2] transcribing with {a.model}\n      (first run downloads the model; be patient)...", flush=True)
    kw = {"path_or_hf_repo": a.model}
    if a.lang:
        kw["language"] = a.lang
    res = mlx_whisper.transcribe(str(audio), **kw)

    OUT.mkdir(exist_ok=True)
    # filename = generation date + video title + video id (e.g. 2026-06-03_Beyond_the_basics_..._tuY2ChJIx48)
    base = OUT / f"{date.today().isoformat()}_{slug(title)}_{vid}"
    txt = base.with_suffix(".txt")
    txt.write_text(res["text"].strip() + "\n", encoding="utf-8")
    print(f"\n✅ transcript: {txt}", flush=True)

    if a.timestamps:
        lines = []
        for s in res.get("segments", []):
            t = int(s["start"])
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {s['text'].strip()}")
        ts = base.parent / (base.name + ".timestamps.txt")
        ts.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"   + timestamps: {ts}", flush=True)

    print(f"\n--- preview ---\n{res['text'][:600].strip()}", flush=True)


if __name__ == "__main__":
    main()

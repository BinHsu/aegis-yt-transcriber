#!/usr/bin/env python3
"""aegis-yt-transcriber — YouTube URL -> text transcript (works on videos with NO captions).

Captions-first: by default the tool first tries to pull existing captions (manual, then
auto-generated) via yt-dlp — seconds, no model, no audio download. Only when a video has
no usable captions (or they're disabled) does it fall back to local transcription:

  yt-dlp (download best audio) -> ffmpeg (extract mp3) -> Whisper -> clean text.

Two Whisper backends, picked automatically:
  - mlx-whisper    : Apple-Silicon only, Metal-accelerated (fastest on a Mac).
  - faster-whisper : cross-platform (Windows / Linux / Intel-Mac, CPU or CUDA GPU).
On Apple Silicon the tool uses mlx-whisper if installed; everywhere else it uses
faster-whisper. Override with --backend {auto,mlx,faster}. Skip the caption check with
--force-whisper (e.g. when auto-captions are low quality and you want clean ASR).

Usage:
    python yt2txt.py <youtube-url> [--force-whisper] [--backend auto] [--model NAME] [--lang en] [--timestamps]

Default model per backend:
  - mlx    : mlx-community/whisper-large-v3-turbo   (~1.6 GB first run, then cached)
  - faster : large-v3                               (downloaded from Hugging Face, then cached)
Use a smaller model for speed, e.g. --model small  (faster) or
--model mlx-community/whisper-small  (mlx).
"""
from __future__ import annotations
import argparse, platform, re, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
OUT = ROOT / "transcripts"
DEFAULT_MODEL = {
    "mlx": "mlx-community/whisper-large-v3-turbo",
    "faster": "large-v3",
}


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:80] or "video"


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def pick_backend(arg: str) -> str:
    """auto -> mlx on Apple Silicon (if mlx-whisper is importable), else faster-whisper."""
    if arg != "auto":
        return arg
    if is_apple_silicon():
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "faster"


def fmt_timestamps(segments) -> str:
    """Render Whisper segments as [mm:ss]-prefixed lines. mm rolls over at 60s."""
    lines = []
    for s in segments:
        t = int(s["start"])
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {s['text'].strip()}")
    return "\n".join(lines) + "\n"


def parse_json3(data: dict) -> list:
    """Parse YouTube json3 caption data into [{start, text}] segments. Pure (unit-tested)."""
    segs = []
    for ev in data.get("events", []):
        raw = "".join(s.get("utf8", "") for s in (ev.get("segs") or []) if s)
        text = " ".join(raw.split())  # collapse intra-cue newlines / runs of whitespace
        if text:
            segs.append({"start": ev.get("tStartMs", 0) / 1000.0, "text": text})
    return segs


def fetch_captions(url: str, lang: str | None):
    """Try to pull existing captions (manual preferred, then auto-generated) via yt-dlp's
    json3 track — no audio download, no Whisper. Returns {text, segments, vid, title} or None."""
    import yt_dlp  # lazy
    import json, urllib.request
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    vid = info["id"]
    title = info.get("title", vid)
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    wanted = [lang] if lang else ["en", "en-US", "en-GB", "en-orig"]

    track = None
    for source in (manual, auto):          # human-written captions beat ASR
        for l in wanted:
            if l in source:
                track = source[l]
                break
        if track:
            break
    if track is None and not lang:         # no language preference matched -> take anything
        for source in (manual, auto):
            if source:
                track = next(iter(source.values()))
                break
    if not track:
        return None

    fmt = next((f for f in track if f.get("ext") == "json3"), None)
    if fmt is None:                        # json3 not offered -> let Whisper handle it cleanly
        return None
    raw = urllib.request.urlopen(fmt["url"], timeout=30).read().decode("utf-8")
    segs = parse_json3(json.loads(raw))
    if not segs:
        return None
    return {
        "text": " ".join(s["text"] for s in segs).strip() + "\n",
        "segments": segs,
        "vid": vid,
        "title": title,
    }


def download_audio(url: str):
    import yt_dlp  # lazy: keeps `import yt2txt` dependency-free for unit tests
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


def transcribe_mlx(audio: Path, model: str, lang: str | None) -> dict:
    """Apple-Silicon Metal backend. Returns {text, segments:[{start,text}]}."""
    import mlx_whisper
    kw = {"path_or_hf_repo": model}
    if lang:
        kw["language"] = lang
    res = mlx_whisper.transcribe(str(audio), **kw)
    segs = [{"start": s["start"], "text": s["text"]} for s in res.get("segments", [])]
    return {"text": res["text"], "segments": segs}


def transcribe_faster(audio: Path, model: str, lang: str | None) -> dict:
    """Cross-platform CTranslate2 backend (CPU or CUDA). Returns {text, segments:[{start,text}]}."""
    from faster_whisper import WhisperModel
    # device/compute "auto" -> CUDA float16 if a GPU is present, else CPU int8.
    m = WhisperModel(model, device="auto", compute_type="auto")
    segments, _info = m.transcribe(str(audio), language=lang)
    segs = [{"start": s.start, "text": s.text} for s in segments]  # generator -> materialize
    return {"text": "".join(s["text"] for s in segs).strip(), "segments": segs}


def main() -> None:
    ap = argparse.ArgumentParser(description="YouTube URL -> transcript (no-caption friendly)")
    ap.add_argument("url")
    ap.add_argument("--force-whisper", action="store_true",
                    help="skip the caption check and always transcribe the audio (use when auto-captions are poor)")
    ap.add_argument("--backend", choices=["auto", "mlx", "faster"], default="auto",
                    help="Whisper backend (default: auto — mlx on Apple Silicon, faster-whisper elsewhere)")
    ap.add_argument("--model", default=None, help="model name/repo (default: per-backend, see --help)")
    ap.add_argument("--lang", default=None, help="force language code, e.g. en (default: auto-detect)")
    ap.add_argument("--timestamps", action="store_true", help="also write a [mm:ss] segmented file")
    a = ap.parse_args()

    res = vid = title = None

    # Captions-first: if the video already has captions, use them (seconds, no model). Skip with --force-whisper.
    if not a.force_whisper:
        print(f"[1/2] checking for existing captions: {a.url}", flush=True)
        cap = fetch_captions(a.url, a.lang)
        if cap:
            res, vid, title = cap, cap["vid"], cap["title"]
            print(f"      ✓ captions found ({len(cap['segments'])} cues) — using them, skipping Whisper", flush=True)
        else:
            print("      no usable captions (or disabled) — falling back to audio + Whisper", flush=True)

    # Fallback (or --force-whisper): download audio and transcribe locally.
    if res is None:
        backend = pick_backend(a.backend)
        model = a.model or DEFAULT_MODEL[backend]
        step = "[1/2]" if a.force_whisper else "[2/3]"
        print(f"{step} downloading audio: {a.url}", flush=True)
        audio, vid, title = download_audio(a.url)
        print(f"      got: {audio.name}  ({title})", flush=True)
        step = "[2/2]" if a.force_whisper else "[3/3]"
        print(f"{step} transcribing with backend={backend}, model={model}\n"
              f"      (first run downloads the model; be patient)...", flush=True)
        res = transcribe_mlx(audio, model, a.lang) if backend == "mlx" else transcribe_faster(audio, model, a.lang)

    OUT.mkdir(exist_ok=True)
    # filename = generation date + video title + video id (e.g. 2026-06-03_Beyond_the_basics_..._tuY2ChJIx48)
    base = OUT / f"{date.today().isoformat()}_{slug(title)}_{vid}"
    txt = base.with_suffix(".txt")
    txt.write_text(res["text"].strip() + "\n", encoding="utf-8")
    print(f"\n✅ transcript: {txt}", flush=True)

    if a.timestamps:
        ts = base.parent / (base.name + ".timestamps.txt")
        ts.write_text(fmt_timestamps(res.get("segments", [])), encoding="utf-8")
        print(f"   + timestamps: {ts}", flush=True)

    print(f"\n--- preview ---\n{res['text'][:600].strip()}", flush=True)


if __name__ == "__main__":
    main()

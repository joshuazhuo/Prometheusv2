# Prometheus v2 — TikTok Transcript Extractor

## Project Overview
Extracts transcripts from TikTok videos using yt-dlp + OpenAI Whisper, saves results back to CSV.

## Python Setup
- `python` → Python 3.8 (old, avoid)
- `py` → resolves to Python 3.8 via shebang (avoid for this project)
- `py -3.14` → Python 3.14 (correct version to use)
- All dependencies (yt-dlp, openai-whisper, ffmpeg) are installed under Python 3.14

## Always Run With
```
py -3.14 extract_transcripts.py
```

## Key Files
- `extract_transcripts.py` — main script
- `tiktok_videos.csv` — input/output data (25 TikTok videos, transcript column populated on run)

## Dependencies
- `yt-dlp` — downloads audio from TikTok URLs
- `openai-whisper` — local speech-to-text transcription
- `ffmpeg` — audio processing (installed via winget)

## Known Issues / Notes
- Script uses `--cookies-from-browser chrome` for yt-dlp; Chrome should be closed when running
- TikTok extraction can fail if yt-dlp is outdated — update with `py -3.14 -m pip install -U yt-dlp`
- Whisper `base` model is default; use `--model medium` or `--model large` for better accuracy

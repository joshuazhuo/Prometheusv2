#!/usr/bin/env python3
"""
Extract transcripts from TikTok videos and add them to the CSV.

Supports two methods:
  1. Whisper (default): Downloads audio via yt-dlp, transcribes with OpenAI Whisper locally.
  2. Google Cloud Speech-to-Text: Uses GCP credentials for transcription.

Requirements:
  pip install yt-dlp openai-whisper
  # Also needs ffmpeg installed on the system

Usage:
  python extract_transcripts.py                    # Use Whisper (default)
  python extract_transcripts.py --method google    # Use Google Cloud Speech-to-Text
  python extract_transcripts.py --model medium     # Use a larger Whisper model for better accuracy
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time


CSV_PATH = "tiktok_videos.csv"
OUTPUT_PATH = "tiktok_videos.csv"


def download_audio(url, output_path):
    """Download audio from a TikTok video URL using yt-dlp."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--cookies-from-browser", "chrome",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    # yt-dlp may create the file with a different extension
    if os.path.exists(output_path):
        return output_path
    # Check for common variants
    for ext in [".wav", ".wav.wav", ".mp3", ".m4a", ".webm"]:
        candidate = output_path + ext
        if os.path.exists(candidate):
            return candidate
    # Try glob pattern
    base = os.path.splitext(output_path)[0]
    matches = glob.glob(f"{base}*")
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Audio file not found after download: {output_path}")


def transcribe_with_whisper(audio_path, model):
    """Transcribe audio using OpenAI Whisper."""
    result = model.transcribe(audio_path, language="en")
    return result["text"].strip()


def transcribe_with_google(audio_path, credentials_path="credentials.json"):
    """Transcribe audio using Google Cloud Speech-to-Text API."""
    from google.cloud import speech
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = speech.SpeechClient(credentials=credentials)

    # Convert to mono 16kHz WAV for best results
    mono_path = audio_path + ".mono.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", mono_path],
        capture_output=True, check=True,
    )

    with open(mono_path, "rb") as f:
        content = f.read()
    os.remove(mono_path)

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    # Use long_running_recognize for audio > 1 min
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=300)

    transcript = " ".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    )
    return transcript.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract TikTok video transcripts")
    parser.add_argument(
        "--method", choices=["whisper", "google"], default="whisper",
        help="Transcription method (default: whisper)",
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    parser.add_argument(
        "--credentials", default="credentials.json",
        help="Path to Google Cloud service account JSON (for --method google)",
    )
    args = parser.parse_args()

    audio_dir = tempfile.mkdtemp(prefix="tiktok_audio_")

    # Read existing CSV
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add transcript column
    if "transcript" not in fieldnames:
        fieldnames.append("transcript")

    # Load Whisper model if needed
    whisper_model = None
    if args.method == "whisper":
        import whisper
        print(f"Loading Whisper model ({args.model})...")
        whisper_model = whisper.load_model(args.model)

    success_count = 0
    fail_count = 0

    for i, row in enumerate(rows):
        url = row["webVideoUrl"]
        title = row["text"][:70]
        print(f"\n[{i+1}/{len(rows)}] {title}...")

        if row.get("transcript"):
            print("  Already has transcript, skipping.")
            success_count += 1
            continue

        audio_path = os.path.join(audio_dir, f"video_{i}.wav")
        try:
            print("  Downloading audio...")
            actual_path = download_audio(url, audio_path)
            print(f"  Downloaded: {os.path.basename(actual_path)}")

            print("  Transcribing...")
            if args.method == "whisper":
                transcript = transcribe_with_whisper(actual_path, whisper_model)
            else:
                transcript = transcribe_with_google(actual_path, args.credentials)

            row["transcript"] = transcript
            success_count += 1
            print(f"  OK ({len(transcript)} chars): {transcript[:100]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            row["transcript"] = ""
            fail_count += 1

        finally:
            # Clean up audio files
            for f in glob.glob(os.path.join(audio_dir, f"video_{i}*")):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Save after each video (resume-friendly)
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Cleanup temp dir
    try:
        os.rmdir(audio_dir)
    except OSError:
        pass

    print(f"\nDone! {success_count} succeeded, {fail_count} failed.")
    print(f"Transcripts saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract individual products mentioned in TikTok video transcripts.

Reads tiktok_videos.csv, calls Claude API for each transcript to identify
products, and writes products.csv with dropship-relevant fields.

Requirements:
  pip install anthropic

Usage:
  py -3.14 extract_products.py
  py -3.14 extract_products.py --input tiktok_videos.csv --output products.csv
"""

import argparse
import csv
import json
import os
import sys

import anthropic

INPUT_CSV = "tiktok_videos.csv"
OUTPUT_CSV = "products.csv"

OUTPUT_FIELDNAMES = [
    "video_index",
    "video_title",
    "video_url",
    "product_name",
    "product_category",
    "description",
    "problem_solved",
    "selling_tone",
    "selling_method",
    "desirability_tactics",
    "authority_signals",
    "urgency_or_scarcity",
    "target_audience",
    "price_positioning",
    "dropship_notes",
]

SYSTEM_PROMPT = """You are a product research analyst specializing in dropshipping and e-commerce.
Your task is to extract every individual product mentioned in a TikTok video transcript and analyze
how each product was sold to the audience. Be thorough — include both recommended AND avoided products,
since avoided products still reveal market demand.

Return a JSON array. Each element must have exactly these fields:
- product_name: Exact name or clear description of the product
- product_category: e.g. "health supplement", "survival gear", "food", "personal care"
- description: What the product is and what it does (1-2 sentences)
- problem_solved: The pain point or need this product addresses
- selling_tone: e.g. "fear-based", "aspirational", "authoritative", "urgent", "educational"
- selling_method: How it was presented — e.g. "compare good vs bad brands", "list format", "expert endorsement", "before/after contrast"
- desirability_tactics: Specific phrases or techniques used to make it desirable (quote key phrases if possible)
- authority_signals: Any credibility cues used — e.g. "military veterans", "ER doctors", "FDA", "independent testing"
- urgency_or_scarcity: Any time pressure or scarcity messaging (or "none")
- target_audience: Who this product is aimed at
- price_positioning: Any pricing signals — e.g. "budget", "premium", "value vs expensive alternative" (or "not mentioned")
- dropship_notes: 1-2 sentences on dropship potential — demand signals, competition, niche fit

Return ONLY valid JSON array, no markdown, no explanation."""


def extract_products_from_transcript(
    client: anthropic.Anthropic,
    video_index: str,
    video_title: str,
    video_url: str,
    transcript: str,
) -> list[dict]:
    """Call Claude API to extract products from a single transcript."""

    user_message = f"""Video title: {video_title}

Transcript:
{transcript}

Extract all products mentioned in this transcript."""

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final = stream.get_final_message()

    # Extract text from response, stripping markdown code fences if present
    text = next(
        (block.text for block in final.content if block.type == "text"),
        ""
    ).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # remove opening fence line
        text = text.rsplit("```", 1)[0].strip()  # remove closing fence

    try:
        products = json.loads(text)
        if not isinstance(products, list):
            products = [products]
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse failed ({e}). Raw response saved.")
        print(f"  Raw: {text[:200]}")
        return []

    # Attach video metadata to each product
    for p in products:
        p["video_index"] = video_index
        p["video_title"] = video_title
        p["video_url"] = video_url

    return products


def main():
    parser = argparse.ArgumentParser(description="Extract products from TikTok transcripts")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: $env:ANTHROPIC_API_KEY = 'your-key-here'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Read transcripts
    with open(args.input, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))

    rows_with_transcripts = [r for r in rows if r.get("transcript", "").strip()]
    print(f"Found {len(rows_with_transcripts)} videos with transcripts.\n")

    # Load already-processed video URLs to support resuming
    processed_urls = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                processed_urls.add(existing.get("video_url", ""))
        print(f"Resuming — {len(processed_urls)} videos already processed.\n")

    all_products = []
    success = 0
    fail = 0

    for i, row in enumerate(rows_with_transcripts):
        url = row.get("webVideoUrl", "")
        title = row.get("text", "")[:80]
        index = row.get("index", str(i))

        if url in processed_urls:
            print(f"[{i+1}/{len(rows_with_transcripts)}] {title}... skipping (already done)")
            continue

        print(f"[{i+1}/{len(rows_with_transcripts)}] {title}...")

        try:
            products = extract_products_from_transcript(
                client, index, row.get("text", ""), url, row["transcript"]
            )
            all_products.extend(products)
            success += 1
            print(f"  Found {len(products)} products.")
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1

    # Write output CSV
    write_header = not os.path.exists(args.output) or not processed_urls
    with open(args.output, "a" if processed_urls else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(all_products)

    print(f"\nDone! {success} videos processed, {fail} failed.")
    print(f"Products saved to {args.output}")
    total = len(all_products)
    print(f"Total products extracted this run: {total}")


if __name__ == "__main__":
    main()

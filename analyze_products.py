#!/usr/bin/env python3
"""
Analyze products.csv to:
  1. Assign funnel tiers (Tier 1-4) using Claude
  2. Pull Google Trends demand scores via pytrends
  3. Output analyzed_products.csv ranked by ad potential

Funnel tiers (prepper/survival niche):
  Tier 1 - Must Have Now   : water, food, core medical — widest ad targeting
  Tier 2 - Protect & Secure: home security, wound care, comms — retargeting
  Tier 3 - Optimize        : better gear, monitoring, hygiene — warm audience
  Tier 4 - Niche Expert    : specific brands, specialty items — converted buyers

Usage:
  py -3.14 analyze_products.py
"""

import csv
import json
import os
import time

import anthropic
from pytrends.request import TrendReq

INPUT_CSV = "products.csv"
OUTPUT_CSV = "analyzed_products.csv"

FUNNEL_PROMPT = """You are an e-commerce strategist specializing in the prepper, survival, and emergency preparedness niche.

Analyze these products and assign each one:
1. funnel_tier: 1, 2, 3, or 4
   - Tier 1 "Must Have Now": Core survival essentials (clean water, emergency food, critical medical).
     Widest audience. First thing a new prepper buys. Run cold-traffic ads here.
   - Tier 2 "Protect & Secure": Home security, wound care, emergency comms, fire/heat.
     Second purchase. Retarget people who engaged with Tier 1 content.
   - Tier 3 "Optimize & Prepare": Better versions of basics, health monitoring, hygiene, child safety.
     Warm audience. People already committed to preparedness.
   - Tier 4 "Niche Expert": Specific brand comparisons, specialty food products, highly specific gear.
     Converted buyers. Loyal audience. Email/retargeting only.

2. funnel_tier_name: one of "Must Have Now", "Protect & Secure", "Optimize & Prepare", "Niche Expert"

3. competition_level: "low", "medium", or "high" (Amazon/Walmart shelf competition)

4. ad_potential: score 1-10 (10 = best) considering:
   - Emotional hook strength
   - Visual ad potential (can you SHOW the problem/solution?)
   - Impulse buy likelihood
   - Profit margin potential for dropshipping

5. best_platform: "TikTok", "Facebook", "Both", or "Instagram"

6. ad_angle: One punchy sentence describing the best ad hook for this product.
   Example: "Show a wound getting infected vs. properly irrigated — 'ER doctors carry this for a reason'"

7. search_query: The best 2-4 word Google Trends query to measure demand for this product.
   Keep it generic enough to get results (e.g. "irrigation syringe" not "35mL irrigation syringe with 18 gauge tip")

Return a JSON array. One object per product, preserving the original product_name exactly.
Each object must have: product_name, funnel_tier, funnel_tier_name, competition_level, ad_potential, best_platform, ad_angle, search_query

Products to analyze:
"""


def analyze_batch(client, products):
    """Analyze a single batch of products with Claude."""
    product_list = [
        {
            "product_name": p["product_name"],
            "product_category": p["product_category"],
            "description": p["description"][:150],
            "problem_solved": p["problem_solved"][:100],
            "selling_tone": p["selling_tone"],
            "target_audience": p["target_audience"][:100],
            "dropship_notes": p["dropship_notes"][:150],
        }
        for p in products
    ]

    prompt = FUNNEL_PROMPT + json.dumps(product_list, indent=2)

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        final = stream.get_final_message()

    text = next(
        (b.text for b in final.content if b.type == "text"), ""
    ).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()

    return json.loads(text)


def analyze_with_claude(client, products, batch_size=45):
    """Analyze all products in batches to avoid token limits."""
    all_results = []
    batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]
    print(f"Sending {len(products)} products to Claude in {len(batches)} batches...")
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)} ({len(batch)} products)...")
        results = analyze_batch(client, batch)
        all_results.extend(results)
    return all_results


def get_trends_score(pytrends, query, retries=3):
    """Get Google Trends interest score (0-100) for a search query."""
    for attempt in range(retries):
        try:
            pytrends.build_payload([query], timeframe="today 12-m", geo="US")
            df = pytrends.interest_over_time()
            if df.empty or query not in df.columns:
                return 0, "no data"
            avg = int(df[query].mean())
            last = int(df[query].iloc[-4:].mean())
            first = int(df[query].iloc[:4].mean())
            if last > first * 1.15:
                direction = "rising"
            elif last < first * 0.85:
                direction = "declining"
            else:
                direction = "stable"
            return avg, direction
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return 0, "error"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Read products
    with open(INPUT_CSV, newline="", encoding="utf-8-sig", errors="replace") as f:
        products = list(csv.DictReader(f))
    print(f"Loaded {len(products)} products from {INPUT_CSV}\n")

    # Step 1: Claude funnel analysis (one batch call)
    analysis = analyze_with_claude(client, products)
    analysis_map = {a["product_name"]: a for a in analysis}
    print(f"Received analysis for {len(analysis)} products.\n")

    # Step 2: Google Trends scores
    print("Fetching Google Trends data (this takes a few minutes)...")
    pytrends = TrendReq(hl="en-US", tz=360)

    # Deduplicate queries to avoid redundant API calls
    query_map = {}
    for a in analysis:
        q = a.get("search_query", "").strip()
        if q and q not in query_map:
            query_map[q] = None

    print(f"  Querying {len(query_map)} unique search terms...")
    for i, q in enumerate(query_map):
        score, direction = get_trends_score(pytrends, q)
        query_map[q] = (score, direction)
        print(f"  [{i+1}/{len(query_map)}] '{q}': {score}/100 ({direction})")
        time.sleep(2)  # be polite to Google

    # Step 3: Merge and write output
    output_fieldnames = list(products[0].keys()) + [
        "funnel_tier", "funnel_tier_name", "competition_level",
        "ad_potential", "best_platform", "ad_angle",
        "search_query", "trends_score", "trend_direction",
    ]

    enriched = []
    for p in products:
        row = dict(p)
        a = analysis_map.get(p["product_name"], {})
        row["funnel_tier"] = a.get("funnel_tier", "")
        row["funnel_tier_name"] = a.get("funnel_tier_name", "")
        row["competition_level"] = a.get("competition_level", "")
        row["ad_potential"] = a.get("ad_potential", "")
        row["best_platform"] = a.get("best_platform", "")
        row["ad_angle"] = a.get("ad_angle", "")
        row["search_query"] = a.get("search_query", "")
        q = a.get("search_query", "")
        score, direction = query_map.get(q, (0, "no data"))
        row["trends_score"] = score
        row["trend_direction"] = direction
        enriched.append(row)

    # Sort: tier first, then ad_potential desc, then trends_score desc
    enriched.sort(key=lambda r: (
        int(r["funnel_tier"]) if str(r["funnel_tier"]).isdigit() else 9,
        -int(r["ad_potential"]) if str(r["ad_potential"]).isdigit() else 0,
        -int(r["trends_score"]) if str(r["trends_score"]).isdigit() else 0,
    ))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    # Print summary by tier
    print("\n" + "=" * 60)
    print(f"OUTPUT: {OUTPUT_CSV}  ({len(enriched)} products)")
    print("=" * 60)
    for tier in [1, 2, 3, 4]:
        tier_products = [r for r in enriched if str(r["funnel_tier"]) == str(tier)]
        if not tier_products:
            continue
        name = tier_products[0]["funnel_tier_name"] if tier_products else ""
        print(f"\nTier {tier} - {name} ({len(tier_products)} products)")
        top = sorted(tier_products,
                     key=lambda r: (-int(r["ad_potential"]) if str(r["ad_potential"]).isdigit() else 0,
                                    -int(r["trends_score"]) if str(r["trends_score"]).isdigit() else 0))[:5]
        for p in top:
            print(f"  [{p['ad_potential']}/10 | trends:{p['trends_score']} {p['trend_direction']}] "
                  f"{p['product_name'][:50]}")
            print(f"    -> {p['ad_angle'][:80]}")


if __name__ == "__main__":
    main()

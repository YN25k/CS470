from __future__ import annotations

import argparse
import json
import time
from typing import Any

import requests

from utils import db_cursor, normalize_timestamp, parse_json_field, safe_float, unix_to_iso8601

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Polymarket metadata and price history.")
    parser.add_argument("--limit", type=int, default=500, help="Number of closed markets to fetch per batch.")
    parser.add_argument("--min-clean", type=int, default=500, help="Keep fetching batches until at least this many markets survive cleaning.")
    return parser.parse_args()


def request_json(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    for attempt in range(5):
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def derive_outcome_binary(outcome_prices: Any) -> tuple[int | None, float | None]:
    prices = parse_json_field(outcome_prices, default=[])
    if not isinstance(prices, list) or not prices:
        return None, None
    yes_price = safe_float(prices[0])
    if yes_price is None:
        return None, None
    if yes_price > 0.95:
        return 1, yes_price
    if yes_price < 0.05:
        return 0, yes_price
    return None, yes_price


def parse_yes_token_id(clob_token_ids: Any) -> str | None:
    token_ids = parse_json_field(clob_token_ids, default=[])
    if isinstance(token_ids, list) and token_ids:
        return str(token_ids[0]).strip() or None
    return None


def fetch_markets(limit: int, start_offset: int = 0) -> tuple[list[dict[str, Any]], bool]:
    """Fetch up to *limit* markets starting at *start_offset*.

    Returns (markets, has_more) where has_more is False when the API
    returned fewer results than requested (i.e. we've exhausted the data).
    """
    session = requests.Session()
    fetched: list[dict[str, Any]] = []
    has_more = True
    for offset in range(start_offset, start_offset + limit, PAGE_SIZE):
        params = {
            "closed": "true",
            "limit": PAGE_SIZE,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        page = request_json(session, GAMMA_URL, params)
        if not isinstance(page, list) or not page:
            has_more = False
            break
        fetched.extend(page)
        print(f"Fetched {len(fetched)} markets so far (offset {start_offset})...")
        if len(page) < PAGE_SIZE:
            has_more = False
            break
        if len(fetched) >= limit:
            break
    return fetched[:limit], has_more


def insert_markets(markets: list[dict[str, Any]]) -> int:
    inserted = 0
    with db_cursor() as connection:
        for market in markets:
            outcome_binary, _ = derive_outcome_binary(market.get("outcomePrices"))
            yes_token_id = parse_yes_token_id(market.get("clobTokenIds"))
            volume_total = safe_float(market.get("volumeNum"))
            if volume_total is None:
                volume_total = safe_float(market.get("volume"))
            liquidity_raw = safe_float(market.get("liquidityNum"))
            if liquidity_raw is None:
                liquidity_raw = safe_float(market.get("liquidity"))
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO raw_markets (
                    market_id, platform, question, description, outcomes, outcome_prices,
                    outcome_binary, resolve_ts, end_ts, volume_total, liquidity_raw,
                    category, yes_token_id, slug, closed, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(market.get("id")),
                    "polymarket",
                    market.get("question") or "",
                    market.get("description"),
                    market.get("outcomes") if isinstance(market.get("outcomes"), str) else json.dumps(market.get("outcomes")),
                    market.get("outcomePrices") if isinstance(market.get("outcomePrices"), str) else json.dumps(market.get("outcomePrices")),
                    outcome_binary,
                    normalize_timestamp(market.get("closedTime")),
                    normalize_timestamp(market.get("endDate")),
                    volume_total,
                    liquidity_raw,
                    market.get("category"),
                    yes_token_id,
                    market.get("slug"),
                    int(bool(market.get("closed"))) if market.get("closed") is not None else None,
                    int(bool(market.get("active"))) if market.get("active") is not None else None,
                ),
            )
            inserted += cursor.rowcount
    return inserted


def fetch_missing_histories() -> list[dict[str, Any]]:
    with db_cursor() as connection:
        rows = connection.execute(
            """
            SELECT rm.market_id, rm.yes_token_id
            FROM raw_markets rm
            LEFT JOIN (
                SELECT DISTINCT market_id FROM raw_price_history
            ) rph ON rm.market_id = rph.market_id
            WHERE rm.yes_token_id IS NOT NULL
              AND TRIM(rm.yes_token_id) <> ''
              AND rph.market_id IS NULL
            ORDER BY (rm.volume_total IS NULL), rm.volume_total DESC, rm.market_id
            """
        ).fetchall()
    return [{"market_id": row["market_id"], "yes_token_id": row["yes_token_id"]} for row in rows]


def insert_price_history(market_id: str, history: list[dict[str, Any]]) -> int:
    inserted = 0
    with db_cursor() as connection:
        for point in history:
            if "t" not in point or "p" not in point:
                continue
            probability = safe_float(point["p"])
            if probability is None:
                continue
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO raw_price_history (market_id, prob_ts, probability)
                VALUES (?, ?, ?)
                """,
                (market_id, unix_to_iso8601(point["t"]), probability),
            )
            inserted += cursor.rowcount
    return inserted


def ensure_price_history_index() -> None:
    with db_cursor() as connection:
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_price_history_market_prob_unique
            ON raw_price_history(market_id, prob_ts)
            """
        )


def fetch_all_price_histories() -> None:
    """Fetch CLOB price history for every raw_market that doesn't have it yet."""
    session = requests.Session()
    pending = fetch_missing_histories()
    if not pending:
        return
    print(f"Need to fetch price history for {len(pending)} markets.")
    total_points = 0
    for index, item in enumerate(pending, start=1):
        payload = request_json(
            session,
            CLOB_URL,
            {"market": item["yes_token_id"], "interval": "max", "fidelity": 60},
        )
        history = payload.get("history", []) if isinstance(payload, dict) else []
        inserted = insert_price_history(item["market_id"], history)
        total_points += inserted
        if index % 50 == 0 or index == len(pending):
            print(f"  Processed {index}/{len(pending)} markets; inserted {total_points} history rows.")


def main() -> None:
    import importlib
    clean_module = importlib.import_module("02_clean")
    run_cleaning = clean_module.run_cleaning

    args = parse_args()
    ensure_price_history_index()
    batch_size = args.limit
    min_clean = args.min_clean
    offset = 0
    batch_number = 0

    while True:
        batch_number += 1
        print(f"\n=== Batch {batch_number}: fetching {batch_size} markets (offset {offset}) ===")
        markets, has_more = fetch_markets(batch_size, start_offset=offset)
        if not markets:
            print("No more markets available from the API.")
            break
        print(f"Fetched {len(markets)} market records from Gamma.")
        inserted_markets = insert_markets(markets)
        print(f"Inserted {inserted_markets} new rows into raw_markets.")

        fetch_all_price_histories()

        clean_count = run_cleaning(verbose=True)
        print(f"\n--- Clean markets so far: {clean_count} / {min_clean} target ---")
        if clean_count >= min_clean:
            print(f"Target reached! {clean_count} clean markets.")
            break
        if not has_more:
            print(f"API exhausted. Only {clean_count} clean markets available.")
            break
        offset += batch_size

    print("Collection complete.")


if __name__ == "__main__":
    main()

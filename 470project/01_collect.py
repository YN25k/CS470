from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any

import requests

from utils import db_cursor, normalize_timestamp, parse_json_field, safe_float, unix_to_iso8601

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 100

GENRE_TAGS = {
    "politics": ["politics", "elections", "government"],
    "sports": ["sports", "nfl", "nba", "mlb", "soccer"],
    "economics": ["economics", "crypto", "finance", "business"],
    "culture": ["entertainment", "culture", "media", "music", "movies"],
}

CATEGORY_HINTS = {
    "politics": ["politic", "elect", "government", "congress", "senate", "policy", "world"],
    "sports": ["sport", "nfl", "nba", "mlb", "soccer", "football", "basketball", "baseball", "tennis", "golf"],
    "economics": ["econom", "crypto", "finance", "business", "market", "stocks"],
    "culture": ["entertainment", "culture", "media", "music", "movie", "film", "celebrity"],
}


class HardStopError(RuntimeError):
    pass


class RequestManager:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.global_delay = 0.0
        self.clob_delay = 0.1

    def _sleep_before_request(self, url: str) -> None:
        delay = self.global_delay
        if url == CLOB_URL:
            delay = max(delay, self.clob_delay)
        if delay > 0:
            time.sleep(delay)

    def _handle_429(self, retry_number: int) -> None:
        print("Rate limited. Waiting 30s before retry...")
        time.sleep(30)
        self.global_delay += 0.1
        self.clob_delay = max(self.clob_delay, self.global_delay)
        print(f"Increased delay to {self.global_delay:.1f}s")
        if retry_number >= 3:
            raise HardStopError(
                "ERROR: Still rate limited after 3 retries. Wait a few minutes and re-run. "
                "Your progress is saved — the script will resume where it left off."
            )

    def request_json(self, url: str, params: dict[str, Any], request_label: str) -> Any | None:
        rate_limit_retries = 0
        server_retried = False
        network_retried = False
        while True:
            self._sleep_before_request(url)
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                if network_retried:
                    raise HardStopError("ERROR: Network connection failed. Check your internet and re-run.") from exc
                network_retried = True
                print(f"Network error for {request_label}. Retrying in 5s...")
                time.sleep(5)
                continue

            status = response.status_code
            if status == 429:
                rate_limit_retries += 1
                self._handle_429(rate_limit_retries)
                continue
            if status == 403:
                raise HardStopError(
                    "ERROR: 403 Forbidden — you may be IP blocked. Wait 10-15 minutes and try again. "
                    "If this persists, increase sleep time between requests."
                )
            if 500 <= status <= 599:
                if server_retried:
                    print(f"Server error {status} for {request_label}. Skipping this request.")
                    return None
                server_retried = True
                print(f"Server error {status} for {request_label}. Retrying in 5s...")
                time.sleep(5)
                continue
            if status != 200:
                print(f"Unexpected status {status} for {request_label}.")
                print(response.text[:500])
                return None
            try:
                return response.json()
            except ValueError:
                print(f"Invalid JSON for {request_label}. Skipping this request.")
                return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Polymarket metadata and price history.")
    parser.add_argument("--per-genre", type=int, default=1500, help="Target number of markets to fetch per genre group.")
    parser.add_argument("--min-per-genre", type=int, default=50, help="Warn if any genre falls below this count after collection.")
    return parser.parse_args()


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


def infer_genre_from_category(category_value: Any) -> str | None:
    category_text = str(category_value or "").lower()
    if not category_text:
        return None
    for genre, hints in CATEGORY_HINTS.items():
        if any(hint in category_text for hint in hints):
            return genre
    return None


def market_to_row(market: dict[str, Any]) -> tuple[Any, ...]:
    outcome_binary, _ = derive_outcome_binary(market.get("outcomePrices"))
    yes_token_id = parse_yes_token_id(market.get("clobTokenIds"))
    volume_total = safe_float(market.get("volumeNum"))
    if volume_total is None:
        volume_total = safe_float(market.get("volume"))
    liquidity_raw = safe_float(market.get("liquidityNum"))
    if liquidity_raw is None:
        liquidity_raw = safe_float(market.get("liquidity"))
    return (
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
    )


def insert_markets(markets: list[dict[str, Any]]) -> int:
    inserted = 0
    with db_cursor() as connection:
        for market in markets:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO raw_markets (
                    market_id, platform, question, description, outcomes, outcome_prices,
                    outcome_binary, resolve_ts, end_ts, volume_total, liquidity_raw,
                    category, yes_token_id, slug, closed, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                market_to_row(market),
            )
            inserted += cursor.rowcount
    return inserted


def fetch_tagged_markets(requests_mgr: RequestManager, tag: str, per_genre: int) -> list[dict[str, Any]]:
    fetched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for offset in range(0, per_genre, PAGE_SIZE):
        payload = requests_mgr.request_json(
            GAMMA_URL,
            {
                "closed": "true",
                "limit": PAGE_SIZE,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
                "tag": tag,
            },
            f"Gamma markets tag={tag} offset={offset}",
        )
        if payload is None:
            break
        if not isinstance(payload, list) or not payload:
            break
        for market in payload:
            market_id = str(market.get("id"))
            if market_id not in seen_ids:
                fetched.append(market)
                seen_ids.add(market_id)
        if len(payload) < PAGE_SIZE or len(fetched) >= per_genre:
            break
    return fetched[:per_genre]


def collect_balanced_markets(requests_mgr: RequestManager, per_genre: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    combined: dict[str, dict[str, Any]] = {}
    per_genre_counts: Counter[str] = Counter()

    for genre, tags in GENRE_TAGS.items():
        genre_market_ids: set[str] = set()
        print(f"\nCollecting {genre} markets...")
        for tag in tags:
            if len(genre_market_ids) >= per_genre:
                break
            tagged = fetch_tagged_markets(requests_mgr, tag, per_genre)
            if not tagged:
                continue
            for market in tagged:
                market_id = str(market.get("id"))
                combined[market_id] = market
                if market_id not in genre_market_ids:
                    genre_market_ids.add(market_id)
                if len(genre_market_ids) >= per_genre:
                    break
            print(f"  tag={tag}: genre pool now {len(genre_market_ids)}")
        if len(genre_market_ids) == 0:
            print(f"  No useful tagged results for {genre}; relying on category fallback.")
        per_genre_counts[genre] = len(genre_market_ids)

    general_limit = per_genre * len(GENRE_TAGS)
    print(f"\nRunning general pull for up to {general_limit} markets to catch untagged markets...")
    for offset in range(0, general_limit, PAGE_SIZE):
        payload = requests_mgr.request_json(
            GAMMA_URL,
            {
                "closed": "true",
                "limit": PAGE_SIZE,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            },
            f"Gamma markets general offset={offset}",
        )
        if payload is None:
            break
        if not isinstance(payload, list) or not payload:
            break
        for market in payload:
            market_id = str(market.get("id"))
            combined.setdefault(market_id, market)
        if len(payload) < PAGE_SIZE:
            break

    inferred_counts: Counter[str] = Counter()
    for market in combined.values():
        inferred_genre = infer_genre_from_category(market.get("category"))
        if inferred_genre is not None:
            inferred_counts[inferred_genre] += 1

    print("\nRaw collection counts by inferred category:")
    for genre in GENRE_TAGS:
        print(f"  {genre}: {inferred_counts.get(genre, 0)}")

    return list(combined.values()), dict(inferred_counts)


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


def fetch_all_price_histories(requests_mgr: RequestManager) -> None:
    pending = fetch_missing_histories()
    if not pending:
        print("No missing price histories.")
        return
    print(f"Need to fetch price history for {len(pending)} markets.")
    total_points = 0
    skipped_markets: list[str] = []
    for index, item in enumerate(pending, start=1):
        payload = requests_mgr.request_json(
            CLOB_URL,
            {"market": item["yes_token_id"], "interval": "max", "fidelity": 10},
            f"CLOB price history market_id={item['market_id']}",
        )
        if payload is None:
            skipped_markets.append(item["market_id"])
            continue
        history = payload.get("history", []) if isinstance(payload, dict) else []
        inserted = insert_price_history(item["market_id"], history)
        total_points += inserted
        if index % 50 == 0 or index == len(pending):
            print(f"Processed {index}/{len(pending)} markets; inserted {total_points} history rows.")
    if skipped_markets:
        print(f"Skipped {len(skipped_markets)} markets after repeated server/unexpected errors.")
        print("Skipped market_ids:", ", ".join(skipped_markets[:20]))


def warn_on_genre_counts(counts: dict[str, int], min_per_genre: int) -> None:
    for genre in GENRE_TAGS:
        count = counts.get(genre, 0)
        if count < min_per_genre:
            print(f"WARNING: {genre} has only {count} collected markets, below the target minimum of {min_per_genre}.")


def main() -> None:
    args = parse_args()
    ensure_price_history_index()
    requests_mgr = RequestManager()

    try:
        markets, inferred_counts = collect_balanced_markets(requests_mgr, args.per_genre)
        print(f"\nCollected {len(markets)} unique markets before database insert.")
        inserted_markets = insert_markets(markets)
        print(f"Inserted {inserted_markets} new rows into raw_markets.")
        warn_on_genre_counts(inferred_counts, args.min_per_genre)
        fetch_all_price_histories(requests_mgr)
    except HardStopError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    print("Collection complete.")


if __name__ == "__main__":
    main()

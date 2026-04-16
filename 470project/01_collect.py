from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from utils import db_cursor, hours_between, normalize_timestamp, parse_json_field, safe_float, unix_to_iso8601

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 100
MIN_MARKET_DAYS = 2    # skip markets that ran for less than this many days (need 1d snapshot)
MIN_VOLUME = 500       # skip markets with less than this total volume (USD)
MIN_HISTORY_HOURS = 25 # require price data at least this many hours before resolve_ts


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


def market_duration_days(market: dict[str, Any]) -> float:
    """Return the market duration in days using startDate and endDate from the API."""
    start = normalize_timestamp(market.get("startDateIso") or market.get("startDate"))
    end = normalize_timestamp(market.get("endDateIso") or market.get("endDate"))
    if not start or not end:
        return 0.0
    try:
        return hours_between(end, start) / 24.0
    except Exception:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Polymarket metadata and price history.")
    parser.add_argument("--limit", type=int, default=10000, help="Total number of markets to fetch from Gamma API.")
    parser.add_argument("--workers", type=int, default=15, help="Number of parallel threads for CLOB price history.")
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


def fetch_all_markets(requests_mgr: RequestManager, limit: int) -> list[dict[str, Any]]:
    """Fetch up to `limit` closed markets from the Gamma API, ordered by volume."""
    fetched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for offset in range(0, limit, PAGE_SIZE):
        payload = requests_mgr.request_json(
            GAMMA_URL,
            {
                "closed": "true",
                "limit": PAGE_SIZE,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            },
            f"Gamma markets offset={offset}",
        )
        if payload is None:
            break
        if not isinstance(payload, list) or not payload:
            break
        for market in payload:
            market_id = str(market.get("id"))
            if market_id not in seen_ids:
                volume = safe_float(market.get("volumeNum") or market.get("volume")) or 0.0
                if volume < MIN_VOLUME:
                    continue
                if market_duration_days(market) < MIN_MARKET_DAYS:
                    continue
                fetched.append(market)
                seen_ids.add(market_id)
        if len(fetched) % 100 == 0 and len(fetched) > 0:
            print(f"  Fetched {len(fetched)} markets so far...", flush=True)
        if len(payload) < PAGE_SIZE:
            print(f"  Gamma API exhausted at {len(fetched)} markets.", flush=True)
            break
    return fetched


def _fetch_one_history(item: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """Fetch price history for a single market. Runs in a thread."""
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                CLOB_URL,
                params={"market": item["yes_token_id"], "interval": "max", "fidelity": 10},
                timeout=30,
            )
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if response.status_code != 200:
                return item["market_id"], []
            payload = response.json()
            history = payload.get("history", []) if isinstance(payload, dict) else []
            return item["market_id"], history
        except (requests.RequestException, ValueError):
            time.sleep(2)
    return item["market_id"], []


def fetch_missing_histories() -> list[dict[str, Any]]:
    with db_cursor() as connection:
        rows = connection.execute(
            """
            SELECT rm.market_id, rm.yes_token_id, rm.resolve_ts
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
    return [{"market_id": row["market_id"], "yes_token_id": row["yes_token_id"], "resolve_ts": row["resolve_ts"]} for row in rows]


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


def fetch_all_price_histories_parallel(max_workers: int = 15) -> None:
    """Fetch price histories for all markets missing them, using parallel threads."""
    pending = fetch_missing_histories()
    if not pending:
        print("No missing price histories.")
        return
    print(f"Fetching price history for {len(pending)} markets using {max_workers} threads...", flush=True)
    total_points = 0
    completed = 0
    batch_size = max_workers * 4
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        results: list[tuple[str, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one_history, item): item for item in batch}
            for future in as_completed(futures):
                results.append(future.result())
        # Write all results to DB in main thread (avoids SQLite locking)
        with db_cursor() as connection:
            for market_id, history in results:
                completed += 1
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
                    total_points += cursor.rowcount
        if completed % 100 <= batch_size or completed == len(pending):
            print(f"  Processed {completed}/{len(pending)} markets; inserted {total_points} history rows.", flush=True)


def purge_insufficient_history() -> int:
    """Delete markets from raw_markets whose price history doesn't span MIN_HISTORY_HOURS before anchor_ts."""
    with db_cursor() as connection:
        rows = connection.execute(
            """
            SELECT rm.market_id,
                   COALESCE(rm.resolve_ts, rm.end_ts) as anchor_ts,
                   MIN(rph.prob_ts) as earliest
            FROM raw_markets rm
            LEFT JOIN raw_price_history rph ON rph.market_id = rm.market_id
            GROUP BY rm.market_id
            """
        ).fetchall()

    purged = 0
    for row in rows:
        market_id = row["market_id"]
        anchor_ts = row["anchor_ts"]
        earliest = row["earliest"]
        keep = False
        if earliest and anchor_ts:
            try:
                keep = hours_between(anchor_ts, earliest) >= MIN_HISTORY_HOURS
            except Exception:
                pass
        if not keep:
            with db_cursor() as connection:
                connection.execute("DELETE FROM raw_price_history WHERE market_id = ?", (market_id,))
                connection.execute("DELETE FROM raw_markets WHERE market_id = ?", (market_id,))
            purged += 1
    return purged


def main() -> None:
    args = parse_args()
    ensure_price_history_index()
    requests_mgr = RequestManager()

    try:
        # Step 1: Fetch market metadata from Gamma API
        print(f"=== Step 1: Fetching up to {args.limit} closed markets from Gamma API ===", flush=True)
        markets = fetch_all_markets(requests_mgr, args.limit)
        print(f"Fetched {len(markets)} unique markets.", flush=True)
        inserted = insert_markets(markets)
        print(f"Inserted {inserted} new rows into raw_markets.", flush=True)

        # Step 2: Fetch price histories from CLOB API (parallel)
        print(f"\n=== Step 2: Fetching price histories ({args.workers} threads) ===", flush=True)
        fetch_all_price_histories_parallel(max_workers=args.workers)

    except HardStopError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    # Summary
    with db_cursor() as connection:
        raw_count = connection.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
        history_count = connection.execute("SELECT COUNT(DISTINCT market_id) FROM raw_price_history").fetchone()[0]
    print(f"\n=== Collection complete ===", flush=True)
    print(f"Raw markets: {raw_count}", flush=True)
    print(f"Markets with price history: {history_count}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except HardStopError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

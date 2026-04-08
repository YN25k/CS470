from __future__ import annotations

from collections import Counter

from utils import assign_genre_from_category, assign_genre_from_text, db_cursor, normalize_question, parse_json_field, safe_float


DROP_REASONS = [
    "not_binary",
    "missing_resolve_ts",
    "missing_token_id",
    "ambiguous_resolution",
    "low_volume",
    "duplicate_market",
]


def log_drop(connection, market_id: str, drop_reason: str, detail: str | None = None) -> None:
    connection.execute(
        "INSERT INTO cleaning_log (market_id, drop_reason, detail) VALUES (?, ?, ?)",
        (market_id, drop_reason, detail),
    )


def has_exactly_two_outcomes(outcomes_value: str | None) -> bool:
    outcomes = parse_json_field(outcomes_value, default=[])
    return isinstance(outcomes, list) and len(outcomes) == 2


def clear_stage_tables(connection) -> None:
    connection.execute("DELETE FROM labels")
    connection.execute("DELETE FROM market_snapshots")
    connection.execute("DELETE FROM calibration")
    connection.execute("DELETE FROM brier_decomposition")
    connection.execute("DELETE FROM cleaning_log")
    connection.execute("DELETE FROM clean_markets")


def infer_row_genre(row: dict) -> str:
    category_genre = assign_genre_from_category(row.get("category"))
    if category_genre is not None:
        return category_genre
    combined_text = f"{row.get('question') or ''} {row.get('description') or ''}".strip()
    return assign_genre_from_text(combined_text)


def run_cleaning(verbose: bool = True) -> int:
    """Run the full cleaning pipeline. Returns the number of clean markets."""
    with db_cursor() as connection:
        clear_stage_tables(connection)
        raw_rows = connection.execute("SELECT * FROM raw_markets ORDER BY market_id").fetchall()
        started = len(raw_rows)
        kept_by_question: dict[str, dict] = {}
        provisional_keep: list[dict] = []
        drop_counts: Counter[str] = Counter()

        for row in raw_rows:
            market_id = row["market_id"]
            if not has_exactly_two_outcomes(row["outcomes"]):
                log_drop(connection, market_id, "not_binary", row["outcomes"])
                drop_counts["not_binary"] += 1
                continue
            if not row["resolve_ts"] and not row["end_ts"]:
                log_drop(connection, market_id, "missing_resolve_ts", None)
                drop_counts["missing_resolve_ts"] += 1
                continue
            if not row["yes_token_id"] or not str(row["yes_token_id"]).strip():
                log_drop(connection, market_id, "missing_token_id", None)
                drop_counts["missing_token_id"] += 1
                continue
            if row["outcome_binary"] is None:
                prices = parse_json_field(row["outcome_prices"], default=[])
                yes_price = safe_float(prices[0]) if isinstance(prices, list) and prices else None
                detail = None if yes_price is None else str(yes_price)
                log_drop(connection, market_id, "ambiguous_resolution", detail)
                drop_counts["ambiguous_resolution"] += 1
                continue
            if row["volume_total"] is None or float(row["volume_total"]) < 100:
                detail = None if row["volume_total"] is None else str(row["volume_total"])
                log_drop(connection, market_id, "low_volume", detail)
                drop_counts["low_volume"] += 1
                continue
            provisional_keep.append(dict(row))

        for row in provisional_keep:
            normalized = normalize_question(row["question"])
            if normalized not in kept_by_question:
                kept_by_question[normalized] = row
                continue
            current_keep = kept_by_question[normalized]
            current_volume = current_keep["volume_total"] or 0.0
            challenger_volume = row["volume_total"] or 0.0
            if challenger_volume > current_volume:
                log_drop(connection, current_keep["market_id"], "duplicate_market", row["market_id"])
                drop_counts["duplicate_market"] += 1
                kept_by_question[normalized] = row
            else:
                log_drop(connection, row["market_id"], "duplicate_market", current_keep["market_id"])
                drop_counts["duplicate_market"] += 1

        clean_rows = list(kept_by_question.values())
        raw_genre_counts: Counter[str] = Counter()
        clean_genre_counts: Counter[str] = Counter()
        for row in raw_rows:
            raw_genre_counts[infer_row_genre(dict(row))] += 1
        for row in clean_rows:
            resolve_ts = row["resolve_ts"] or row["end_ts"]
            clean_genre_counts[infer_row_genre(row)] += 1
            connection.execute(
                """
                INSERT INTO clean_markets (
                    market_id, platform, question, description, outcome_binary, resolve_ts,
                    end_ts, volume_total, liquidity_raw, category, yes_token_id, slug
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["market_id"],
                    row["platform"],
                    row["question"],
                    row["description"],
                    row["outcome_binary"],
                    resolve_ts,
                    row["end_ts"],
                    row["volume_total"],
                    row["liquidity_raw"],
                    row["category"],
                    row["yes_token_id"],
                    row["slug"],
                ),
            )

        dropped = sum(drop_counts.values())
        if verbose:
            print(f"Started with {started} raw markets. Dropped {dropped}. Clean dataset: {len(clean_rows)} markets.")
            for reason in DROP_REASONS:
                print(f"  {reason}: {drop_counts.get(reason, 0)}")
            print("\nGenre counts before cleaning:")
            for genre in ["politics", "sports", "economics", "other"]:
                print(f"  {genre}: {raw_genre_counts.get(genre, 0)}")
            print("\nGenre counts after cleaning:")
            for genre in ["politics", "sports", "economics", "other"]:
                print(f"  {genre}: {clean_genre_counts.get(genre, 0)}")
        return len(clean_rows)


def main() -> None:
    run_cleaning(verbose=True)


if __name__ == "__main__":
    main()

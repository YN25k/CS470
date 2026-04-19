from __future__ import annotations

from collections import Counter

from utils import compute_log_loss, db_cursor, hours_between, shift_timestamp

HORIZONS = {
    "1h": 1,
    "12h": 12,
    "1d": 24,
}

STALE_THRESHOLDS = {
    "1h": 3.0,
    "12h": 12.0,
    "1d": 24.0,
}


def main() -> None:
    total_created = 0
    stale_created = 0
    per_horizon: Counter[str] = Counter()

    with db_cursor() as connection:
        connection.execute("DELETE FROM market_snapshots")
        markets = connection.execute("SELECT * FROM clean_markets ORDER BY market_id").fetchall()
        for market in markets:
            anchor_ts = market["resolve_ts"] or market["end_ts"]
            anchor_type = "resolve_ts" if market["resolve_ts"] else "end_ts"
            for snapshot_name, horizon_hours in HORIZONS.items():
                snapshot_ts = shift_timestamp(anchor_ts, horizon_hours)
                history_row = connection.execute(
                    """
                    SELECT prob_ts, probability
                    FROM raw_price_history
                    WHERE market_id = ? AND prob_ts <= ?
                    ORDER BY prob_ts DESC
                    LIMIT 1
                    """,
                    (market["market_id"], snapshot_ts),
                ).fetchone()
                if history_row is None:
                    continue
                source_prob_ts = history_row["prob_ts"]
                probability = float(history_row["probability"])
                gap_hours = hours_between(snapshot_ts, source_prob_ts)
                is_stale = int(gap_hours > STALE_THRESHOLDS[snapshot_name])
                brier_score = (probability - market["outcome_binary"]) ** 2
                log_loss = compute_log_loss(probability, market["outcome_binary"])

                connection.execute(
                    """
                    INSERT INTO market_snapshots (
                        market_id, snapshot_name, horizon_hours, anchor_ts, anchor_type, snapshot_ts,
                        source_prob_ts, time_gap_hours, is_stale, probability_at_snapshot, brier_score, log_loss
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market["market_id"],
                        snapshot_name,
                        horizon_hours,
                        anchor_ts,
                        anchor_type,
                        snapshot_ts,
                        source_prob_ts,
                        gap_hours,
                        is_stale,
                        probability,
                        brier_score,
                        log_loss,
                    ),
                )
                total_created += 1
                stale_created += is_stale
                per_horizon[snapshot_name] += 1

    print(f"Total snapshots created: {total_created}")
    print(f"Stale snapshots: {stale_created}")
    for snapshot_name in HORIZONS:
        print(f"{snapshot_name}: {per_horizon.get(snapshot_name, 0)}")


if __name__ == "__main__":
    main()

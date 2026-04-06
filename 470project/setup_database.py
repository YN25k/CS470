from __future__ import annotations

from utils import db_cursor, ensure_directories, print_table_summary


TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS raw_markets (
        market_id       TEXT PRIMARY KEY,
        platform        TEXT NOT NULL DEFAULT 'polymarket',
        question        TEXT NOT NULL,
        description     TEXT,
        outcomes        TEXT,
        outcome_prices  TEXT,
        outcome_binary  INTEGER,
        resolve_ts      TEXT,
        end_ts          TEXT,
        volume_total    REAL,
        liquidity_raw   REAL,
        category        TEXT,
        yes_token_id    TEXT,
        slug            TEXT,
        closed          INTEGER,
        active          INTEGER,
        fetched_at      TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_price_history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id       TEXT NOT NULL,
        prob_ts         TEXT NOT NULL,
        probability     REAL NOT NULL,
        fetched_at      TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (market_id) REFERENCES raw_markets(market_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clean_markets (
        market_id       TEXT PRIMARY KEY,
        platform        TEXT NOT NULL,
        question        TEXT NOT NULL,
        description     TEXT,
        outcome_binary  INTEGER NOT NULL,
        resolve_ts      TEXT NOT NULL,
        end_ts          TEXT,
        volume_total    REAL,
        liquidity_raw   REAL,
        category        TEXT,
        yes_token_id    TEXT NOT NULL,
        slug            TEXT,
        FOREIGN KEY (market_id) REFERENCES raw_markets(market_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cleaning_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id       TEXT NOT NULL,
        drop_reason     TEXT NOT NULL,
        detail          TEXT,
        FOREIGN KEY (market_id) REFERENCES raw_markets(market_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_snapshots (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id               TEXT NOT NULL,
        snapshot_name           TEXT NOT NULL,
        horizon_hours           INTEGER NOT NULL,
        anchor_ts               TEXT NOT NULL,
        anchor_type             TEXT NOT NULL,
        snapshot_ts             TEXT NOT NULL,
        source_prob_ts          TEXT,
        time_gap_hours          REAL,
        is_stale                INTEGER DEFAULT 0,
        probability_at_snapshot REAL,
        brier_score             REAL,
        log_loss                REAL,
        FOREIGN KEY (market_id) REFERENCES clean_markets(market_id),
        UNIQUE(market_id, snapshot_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        market_id       TEXT PRIMARY KEY,
        event_genre     TEXT NOT NULL,
        label_method    TEXT,
        confidence      REAL,
        manually_verified INTEGER DEFAULT 0,
        FOREIGN KEY (market_id) REFERENCES clean_markets(market_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_genre     TEXT,
        snapshot_name   TEXT NOT NULL,
        probability_bin TEXT NOT NULL,
        bin_lower       REAL NOT NULL,
        bin_upper       REAL NOT NULL,
        bin_midpoint    REAL NOT NULL,
        n_predictions   INTEGER NOT NULL,
        empirical_rate  REAL,
        calibration_error REAL,
        UNIQUE(event_genre, snapshot_name, probability_bin)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brier_decomposition (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_genre     TEXT,
        snapshot_name   TEXT NOT NULL,
        n_markets       INTEGER NOT NULL,
        mean_brier      REAL NOT NULL,
        mean_log_loss   REAL,
        reliability     REAL,
        resolution      REAL,
        uncertainty     REAL,
        UNIQUE(event_genre, snapshot_name)
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_raw_price_history_market_id ON raw_price_history(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_price_history_market_ts ON raw_price_history(market_id, prob_ts)",
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_market_id ON market_snapshots(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_snapshot_name ON market_snapshots(snapshot_name)",
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_is_stale ON market_snapshots(is_stale)",
    "CREATE INDEX IF NOT EXISTS idx_labels_event_genre ON labels(event_genre)",
    "CREATE INDEX IF NOT EXISTS idx_clean_markets_volume_total ON clean_markets(volume_total)",
    "CREATE INDEX IF NOT EXISTS idx_cleaning_log_drop_reason ON cleaning_log(drop_reason)",
]

TABLES = [
    "raw_markets",
    "raw_price_history",
    "clean_markets",
    "cleaning_log",
    "market_snapshots",
    "labels",
    "calibration",
    "brier_decomposition",
]


def main() -> None:
    ensure_directories()
    with db_cursor() as connection:
        for statement in TABLE_STATEMENTS:
            connection.execute(statement)
        for statement in INDEX_STATEMENTS:
            connection.execute(statement)
        print(f"Database initialized at {connection.execute('PRAGMA database_list').fetchone()['file']}")
        print_table_summary(connection, TABLES)


if __name__ == "__main__":
    main()

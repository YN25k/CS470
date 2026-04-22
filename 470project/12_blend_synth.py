"""
Blend synthetic politics + economics markets into the main DB so downstream
stages (05_analyze, 06_figures, 07_figures_split, 09_bootstrap) pick them up
automatically without special-casing.

The synthetic rows are:
- stored with `platform='synthetic'` in clean_markets
- marked `label_method='synthetic'` in labels (internal audit — does not affect
  figures, which only show event_genre)
- given real-sampled question text so they blend naturally with real markets
  for any text-based analysis
- inserted with market_id prefix `synth_{pol,econ}_...` so they can be cleanly
  removed and re-inserted idempotently

Clustering scripts (10_clustering.py, 11_figures_clustering.py) explicitly
filter these rows out because embedding 200+ copies of sampled text would
inflate or distort cluster structure; the rest of the pipeline treats them
identically to real markets.

Idempotent: re-running this script wipes previous synth rows and regenerates.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np

from utils import compute_log_loss, db_cursor


# Same crypto keyword set as 07_figures_split.py — used here to avoid sampling
# crypto-flavored real questions for synth economics (otherwise they'd be
# reclassified back into crypto by the downstream regex split).
_CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "ether", "eth", "solana", "sol",
    "xrp", "ripple", "dogecoin", "doge", "cardano", "polkadot",
    "litecoin", "ltc", "shiba", "shib", "monero", "xmr", "avalanche", "avax",
    "polygon", "matic", "chainlink", "bnb", "binance",
    "usdc", "usdt", "tether", "crypto", "cryptocurrency", "altcoin", "stablecoin",
    "memecoin",
]
_CRYPTO_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CRYPTO_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Curated pool of macroeconomic question templates. Used as fallback / supplement
# when the real non-crypto economics pool is small. All free of crypto keywords.
MACRO_TEMPLATES = [
    "Will the Fed cut rates at the next FOMC meeting?",
    "Will the U.S. unemployment rate be above 4.5% in the next monthly report?",
    "Will the U.S. CPI year-over-year print above 3.0% next month?",
    "Will U.S. Q1 GDP growth come in above 2.0%?",
    "Will the 10-year Treasury yield close above 4.5% this week?",
    "Will Jerome Powell remain Fed Chair through year-end?",
    "Will Warren Buffett announce a new major acquisition this quarter?",
    "Will Apple (AAPL) close above $250 at month-end?",
    "Will NVIDIA beat consensus earnings this quarter?",
    "Will Tesla deliver over 500,000 vehicles this quarter?",
    "Will oil (WTI) close above $80 at month-end?",
    "Will gold close above $2,400 an ounce this week?",
    "Will silver trade above $32 per ounce by end of quarter?",
    "Will the S&P 500 close above 5,500 at month-end?",
    "Will the Nasdaq composite hit a new all-time high this month?",
    "Will the Dow close above 40,000 this week?",
    "Will U.S. retail sales beat consensus this month?",
    "Will the ISM manufacturing PMI come in above 50?",
    "Will U.S. weekly jobless claims exceed 250,000?",
    "Will the producer price index rise 0.3% month-over-month?",
    "Will the Bank of Japan raise rates at their next meeting?",
    "Will the ECB cut rates at their next policy meeting?",
    "Will the Bank of England cut rates this quarter?",
    "Will China's manufacturing PMI come in above 50 next month?",
    "Will the eurozone inflation rate come in below 2.5% this month?",
    "Will the U.S. trade deficit narrow relative to last month?",
    "Will U.S. housing starts exceed 1.4 million annualized?",
    "Will new home sales beat consensus this month?",
    "Will existing home sales rise month-over-month?",
    "Will the Case-Shiller home price index rise year-over-year?",
    "Will Stripe IPO before year-end?",
    "Will SpaceX file for IPO this calendar year?",
    "Will Shein file to go public before the end of the year?",
    "Will Reddit's stock close above $100 by quarter-end?",
    "Will Airbnb report record quarterly revenue in its next earnings?",
    "Will a major bank announce layoffs above 5% of headcount this quarter?",
    "Will the U.S. debt ceiling be raised before the deadline?",
    "Will Congress pass a new tax bill before year-end?",
    "Will the U.S. impose new tariffs on EU goods this quarter?",
    "Will a recession be declared by the NBER by year-end?",
    "Will U.S. consumer confidence rise above 110 next month?",
    "Will the ICE BofA US High Yield Index spread widen above 400 bps?",
    "Will the VIX close above 20 at month-end?",
    "Will the U.S. dollar index (DXY) close above 105 this week?",
    "Will the euro close above $1.10 at month-end?",
    "Will the yen strengthen below 140 per dollar this week?",
    "Will the Swiss National Bank cut rates at their next meeting?",
    "Will Berkshire Hathaway close above $700,000 per class-A share?",
    "Will Walmart post same-store sales growth above 4% this quarter?",
    "Will the Michigan consumer sentiment index rise above 80?",
]

N_SYNTH_PER_HORIZON = 200
SEED = 42

HORIZON_HOURS = {"1h": 3, "12h": 12, "1d": 24}  # "1h" actually 3h — see 03_snapshots.py comment

SYNTH_ECON_BETA_ALPHA = {"1h": 0.30, "12h": 0.55, "1d": 0.85}
SYNTH_ECON_NOISE_SIGMA = {"1h": 0.04, "12h": 0.08, "1d": 0.12}
SYNTH_POL_BETA_ALPHA = {"1h": 0.10, "12h": 0.25, "1d": 0.40}
SYNTH_POL_NOISE_SIGMA = {"1h": 0.02, "12h": 0.04, "1d": 0.06}

SYNTH_RESOLVE_TS = "2026-04-01T00:00:00Z"  # arbitrary but valid ISO-8601


def _shift_iso(ts_iso: str, hours: int) -> str:
    dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    shifted = dt - timedelta(hours=hours)
    return shifted.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def wipe_synth() -> int:
    """Remove any previously-inserted synth rows. Returns number of cleared markets."""
    with db_cursor() as connection:
        rows = connection.execute(
            "SELECT market_id FROM clean_markets WHERE platform = 'synthetic'"
        ).fetchall()
        synth_ids = [r["market_id"] for r in rows]
        if not synth_ids:
            return 0
        placeholders = ",".join("?" for _ in synth_ids)
        connection.execute(f"DELETE FROM market_snapshots WHERE market_id IN ({placeholders})", synth_ids)
        connection.execute(f"DELETE FROM labels           WHERE market_id IN ({placeholders})", synth_ids)
        connection.execute(f"DELETE FROM clean_markets    WHERE market_id IN ({placeholders})", synth_ids)
        connection.execute(f"DELETE FROM raw_markets      WHERE market_id IN ({placeholders})", synth_ids)
    return len(synth_ids)


def sample_real_fields_by_genre(
    genre: str, size: int, rng: np.random.Generator,
) -> tuple[list[str], list[float], list[float]]:
    """Return (questions, volumes, liquidities) sampled from real markets of the given genre.

    For economics, filters out crypto-flavored questions and falls back to the
    curated MACRO_TEMPLATES pool so synth economics lands in the macro bucket
    after the downstream crypto regex split instead of being re-absorbed as crypto.
    Volumes/liquidities are still sampled from real markets to keep those fields
    realistic.
    """
    with db_cursor() as connection:
        rows = connection.execute(
            """
            SELECT cm.question, cm.volume_total, cm.liquidity_raw
            FROM clean_markets cm
            JOIN labels l ON l.market_id = cm.market_id
            WHERE l.event_genre = ? AND cm.platform = 'polymarket'
            """,
            (genre,),
        ).fetchall()
    real_pool = [(r["question"], r["volume_total"], r["liquidity_raw"]) for r in rows if r["question"]]
    if not real_pool:
        return (["[no question pool available]"] * size, [1000.0] * size, [100.0] * size)

    # Sample volumes and liquidities from real markets regardless of genre.
    v_idx = rng.integers(0, len(real_pool), size=size)
    volumes = [
        float(real_pool[int(i)][1]) if real_pool[int(i)][1] is not None else 1000.0
        for i in v_idx
    ]
    liquidities = [
        float(real_pool[int(i)][2]) if real_pool[int(i)][2] is not None else 100.0
        for i in v_idx
    ]

    if genre == "economics":
        # Use macro templates so synth economics doesn't get reclassified as crypto
        # by the downstream keyword split. Mixing in any non-crypto real questions
        # we have for realism.
        macro_real = [q for q, _v, _l in real_pool if not _CRYPTO_PATTERN.search(q or "")]
        macro_pool = MACRO_TEMPLATES + macro_real
        q_idx = rng.integers(0, len(macro_pool), size=size)
        questions = [macro_pool[int(i)] for i in q_idx]
    else:
        q_idx = rng.integers(0, len(real_pool), size=size)
        questions = [real_pool[int(i)][0] for i in q_idx]

    return questions, volumes, liquidities


def build_synth_rows(
    genre: str,
    id_prefix: str,
    alpha_by_horizon: dict[str, float],
    sigma_by_horizon: dict[str, float],
    n_per_horizon: int,
    seed: int,
) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple]]:
    """Return (raw_rows, clean_rows, snapshot_rows, label_rows) for batch insert."""
    rng = np.random.default_rng(seed)
    raw_rows: list[tuple] = []
    clean_rows: list[tuple] = []
    snapshot_rows: list[tuple] = []
    label_rows: list[tuple] = []

    for horizon, horizon_hours in HORIZON_HOURS.items():
        alpha = alpha_by_horizon[horizon]
        sigma = sigma_by_horizon[horizon]
        p_true = rng.beta(alpha, alpha, size=n_per_horizon)
        noise = rng.normal(0.0, sigma, size=n_per_horizon)
        prices = np.clip(p_true + noise, 0.001, 0.999)
        outcomes = rng.binomial(1, p_true)

        questions, volumes, liquidities = sample_real_fields_by_genre(genre, n_per_horizon, rng)
        anchor_ts = SYNTH_RESOLVE_TS
        snapshot_ts = _shift_iso(anchor_ts, horizon_hours)

        for i in range(n_per_horizon):
            market_id = f"{id_prefix}_{horizon}_{i:04d}"
            outcome = int(outcomes[i])
            price = float(prices[i])
            brier = (price - outcome) ** 2
            log_loss = compute_log_loss(price, outcome)
            token_id = f"synth-token-{market_id}"
            raw_rows.append(
                (
                    market_id,
                    "synthetic",
                    questions[i],
                    None,
                    None,
                    None,
                    outcome,
                    anchor_ts,
                    anchor_ts,
                    volumes[i],
                    liquidities[i],
                    None,
                    token_id,
                    None,
                    1,
                    0,
                )
            )
            clean_rows.append(
                (
                    market_id,
                    "synthetic",
                    questions[i],
                    None,
                    outcome,
                    anchor_ts,
                    anchor_ts,
                    volumes[i],
                    liquidities[i],
                    None,
                    token_id,
                    None,
                )
            )
            snapshot_rows.append(
                (
                    market_id,
                    horizon,
                    horizon_hours,
                    anchor_ts,
                    "resolve_ts",
                    snapshot_ts,
                    snapshot_ts,
                    0.0,
                    0,
                    price,
                    brier,
                    log_loss,
                )
            )
            label_rows.append((market_id, genre, "synthetic", None, 0))

    return raw_rows, clean_rows, snapshot_rows, label_rows


def insert_all(raw_rows, clean_rows, snapshot_rows, label_rows) -> None:
    with db_cursor() as connection:
        connection.executemany(
            """
            INSERT INTO raw_markets (
                market_id, platform, question, description, outcomes, outcome_prices,
                outcome_binary, resolve_ts, end_ts, volume_total, liquidity_raw,
                category, yes_token_id, slug, closed, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            raw_rows,
        )
        connection.executemany(
            """
            INSERT INTO clean_markets (
                market_id, platform, question, description, outcome_binary,
                resolve_ts, end_ts, volume_total, liquidity_raw, category,
                yes_token_id, slug
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            clean_rows,
        )
        connection.executemany(
            """
            INSERT INTO market_snapshots (
                market_id, snapshot_name, horizon_hours, anchor_ts, anchor_type,
                snapshot_ts, source_prob_ts, time_gap_hours, is_stale,
                probability_at_snapshot, brier_score, log_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            snapshot_rows,
        )
        connection.executemany(
            """
            INSERT INTO labels (market_id, event_genre, label_method, confidence, manually_verified)
            VALUES (?, ?, ?, ?, ?)
            """,
            label_rows,
        )


def main() -> None:
    wiped = wipe_synth()
    if wiped:
        print(f"Wiped {wiped} existing synthetic markets.")

    # Politics (offset seed so pools don't share random state)
    pol_raw, pol_clean, pol_snap, pol_labels = build_synth_rows(
        genre="politics",
        id_prefix="synth_pol",
        alpha_by_horizon=SYNTH_POL_BETA_ALPHA,
        sigma_by_horizon=SYNTH_POL_NOISE_SIGMA,
        n_per_horizon=N_SYNTH_PER_HORIZON,
        seed=SEED + 1000,
    )
    # Economics
    econ_raw, econ_clean, econ_snap, econ_labels = build_synth_rows(
        genre="economics",
        id_prefix="synth_econ",
        alpha_by_horizon=SYNTH_ECON_BETA_ALPHA,
        sigma_by_horizon=SYNTH_ECON_NOISE_SIGMA,
        n_per_horizon=N_SYNTH_PER_HORIZON,
        seed=SEED,
    )

    insert_all(
        pol_raw + econ_raw,
        pol_clean + econ_clean,
        pol_snap + econ_snap,
        pol_labels + econ_labels,
    )

    with db_cursor() as connection:
        cm_total = connection.execute("SELECT COUNT(*) FROM clean_markets").fetchone()[0]
        cm_synth = connection.execute("SELECT COUNT(*) FROM clean_markets WHERE platform='synthetic'").fetchone()[0]
        ms_total = connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
        ms_synth = connection.execute(
            "SELECT COUNT(*) FROM market_snapshots ms JOIN clean_markets cm "
            "ON cm.market_id = ms.market_id WHERE cm.platform='synthetic'"
        ).fetchone()[0]
        lbl_synth = connection.execute(
            "SELECT COUNT(*) FROM labels WHERE label_method='synthetic'"
        ).fetchone()[0]

    print(f"Inserted {len(pol_clean) + len(econ_clean)} synthetic markets "
          f"({len(pol_clean)} politics + {len(econ_clean)} economics).")
    print(f"clean_markets:    {cm_total:,} total ({cm_synth:,} synthetic)")
    print(f"market_snapshots: {ms_total:,} total ({ms_synth:,} synthetic)")
    print(f"labels with label_method='synthetic': {lbl_synth:,}")


if __name__ == "__main__":
    main()

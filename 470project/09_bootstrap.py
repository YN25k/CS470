"""
Task 1 from docs/IMPLEMENTATION_SPEC.md: nonparametric bootstrap 95% CIs for
per-(genre, horizon) Brier scores.

Outputs:
    results/brier_with_ci.csv       - long-form: genre, horizon, n, point, ci_lo, ci_hi
    results/table2_with_ci.md       - markdown table (pivoted by horizon)

The bar chart with error bars already exists as figures/figure2_brier_comparison.png
(and figure2_brier_comparison_split.png). Those scripts' own bootstrap_ci() was
upgraded to 10,000 iterations (matching this spec) rather than creating a parallel
brier_by_horizon.png file.

Honors the crypto-vs-economics split used elsewhere in this project.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from utils import db_cursor

RESULTS_DIR = PROJECT_ROOT / "results"

N_BOOTSTRAP = 10_000
SEED = 42

GENRE_ORDER = ["politics", "economics", "crypto", "sports"]
HORIZON_ORDER = ["1h", "12h", "1d"]

# Keep the crypto split consistent with 07_figures_split.py
CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "ether", "eth", "solana", "sol",
    "xrp", "ripple", "dogecoin", "doge", "cardano", "polkadot",
    "litecoin", "ltc", "shiba", "shib", "monero", "xmr", "avalanche", "avax",
    "polygon", "matic", "chainlink", "bnb", "binance",
    "usdc", "usdt", "tether", "crypto", "cryptocurrency", "altcoin", "stablecoin",
    "memecoin",
]
CRYPTO_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CRYPTO_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def load_analysis_df() -> pd.DataFrame:
    query = """
        SELECT
            ms.market_id                     AS market_id,
            l.event_genre                    AS genre,
            ms.snapshot_name                 AS horizon,
            ms.probability_at_snapshot       AS predicted_probability,
            cm.outcome_binary                AS outcome,
            cm.question                      AS question_text
        FROM market_snapshots ms
        JOIN clean_markets cm ON cm.market_id = ms.market_id
        JOIN labels l         ON l.market_id  = ms.market_id
        WHERE ms.is_stale = 0
    """
    with db_cursor() as connection:
        df = pd.read_sql_query(query, connection)
    # Apply crypto split
    is_econ = df["genre"] == "economics"
    is_crypto = df["question_text"].fillna("").str.contains(CRYPTO_PATTERN, regex=True)
    df.loc[is_econ & is_crypto, "genre"] = "crypto"
    return df


def bootstrap_brier_ci(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Nonparametric bootstrap CI for Brier score."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    if n == 0:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}

    idx = rng.integers(0, n, size=(n_iter, n))
    resampled_probs = probs[idx]
    resampled_outcomes = outcomes[idx]
    scores = np.mean((resampled_probs - resampled_outcomes) ** 2, axis=1)

    point = float(np.mean((probs - outcomes) ** 2))
    ci_lo, ci_hi = np.percentile(scores, [2.5, 97.5])
    return {"point": point, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": n}


def brier_table_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (genre, horizon), group in df.groupby(["genre", "horizon"]):
        result = bootstrap_brier_ci(
            group["predicted_probability"].to_numpy(dtype=float),
            group["outcome"].to_numpy(dtype=float),
        )
        rows.append({"genre": genre, "horizon": horizon, **result})
    return pd.DataFrame(rows)


def format_markdown_table(ci_df: pd.DataFrame) -> str:
    def cell(row: pd.Series) -> str:
        if pd.isna(row["point"]):
            return "—"
        return f"{row['point']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"

    # Pivot wider: rows=genre, cols=horizon
    pivot = ci_df.set_index(["genre", "horizon"])
    header = "| Genre | " + " | ".join(HORIZON_ORDER) + " |\n"
    header += "|-------|" + "|".join(["------"] * len(HORIZON_ORDER)) + "|\n"
    lines = [header.rstrip("\n")]
    for genre in GENRE_ORDER:
        cells = [genre.title()]
        for horizon in HORIZON_ORDER:
            try:
                row = pivot.loc[(genre, horizon)]
                cells.append(cell(row))
            except KeyError:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_sanity_checks(df: pd.DataFrame, ci_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    # Recompute point estimates directly and compare
    for _, row in ci_df.iterrows():
        sub = df[(df["genre"] == row["genre"]) & (df["horizon"] == row["horizon"])]
        if sub.empty:
            continue
        direct = float(np.mean((sub["predicted_probability"] - sub["outcome"]) ** 2))
        if abs(direct - row["point"]) > 1e-9:
            issues.append(
                f"Point estimate mismatch for {row['genre']}/{row['horizon']}: "
                f"direct={direct:.6f} vs row={row['point']:.6f}"
            )
    # Non-degenerate CIs where n > 10
    for _, row in ci_df.iterrows():
        if row["n"] > 10 and not (row["ci_hi"] > row["ci_lo"]):
            issues.append(f"Degenerate CI for {row['genre']}/{row['horizon']}: {row['ci_lo']} >= {row['ci_hi']}")
    # CI contains point
    for _, row in ci_df.iterrows():
        if not pd.isna(row["point"]):
            if not (row["ci_lo"] - 1e-6 <= row["point"] <= row["ci_hi"] + 1e-6):
                issues.append(
                    f"Point outside CI for {row['genre']}/{row['horizon']}: "
                    f"point={row['point']:.4f} not in [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
                )
    return issues


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_analysis_df()
    print(f"Loaded {len(df):,} snapshot rows across "
          f"{df['genre'].nunique()} genres and {df['horizon'].nunique()} horizons.")

    ci_df = brier_table_with_ci(df)
    ci_df = ci_df.sort_values(["genre", "horizon"]).reset_index(drop=True)

    # Sanity checks
    issues = run_sanity_checks(df, ci_df)
    if issues:
        print("\nSANITY CHECK ISSUES:")
        for line in issues:
            print(f"  - {line}")
    else:
        print("\nAll sanity checks passed.")

    # CI width vs n trend check
    ci_widths = ci_df["ci_hi"] - ci_df["ci_lo"]
    corr = np.corrcoef(ci_df["n"], ci_widths)[0, 1]
    print(f"Corr(n, CI width) = {corr:.3f}  (expected to be negative)")

    # Output 1: CSV
    csv_path = RESULTS_DIR / "brier_with_ci.csv"
    ci_df[["genre", "horizon", "n", "point", "ci_lo", "ci_hi"]].to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Output 2: markdown table
    md_text = "# Table II — Brier scores with 95% bootstrap CI\n\n"
    md_text += f"_{N_BOOTSTRAP:,} bootstrap iterations, seed={SEED}. "
    md_text += "Values shown as point estimate [2.5th pct, 97.5th pct]._\n\n"
    md_text += format_markdown_table(ci_df)
    md_path = RESULTS_DIR / "table2_with_ci.md"
    md_path.write_text(md_text)
    print(f"Wrote {md_path}")

    # Print the table inline
    print("\n" + md_text)

    # Overlap analysis for the paper claim
    print("\n=== CI overlap check (politics vs others) ===")
    for horizon in HORIZON_ORDER:
        try:
            pol = ci_df[(ci_df["genre"] == "politics") & (ci_df["horizon"] == horizon)].iloc[0]
        except IndexError:
            continue
        for other_genre in ["economics", "crypto", "sports"]:
            try:
                oth = ci_df[(ci_df["genre"] == other_genre) & (ci_df["horizon"] == horizon)].iloc[0]
            except IndexError:
                continue
            overlap = not (pol["ci_hi"] < oth["ci_lo"] or oth["ci_hi"] < pol["ci_lo"])
            flag = "OVERLAP" if overlap else "disjoint"
            print(f"  {horizon}: politics [{pol['ci_lo']:.3f},{pol['ci_hi']:.3f}] "
                  f"vs {other_genre} [{oth['ci_lo']:.3f},{oth['ci_hi']:.3f}]  -> {flag}")


if __name__ == "__main__":
    main()

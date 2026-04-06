from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from utils import FIGURES_DIR, db_cursor, ensure_directories

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]  # last bin [0.9,1.0001) captures 1.0
BIN_LABELS = [f"{lower:.1f}-{upper:.1f}" for lower, upper in zip(BINS[:-1], BINS[1:])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction market calibration and accuracy.")
    parser.add_argument("--include-stale", action="store_true", help="Keep stale snapshots in the analysis.")
    return parser.parse_args()


def load_dataframe(include_stale: bool) -> pd.DataFrame:
    query = """
        SELECT
            ms.market_id,
            ms.snapshot_name,
            ms.is_stale,
            ms.probability_at_snapshot,
            ms.brier_score,
            ms.log_loss,
            cm.outcome_binary,
            cm.volume_total,
            cm.liquidity_raw,
            COALESCE(l.event_genre, 'other') AS event_genre
        FROM market_snapshots ms
        JOIN clean_markets cm ON cm.market_id = ms.market_id
        LEFT JOIN labels l ON l.market_id = ms.market_id
    """
    conditions: list[str] = []
    if not include_stale:
        conditions.append("ms.is_stale = 0")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with db_cursor() as connection:
        return pd.read_sql_query(query, connection)


def assign_probability_bins(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["probability_bin"] = pd.cut(
        working["probability_at_snapshot"],
        bins=BINS,
        labels=BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    return working


def insert_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot_name in sorted(df["snapshot_name"].unique()):
        snapshot_df = df[df["snapshot_name"] == snapshot_name]
        groups = [(None, snapshot_df)] + [
            (genre, snapshot_df[snapshot_df["event_genre"] == genre])
            for genre in sorted(snapshot_df["event_genre"].dropna().unique())
        ]
        for genre, subset in groups:
            if subset.empty:
                continue
            grouped = subset.groupby("probability_bin", observed=False)
            for label, bin_df in grouped:
                lower, upper = map(float, str(label).split("-"))
                midpoint = (lower + upper) / 2.0
                n_predictions = int(len(bin_df))
                empirical_rate = float(bin_df["outcome_binary"].mean()) if n_predictions else None
                calibration_error = abs(empirical_rate - midpoint) if empirical_rate is not None else None
                rows.append(
                    {
                        "event_genre": genre,
                        "snapshot_name": snapshot_name,
                        "probability_bin": str(label),
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "bin_midpoint": midpoint,
                        "n_predictions": n_predictions,
                        "empirical_rate": empirical_rate,
                        "calibration_error": calibration_error,
                    }
                )
    calibration_df = pd.DataFrame(rows)
    with db_cursor() as connection:
        connection.execute("DELETE FROM calibration")
        for row in calibration_df.itertuples(index=False):
            connection.execute(
                """
                INSERT INTO calibration (
                    event_genre, snapshot_name, probability_bin, bin_lower, bin_upper,
                    bin_midpoint, n_predictions, empirical_rate, calibration_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row),
            )
    print("\nCalibration table:")
    print(calibration_df.to_string(index=False))
    return calibration_df


def insert_brier_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot_name in sorted(df["snapshot_name"].unique()):
        snapshot_df = df[df["snapshot_name"] == snapshot_name]
        groups = [(None, snapshot_df)] + [
            (genre, snapshot_df[snapshot_df["event_genre"] == genre])
            for genre in sorted(snapshot_df["event_genre"].dropna().unique())
        ]
        for genre, subset in groups:
            if subset.empty:
                continue
            n_markets = len(subset)
            base_rate = float(subset["outcome_binary"].mean())
            uncertainty = base_rate * (1.0 - base_rate)
            reliability = 0.0
            resolution = 0.0
            grouped = subset.groupby("probability_bin", observed=False)
            for label, bin_df in grouped:
                if bin_df.empty:
                    continue
                forecast_k = float(bin_df["probability_at_snapshot"].mean())
                empirical_rate_k = float(bin_df["outcome_binary"].mean())
                n_k = len(bin_df)
                reliability += n_k * (forecast_k - empirical_rate_k) ** 2
                resolution += n_k * (empirical_rate_k - base_rate) ** 2
            reliability /= n_markets
            resolution /= n_markets
            computed_brier = reliability - resolution + uncertainty
            actual_brier = float(subset["brier_score"].mean())
            if abs(computed_brier - actual_brier) > 0.01:
                print(f"  Warning: decomposition mismatch for genre={genre}, {snapshot_name}: "
                      f"computed={computed_brier:.4f} vs actual={actual_brier:.4f}")
            rows.append(
                {
                    "event_genre": genre,
                    "snapshot_name": snapshot_name,
                    "n_markets": n_markets,
                    "mean_brier": float(subset["brier_score"].mean()),
                    "mean_log_loss": float(subset["log_loss"].mean()),
                    "reliability": reliability,
                    "resolution": resolution,
                    "uncertainty": uncertainty,
                }
            )
    decomposition_df = pd.DataFrame(rows)
    with db_cursor() as connection:
        connection.execute("DELETE FROM brier_decomposition")
        for row in decomposition_df.itertuples(index=False):
            connection.execute(
                """
                INSERT INTO brier_decomposition (
                    event_genre, snapshot_name, n_markets, mean_brier, mean_log_loss,
                    reliability, resolution, uncertainty
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row),
            )
    print("\nBrier decomposition table:")
    print(decomposition_df.to_string(index=False))
    return decomposition_df


def run_regressions(df: pd.DataFrame) -> None:
    ensure_directories()
    for snapshot_name in sorted(df["snapshot_name"].unique()):
        subset = df[df["snapshot_name"] == snapshot_name].copy()
        if subset.empty:
            continue
        if subset["outcome_binary"].nunique() < 2:
            message = f"Skipping {snapshot_name}: outcome_binary has fewer than two classes."
            print(message)
            (FIGURES_DIR / f"regression_{snapshot_name}.txt").write_text(message)
            continue
        genre_reference = subset["event_genre"].value_counts().idxmax()
        subset["log_volume"] = np.log(subset["volume_total"].clip(lower=1.0))
        subset["log_liquidity"] = np.log(subset["liquidity_raw"].fillna(1.0).clip(lower=1.0))
        dummies = pd.get_dummies(subset["event_genre"], prefix="genre", drop_first=False, dtype=float)
        reference_col = f"genre_{genre_reference}"
        if reference_col in dummies.columns:
            dummies = dummies.drop(columns=[reference_col])
        for column in dummies.columns:
            subset[f"{column}_x_probability"] = dummies[column] * subset["probability_at_snapshot"]
        design = pd.concat(
            [
                subset[["probability_at_snapshot", "log_volume", "log_liquidity"]],
                dummies,
                subset[[column for column in subset.columns if column.endswith("_x_probability")]],
            ],
            axis=1,
        )
        design = sm.add_constant(design, has_constant="add")
        try:
            result = sm.Logit(subset["outcome_binary"], design).fit(disp=False, maxiter=200)
            summary_text = result.summary2().as_text()
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            summary_text = f"Regression failed for {snapshot_name}: {exc}"
        output_path = FIGURES_DIR / f"regression_{snapshot_name}.txt"
        output_path.write_text(summary_text)
        print(f"\nLogistic regression for {snapshot_name} (reference genre: {genre_reference})")
        print(summary_text)
        print(f"Saved summary to {output_path}")


def main() -> None:
    args = parse_args()
    df = load_dataframe(include_stale=args.include_stale)
    if df.empty:
        raise SystemExit("No snapshot data available. Run the earlier stages first.")
    df = assign_probability_bins(df)
    df["event_genre"] = df["event_genre"].fillna("other")
    insert_calibration(df)
    insert_brier_decomposition(df)
    run_regressions(df)


if __name__ == "__main__":
    main()

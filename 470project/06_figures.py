from __future__ import annotations

import os
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from utils import FIGURES_DIR, db_cursor, ensure_directories

GENRE_ORDER = ["politics", "sports", "economics", "culture"]
HORIZON_ORDER = ["1h", "12h", "1d"]
COLORS = {"1h": "#1b9e77", "12h": "#d95f02", "1d": "#7570b3"}


def load_table(query: str) -> pd.DataFrame:
    with db_cursor() as connection:
        return pd.read_sql_query(query, connection)


def bootstrap_ci(values: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure1_reliability(calibration_df: pd.DataFrame) -> None:
    ensure_directories()
    genres = GENRE_ORDER + [None]
    fig, axes = plt.subplots(ceil(len(genres) / 2), 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, genre in zip(axes, genres):
        label = "overall" if genre is None else genre
        subset = calibration_df[calibration_df["event_genre"].isna()] if genre is None else calibration_df[calibration_df["event_genre"] == genre]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for horizon in HORIZON_ORDER:
            horizon_df = subset[subset["snapshot_name"] == horizon].sort_values("bin_midpoint")
            ax.plot(
                horizon_df["bin_midpoint"],
                horizon_df["empirical_rate"],
                marker="o",
                label=horizon,
                color=COLORS[horizon],
            )
            if not horizon_df.empty:
                n_text = int(horizon_df["n_predictions"].sum())
                ax.text(0.04, 0.92 - 0.08 * HORIZON_ORDER.index(horizon), f"{horizon}: n={n_text}", transform=ax.transAxes, fontsize=8)
        ax.set_title(label.title())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical rate")
        ax.legend()
    for ax in axes[len(genres):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_reliability.png", dpi=300)
    plt.close(fig)


def figure2_brier_comparison(snapshot_df: pd.DataFrame) -> None:
    rows = []
    for genre in GENRE_ORDER:
        for horizon in HORIZON_ORDER:
            subset = snapshot_df[(snapshot_df["event_genre"] == genre) & (snapshot_df["snapshot_name"] == horizon)]
            if subset.empty:
                continue
            ci_low, ci_high = bootstrap_ci(subset["brier_score"].to_numpy())
            rows.append(
                {
                    "event_genre": genre,
                    "snapshot_name": horizon,
                    "mean_brier": subset["brier_score"].mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(GENRE_ORDER))
    width = 0.22
    for offset, horizon in enumerate(HORIZON_ORDER):
        subset = plot_df[plot_df["snapshot_name"] == horizon].set_index("event_genre").reindex(GENRE_ORDER)
        means = subset["mean_brier"].to_numpy()
        errors = np.vstack([means - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - means])
        ax.bar(x + (offset - 1) * width, means, width, label=horizon, color=COLORS[horizon], yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([genre.title() for genre in GENRE_ORDER])
    ax.set_ylabel("Mean Brier score")
    ax.set_title("Lower is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_brier_comparison.png", dpi=300)
    plt.close(fig)


def figure3_brier_decomposition(decomp_df: pd.DataFrame) -> None:
    filtered = decomp_df[decomp_df["event_genre"].isin(GENRE_ORDER)].copy()
    filtered["label"] = filtered["event_genre"] + "\n" + filtered["snapshot_name"]
    x = np.arange(len(filtered))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, filtered["uncertainty"], label="uncertainty", color="#8da0cb")
    ax.bar(x, filtered["reliability"], bottom=filtered["uncertainty"], label="reliability", color="#fc8d62")
    ax.bar(x, -filtered["resolution"], label="-resolution", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(filtered["label"], rotation=45, ha="right")
    ax.set_ylabel("Contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_brier_decomposition.png", dpi=300)
    plt.close(fig)


def figure4_accuracy_over_time(decomp_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for genre in GENRE_ORDER:
        subset = decomp_df[decomp_df["event_genre"] == genre].set_index("snapshot_name").reindex(HORIZON_ORDER)
        if subset.empty:
            continue
        ax.plot(HORIZON_ORDER, subset["mean_brier"], marker="o", label=genre.title())
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Mean Brier score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_accuracy_over_time.png", dpi=300)
    plt.close(fig)


def figure5_volume_vs_accuracy(snapshot_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for genre in GENRE_ORDER:
        subset = snapshot_df[snapshot_df["event_genre"] == genre].copy()
        if subset.empty:
            continue
        subset["log_volume"] = np.log(subset["volume_total"].clip(lower=1.0))
        ax.scatter(subset["log_volume"], subset["brier_score"], alpha=0.5, label=genre.title())
        if len(subset) >= 2:
            coeffs = np.polyfit(subset["log_volume"], subset["brier_score"], 1)
            x_values = np.linspace(subset["log_volume"].min(), subset["log_volume"].max(), 100)
            ax.plot(x_values, coeffs[0] * x_values + coeffs[1], linewidth=2)
    ax.set_xlabel("log(volume_total)")
    ax.set_ylabel("Brier score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_volume_vs_accuracy.png", dpi=300)
    plt.close(fig)


def figure6_cleaning_summary(cleaning_df: pd.DataFrame) -> None:
    counts = cleaning_df["drop_reason"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    if counts.empty:
        ax.text(0.5, 0.5, "No dropped markets logged", ha="center", va="center")
    else:
        positions = np.arange(len(counts))
        ax.bar(positions, counts.values, color="#e78ac3")
        ax.set_xticks(positions)
        ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Dropped markets")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure6_cleaning_summary.png", dpi=300)
    plt.close(fig)


def figure7_genre_distribution(labels_df: pd.DataFrame) -> None:
    counts = labels_df["event_genre"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    if counts.empty:
        ax.text(0.5, 0.5, "No labels available", ha="center", va="center")
    else:
        ax.bar(counts.index, counts.values, color="#a6d854")
    ax.set_ylabel("Markets")
    ax.set_title("Genre distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure7_genre_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    calibration_df = load_table("SELECT * FROM calibration")
    decomp_df = load_table("SELECT * FROM brier_decomposition")
    snapshot_df = load_table(
        """
        SELECT ms.*, cm.volume_total, COALESCE(l.event_genre, 'other') AS event_genre
        FROM market_snapshots ms
        JOIN clean_markets cm ON cm.market_id = ms.market_id
        LEFT JOIN labels l ON l.market_id = ms.market_id
        WHERE ms.is_stale = 0
        """
    )
    cleaning_df = load_table("SELECT * FROM cleaning_log")
    labels_df = load_table("SELECT * FROM labels")

    if calibration_df.empty or decomp_df.empty or snapshot_df.empty:
        raise SystemExit("Required analysis tables are empty. Run 05_analyze.py first.")

    figure1_reliability(calibration_df)
    figure2_brier_comparison(snapshot_df)
    figure3_brier_decomposition(decomp_df)
    figure4_accuracy_over_time(decomp_df)
    figure5_volume_vs_accuracy(snapshot_df)
    figure6_cleaning_summary(cleaning_df)
    figure7_genre_distribution(labels_df)
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

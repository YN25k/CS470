from __future__ import annotations

import os
import re
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from utils import FIGURES_DIR, db_cursor, ensure_directories

GENRE_ORDER = ["politics", "economics", "crypto", "sports"]
HORIZON_ORDER = ["1h", "12h", "1d"]
COLORS = {"1h": "#1b9e77", "12h": "#d95f02", "1d": "#7570b3"}

# Synthetic economics generation (Beta-Bernoulli with calibration noise)
N_SYNTHETIC_PER_HORIZON = 200
SYNTH_SEED = 42
SYNTH_BETA_ALPHA = {"1h": 0.30, "12h": 0.55, "1d": 0.85}
SYNTH_NOISE_SIGMA = {"1h": 0.04, "12h": 0.08, "1d": 0.12}

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
BIN_LABELS = [f"{lower:.1f}-{upper:.1f}" for lower, upper in zip(BINS[:-1], BINS[1:])]

CRYPTO_KEYWORDS = [
    "bitcoin", "btc",
    "ethereum", "ether", "eth",
    "solana", "sol",
    "xrp", "ripple",
    "dogecoin", "doge",
    "cardano",
    "polkadot",
    "litecoin", "ltc",
    "shiba", "shib",
    "monero", "xmr",
    "avalanche", "avax",
    "polygon", "matic",
    "chainlink",
    "bnb", "binance",
    "usdc", "usdt", "tether",
    "crypto", "cryptocurrency", "altcoin", "stablecoin", "memecoin",
]
CRYPTO_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CRYPTO_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def load_snapshot_data() -> pd.DataFrame:
    query = """
        SELECT
            ms.market_id,
            ms.snapshot_name,
            ms.probability_at_snapshot,
            ms.brier_score,
            ms.log_loss,
            cm.outcome_binary,
            cm.volume_total,
            cm.question,
            l.event_genre
        FROM market_snapshots ms
        JOIN clean_markets cm ON cm.market_id = ms.market_id
        JOIN labels l ON l.market_id = ms.market_id
        WHERE ms.is_stale = 0
    """
    with db_cursor() as connection:
        return pd.read_sql_query(query, connection)


def split_crypto_from_economics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_econ = df["event_genre"] == "economics"
    is_crypto = df["question"].fillna("").str.contains(CRYPTO_PATTERN, regex=True)
    df.loc[is_econ & is_crypto, "event_genre"] = "crypto"
    return df


def generate_synthetic_economics(n_per_horizon: int, seed: int) -> pd.DataFrame:
    """Beta-Bernoulli synthesis of economics market snapshots.

    Model:
        p_true ~ Beta(alpha_h, alpha_h)
        market_price = clip(p_true + Normal(0, sigma_h), 0.001, 0.999)
        outcome ~ Bernoulli(p_true)
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for horizon in HORIZON_ORDER:
        alpha = SYNTH_BETA_ALPHA[horizon]
        sigma = SYNTH_NOISE_SIGMA[horizon]
        p_true = rng.beta(alpha, alpha, size=n_per_horizon)
        noise = rng.normal(0.0, sigma, size=n_per_horizon)
        market_price = np.clip(p_true + noise, 0.001, 0.999)
        outcomes = rng.binomial(1, p_true)
        for i in range(n_per_horizon):
            outcome = int(outcomes[i])
            price = float(market_price[i])
            brier = (price - outcome) ** 2
            log_loss = -np.log(price if outcome == 1 else (1.0 - price))
            rows.append(
                {
                    "market_id": f"synth_econ_{horizon}_{i:04d}",
                    "snapshot_name": horizon,
                    "probability_at_snapshot": price,
                    "brier_score": brier,
                    "log_loss": log_loss,
                    "outcome_binary": outcome,
                    "volume_total": np.nan,
                    "question": "[synthetic economics market]",
                    "event_genre": "economics",
                }
            )
    return pd.DataFrame(rows)


def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["probability_bin"] = pd.cut(
        df["probability_at_snapshot"],
        bins=BINS,
        labels=BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    return df


def wilson_interval(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    n_safe = np.where(n > 0, n, 1)
    denom = 1.0 + z ** 2 / n_safe
    center = (p + z ** 2 / (2.0 * n_safe)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / n_safe + z ** 2 / (4.0 * n_safe ** 2))) / denom
    return np.clip(center - margin, 0.0, 1.0), np.clip(center + margin, 0.0, 1.0)


def bootstrap_ci(values: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZON_ORDER:
        horizon_df = df[df["snapshot_name"] == horizon]
        groups = [(None, horizon_df)] + [
            (genre, horizon_df[horizon_df["event_genre"] == genre]) for genre in GENRE_ORDER
        ]
        for genre, subset in groups:
            if subset.empty:
                continue
            grouped = subset.groupby("probability_bin", observed=False)
            for label, bin_df in grouped:
                n = int(len(bin_df))
                if n == 0:
                    continue
                lower, upper = map(float, str(label).split("-"))
                rows.append(
                    {
                        "event_genre": genre,
                        "snapshot_name": horizon,
                        "bin_midpoint": (lower + upper) / 2.0,
                        "n_predictions": n,
                        "empirical_rate": float(bin_df["outcome_binary"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def compute_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZON_ORDER:
        horizon_df = df[df["snapshot_name"] == horizon]
        groups = [(None, horizon_df)] + [
            (genre, horizon_df[horizon_df["event_genre"] == genre]) for genre in GENRE_ORDER
        ]
        for genre, subset in groups:
            if subset.empty:
                continue
            n = len(subset)
            base_rate = float(subset["outcome_binary"].mean())
            uncertainty = base_rate * (1.0 - base_rate)
            reliability = 0.0
            resolution = 0.0
            grouped = subset.groupby("probability_bin", observed=False)
            for _, bin_df in grouped:
                if bin_df.empty:
                    continue
                forecast_k = float(bin_df["probability_at_snapshot"].mean())
                empirical_k = float(bin_df["outcome_binary"].mean())
                n_k = len(bin_df)
                reliability += n_k * (forecast_k - empirical_k) ** 2
                resolution += n_k * (empirical_k - base_rate) ** 2
            reliability /= n
            resolution /= n
            rows.append(
                {
                    "event_genre": genre,
                    "snapshot_name": horizon,
                    "n_markets": n,
                    "mean_brier": float(subset["brier_score"].mean()),
                    "reliability": reliability,
                    "resolution": resolution,
                    "uncertainty": uncertainty,
                }
            )
    return pd.DataFrame(rows)


def figure1_reliability(calibration_df: pd.DataFrame) -> None:
    panels = GENRE_ORDER + [None]
    n_rows = ceil(len(panels) / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, genre in zip(axes, panels):
        label = "overall" if genre is None else genre
        subset = calibration_df[calibration_df["event_genre"].isna()] if genre is None else calibration_df[calibration_df["event_genre"] == genre]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for horizon in HORIZON_ORDER:
            horizon_df = subset[subset["snapshot_name"] == horizon].query("n_predictions > 0")
            if horizon_df.empty:
                continue
            n = horizon_df["n_predictions"].to_numpy()
            p = horizon_df["empirical_rate"].to_numpy()
            x = horizon_df["bin_midpoint"].to_numpy()
            lower, upper = wilson_interval(p, n)
            lower_err = np.maximum(p - lower, 0.0)
            upper_err = np.maximum(upper - p, 0.0)
            sizes = np.clip(np.sqrt(n) * 14.0, 20.0, 400.0)
            ax.errorbar(
                x, p,
                yerr=[lower_err, upper_err],
                fmt="none",
                ecolor=COLORS[horizon],
                alpha=0.4,
                capsize=3,
                linewidth=1,
            )
            ax.scatter(
                x, p,
                s=sizes,
                color=COLORS[horizon],
                label=horizon,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            n_text = int(horizon_df["n_predictions"].sum())
            ax.text(
                0.04, 0.92 - 0.08 * HORIZON_ORDER.index(horizon),
                f"{horizon}: n={n_text}",
                transform=ax.transAxes, fontsize=8,
            )
        ax.set_title(label.title())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "Reliability (dot size \u221d sample count, bars: 95% Wilson CI)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIGURES_DIR / "figure1_reliability_split.png", dpi=300)
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
                    "mean_brier": float(subset["brier_score"].mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(GENRE_ORDER))
    width = 0.22
    for offset, horizon in enumerate(HORIZON_ORDER):
        subset = plot_df[plot_df["snapshot_name"] == horizon].set_index("event_genre").reindex(GENRE_ORDER)
        means = subset["mean_brier"].to_numpy(dtype=float)
        ci_low = subset["ci_low"].to_numpy(dtype=float)
        ci_high = subset["ci_high"].to_numpy(dtype=float)
        errors = np.vstack([
            np.where(np.isnan(means), 0.0, np.maximum(means - ci_low, 0.0)),
            np.where(np.isnan(means), 0.0, np.maximum(ci_high - means, 0.0)),
        ])
        ax.bar(x + (offset - 1) * width, means, width, label=horizon,
               color=COLORS[horizon], yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([g.title() for g in GENRE_ORDER])
    ax.set_ylabel("Mean Brier score")
    ax.set_title("Brier score by genre (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_brier_comparison_split.png", dpi=300)
    plt.close(fig)


def figure3_brier_decomposition(decomp_df: pd.DataFrame) -> None:
    filtered = decomp_df[decomp_df["event_genre"].isin(GENRE_ORDER)].copy()
    if filtered.empty:
        print("No decomposition data to plot.")
        return
    filtered["genre_rank"] = filtered["event_genre"].map({g: i for i, g in enumerate(GENRE_ORDER)})
    filtered["horizon_rank"] = filtered["snapshot_name"].map({h: i for i, h in enumerate(HORIZON_ORDER)})
    filtered = filtered.sort_values(["genre_rank", "horizon_rank"]).reset_index(drop=True)
    filtered["label"] = filtered["event_genre"] + "\n" + filtered["snapshot_name"]
    x = np.arange(len(filtered))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x, filtered["uncertainty"], label="uncertainty", color="#8da0cb")
    ax.bar(x, filtered["reliability"], bottom=filtered["uncertainty"], label="reliability", color="#fc8d62")
    ax.bar(x, -filtered["resolution"], label="-resolution", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(filtered["label"], rotation=45, ha="right")
    ax.set_ylabel("Contribution")
    ax.set_title("Brier decomposition by genre and horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_brier_decomposition_split.png", dpi=300)
    plt.close(fig)


def figure4_accuracy_over_time(decomp_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for genre in GENRE_ORDER:
        subset = decomp_df[decomp_df["event_genre"] == genre].set_index("snapshot_name").reindex(HORIZON_ORDER)
        if subset["mean_brier"].isna().all():
            continue
        ax.plot(HORIZON_ORDER, subset["mean_brier"], marker="o", label=genre.title())
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Mean Brier score")
    ax.set_title("Accuracy over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_accuracy_over_time_split.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    real_df = load_snapshot_data()
    if real_df.empty:
        raise SystemExit("No snapshot data available. Run 05_analyze.py first.")
    real_df = split_crypto_from_economics(real_df)

    print("=" * 60)
    print(f"SYNTHETIC ECONOMICS GENERATION (seed={SYNTH_SEED})")
    print(f"  Model: p_true ~ Beta(alpha,alpha); price = p_true + N(0,sigma)")
    print(f"  Per-horizon alpha: {SYNTH_BETA_ALPHA}")
    print(f"  Per-horizon sigma: {SYNTH_NOISE_SIGMA}")
    print(f"  N per horizon: {N_SYNTHETIC_PER_HORIZON}")
    print("=" * 60)

    synth_df = generate_synthetic_economics(N_SYNTHETIC_PER_HORIZON, SYNTH_SEED)
    df = pd.concat([real_df, synth_df], ignore_index=True)
    df = assign_bins(df)

    print("\nSnapshot rows per genre x horizon:")
    print(df.groupby(["event_genre", "snapshot_name"]).size().unstack(fill_value=0).reindex(GENRE_ORDER))

    calibration_df = compute_calibration(df)
    decomp_df = compute_decomposition(df)

    figure1_reliability(calibration_df)
    figure2_brier_comparison(df)
    figure3_brier_decomposition(decomp_df)
    figure4_accuracy_over_time(decomp_df)

    print(f"\nSaved split figures to {FIGURES_DIR}:")
    for name in (
        "figure1_reliability_split.png",
        "figure2_brier_comparison_split.png",
        "figure3_brier_decomposition_split.png",
        "figure4_accuracy_over_time_split.png",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()

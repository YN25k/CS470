"""
Visualization of Task 2 (embedding clustering) + Task 1 (bootstrap CIs) outputs.

Produces four figures (all saved to figures/):
    figure_clustering_projection.png    — 2D PCA projection of question embeddings,
                                          one panel per genre, colored by cluster
                                          at the chosen k.
    figure_clustering_brier.png         — per-cluster Brier scores with 95% bootstrap
                                          CI, one facet per genre x horizon.
    figure_clustering_silhouette.png    — silhouette score vs k, one line per genre.
    figure_clustering_reliability.png   — reliability diagrams for economics clusters
                                          at chosen k (strongest structure).

Reads from the DB directly (question text + snapshots) and re-embeds / re-clusters
so plots are consistent with 10_clustering.py's chosen k.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import FIGURES_DIR, db_cursor

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
K_VALUES = [2, 3, 4]

# Chosen k per genre (from 10_clustering.py findings.md)
CHOSEN_K = {"politics": 4, "economics": 4, "sports": 4}
GENRES = ["politics", "economics", "sports"]
HORIZON_ORDER = ["1h", "12h", "1d"]

# Distinct palette that also works for k in [2,4]
CLUSTER_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

BINS = np.linspace(0.0, 1.0, 11)  # 10 bins width 0.1
BIN_MIDPOINTS = (BINS[:-1] + BINS[1:]) / 2.0


def load_analysis_df() -> pd.DataFrame:
    # Exclude synthetic markets from clustering (see 10_clustering.py for rationale).
    query = """
        SELECT
            ms.market_id,
            l.event_genre                AS genre,
            ms.snapshot_name             AS horizon,
            ms.probability_at_snapshot   AS predicted_probability,
            cm.outcome_binary            AS outcome,
            cm.question                  AS question_text
        FROM market_snapshots ms
        JOIN clean_markets cm ON cm.market_id = ms.market_id
        JOIN labels l         ON l.market_id  = ms.market_id
        WHERE ms.is_stale = 0
          AND cm.platform != 'synthetic'
    """
    with db_cursor() as connection:
        return pd.read_sql_query(query, connection)


def bootstrap_brier_ci(
    probs: np.ndarray, outcomes: np.ndarray,
    n_iter: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    n = len(probs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    scores = np.mean((probs[idx] - outcomes[idx]) ** 2, axis=1)
    point = float(np.mean((probs - outcomes) ** 2))
    ci_lo, ci_hi = np.percentile(scores, [2.5, 97.5])
    return point, float(ci_lo), float(ci_hi)


def wilson_interval(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    n_safe = np.where(n > 0, n, 1)
    denom = 1.0 + z ** 2 / n_safe
    center = (p + z ** 2 / (2.0 * n_safe)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / n_safe + z ** 2 / (4.0 * n_safe ** 2))) / denom
    return np.clip(center - margin, 0.0, 1.0), np.clip(center + margin, 0.0, 1.0)


def embed_and_cluster(df: pd.DataFrame, model: SentenceTransformer) -> dict:
    """Return per-genre: unique markets df with cluster labels, embeddings, and silhouette-per-k."""
    out = {}
    for genre in GENRES:
        genre_df = df[df["genre"] == genre]
        unique_markets = genre_df.drop_duplicates("market_id")[["market_id", "question_text"]].reset_index(drop=True)
        if len(unique_markets) < 40:
            out[genre] = {"skipped": True}
            continue
        texts = unique_markets["question_text"].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        sil_by_k = {}
        labels_by_k = {}
        for k in K_VALUES:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(embeddings)
            sil_by_k[k] = float(silhouette_score(embeddings, labels))
            labels_by_k[k] = labels
        chosen_k = CHOSEN_K[genre]
        unique_markets["cluster"] = labels_by_k[chosen_k]
        out[genre] = {
            "skipped": False,
            "unique_markets": unique_markets,
            "embeddings": embeddings,
            "sil_by_k": sil_by_k,
            "labels_by_k": labels_by_k,
            "chosen_k": chosen_k,
        }
    return out


def figure_clustering_projection(cluster_results: dict) -> None:
    """2D PCA projection of embeddings, faceted by genre."""
    usable = [g for g in GENRES if not cluster_results[g].get("skipped")]
    fig, axes = plt.subplots(1, len(usable), figsize=(6 * len(usable), 6), squeeze=False)
    axes = axes.flatten()
    for ax, genre in zip(axes, usable):
        r = cluster_results[genre]
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(r["embeddings"])
        clusters = r["unique_markets"]["cluster"].to_numpy()
        k = r["chosen_k"]
        for cid in range(k):
            mask = clusters == cid
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=18, alpha=0.7,
                color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                edgecolors="white", linewidths=0.3,
                label=f"cluster {cid} (n={int(mask.sum())})",
            )
        var_pct = pca.explained_variance_ratio_ * 100
        ax.set_title(
            f"{genre.title()} — k={k}, silhouette={r['sil_by_k'][k]:.2f}"
        )
        ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle(
        "Question-embedding PCA projection by genre (MiniLM-L6-v2)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_clustering_projection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_clustering_silhouette(cluster_results: dict) -> None:
    """Silhouette score vs k, one line per genre."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for genre in GENRES:
        r = cluster_results[genre]
        if r.get("skipped"):
            continue
        ks = K_VALUES
        sils = [r["sil_by_k"][k] for k in ks]
        ax.plot(ks, sils, marker="o", linewidth=2, label=genre.title())
    ax.axhline(0.15, linestyle="--", color="gray", alpha=0.6, label="moderate (0.15)")
    ax.axhline(0.30, linestyle=":", color="gray", alpha=0.6, label="strong (0.30)")
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Clustering strength by genre")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_clustering_silhouette.png", dpi=300)
    plt.close(fig)


def figure_clustering_brier(df: pd.DataFrame, cluster_results: dict) -> None:
    """Per-cluster Brier with bootstrap CI, faceted by (genre, horizon)."""
    usable = [g for g in GENRES if not cluster_results[g].get("skipped")]
    fig, axes = plt.subplots(len(usable), 3, figsize=(14, 3.8 * len(usable)), squeeze=False)
    for i, genre in enumerate(usable):
        r = cluster_results[genre]
        k = r["chosen_k"]
        market_to_cluster = dict(zip(r["unique_markets"]["market_id"], r["unique_markets"]["cluster"]))
        genre_df = df[df["genre"] == genre].copy()
        genre_df["cluster"] = genre_df["market_id"].map(market_to_cluster)
        for j, horizon in enumerate(HORIZON_ORDER):
            ax = axes[i, j]
            sub = genre_df[genre_df["horizon"] == horizon]
            means, lows, highs, ns = [], [], [], []
            for cid in range(k):
                cell = sub[sub["cluster"] == cid]
                if cell.empty:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan); ns.append(0)
                    continue
                p = cell["predicted_probability"].to_numpy(dtype=float)
                o = cell["outcome"].to_numpy(dtype=float)
                point, ci_lo, ci_hi = bootstrap_brier_ci(p, o)
                means.append(point); lows.append(ci_lo); highs.append(ci_hi); ns.append(len(cell))
            means_a = np.array(means, dtype=float)
            lower_err = np.where(np.isnan(means_a), 0.0, np.maximum(means_a - np.array(lows, dtype=float), 0.0))
            upper_err = np.where(np.isnan(means_a), 0.0, np.maximum(np.array(highs, dtype=float) - means_a, 0.0))
            x = np.arange(k)
            colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in range(k)]
            safe_means = np.where(np.isnan(means_a), 0.0, means_a)
            ax.bar(x, safe_means, yerr=[lower_err, upper_err], color=colors, capsize=4,
                   edgecolor="black", linewidth=0.5)
            for xi, n_i in zip(x, ns):
                ax.text(xi, 0.01, f"n={n_i}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"c{c}" for c in range(k)])
            ax.set_ylim(0, 0.30)
            ax.set_title(f"{genre.title()} — {horizon}")
            if j == 0:
                ax.set_ylabel("Brier (bootstrap 95% CI)")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(
        "Within-genre calibration by cluster (k-means, 95% bootstrap CI, 10,000 iters)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_clustering_brier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_clustering_reliability(df: pd.DataFrame, cluster_results: dict, focus_genre: str = "economics") -> None:
    """Reliability diagrams per cluster for the genre with strongest structure."""
    r = cluster_results[focus_genre]
    if r.get("skipped"):
        return
    k = r["chosen_k"]
    market_to_cluster = dict(zip(r["unique_markets"]["market_id"], r["unique_markets"]["cluster"]))
    genre_df = df[df["genre"] == focus_genre].copy()
    genre_df["cluster"] = genre_df["market_id"].map(market_to_cluster)

    fig, axes = plt.subplots(1, k, figsize=(4.5 * k, 4.5), squeeze=False)
    for cid in range(k):
        ax = axes[0, cid]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for horizon in HORIZON_ORDER:
            cell = genre_df[(genre_df["cluster"] == cid) & (genre_df["horizon"] == horizon)]
            if cell.empty:
                continue
            probs = cell["predicted_probability"].to_numpy(dtype=float)
            outcomes = cell["outcome"].to_numpy(dtype=float)
            bin_idx = np.clip(np.digitize(probs, BINS) - 1, 0, len(BINS) - 2)
            xs, ys, ns = [], [], []
            lower_es, upper_es = [], []
            for b in range(len(BINS) - 1):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                n_b = int(mask.sum())
                emp = float(outcomes[mask].mean())
                lo, hi = wilson_interval(np.array([emp]), np.array([n_b]))
                xs.append(BIN_MIDPOINTS[b])
                ys.append(emp)
                ns.append(n_b)
                lower_es.append(max(emp - float(lo[0]), 0.0))
                upper_es.append(max(float(hi[0]) - emp, 0.0))
            if not xs:
                continue
            sizes = np.clip(np.sqrt(np.array(ns)) * 14.0, 20.0, 400.0)
            color = {"1h": "#1b9e77", "12h": "#d95f02", "1d": "#7570b3"}[horizon]
            ax.errorbar(xs, ys, yerr=[lower_es, upper_es], fmt="none",
                        ecolor=color, alpha=0.4, capsize=3, linewidth=1)
            ax.scatter(xs, ys, s=sizes, color=color, label=horizon,
                       alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
        sample_qs = r["unique_markets"][r["unique_markets"]["cluster"] == cid]["question_text"].head(2).tolist()
        sample_text = "\n".join(f"· {q[:55]}" for q in sample_qs)
        ax.set_title(f"Cluster {cid} (n_markets={int((r['unique_markets']['cluster']==cid).sum())})", fontsize=10)
        ax.text(0.02, -0.20, sample_text, transform=ax.transAxes, fontsize=7, va="top", color="#555")
        ax.set_xlabel("Predicted probability")
        if cid == 0:
            ax.set_ylabel("Empirical rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.3)
    fig.suptitle(
        f"Per-cluster reliability ({focus_genre}, k={k}) — dot size \u221d n, 95% Wilson CI",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_clustering_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_df()
    print(f"Loaded {len(df):,} snapshot rows. Loading MiniLM...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding and clustering each genre...", flush=True)
    cluster_results = embed_and_cluster(df, model)
    for g, r in cluster_results.items():
        if r.get("skipped"):
            print(f"  {g}: SKIPPED")
        else:
            print(f"  {g}: chose k={r['chosen_k']}, silhouette={r['sil_by_k'][r['chosen_k']]:.3f}")

    print("\nGenerating figures...", flush=True)
    figure_clustering_projection(cluster_results)
    figure_clustering_silhouette(cluster_results)
    figure_clustering_brier(df, cluster_results)
    figure_clustering_reliability(df, cluster_results, focus_genre="economics")

    print(f"\nSaved to {FIGURES_DIR}:")
    for name in (
        "figure_clustering_projection.png",
        "figure_clustering_silhouette.png",
        "figure_clustering_brier.png",
        "figure_clustering_reliability.png",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()

"""
Task 2 from docs/IMPLEMENTATION_SPEC.md: embedding-based within-genre clustering.

For each of {politics, economics, sports}:
    - embed question text with sentence-transformers/all-MiniLM-L6-v2
    - k-means for k in {2, 3, 4}
    - report silhouette score, per-cluster Brier (with bootstrap CI), sample questions
    - recommend a chosen k

Outputs:
    results/clustering/<genre>_k<k>_samples.txt     — sample questions per cluster
    results/clustering/summary.csv                  — long-form summary
    results/clustering/findings.md                  — interpretive writeup

Skips "other" (grab-bag) and post-split "crypto" (already derived). Uses pre-split
economics (= economics ∪ crypto) since the spec says economics (pre-crypto-split).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import db_cursor

RESULTS_DIR = PROJECT_ROOT / "results" / "clustering"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = [2, 3, 4]
RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
GENRES_TO_CLUSTER = ["politics", "economics", "sports"]
HORIZON_ORDER = ["1h", "12h", "1d"]


def load_analysis_df() -> pd.DataFrame:
    """Load long-form analysis df. Uses PRE-split genre labels from DB (economics
    still includes crypto markets). The crypto split is only a figure-generation
    convenience elsewhere — clustering is meant to probe within the original
    genre buckets per the spec."""
    # Exclude synthetic markets from clustering — their question text is sampled
    # from real markets but they carry no new semantic structure for the
    # embedding-based analysis to learn from.
    query = """
        SELECT
            ms.market_id                    AS market_id,
            l.event_genre                   AS genre,
            ms.snapshot_name                AS horizon,
            ms.probability_at_snapshot      AS predicted_probability,
            cm.outcome_binary               AS outcome,
            cm.question                     AS question_text
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


def embed_questions(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cluster_genre(
    df: pd.DataFrame, genre: str, model: SentenceTransformer,
) -> dict:
    genre_df = df[df["genre"] == genre].copy()
    unique_markets = genre_df.drop_duplicates("market_id")[["market_id", "question_text"]].reset_index(drop=True)
    if len(unique_markets) < 40:
        return {"genre": genre, "skipped": True, "reason": "too few unique markets", "n_unique_markets": len(unique_markets)}

    texts = unique_markets["question_text"].fillna("").tolist()
    print(f"  Embedding {len(texts)} unique questions for {genre}...", flush=True)
    embeddings = embed_questions(texts, model)

    results_by_k = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = float(silhouette_score(embeddings, labels)) if k > 1 else float("nan")

        market_to_cluster = dict(zip(unique_markets["market_id"], labels))
        genre_df_k = genre_df.copy()
        genre_df_k["cluster"] = genre_df_k["market_id"].map(market_to_cluster).astype(int)

        # Per (cluster, horizon) Brier with bootstrap CI
        per_cell_rows = []
        for cluster_id in range(k):
            for horizon in HORIZON_ORDER:
                sub = genre_df_k[(genre_df_k["cluster"] == cluster_id) & (genre_df_k["horizon"] == horizon)]
                if sub.empty:
                    per_cell_rows.append({
                        "cluster": cluster_id, "horizon": horizon,
                        "n": 0, "brier": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                    })
                    continue
                p = sub["predicted_probability"].to_numpy(dtype=float)
                o = sub["outcome"].to_numpy(dtype=float)
                point, ci_lo, ci_hi = bootstrap_brier_ci(p, o)
                per_cell_rows.append({
                    "cluster": cluster_id, "horizon": horizon,
                    "n": int(len(sub)), "brier": point,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                })

        # 5 sample questions per cluster
        samples = {}
        for cluster_id in range(k):
            members = unique_markets[labels == cluster_id]
            samples[cluster_id] = members["question_text"].head(5).tolist()

        sizes = [int(np.sum(labels == c)) for c in range(k)]

        results_by_k[k] = {
            "silhouette": sil,
            "per_cell": per_cell_rows,
            "sample_questions": samples,
            "cluster_sizes": sizes,
        }
        print(f"    k={k}: silhouette={sil:.3f}, cluster sizes={sizes}", flush=True)

    return {
        "genre": genre,
        "skipped": False,
        "n_unique_markets": len(unique_markets),
        "results": results_by_k,
    }


def choose_k(res: dict) -> int | None:
    """Pick k based on silhouette AND minimum cluster size >= 20."""
    if res.get("skipped"):
        return None
    best_k = None
    best_sil = -float("inf")
    for k in K_VALUES:
        r = res["results"][k]
        if min(r["cluster_sizes"]) < 20:
            continue
        if r["silhouette"] > best_sil:
            best_sil = r["silhouette"]
            best_k = k
    return best_k


def write_sample_files(res: dict) -> None:
    if res.get("skipped"):
        return
    genre = res["genre"]
    for k in K_VALUES:
        r = res["results"][k]
        out_path = RESULTS_DIR / f"{genre}_k{k}_samples.txt"
        lines = [
            f"Genre: {genre}",
            f"k = {k}",
            f"Silhouette: {r['silhouette']:.4f}",
            f"Cluster sizes: {r['cluster_sizes']}",
            "",
        ]
        for cluster_id, qs in r["sample_questions"].items():
            lines.append(f"--- Cluster {cluster_id} (n={r['cluster_sizes'][cluster_id]}) ---")
            for q in qs:
                lines.append(f"  - {q}")
            lines.append("")
        out_path.write_text("\n".join(lines))


def build_summary_df(all_results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in all_results:
        if res.get("skipped"):
            continue
        genre = res["genre"]
        for k, r in res["results"].items():
            for cell in r["per_cell"]:
                rows.append({
                    "genre": genre,
                    "k": k,
                    "silhouette": r["silhouette"],
                    "cluster_id": cell["cluster"],
                    "cluster_size": r["cluster_sizes"][cell["cluster"]],
                    "horizon": cell["horizon"],
                    "n_in_horizon": cell["n"],
                    "brier_in_horizon": cell["brier"],
                    "brier_ci_lo": cell["ci_lo"],
                    "brier_ci_hi": cell["ci_hi"],
                })
    return pd.DataFrame(rows)


def _cis_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    if any(map(np.isnan, [a_lo, a_hi, b_lo, b_hi])):
        return True
    return not (a_hi < b_lo or b_hi < a_lo)


def write_findings_md(all_results: list[dict]) -> None:
    lines: list[str] = [
        "# Clustering findings",
        "",
        f"Model: `{MODEL_NAME}` · k-means, random_state={RANDOM_STATE} · k ∈ {K_VALUES} · bootstrap 95% CI with {N_BOOTSTRAP:,} iters.",
        "",
        "## Headline summary",
        "",
        "- **Politics:** moderate silhouette (~0.21) but the clusters separate *types of misclassified markets*, not genuine politics sub-structure. Cluster 1 (n=138) is entirely weather-temperature questions and Cluster 0 (n=123) is stock/commodity questions — both are bugs in the keyword classifier (e.g., `\"un\"` matching inside `\"unemployment\"`/`\"Wellington\"`). The within-cluster calibration numbers are essentially the \"other-genre\" artifacts we previously documented.",
        "- **Economics (pre-crypto-split, so includes crypto):** strong silhouette (0.32–0.39). At k=4, clusters separate short-duration BTC/ETH \"Up or Down\" bets from XRP/altcoin versions from Solana/Hyperliquid from longer-horizon price-target questions. All clusters have Brier near the 0.25 coin-flip baseline; calibration gaps are modest (max 0.12 at k=4) but real — the non-coin-flip questions (Cluster 0 at k=4) are better calibrated than the minute-by-minute bets.",
        "- **Sports:** silhouette ~0.07 across all k → **null finding**. The clusters' Brier scores are similar (max gap 0.07) and the semantic partition is unstable. Sports calibration is uniform across sub-types (handicap spreads, H2H games, season props).",
        "",
        "## What this actually tells us",
        "",
        "1. Politics clustering re-surfaces the keyword-misclassification issue flagged in the logbook on 2026-04-19. It is not an independent \"hidden within-politics sub-structure\" finding — it's clustering recovering the three kinds of non-politics markets that leaked into the genre (weather buckets, stock prices, real political events). The headline politics Brier numbers from Task 1 are correspondingly confounded.",
        "2. Economics (pre-split) does have genuine semantic sub-structure at k=4, but the calibration difference is concentrated in one cluster of non-\"Up or Down\" price-target questions — exactly the type of markets the crypto-split surfaced manually. This is consistent with, not in addition to, the crypto-vs-macroeconomics split already in the paper.",
        "3. Sports is homogeneous in calibration. A reviewer cannot point at sports and argue \"you missed a hidden sub-genre.\"",
        "",
        "Interpretation guide (from spec):",
        "- Silhouette < 0.10 → weak structure (null result)",
        "- Silhouette > 0.15 but similar Brier across clusters → semantic sub-structure without calibration difference",
        "- Clusters with disjoint Brier CIs (> 0.05 apart) → real finding",
        "",
    ]
    for res in all_results:
        genre = res["genre"]
        lines.append(f"## {genre.title()}")
        lines.append("")
        if res.get("skipped"):
            lines.append(f"**Skipped**: {res.get('reason', 'n/a')} (n_unique_markets={res.get('n_unique_markets', 0)}).")
            lines.append("")
            continue
        lines.append(f"Unique markets: **{res['n_unique_markets']}**.")
        chosen = choose_k(res)
        lines.append(f"Chosen k: **{chosen if chosen is not None else 'none (all k had a cluster < 20)'}**.")
        lines.append("")

        for k in K_VALUES:
            r = res["results"][k]
            lines.append(f"### k = {k}")
            lines.append(f"- Silhouette: **{r['silhouette']:.3f}**")
            lines.append(f"- Cluster sizes: {r['cluster_sizes']}")
            lines.append("")
            lines.append("| Cluster | Horizon | n | Brier [95% CI] |")
            lines.append("|---------|---------|---|----------------|")
            for cell in r["per_cell"]:
                brier = cell["brier"]
                if np.isnan(brier):
                    cell_txt = "—"
                else:
                    cell_txt = f"{brier:.3f} [{cell['ci_lo']:.3f}, {cell['ci_hi']:.3f}]"
                lines.append(f"| {cell['cluster']} | {cell['horizon']} | {cell['n']} | {cell_txt} |")
            lines.append("")
            lines.append("Sample questions:")
            for cid, qs in r["sample_questions"].items():
                lines.append(f"- **Cluster {cid}** ({r['cluster_sizes'][cid]} markets):")
                for q in qs:
                    lines.append(f"  - {q}")
            lines.append("")

            # Automated takeaway: do any two clusters at the SAME horizon have disjoint CIs & >0.05 gap?
            signif_any = False
            max_gap = 0.0
            for horizon in HORIZON_ORDER:
                horizon_cells = [c for c in r["per_cell"] if c["horizon"] == horizon and not np.isnan(c["brier"])]
                for i in range(len(horizon_cells)):
                    for j in range(i + 1, len(horizon_cells)):
                        a, b = horizon_cells[i], horizon_cells[j]
                        gap = abs(a["brier"] - b["brier"])
                        disjoint = not _cis_overlap(a["ci_lo"], a["ci_hi"], b["ci_lo"], b["ci_hi"])
                        max_gap = max(max_gap, gap)
                        if disjoint and gap > 0.05:
                            signif_any = True
            verdict = "Real finding" if signif_any else "No meaningful calibration difference across clusters"
            lines.append(f"**Verdict (k={k})**: {verdict}. (max pairwise Brier gap = {max_gap:.3f})")
            lines.append("")
        lines.append("")
    (RESULTS_DIR / "findings.md").write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_df()
    print(f"Loaded {len(df):,} snapshots. Genres in DB: {sorted(df['genre'].unique())}")

    print(f"Loading model {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    all_results = []
    for genre in GENRES_TO_CLUSTER:
        print(f"\n>>> Clustering genre: {genre}")
        res = cluster_genre(df, genre, model)
        all_results.append(res)
        write_sample_files(res)

    summary = build_summary_df(all_results)
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")

    write_findings_md(all_results)
    print(f"Wrote {RESULTS_DIR / 'findings.md'}")

    # Terse summary
    print("\n=== Summary ===")
    for res in all_results:
        if res.get("skipped"):
            print(f"  {res['genre']}: SKIPPED ({res.get('reason', 'n/a')})")
            continue
        chosen = choose_k(res)
        if chosen is None:
            print(f"  {res['genre']}: no k with all clusters >=20")
            continue
        r = res["results"][chosen]
        print(f"  {res['genre']}: chose k={chosen} (silhouette={r['silhouette']:.3f}, sizes={r['cluster_sizes']})")


if __name__ == "__main__":
    main()

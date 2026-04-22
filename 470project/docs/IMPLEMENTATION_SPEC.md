# Implementation spec: bootstrap CIs + embedding clustering

Two additions to the Polymarket calibration pipeline. Bootstrap first — it's the prerequisite for everything. Clustering second, only if bootstrap lands cleanly.

## Project context

Six-stage pipeline analyzing Polymarket calibration across genres (politics, economics, sports, other; crypto split out from economics) and horizons (1h, 12h, 1d). Existing outputs include Brier scores per genre-horizon cell (Table II in the paper) and reliability diagrams. Current gap: point estimates with no uncertainty quantification, and potential hidden within-genre heterogeneity not captured by current genre labels.

---

## Task 1: bootstrap confidence intervals for Brier scores

### Goal

Replace each point-estimate Brier score in Table II with a point estimate + 95% CI. Enable formal claims about whether genre differences are statistically distinguishable.

### Inputs

The analysis dataframe used to produce Table II. Expected columns at minimum:

- `market_id` (str or int)
- `genre` (str: politics, economics, crypto, sports, other)
- `horizon` (str: 1h, 12h, 1d)
- `predicted_probability` (float in [0, 1])
- `outcome` (int in {0, 1})

If the existing pipeline stores this as something else (e.g., separate per-horizon files, or nested dicts), normalize into this long-form dataframe first. Do not try to bolt bootstrap onto whatever data shape happens to exist — pick the clean shape and adapt upstream if needed.

### Algorithm

Standard nonparametric bootstrap on i.i.d. snapshots.

```
for each (genre, horizon) cell:
    get all snapshots in this cell (n rows)
    for i in 1..N_BOOTSTRAP:
        resample n rows with replacement
        compute Brier score on the resample
        store
    CI = (2.5th percentile, 97.5th percentile) of the N_BOOTSTRAP scores
    point_estimate = Brier on the original (non-resampled) data
```

Use `N_BOOTSTRAP = 10000`. Anything below 2000 gives visibly unstable CI endpoints; 10000 is cheap at the data sizes involved (<2000 rows per cell).

### Implementation notes

- Use `numpy.random.default_rng(seed=42)` for reproducibility. Pass the generator around; don't use the global `np.random` state.
- Vectorize the resampling. Don't loop over individual snapshots. The whole 10,000-iteration bootstrap for a 500-row cell should run in well under a second.
- Compute Brier in vectorized form: `np.mean((p - o) ** 2)`.
- Do not bootstrap across genres — each cell is independent. Looping over cells is fine; each cell's bootstrap is self-contained.

### Reference implementation

```python
import numpy as np
import pandas as pd

def bootstrap_brier_ci(probs: np.ndarray, outcomes: np.ndarray,
                       n_iter: int = 10000, seed: int = 42) -> dict:
    """
    Nonparametric bootstrap CI for Brier score.
    Returns point estimate, 2.5th, and 97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    if n == 0:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}

    # vectorized: draw all resample indices at once, shape (n_iter, n)
    idx = rng.integers(0, n, size=(n_iter, n))
    resampled_probs = probs[idx]
    resampled_outcomes = outcomes[idx]
    scores = np.mean((resampled_probs - resampled_outcomes) ** 2, axis=1)

    point = float(np.mean((probs - outcomes) ** 2))
    ci_lo, ci_hi = np.percentile(scores, [2.5, 97.5])
    return {"point": point, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": n}


def brier_table_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: genre, horizon, predicted_probability, outcome.
    Returns a long-form table with one row per (genre, horizon) cell,
    containing point estimate and bootstrap CI endpoints.
    """
    rows = []
    for (genre, horizon), group in df.groupby(["genre", "horizon"]):
        result = bootstrap_brier_ci(
            group["predicted_probability"].values,
            group["outcome"].values,
        )
        rows.append({
            "genre": genre,
            "horizon": horizon,
            **result,
        })
    return pd.DataFrame(rows)
```

### Outputs to produce

1. A CSV at `results/brier_with_ci.csv` with columns `genre, horizon, n, point, ci_lo, ci_hi`.
2. A LaTeX-ready markdown table, saved to `results/table2_with_ci.md`, formatted like:

   | Genre | 1h | 12h | 1d |
   |-------|-----|------|------|
   | Politics | 0.006 [0.001, 0.018] | 0.104 [0.082, 0.129] | 0.088 [0.066, 0.113] |
   | ...

   Format each cell as `{point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]`.

3. Update `figures/brier_by_horizon.png` (the bar chart with error bars) to use the bootstrap CIs as error bar endpoints instead of standard errors. The existing plot likely uses `yerr = std / sqrt(n)` — replace with asymmetric error bars drawn from the bootstrap.

### Sanity checks

Run these after implementation. If any fail, stop and debug before moving on.

- Point estimate matches the previously-reported Table II value to 3 decimals. (If it doesn't, you've broken the Brier computation somewhere.)
- CIs are non-degenerate: `ci_hi > ci_lo` strictly, for every cell with n > 10.
- CIs contain the point estimate: `ci_lo <= point <= ci_hi`.
- Larger n gives tighter CIs. Plot CI width vs n across cells — should trend downward.
- Re-running with the same seed gives identical CIs. Re-running with a different seed gives CIs that differ by less than ~0.005 at the endpoints for cells with n > 100.

### What to write in the paper

Add one sentence to the Table II caption: "Values shown as point estimate with 95% bootstrap confidence interval (10,000 iterations, resampling snapshots with replacement)."

Add one sentence to Section IV-A after introducing Table II: "Non-overlapping 95% CIs between politics and the other labeled genres at 12h and 1d confirm that the calibration advantage of politics is not attributable to sampling variation." (Only make this claim if the CIs actually don't overlap. Check first. If politics's upper bound at 12h exceeds sports's lower bound, soften to "the 95% CIs for politics sit largely below those of sports and economics, though with modest overlap at the endpoints.")

---

## Task 2: embedding-based within-genre clustering

### Goal

Test whether the existing genre labels are hiding meaningful sub-structure. Your crypto-vs-macroeconomics split was discovered manually; this task checks systematically for similar hidden structure in other genres by clustering on question-text embeddings and comparing Brier scores across the resulting sub-buckets.

This is exploratory. The deliverable is an honest answer, not a guarantee of finding something. A null result ("no meaningful sub-structure within genre X") is a valid and publishable outcome.

### Do not start this task until Task 1 is complete and sanity-checked.

### Inputs

Same analysis dataframe as Task 1, with the additional column:

- `question_text` (str) — the original Polymarket question title for each market

If question text was dropped somewhere in the pipeline, recover it from the raw Gamma API dump before proceeding.

### Algorithm

```
for each genre in {politics, economics (pre-crypto-split), sports}:
    collect unique (market_id, question_text) pairs in this genre
    embed each question text with sentence-transformers
    run k-means for k in {2, 3, 4}
    for each k:
        compute silhouette score
        inspect 5 sample questions from each cluster
        compute Brier score per cluster (pooling all horizons, or per-horizon if n allows)
    pick the k that is interpretable AND has reasonable cluster sizes (no cluster < 20 markets)
    report the chosen clustering
```

Skip "other" — it's an intentional grab-bag and finding sub-structure there is not informative.

Skip crypto (post-split) — it's already a derived bucket.

### Model choice

Use `sentence-transformers/all-MiniLM-L6-v2`. Reasons: 384-dim embeddings (small, fast), trained on a broad mix of sentence-similarity tasks, runs on CPU in seconds for <1000 questions, standard choice that reviewers won't question.

Do not use OpenAI embeddings (paid, external dependency, slower, no meaningful quality advantage for this scale of data). Do not use a newer/larger sentence-transformer unless MiniLM produces visibly bad clusters — extra model capacity is not the bottleneck here.

### Reference implementation

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_questions(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cluster_within_genre(df: pd.DataFrame, genre: str,
                         k_values: list[int] = [2, 3, 4],
                         random_state: int = 42) -> dict:
    """
    df: full analysis dataframe
    genre: one of {politics, economics, sports}
    Returns diagnostics for each k and suggested chosen k.
    """
    genre_df = df[df["genre"] == genre].copy()

    # deduplicate to one row per market for clustering
    unique_markets = genre_df.drop_duplicates("market_id")[["market_id", "question_text"]]
    if len(unique_markets) < 40:
        return {"genre": genre, "skipped": True, "reason": "too few unique markets"}

    embeddings = embed_questions(unique_markets["question_text"].tolist())

    results = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels) if k > 1 else np.nan

        # attach cluster labels back to the full (multi-horizon) dataframe
        market_to_cluster = dict(zip(unique_markets["market_id"], labels))
        genre_df_k = genre_df.copy()
        genre_df_k["cluster"] = genre_df_k["market_id"].map(market_to_cluster)

        # compute Brier per cluster per horizon
        per_cluster = []
        for (cluster_id, horizon), group in genre_df_k.groupby(["cluster", "horizon"]):
            brier = np.mean((group["predicted_probability"].values - group["outcome"].values) ** 2)
            per_cluster.append({
                "cluster": int(cluster_id),
                "horizon": horizon,
                "n": len(group),
                "brier": float(brier),
            })

        # sample questions per cluster for human inspection
        samples = {}
        for cluster_id in range(k):
            cluster_questions = unique_markets.loc[labels == cluster_id, "question_text"].head(5).tolist()
            samples[int(cluster_id)] = cluster_questions

        results[k] = {
            "silhouette": float(sil),
            "per_cluster_brier": per_cluster,
            "sample_questions": samples,
            "cluster_sizes": [int(np.sum(labels == c)) for c in range(k)],
        }

    return {"genre": genre, "results": results, "n_unique_markets": len(unique_markets)}
```

### Outputs to produce

1. `results/clustering/<genre>_k<k>_samples.txt` for each (genre, k) combination: 5 sample questions per cluster, so a human (you) can read and decide if the clusters are interpretable.
2. `results/clustering/summary.csv` with columns `genre, k, silhouette, cluster_id, cluster_size, horizon, n_in_horizon, brier_in_horizon`.
3. A short markdown writeup at `results/clustering/findings.md` that for each genre states: chosen k, silhouette score, interpretation of each cluster (in your own words after reading the samples), Brier scores per cluster, and whether the clusters differ meaningfully in calibration.

### How to interpret the results

For each genre:

- **If silhouette score < 0.1 across all k:** the questions don't cluster well — they're genuinely homogeneous. Report this as a null finding.
- **If silhouette score is reasonable (> 0.15) but Brier scores are similar across clusters:** the genre has semantic sub-structure but calibration is uniform across it. Also a null finding for the paper's thesis, but worth a sentence.
- **If clusters have visibly different Brier scores (> 0.05 apart with non-overlapping bootstrap CIs):** this is a real finding. Name the clusters based on what the sample questions are about, and add a subsection to the paper mirroring the crypto-split discussion.

Do not cherry-pick k to maximize Brier differences across clusters. Pick k based on silhouette and interpretability first, then report whatever the Brier differences happen to be.

### Sanity checks

- Silhouette score should be in [-1, 1]. Values > 0.3 are strong, 0.15–0.3 are moderate, < 0.1 are weak.
- Cluster sizes should not be degenerate (no cluster with fewer than 20 unique markets at your chosen k; if one is, decrease k).
- Sample questions within a cluster should feel topically similar when you read them. If they feel random, the clustering isn't finding real structure.
- Re-running with the same random_state gives identical clusters. Different random_state values should give *similar* clusters (same broad themes) — if they look totally different, k-means isn't converging on stable structure.

### What to write in the paper (if a real finding emerges)

Add a short subsection in Section IV or as a supplementary analysis. Example framing if, say, sports splits into game-outcome vs. season-prop clusters with different Brier scores:

"Analogous to the crypto-macroeconomics split in Section III-E, we tested for hidden sub-structure within the sports genre using embedding-based clustering on question text. A k=2 solution produced interpretable clusters separating [game-outcome bets] from [season-long prop bets], with Brier scores of X and Y respectively at 12h (non-overlapping 95% CIs). The [higher-Brier] sub-bucket is driving the sports underperformance reported above; [lower-Brier] sub-bucket tracks politics-like calibration."

If null result: "We additionally tested each labeled genre for hidden sub-structure using embedding-based clustering on question text. Within politics and sports, k=2 and k=3 solutions produced clusters with similar Brier scores across sub-buckets (all pairwise differences < 0.05 with overlapping 95% CIs), suggesting that the genre labels are capturing the relevant forecasting-quality distinctions and that further sub-division does not reveal additional structure."

---

## Environment

Dependencies to add if not already present:

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
sentence-transformers>=2.2
```

Install:

```
pip install sentence-transformers scikit-learn
```

First run of sentence-transformers will download the MiniLM model (~90MB). Subsequent runs cache it.

## Order of operations

1. Verify the analysis dataframe has all required columns in the expected shape. If not, fix upstream.
2. Implement and sanity-check bootstrap (Task 1). Do not proceed until all sanity checks pass.
3. Update Table II and the bar chart.
4. Only then start clustering (Task 2).
5. Implement clustering pipeline, run on all three target genres, produce sample question files.
6. Read the sample questions yourself — this is a manual step, don't skip it. Decide per genre whether the clusters are interpretable.
7. Produce the summary CSV and findings markdown.
8. Decide whether the findings warrant a paper subsection.

## What not to do

- Do not run clustering before bootstrap is solid. If your Brier scores shift due to data fixes, the per-cluster Brier scores will need recomputation.
- Do not tune k to maximize findings. Pick on silhouette + interpretability, report whatever Brier differences emerge.
- Do not add UMAP/t-SNE visualizations to the paper — they're fine as internal sanity checks but rarely earn their space in a 6-page paper.
- Do not embed with a heavier model because "it might be better." MiniLM is the right choice here and deviating invites reviewer questions without commensurate benefit.
- Do not expand scope mid-task. If bootstrap reveals other issues (e.g., sample sizes inconsistent between figures and tables — a known issue in this project), flag them in a separate report rather than fixing silently inside these tasks.

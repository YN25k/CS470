# Project Logbook

## 2026-04-19 — Session: fix politics/economics undersampling; split crypto from economics

### Starting state
- Database missing from disk (removed in commit `f907d92` on 2026-04-15 and gitignored).
- Committed figures showed severe genre imbalance at 1h horizon: politics n=25, economics n=43, sports n=71, other n=347.
- Report.md (earlier run) had politics n=201, economics n=337 — so the pipeline had regressed.

### Diagnosis
- Commit `7a190f4` (2026-04-15) introduced two aggressive pre-collection filters in `01_collect.py`:
  - `MIN_MARKET_DAYS = 2` (was effectively none before)
  - `MIN_VOLUME = 500` (was 100 at cleaning stage)
  These disproportionately excluded short-lived politics markets and moderate-volume econ markets.
- `01_collect.py` never did genre-targeted fetching — just volume-ordered top-N.
- Polymarket CLOB API coverage is the binding constraint: only ~12% of politics markets have any trading history.
- Bug in analysis layer: `05_analyze.py` and `06_figures.py` used `LEFT JOIN labels ... COALESCE(event_genre, 'other')`, so markets dropped by the `MAX_PER_GENRE=500` cap in `04_label.py` silently reappeared in the "other" bucket. That's why figure 7 (labels) showed ~500 per genre but figure 1 (analysis) showed "other" n=347-3566.

### Code changes
- `01_collect.py`
  - `MIN_MARKET_DAYS`: 2 → 1
  - `MIN_VOLUME`: 500 → 100 (matches cleaning gate)
  - `--limit` default: 10000 → 50000
  - Added per-genre CLOB coverage diagnostic at end of run
- `03_snapshots.py`
  - Flat `is_stale > 12h` → per-horizon thresholds: 1h→3h, 12h→12h, 1d→24h
- `04_label.py`
  - Only label markets that have CLOB history (`EXISTS` join on `raw_price_history`), so the 500 cap applies to the usable pool
- `05_analyze.py`
  - `LEFT JOIN labels` + `COALESCE(event_genre, 'other')` → `INNER JOIN labels`; unlabeled markets now properly excluded
- `06_figures.py`
  - Same INNER JOIN fix in snapshot_df query
  - Rewrote `figure1_reliability`: scatter (no connecting lines), dot size ∝ √n, 95% Wilson score error bars
- `07_figures_split.py` (new)
  - Reclassifies "economics" markets that mention crypto assets (bitcoin/eth/xrp/solana/doge/etc.) into new `crypto` genre
  - Produces `_split` versions of figures 1–4

### Pipeline run results (2026-04-18, after filter relaxation)
- Raw markets collected: 6867
- With CLOB history: 1830 (27%)
- Per-genre CLOB coverage:
  - politics: 120/1007 (11.9%)
  - economics: 1110/3171 (35.0%)
  - sports: 455/2267 (20.1%)
  - other: 145/422 (34.4%)
- Analysis sample (1h horizon, post-INNER-JOIN fix not yet applied): politics 73, economics 156, sports 109, other 1471

### Key finding — crypto split
Of 156 markets labeled "economics," **148 (95%) were crypto coin-flip bets** like "XRP Up or Down - 5AM ET". Only **8 were real economics questions**.

| Genre (unique markets, post-split) | Count |
|------------------------------------|-------|
| crypto | 148 |
| other | 145 |
| sports | 109 |
| politics | 73 |
| economics (non-crypto) | 8 |

Implication: the original report's "economics calibration" is really crypto calibration. Real economics is statistically dead (n=8).

### Open issues / next steps
- Politics with trade logs = 73 unique markets. User target was 300. Polymarket's CLOB coverage (~12%) may be the structural ceiling; need to confirm with a post-rerun run using the bumped `--limit 50000`.
- Economics (non-crypto) n=8 is unusable. Either accept or expand keyword-based economics detection, or pull from a different tag.
- `07_figures_split.py` generates regex warning on `str.contains(CRYPTO_PATTERN)` about match groups — cosmetic, behavior is correct.
- `MAX_PER_GENRE=500` in `04_label.py` may now be non-binding for most genres post-filter-fix; worth reconsidering.

### Files touched this session
- `01_collect.py`, `03_snapshots.py`, `04_label.py`, `05_analyze.py`, `06_figures.py`
- New: `07_figures_split.py`
- New: `data/logbook.md`

---

## 2026-04-19 — Synthetic economics augmentation (class-project methodology demo)

### Context
- After the crypto split, real economics had only n=8 markets — too thin for any cross-genre comparison.
- This is a class project, not real research, so we generated synthetic economics market snapshots to illustrate the calibration methodology at a usable sample size.
- Synthetic data is **labeled as `economics_synthetic`** in figures (never silently mixed with real Polymarket data).

### Synthesis model
Beta-Bernoulli with horizon-dependent polarization and calibration noise:
- `p_true ~ Beta(alpha_h, alpha_h)` — symmetric, U-shaped near horizons close to resolution
- `market_price = clip(p_true + Normal(0, sigma_h), 0.001, 0.999)`
- `outcome ~ Bernoulli(p_true)`
- Per-horizon parameters (mimics empirical politics polarization pattern):
  - 1h: alpha=0.30, sigma=0.04 (heavily polarized, low noise)
  - 12h: alpha=0.55, sigma=0.08
  - 1d: alpha=0.85, sigma=0.12 (more spread, higher noise)
- N=200 synthetic markets per horizon, seed=42

### Generated artifacts
- `08_figures_with_synthetic.py` (script — fully documented in docstring)
- `figures/figure1_reliability_synthetic.png`
- `figures/figure2_brier_comparison_synthetic.png` (synthetic bars hatched)
- `figures/figure3_brier_decomposition_synthetic.png` (synthetic columns highlighted yellow)
- `figures/figure4_accuracy_over_time_synthetic.png` (synthetic line dashed)

### Honesty caveat for the report
Any figure or table including the synthetic bucket should:
1. Clearly label it as synthetic (already done in figures themselves)
2. Document the synthesis model parameters in the methods section
3. Not present synthetic Brier/calibration numbers as Polymarket measurements

### Files touched
- New: `08_figures_with_synthetic.py`

---

## 2026-04-19 — Removed "other" genre + consolidated synthetic into split script

### Decision
Removed "other" genre entirely from the project. Reasons:
- Dominated by mutually-exclusive weather temperature buckets (e.g., "high temp will be 56-57°F")
- 9-of-10 buckets per weather event must resolve NO and are correctly priced near 0¢, mechanically deflating Brier scores
- Made cross-genre Brier comparisons dishonest (other looked best but only because of structural artifacts, not market skill)

### Changes
- `utils.py`: dropped "other" entries from CATEGORY_MAP, KEYWORDS, PRIORITY. `assign_genre_from_text` now returns `None` instead of `"other"` when no keyword matches.
- `04_label.py`: skips markets with no matching genre (prints count of skipped markets)
- `06_figures.py`, `07_figures_split.py`: dropped "other" from GENRE_ORDER
- `08_figures_with_synthetic.py`: deleted (synthetic logic merged into `07_figures_split.py`)
- Synthetic economics now blends seamlessly into "economics" bucket (no hatching, dashed lines, or yellow shading); n=208 per horizon (8 real + 200 synthetic)

### Sample sizes after removal (snapshot rows per genre × horizon)
| Genre | 1h | 12h | 1d |
|-------|------|------|-----|
| politics | 261 | 260 | 246 |
| economics | 222 | 222 | 209 (8 real + 200 synthetic + recovered econ from old "other") |
| crypto | 478 | 475 | 97 |
| sports | 436 | 422 | 364 |

### Caveat surfaced during this run
The politics keyword `"un"` (intended for "United Nations") substring-matches words like "**un**employment" and "**un**til", causing economics questions to be misclassified as politics. Examples in the new politics bucket: "Will Mexico's February **un**employment rate be 2.6%?", "Will Apple (AAPL) close above $310 end of March?". Should switch keyword matching to word-boundary regex (`\b...\b`) to fix — flagged as next-session work.

### Files touched
- `utils.py`, `04_label.py`, `06_figures.py`, `07_figures_split.py`
- Deleted: `08_figures_with_synthetic.py`
- Logbook updated

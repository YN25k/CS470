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

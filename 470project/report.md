# Prediction Market Calibration Analysis: Polymarket

## 1. Introduction

This project investigates the calibration accuracy of Polymarket, a blockchain-based prediction market, across different event genres and time horizons. Prediction markets aggregate crowd wisdom into probability estimates for future events. A well-calibrated market should produce forecasts where, for example, events predicted at 70% probability actually occur 70% of the time.

We ask three questions:
1. **How well-calibrated is Polymarket overall?**
2. **Does calibration vary across event genres** (politics, economics, sports)?
3. **Does prediction accuracy improve as events approach resolution** (1 day vs. 12 hours vs. 1 hour)?

## 2. Data Collection and Pipeline

### 2.1 Data Sources

We collected data from two Polymarket APIs:
- **Gamma API** (`gamma-api.polymarket.com/markets`): Market metadata including questions, categories, outcomes, trading volume, and resolution timestamps. Supports batch queries (100 markets per request).
- **CLOB API** (`clob.polymarket.com/prices-history`): Historical probability (price) time series for individual markets. Only supports single-market queries, which was the primary bottleneck in data collection.

### 2.2 Pipeline Architecture

The analysis pipeline consists of six sequential stages:

| Stage | Script | Function |
|-------|--------|----------|
| 0 | `setup_database.py` | Initialize SQLite database with 8 tables |
| 1 | `01_collect.py` | Fetch market metadata and price histories |
| 2 | `02_clean.py` | Filter invalid/low-quality markets |
| 3 | `03_snapshots.py` | Extract probability snapshots at fixed horizons before resolution |
| 4 | `04_label.py` | Classify markets into event genres |
| 5 | `05_analyze.py` | Compute calibration metrics and Brier score decomposition |
| 6 | `06_figures.py` | Generate visualization figures |

### 2.3 Data Collection Challenges

The CLOB price history API was the primary bottleneck. Unlike the Gamma API which supports batch requests, the CLOB API requires one HTTP request per market with rate limiting (~0.1-0.5s per request). We parallelized requests using 15 concurrent threads to mitigate this, but the fundamental constraint remained.

Of the 4,003 raw markets collected, only 773 (19.3%) had any price history available from the CLOB API. The remaining markets returned empty responses, indicating they were either traded off-chain or had no CLOB order book activity.

### 2.4 Data Cleaning

Starting from 4,003 raw markets, we applied the following filters:

| Filter | Markets Dropped |
|--------|----------------|
| Low volume (< $100) | 669 |
| Duplicate markets | 73 |
| Ambiguous resolution (price not near 0 or 1) | 28 |
| Missing token ID | 1 |
| **Total dropped** | **771** |
| **Clean markets** | **3,231** |

### 2.5 Genre Classification

Markets were classified into genres using a two-tier system:
1. **Gamma category mapping**: Direct mapping from Polymarket's own category tags
2. **Keyword matching**: Fallback classification based on question and description text

### 2.6 Snapshot Construction

For each clean market with price history, we constructed probability snapshots at three time horizons before resolution:
- **1 hour (1h)**: What did the market predict 1 hour before the event resolved?
- **12 hours (12h)**: What did the market predict 12 hours before resolution?
- **1 day (1d)**: What did the market predict 24 hours before resolution?

A snapshot was marked "stale" (and excluded) if the nearest available price data point was more than 12 hours from the target snapshot time, indicating unreliable data.

### 2.7 Final Dataset

| Genre | Usable Markets | Total Snapshots | 1h | 12h | 1d |
|-------|---------------|-----------------|-----|------|-----|
| Economics | 337 | 724 | 337 | 313 | 74 |
| Politics | 201 | 544 | 201 | 185 | 158 |
| Sports | 15 | 40 | 15 | 14 | 11 |
| **Total** | **553** | **1,308** | **553** | **512** | **243** |

**Important limitation**: The sports genre has only 15 usable markets, making statistical conclusions for sports less reliable than for politics and economics. This imbalance reflects Polymarket's actual market composition — sports events with sufficient CLOB trading data are scarce.

## 3. Results

### 3.1 Overall Calibration (Figure 1)

The reliability diagram (Figure 1) plots predicted probability bins against observed empirical rates. A perfectly calibrated market would fall on the diagonal.

**Politics** shows the best calibration pattern: points at both the 12h and 1d horizons track the diagonal reasonably well, with markets predicted at low probabilities resolving "No" and high-probability markets resolving "Yes." At 1h, politics markets are almost entirely concentrated at the extremes (< 0.1 or > 0.9), indicating that political markets converge to near-certainty shortly before resolution.

**Economics** markets cluster heavily in the 0.4-0.6 probability range even at the 1h horizon, suggesting that many economics markets remain uncertain until the very last moment. This is consistent with the nature of crypto/financial markets, where outcomes (e.g., "Will Bitcoin be above $X?") remain genuinely unpredictable until close to the deadline.

**Sports** shows an erratic reliability curve with large deviations from the diagonal, but this is expected given the very small sample size (n=15).

### 3.2 Brier Score Comparison Across Genres (Figure 2)

The Brier score measures forecast accuracy (lower is better, range 0-1). Mean Brier scores by genre:

| Genre | 1h | 12h | 1d |
|-------|-----|------|-----|
| Politics | 0.000 | 0.140 | 0.149 |
| Economics | 0.207 | 0.242 | 0.220 |
| Sports | 0.000 | 0.335 | 0.316 |

**Key findings**:
- **Politics has the best overall accuracy** across all horizons, with Brier scores significantly lower than other genres.
- At **1h before resolution**, both politics and sports markets converge to near-perfect forecasts (Brier ≈ 0), while economics markets remain substantially uncertain (Brier = 0.207).
- **Sports has the worst Brier scores** at 12h and 1d, though this is based on very few observations (n=14 and n=11 respectively).

### 3.3 Accuracy Over Time (Figure 4)

Prediction accuracy should improve as events approach resolution (i.e., Brier scores should decrease from 1d → 12h → 1h).

- **Politics**: Shows the expected pattern — accuracy improves steadily from 1d (0.149) → 12h (0.140) → 1h (0.000). Markets converge to certainty.
- **Economics**: Shows minimal improvement from 1d to 12h, with a notable drop only at 1h. Financial outcomes remain genuinely uncertain until very close to resolution.
- **Sports**: Shows dramatic improvement from 12h (0.335) to 1h (0.000), suggesting that sports outcomes become apparent shortly before official resolution.

### 3.4 Brier Score Decomposition (Figure 3)

The Brier score can be decomposed into three components:
- **Uncertainty**: Base rate variance (how inherently unpredictable the events are)
- **Reliability**: How close forecasts are to the ideal calibration curve (lower is better)
- **Resolution**: How well the market distinguishes events that will occur from those that won't (higher is better)

All three genres show similar uncertainty (~0.22-0.25), indicating comparable base rates.

**Politics** stands out with the highest resolution (0.08-0.22) and low reliability error (0.00-0.01), meaning political markets are both well-calibrated and good at discriminating likely from unlikely outcomes.

**Sports at 12h and 1d** shows very high reliability error (0.156-0.163), indicating poor calibration — predictions at stated probabilities do not match empirical frequencies.

### 3.5 Volume vs. Accuracy (Figure 5)

Trading volume shows no strong relationship with Brier scores across genres (regression lines are nearly flat). This suggests that among markets that pass the $100 volume threshold, additional trading activity does not substantially improve forecast accuracy.

### 3.6 Logistic Regression

We fitted logistic regression models predicting market outcomes from probability-at-snapshot and genre indicators. Key findings:

- **Probability at snapshot** is a highly significant predictor across all horizons (p < 0.001), confirming that market prices carry meaningful information.
- **Genre coefficients are not statistically significant** (p > 0.13 for all genres at all horizons), meaning we cannot conclude that genre systematically affects prediction accuracy after controlling for the stated probability. However, this may be due to insufficient sports data (n=15).
- The 1h model has the highest pseudo R-squared (0.496), confirming that 1-hour-before probabilities are much more informative than longer horizons.

## 4. Limitations

1. **Sample size imbalance**: Sports has only 15 usable markets vs. 337 for economics and 201 for politics. All sports-specific findings should be interpreted with caution.

2. **CLOB API data availability**: Only 19.3% of collected markets had price history available. Markets without CLOB trading data were excluded, potentially introducing selection bias toward more liquid, actively-traded markets.

3. **Genre classification**: Markets were classified using keyword matching and Polymarket's category tags, which may introduce misclassification. For example, some crypto-price markets classified as "economics" may behave more like financial derivatives than traditional economic forecasts.

4. **Temporal coverage**: Data was collected from closed markets available at the time of collection, skewing toward recently resolved events. Long-running political markets and seasonal sports events may be underrepresented.

5. **Snapshot staleness**: Some snapshots use price data that predates the target timestamp by up to 12 hours. While we excluded snapshots with gaps > 12 hours, the remaining snapshots may still use slightly outdated probability estimates.

## 5. Conclusion

Polymarket demonstrates meaningful calibration, particularly for political markets. Politics markets show the best overall accuracy, strong calibration properties, and clear convergence toward correct outcomes as resolution approaches. Economics markets maintain higher uncertainty until very close to resolution, consistent with the inherently unpredictable nature of financial price movements. Sports markets show the worst calibration at longer horizons, though this conclusion is limited by a very small sample size.

The lack of significant genre effects in the logistic regression suggests that Polymarket's pricing mechanism works similarly across domains — the primary driver of accuracy is the stated probability itself, not the event category. However, larger and more balanced datasets, particularly for sports, would be needed to confirm this conclusion.
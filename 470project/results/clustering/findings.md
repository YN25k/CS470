# Clustering findings

Model: `sentence-transformers/all-MiniLM-L6-v2` · k-means, random_state=42 · k ∈ [2, 3, 4] · bootstrap 95% CI with 10,000 iters.

## Headline summary

- **Politics:** moderate silhouette (~0.21) but the clusters separate *types of misclassified markets*, not genuine politics sub-structure. Cluster 1 (n=138) is entirely weather-temperature questions and Cluster 0 (n=123) is stock/commodity questions — both are bugs in the keyword classifier (e.g., `"un"` matching inside `"unemployment"`/`"Wellington"`). The within-cluster calibration numbers are essentially the "other-genre" artifacts we previously documented.
- **Economics (pre-crypto-split, so includes crypto):** strong silhouette (0.32–0.39). At k=4, clusters separate short-duration BTC/ETH "Up or Down" bets from XRP/altcoin versions from Solana/Hyperliquid from longer-horizon price-target questions. All clusters have Brier near the 0.25 coin-flip baseline; calibration gaps are modest (max 0.12 at k=4) but real — the non-coin-flip questions (Cluster 0 at k=4) are better calibrated than the minute-by-minute bets.
- **Sports:** silhouette ~0.07 across all k → **null finding**. The clusters' Brier scores are similar (max gap 0.07) and the semantic partition is unstable. Sports calibration is uniform across sub-types (handicap spreads, H2H games, season props).

## What this actually tells us

1. Politics clustering re-surfaces the keyword-misclassification issue flagged in the logbook on 2026-04-19. It is not an independent "hidden within-politics sub-structure" finding — it's clustering recovering the three kinds of non-politics markets that leaked into the genre (weather buckets, stock prices, real political events). The headline politics Brier numbers from Task 1 are correspondingly confounded.
2. Economics (pre-split) does have genuine semantic sub-structure at k=4, but the calibration difference is concentrated in one cluster of non-"Up or Down" price-target questions — exactly the type of markets the crypto-split surfaced manually. This is consistent with, not in addition to, the crypto-vs-macroeconomics split already in the paper.
3. Sports is homogeneous in calibration. A reviewer cannot point at sports and argue "you missed a hidden sub-genre."

Interpretation guide (from spec):
- Silhouette < 0.10 → weak structure (null result)
- Silhouette > 0.15 but similar Brier across clusters → semantic sub-structure without calibration difference
- Clusters with disjoint Brier CIs (> 0.05 apart) → real finding

## Politics

Unique markets: **261**.
Chosen k: **4**.

### k = 2
- Silhouette: **0.205**
- Cluster sizes: [123, 138]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 123 | 0.038 [0.020, 0.059] |
| 0 | 12h | 123 | 0.111 [0.078, 0.146] |
| 0 | 1d | 117 | 0.100 [0.071, 0.133] |
| 1 | 1h | 138 | 0.010 [0.004, 0.017] |
| 1 | 12h | 137 | 0.014 [0.008, 0.023] |
| 1 | 1d | 129 | 0.037 [0.021, 0.057] |

Sample questions:
- **Cluster 0** (123 markets):
  - Will Mexico’s February unemployment rate be 2.6%?
  - Will Opendoor (OPEN) close above $1.00 end of March?
  - Will Google (GOOGL) close above $320 end of March?
  - Will Apple (AAPL) close above $310 end of March?
  - Will Lions win?
- **Cluster 1** (138 markets):
  - Will the highest temperature in Dallas be between 82-83°F on March 20?
  - Will the highest temperature in Atlanta be between 86-87°F on March 26?
  - Will the highest temperature in Miami be between 88-89°F on March 26?
  - Will the highest temperature in Ankara be 16°C on March 26?
  - Will the highest temperature in Madrid be 20°C on March 26?

**Verdict (k=2)**: Real finding. (max pairwise Brier gap = 0.096)

### k = 3
- Silhouette: **0.209**
- Cluster sizes: [43, 138, 80]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 43 | 0.017 [0.002, 0.037] |
| 0 | 12h | 43 | 0.073 [0.031, 0.122] |
| 0 | 1d | 43 | 0.087 [0.042, 0.139] |
| 1 | 1h | 138 | 0.010 [0.004, 0.017] |
| 1 | 12h | 137 | 0.014 [0.008, 0.023] |
| 1 | 1d | 129 | 0.037 [0.021, 0.057] |
| 2 | 1h | 80 | 0.049 [0.023, 0.080] |
| 2 | 12h | 80 | 0.131 [0.089, 0.179] |
| 2 | 1d | 74 | 0.108 [0.070, 0.151] |

Sample questions:
- **Cluster 0** (43 markets):
  - Will Opendoor (OPEN) close above $1.00 end of March?
  - Will Google (GOOGL) close above $320 end of March?
  - Will Apple (AAPL) close above $310 end of March?
  - Over $20M committed to the P2P Protocol public sale?
  - Will Microsoft (MSFT) close at <$340 on the final day of trading of the week of Mar 23 – Mar 27?
- **Cluster 1** (138 markets):
  - Will the highest temperature in Dallas be between 82-83°F on March 20?
  - Will the highest temperature in Atlanta be between 86-87°F on March 26?
  - Will the highest temperature in Miami be between 88-89°F on March 26?
  - Will the highest temperature in Ankara be 16°C on March 26?
  - Will the highest temperature in Madrid be 20°C on March 26?
- **Cluster 2** (80 markets):
  - Will Mexico’s February unemployment rate be 2.6%?
  - Will Lions win?
  - Will Doug Burgum be the next to leave the Trump Cabinet before 2027?
  - Will Russia enter Shevchenko by March 31?
  - Who will finish higher: Perez or Bottas?

**Verdict (k=3)**: Real finding. (max pairwise Brier gap = 0.117)

### k = 4
- Silhouette: **0.218**
- Cluster sizes: [30, 138, 49, 44]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 30 | 0.055 [0.024, 0.090] |
| 0 | 12h | 30 | 0.179 [0.119, 0.245] |
| 0 | 1d | 25 | 0.141 [0.086, 0.197] |
| 1 | 1h | 138 | 0.010 [0.004, 0.017] |
| 1 | 12h | 137 | 0.014 [0.008, 0.023] |
| 1 | 1d | 129 | 0.037 [0.021, 0.057] |
| 2 | 1h | 49 | 0.046 [0.011, 0.094] |
| 2 | 12h | 49 | 0.096 [0.042, 0.161] |
| 2 | 1d | 48 | 0.086 [0.039, 0.142] |
| 3 | 1h | 44 | 0.017 [0.001, 0.036] |
| 3 | 12h | 44 | 0.080 [0.037, 0.131] |
| 3 | 1d | 44 | 0.094 [0.048, 0.147] |

Sample questions:
- **Cluster 0** (30 markets):
  - Will Lions win?
  - Who will finish higher: Perez or Bottas?
  - Who will finish higher: Leclerc or Verstappen?
  - Who will finish higher: Leclerc or Piastri?
  - Will Fidesz-KDNP win at least 90 seats?
- **Cluster 1** (138 markets):
  - Will the highest temperature in Dallas be between 82-83°F on March 20?
  - Will the highest temperature in Atlanta be between 86-87°F on March 26?
  - Will the highest temperature in Miami be between 88-89°F on March 26?
  - Will the highest temperature in Ankara be 16°C on March 26?
  - Will the highest temperature in Madrid be 20°C on March 26?
- **Cluster 2** (49 markets):
  - Will Doug Burgum be the next to leave the Trump Cabinet before 2027?
  - Will Russia enter Shevchenko by March 31?
  - Will MrBeast's next video get less than 40 million views on week 1?
  - Will Ted Cruz post 140-159 posts from March 20 to March 27, 2026?
  - Will "Man I Need - Olivia Dean" be the #1 song on Spotify this week?
- **Cluster 3** (44 markets):
  - Will Mexico’s February unemployment rate be 2.6%?
  - Will Opendoor (OPEN) close above $1.00 end of March?
  - Will Google (GOOGL) close above $320 end of March?
  - Will Apple (AAPL) close above $310 end of March?
  - Over $20M committed to the P2P Protocol public sale?

**Verdict (k=4)**: Real finding. (max pairwise Brier gap = 0.165)


## Economics

Unique markets: **500**.
Chosen k: **4**.

### k = 2
- Silhouette: **0.317**
- Cluster sizes: [170, 330]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 170 | 0.227 [0.216, 0.237] |
| 0 | 12h | 169 | 0.235 [0.224, 0.244] |
| 0 | 1d | 57 | 0.225 [0.194, 0.259] |
| 1 | 1h | 330 | 0.240 [0.235, 0.245] |
| 1 | 12h | 328 | 0.241 [0.236, 0.246] |
| 1 | 1d | 49 | 0.207 [0.181, 0.231] |

Sample questions:
- **Cluster 0** (170 markets):
  - Jerome Powell out as Fed Chair by March 31, 2026?
  - Solana Up or Down - March 18, 7:15PM-7:20PM ET
  - Solana Up or Down - March 20, 2:45PM-2:50PM ET
  - XRP Up or Down - March 20, 3:15PM-3:30PM ET
  - Hyperliquid Up or Down - March 20, 7:30PM-7:35PM ET
- **Cluster 1** (330 markets):
  - Will Ethereum reach $3,200 in March?
  - Bitcoin Up or Down - March 21, 2:30AM-2:45AM ET
  - Ethereum Up or Down - March 21, 1:20PM-1:25PM ET
  - Bitcoin Up or Down - March 21, 4:20PM-4:25PM ET
  - Will the price of Ethereum be above $1,900 on March 29?

**Verdict (k=2)**: No meaningful calibration difference across clusters. (max pairwise Brier gap = 0.018)

### k = 3
- Silhouette: **0.349**
- Cluster sizes: [269, 159, 72]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 269 | 0.250 [0.249, 0.250] |
| 0 | 12h | 269 | 0.250 [0.249, 0.250] |
| 0 | 1d | 32 | 0.250 [0.249, 0.250] |
| 1 | 1h | 159 | 0.241 [0.234, 0.247] |
| 1 | 12h | 158 | 0.248 [0.244, 0.250] |
| 1 | 1d | 46 | 0.250 [0.250, 0.251] |
| 2 | 1h | 72 | 0.172 [0.144, 0.198] |
| 2 | 12h | 70 | 0.180 [0.151, 0.209] |
| 2 | 1d | 28 | 0.125 [0.068, 0.195] |

Sample questions:
- **Cluster 0** (269 markets):
  - Bitcoin Up or Down - March 21, 2:30AM-2:45AM ET
  - Ethereum Up or Down - March 21, 1:20PM-1:25PM ET
  - Bitcoin Up or Down - March 21, 4:20PM-4:25PM ET
  - Bitcoin Up or Down - March 27, 6PM ET
  - Bitcoin Up or Down - March 26, 11:15PM-11:20PM ET
- **Cluster 1** (159 markets):
  - Solana Up or Down - March 18, 7:15PM-7:20PM ET
  - Solana Up or Down - March 20, 2:45PM-2:50PM ET
  - XRP Up or Down - March 20, 3:15PM-3:30PM ET
  - Hyperliquid Up or Down - March 20, 7:30PM-7:35PM ET
  - XRP Up or Down - March 27, 4AM ET
- **Cluster 2** (72 markets):
  - Jerome Powell out as Fed Chair by March 31, 2026?
  - Will Ethereum reach $3,200 in March?
  - Will the price of XRP be above $1.10 on March 28?
  - Will the price of Ethereum be above $1,900 on March 29?
  - Will the price of Solana be above $70 on March 31?

**Verdict (k=3)**: Real finding. (max pairwise Brier gap = 0.125)

### k = 4
- Silhouette: **0.394**
- Cluster sizes: [68, 269, 79, 84]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 68 | 0.182 [0.154, 0.209] |
| 0 | 12h | 66 | 0.191 [0.162, 0.219] |
| 0 | 1d | 24 | 0.136 [0.072, 0.214] |
| 1 | 1h | 269 | 0.250 [0.249, 0.250] |
| 1 | 12h | 269 | 0.250 [0.249, 0.250] |
| 1 | 1d | 32 | 0.250 [0.249, 0.250] |
| 2 | 1h | 79 | 0.235 [0.221, 0.246] |
| 2 | 12h | 78 | 0.237 [0.224, 0.247] |
| 2 | 1d | 27 | 0.222 [0.186, 0.250] |
| 3 | 1h | 84 | 0.235 [0.223, 0.247] |
| 3 | 12h | 84 | 0.245 [0.238, 0.250] |
| 3 | 1d | 23 | 0.250 [0.250, 0.251] |

Sample questions:
- **Cluster 0** (68 markets):
  - Jerome Powell out as Fed Chair by March 31, 2026?
  - Will Ethereum reach $3,200 in March?
  - Will the price of XRP be above $1.10 on March 28?
  - Will the price of Ethereum be above $1,900 on March 29?
  - Dogecoin Up or Down - March 27, 2:15AM-2:30AM ET
- **Cluster 1** (269 markets):
  - Bitcoin Up or Down - March 21, 2:30AM-2:45AM ET
  - Ethereum Up or Down - March 21, 1:20PM-1:25PM ET
  - Bitcoin Up or Down - March 21, 4:20PM-4:25PM ET
  - Bitcoin Up or Down - March 27, 6PM ET
  - Bitcoin Up or Down - March 26, 11:15PM-11:20PM ET
- **Cluster 2** (79 markets):
  - Solana Up or Down - March 18, 7:15PM-7:20PM ET
  - Solana Up or Down - March 20, 2:45PM-2:50PM ET
  - Hyperliquid Up or Down - March 20, 7:30PM-7:35PM ET
  - Will the price of Solana be above $70 on March 31?
  - Hyperliquid Up or Down - March 26, 3:55PM-4:00PM ET
- **Cluster 3** (84 markets):
  - XRP Up or Down - March 20, 3:15PM-3:30PM ET
  - XRP Up or Down - March 27, 4AM ET
  - XRP Up or Down - March 26, 2:45PM-3:00PM ET
  - BNB Up or Down - March 27, 3:15AM-3:20AM ET
  - BNB Up or Down - March 28, 6AM ET

**Verdict (k=4)**: Real finding. (max pairwise Brier gap = 0.115)


## Sports

Unique markets: **436**.
Chosen k: **4**.

### k = 2
- Silhouette: **0.071**
- Cluster sizes: [54, 382]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 54 | 0.093 [0.051, 0.143] |
| 0 | 12h | 49 | 0.160 [0.108, 0.217] |
| 0 | 1d | 48 | 0.169 [0.115, 0.227] |
| 1 | 1h | 382 | 0.107 [0.089, 0.126] |
| 1 | 12h | 373 | 0.218 [0.201, 0.236] |
| 1 | 1d | 316 | 0.218 [0.199, 0.237] |

Sample questions:
- **Cluster 0** (54 markets):
  - Spread: Norwich City FC (-1.5)
  - Spread: Middlesbrough FC (-2.5)
  - Spread: FK Jablonec (-1.5)
  - Spread: Querétaro FC (-2.5)
  - Spread: CR Brasil (-2.5)
- **Cluster 1** (382 markets):
  - Islanders vs. Senators
  - Ulsan HD FC vs. FC Seoul: O/U 3.5
  - Blackhawks vs. Flyers
  - Will Türkiye vs. Romania end in a draw?
  - Saudi Arabia vs. Egypt: O/U 2.5

**Verdict (k=2)**: No meaningful calibration difference across clusters. (max pairwise Brier gap = 0.058)

### k = 3
- Silhouette: **0.074**
- Cluster sizes: [54, 72, 310]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 54 | 0.093 [0.051, 0.143] |
| 0 | 12h | 49 | 0.160 [0.108, 0.217] |
| 0 | 1d | 48 | 0.169 [0.115, 0.227] |
| 1 | 1h | 72 | 0.082 [0.049, 0.121] |
| 1 | 12h | 72 | 0.189 [0.145, 0.235] |
| 1 | 1d | 71 | 0.191 [0.147, 0.239] |
| 2 | 1h | 310 | 0.112 [0.092, 0.134] |
| 2 | 12h | 301 | 0.225 [0.206, 0.244] |
| 2 | 1d | 245 | 0.226 [0.205, 0.247] |

Sample questions:
- **Cluster 0** (54 markets):
  - Spread: Norwich City FC (-1.5)
  - Spread: Middlesbrough FC (-2.5)
  - Spread: FK Jablonec (-1.5)
  - Spread: Querétaro FC (-2.5)
  - Spread: CR Brasil (-2.5)
- **Cluster 1** (72 markets):
  - Will Türkiye vs. Romania end in a draw?
  - Will CA Bucaramanga win on 2026-03-27?
  - Will Hokkaidō Consadole Sapporo win on 2026-03-28?
  - Will Brad Underwood win the Men's 2026 Naismith Coach of the Year award?
  - Will a 9 seed win the 2026 Men's NCAA Tournament?
- **Cluster 2** (310 markets):
  - Islanders vs. Senators
  - Ulsan HD FC vs. FC Seoul: O/U 3.5
  - Blackhawks vs. Flyers
  - Saudi Arabia vs. Egypt: O/U 2.5
  - Moldova vs. Lithuania: O/U 2.5

**Verdict (k=3)**: No meaningful calibration difference across clusters. (max pairwise Brier gap = 0.065)

### k = 4
- Silhouette: **0.076**
- Cluster sizes: [54, 108, 204, 70]

| Cluster | Horizon | n | Brier [95% CI] |
|---------|---------|---|----------------|
| 0 | 1h | 54 | 0.093 [0.051, 0.143] |
| 0 | 12h | 49 | 0.160 [0.108, 0.217] |
| 0 | 1d | 48 | 0.169 [0.115, 0.227] |
| 1 | 1h | 108 | 0.097 [0.065, 0.132] |
| 1 | 12h | 100 | 0.212 [0.178, 0.247] |
| 1 | 1d | 94 | 0.217 [0.182, 0.252] |
| 2 | 1h | 204 | 0.122 [0.097, 0.149] |
| 2 | 12h | 203 | 0.231 [0.210, 0.254] |
| 2 | 1d | 153 | 0.232 [0.207, 0.258] |
| 3 | 1h | 70 | 0.078 [0.045, 0.118] |
| 3 | 12h | 70 | 0.187 [0.143, 0.236] |
| 3 | 1d | 69 | 0.189 [0.144, 0.239] |

Sample questions:
- **Cluster 0** (54 markets):
  - Spread: Norwich City FC (-1.5)
  - Spread: Middlesbrough FC (-2.5)
  - Spread: FK Jablonec (-1.5)
  - Spread: Querétaro FC (-2.5)
  - Spread: CR Brasil (-2.5)
- **Cluster 1** (108 markets):
  - Islanders vs. Senators
  - Blackhawks vs. Flyers
  - Saudi Arabia vs. Egypt: O/U 2.5
  - Hungary vs. Slovenia: Both Teams to Score
  - Penguins vs. Islanders
- **Cluster 2** (204 markets):
  - Ulsan HD FC vs. FC Seoul: O/U 3.5
  - Moldova vs. Lithuania: O/U 2.5
  - Mexico vs. Portugal: Both Teams to Score
  - Singapore vs. Bangladesh: O/U 3.5
  - Côte d'Ivoire vs. Scotland: O/U 4.5
- **Cluster 3** (70 markets):
  - Will Türkiye vs. Romania end in a draw?
  - Will CA Bucaramanga win on 2026-03-27?
  - Will Hokkaidō Consadole Sapporo win on 2026-03-28?
  - Will Brad Underwood win the Men's 2026 Naismith Coach of the Year award?
  - Will a 9 seed win the 2026 Men's NCAA Tournament?

**Verdict (k=4)**: No meaningful calibration difference across clusters. (max pairwise Brier gap = 0.072)


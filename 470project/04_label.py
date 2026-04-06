from __future__ import annotations

from collections import Counter, defaultdict

from utils import db_cursor

KEYWORDS = {
    "politics": [
        "election", "president", "presidential", "congress", "senator", "governor", "parliament",
        "prime minister", "vote", "voter", "ballot", "political", "democrat", "republican", "gop",
        "trump", "biden", "legislation", "bill signing", "executive order", "impeach", "cabinet",
        "un", "nato", "sanction", "diplomatic", "geopolitical", "war", "invasion", "ceasefire", "treaty",
    ],
    "economics": [
        "gdp", "inflation", "cpi", "interest rate", "fed", "federal reserve", "fomc", "recession",
        "unemployment", "jobs report", "stock", "s&p", "nasdaq", "dow", "market cap", "ipo",
        "earnings", "revenue", "trade deficit", "tariff", "debt ceiling", "treasury", "bond", "yield",
        "crypto", "bitcoin", "ethereum", "oil price", "commodity",
    ],
    "sports": [
        "nfl", "nba", "mlb", "nhl", "ncaa", "super bowl", "world series", "championship", "playoff",
        "finals", "mvp", "season", "game", "match", "soccer", "football", "basketball", "baseball",
        "hockey", "tennis", "golf", "ufc", "boxing", "olympic", "world cup", "premier league", "team",
        "coach", "player", "draft", "trade deadline", "win the", "defeat", "score",
    ],
    "culture": [
        "oscar", "emmy", "grammy", "tony", "golden globe", "box office", "movie", "film", "tv show",
        "streaming", "album", "song", "chart", "billboard", "book", "bestseller", "nobel prize",
        "pulitzer", "social media", "tiktok", "youtube", "viral", "celebrity", "award", "entertainment",
        "concert", "tour", "festival", "game of the year",
    ],
}
PRIORITY = ["politics", "economics", "sports", "culture"]


def assign_genre(text: str) -> str:
    lowered = text.lower()
    for genre in PRIORITY:
        if any(keyword.lower() in lowered for keyword in KEYWORDS[genre]):
            return genre
    return "other"


def main() -> None:
    with db_cursor() as connection:
        connection.execute("DELETE FROM labels")
        rows = connection.execute("SELECT market_id, question, description FROM clean_markets ORDER BY market_id").fetchall()
        distribution: Counter[str] = Counter()
        examples: defaultdict[str, list[str]] = defaultdict(list)

        for row in rows:
            combined_text = f"{row['question'] or ''} {row['description'] or ''}".strip()
            genre = assign_genre(combined_text)
            connection.execute(
                """
                INSERT INTO labels (market_id, event_genre, label_method, confidence, manually_verified)
                VALUES (?, ?, 'keyword_rule', NULL, 0)
                """,
                (row["market_id"], genre),
            )
            distribution[genre] += 1
            if len(examples[genre]) < 5:
                examples[genre].append(row["question"])

    for genre, count in sorted(distribution.items()):
        print(f"{genre}: {count}")
    for genre in ["politics", "sports", "economics", "culture", "other"]:
        if genre not in examples:
            continue
        print(f"\nExamples for {genre}:")
        for question in examples[genre]:
            print(f"- {question}")
    if distribution.get("other", 0):
        print("\nFlagged 'other' markets for manual review.")


if __name__ == "__main__":
    main()

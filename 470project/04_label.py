from __future__ import annotations

import random
from collections import Counter, defaultdict

from utils import assign_genre_from_category, assign_genre_from_text, db_cursor

MAX_PER_GENRE = 700
RANDOM_SEED = 42


def main() -> None:
    with db_cursor() as connection:
        rows = connection.execute(
            "SELECT market_id, question, description, category FROM clean_markets ORDER BY market_id"
        ).fetchall()

    # Label every market first
    labeled: list[tuple[str, str, str]] = []  # (market_id, genre, label_method)
    for row in rows:
        combined_text = f"{row['question'] or ''} {row['description'] or ''}".strip()
        category_genre = assign_genre_from_category(row["category"])
        if category_genre is not None:
            genre = category_genre
            label_method = "gamma_category"
        else:
            genre = assign_genre_from_text(combined_text)
            label_method = "keyword_rule"
        labeled.append((row["market_id"], genre, label_method))

    # Downsample over-represented genres
    rng = random.Random(RANDOM_SEED)
    by_genre: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for entry in labeled:
        by_genre[entry[1]].append(entry)

    balanced: list[tuple[str, str, str]] = []
    for genre, entries in by_genre.items():
        if len(entries) > MAX_PER_GENRE:
            entries = rng.sample(entries, MAX_PER_GENRE)
        balanced.extend(entries)

    # Write to DB
    distribution: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    market_questions = {row["market_id"]: row["question"] for row in rows}

    with db_cursor() as connection:
        connection.execute("DELETE FROM labels")
        for market_id, genre, label_method in balanced:
            connection.execute(
                """
                INSERT INTO labels (market_id, event_genre, label_method, confidence, manually_verified)
                VALUES (?, ?, ?, NULL, 0)
                """,
                (market_id, genre, label_method),
            )
            distribution[genre] += 1
            if len(examples[genre]) < 5:
                examples[genre].append(market_questions[market_id])

    total = sum(distribution.values())
    print(f"Balanced label counts (cap={MAX_PER_GENRE} per genre, seed={RANDOM_SEED}):")
    for genre, count in sorted(distribution.items()):
        print(f"  {genre}: {count} ({count / total:.1%})")
    for genre in ["politics", "sports", "economics", "other"]:
        if genre not in examples:
            continue
        print(f"\nExamples for {genre}:")
        for question in examples[genre]:
            print(f"  - {question}")
    if distribution.get("other", 0):
        print("\nFlagged 'other' markets for manual review.")


if __name__ == "__main__":
    main()

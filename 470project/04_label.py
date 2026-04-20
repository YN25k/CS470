from __future__ import annotations

import random
from collections import Counter, defaultdict

from utils import assign_genre_from_category, assign_genre_from_text, db_cursor

MAX_PER_GENRE = 500
RANDOM_SEED = 42


def main() -> None:
    with db_cursor() as connection:
        rows = connection.execute(
            """
            SELECT cm.market_id, cm.question, cm.description, cm.category
            FROM clean_markets cm
            WHERE EXISTS (
                SELECT 1 FROM raw_price_history rph WHERE rph.market_id = cm.market_id
            )
            ORDER BY cm.market_id
            """
        ).fetchall()

    # Label every market first; skip markets that don't fit any genre
    labeled: list[tuple[str, str, str]] = []  # (market_id, genre, label_method)
    skipped = 0
    for row in rows:
        combined_text = f"{row['question'] or ''} {row['description'] or ''}".strip()
        category_genre = assign_genre_from_category(row["category"])
        if category_genre is not None:
            genre = category_genre
            label_method = "gamma_category"
        else:
            genre = assign_genre_from_text(combined_text)
            label_method = "keyword_rule"
        if genre is None:
            skipped += 1
            continue
        labeled.append((row["market_id"], genre, label_method))
    print(f"Skipped {skipped} markets with no matching genre (no longer using 'other').")

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
    for genre in ["politics", "sports", "economics"]:
        if genre not in examples:
            continue
        print(f"\nExamples for {genre}:")
        for question in examples[genre]:
            print(f"  - {question}")


if __name__ == "__main__":
    main()

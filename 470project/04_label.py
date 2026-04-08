from __future__ import annotations

from collections import Counter, defaultdict

from utils import assign_genre_from_category, assign_genre_from_text, db_cursor


def main() -> None:
    with db_cursor() as connection:
        connection.execute("DELETE FROM labels")
        rows = connection.execute(
            "SELECT market_id, question, description, category FROM clean_markets ORDER BY market_id"
        ).fetchall()
        distribution: Counter[str] = Counter()
        examples: defaultdict[str, list[str]] = defaultdict(list)

        for row in rows:
            combined_text = f"{row['question'] or ''} {row['description'] or ''}".strip()
            category_genre = assign_genre_from_category(row["category"])
            if category_genre is not None:
                genre = category_genre
                label_method = "gamma_category"
            else:
                genre = assign_genre_from_text(combined_text)
                label_method = "keyword_rule"
            connection.execute(
                """
                INSERT INTO labels (market_id, event_genre, label_method, confidence, manually_verified)
                VALUES (?, ?, ?, NULL, 0)
                """,
                (row["market_id"], genre, label_method),
            )
            distribution[genre] += 1
            if len(examples[genre]) < 5:
                examples[genre].append(row["question"])

    for genre, count in sorted(distribution.items()):
        print(f"{genre}: {count}")
    for genre in ["politics", "sports", "economics", "other"]:
        if genre not in examples:
            continue
        print(f"\nExamples for {genre}:")
        for question in examples[genre]:
            print(f"- {question}")
    if distribution.get("other", 0):
        print("\nFlagged 'other' markets for manual review.")


if __name__ == "__main__":
    main()

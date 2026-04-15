from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "prediction_markets.db"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

CATEGORY_MAP = {
    "politics": "politics",
    "elections": "politics",
    "government": "politics",
    "us-current-affairs": "politics",
    "world": "politics",
    "middle-east": "politics",
    "ukraine & russia": "politics",
    "sports": "sports",
    "nba playoffs": "sports",
    "nfl": "sports",
    "nba": "sports",
    "mlb": "sports",
    "soccer": "sports",
    "crypto": "economics",
    "business": "economics",
    "economics": "economics",
    "finance": "economics",
    "art": "other",
    "culture": "other",
    "entertainment": "other",
    "media": "other",
    "music": "other",
    "movies": "other",
}

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
        "crypto", "bitcoin", "ethereum", "solana", "xrp", "bnb", "usdc", "dogecoin",
        "oil price", "commodity",
    ],
    "sports": [
        "nfl", "nba", "mlb", "nhl", "ncaa", "super bowl", "world series", "championship", "playoff",
        "finals", "mvp", "season", "game", "match", "soccer", "football", "basketball", "baseball",
        "hockey", "tennis", "golf", "ufc", "boxing", "olympic", "world cup", "premier league", "team",
        "coach", "player", "draft", "trade deadline", "win the", "defeat", "score",
    ],
    "other": [
        "oscar", "emmy", "grammy", "tony", "golden globe", "box office", "movie", "film", "tv show",
        "streaming", "album", "song", "chart", "billboard", "book", "bestseller", "nobel prize",
        "pulitzer", "social media", "tiktok", "youtube", "viral", "celebrity", "award", "entertainment",
        "concert", "tour", "festival", "game of the year",
        "temperature", "weather", "rainfall", "humidity", "forecast", "celsius", "fahrenheit",
    ],
}

PRIORITY = ["politics", "economics", "sports", "other"]


def ensure_directories() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def db_cursor() -> Iterable[sqlite3.Connection]:
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def parse_json_field(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_timestamp(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unix_to_iso8601(unix_seconds: int | float | str) -> str:
    timestamp = datetime.fromtimestamp(float(unix_seconds), tz=timezone.utc)
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso8601_to_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def shift_timestamp(value: str, hours: int) -> str:
    shifted = iso8601_to_datetime(value) - timedelta(hours=hours)
    return shifted.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hours_between(later: str, earlier: str) -> float:
    delta = iso8601_to_datetime(later) - iso8601_to_datetime(earlier)
    return abs(delta.total_seconds()) / 3600.0


def clamp_probability(probability: float, lower: float = 0.0001, upper: float = 0.9999) -> float:
    return min(max(probability, lower), upper)


def compute_log_loss(probability: float, outcome_binary: int) -> float:
    p = clamp_probability(probability)
    return -math.log(p) if outcome_binary == 1 else -math.log(1.0 - p)


def normalize_question(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def assign_genre_from_text(text: str) -> str:
    lowered = (text or "").lower()
    scores: dict[str, int] = {genre: 0 for genre in PRIORITY}
    for genre, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lowered:
                scores[genre] += 1
    max_score = max(scores.values())
    if max_score == 0:
        return "other"
    # Tiebreak using PRIORITY order
    for genre in PRIORITY:
        if scores[genre] == max_score:
            return genre
    return "other"


def assign_genre_from_category(category: str | None) -> str | None:
    normalized = (category or "").strip().lower()
    if not normalized:
        return None
    if normalized in CATEGORY_MAP:
        return CATEGORY_MAP[normalized]
    for key, genre in CATEGORY_MAP.items():
        if key in normalized:
            return genre
    return None


def list_table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row["name"] for row in rows]


def print_table_summary(connection: sqlite3.Connection, tables: list[str]) -> None:
    for table_name in tables:
        columns = list_table_columns(connection, table_name)
        print(f"{table_name}: {', '.join(columns)}")

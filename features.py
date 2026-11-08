"""Feature engineering for the Netflix Movie-vs-TV-Show classifier.

Target leakage note: the raw `duration` column ("90 min" / "2 Seasons") makes the
task trivial (it literally encodes the label), so it is NOT used. Genre strings that
contain the words "Movies" / "TV Shows" are reduced to content themes instead.
"""
import re
import numpy as np
import pandas as pd

TARGET = "target"  # 1 = TV Show, 0 = Movie

THEMES = {
    "theme_drama": r"drama",
    "theme_comedy": r"comed",
    "theme_action": r"action",
    "theme_documentary": r"docu",
    "theme_kids": r"kids|children|family",
    "theme_romance": r"romant",
    "theme_thriller_horror": r"thriller|horror|mystery",
    "theme_scifi_fantasy": r"sci-fi|fantasy|supernatural",
    "theme_crime": r"crime",
    "theme_reality": r"reality",
    "theme_anime": r"anime",
    "theme_music": r"music|concert",
    "theme_standup": r"stand-up",
    "theme_international": r"international",
}

FEATURE_COLUMNS = [
    "release_year", "year_added", "month_added", "added_lag_years",
    "n_countries", "n_genres", "n_cast", "n_directors",
    "title_len", "title_words", "desc_len", "desc_words",
    "has_director", "has_cast", "has_country", "has_date_added",
    *THEMES.keys(),
    "rating_adult", "rating_teen", "rating_family", "rating_kids",
    "rating_not_rated", "rating_tv_style",
]


def _count_items(s):
    return s.fillna("").apply(lambda x: 0 if x.strip() == "" else len([i for i in x.split(",") if i.strip()]))


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # three rows have a duration string in the rating column (data-entry error)
    df["rating"] = df["rating"].where(~df["rating"].fillna("").str.contains("min"), np.nan)

    added = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["release_year"] = df["release_year"]
    out["year_added"] = added.dt.year
    out["month_added"] = added.dt.month
    out["added_lag_years"] = out["year_added"] - out["release_year"]

    out["n_countries"] = _count_items(df["country"])
    out["n_genres"] = _count_items(df["listed_in"])
    out["n_cast"] = _count_items(df["cast"])
    out["n_directors"] = _count_items(df["director"])

    out["title_len"] = df["title"].str.len()
    out["title_words"] = df["title"].str.split().str.len()
    out["desc_len"] = df["description"].fillna("").str.len()
    out["desc_words"] = df["description"].fillna("").str.split().str.len()

    out["has_director"] = df["director"].notna().astype(int)
    out["has_cast"] = df["cast"].notna().astype(int)
    out["has_country"] = df["country"].notna().astype(int)
    out["has_date_added"] = added.notna().astype(int)

    genres = df["listed_in"].fillna("").str.lower()
    for name, pat in THEMES.items():
        out[name] = genres.str.contains(pat, regex=True).astype(int)

    r = df["rating"].fillna("")
    out["rating_adult"] = r.isin(["TV-MA", "R", "NC-17"]).astype(int)
    out["rating_teen"] = r.isin(["TV-14", "PG-13"]).astype(int)
    out["rating_family"] = r.isin(["TV-PG", "PG"]).astype(int)
    out["rating_kids"] = r.isin(["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"]).astype(int)
    out["rating_not_rated"] = r.isin(["NR", "UR", ""]).astype(int)
    out["rating_tv_style"] = r.str.startswith("TV-").astype(int)

    out = out[FEATURE_COLUMNS]
    out[TARGET] = (df["type"] == "TV Show").astype(int)
    return out

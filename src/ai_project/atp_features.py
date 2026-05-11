"""Shared ATP preprocessing and feature engineering helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATE_COL = "Date"
TARGET_COL = "Favorite_Wins"
REQUIRED_COLUMNS = ["Rank_1", "Rank_2", "Player_1", "Player_2", "Winner", "Surface"]
FEATURE_COLUMNS = ["Rank_Diff", "Surface_Encoded", "Favorite_Form", "Underdog_Form"]
FORM_WINDOW_DEFAULT = 10
FORM_DEFAULT_RATE = 0.5


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {available}"
        )


def filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    ensure_columns(df, [DATE_COL])
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df[(df[DATE_COL].dt.year >= start_year) & (df[DATE_COL].dt.year <= end_year)]
    if df.empty:
        raise ValueError(
            f"No rows found between {start_year} and {end_year}. "
            "Check the dataset or adjust the year filter."
        )
    return df


def validate_and_clean(
    df: pd.DataFrame,
    required: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    required = required or REQUIRED_COLUMNS
    ensure_columns(df, required)
    missing_counts = df[required].isna().sum()
    missing_rows = int(df[required].isna().any(axis=1).sum())
    duplicate_rows = int(df.duplicated().sum())

    summary = {
        "missing_counts": missing_counts,
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
    }

    df = df.drop_duplicates()
    df = df.dropna(subset=required)
    return df, summary


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep form features chronological while preserving source-row tie breaks."""
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    return df.sort_values([DATE_COL, "_row_id"]).reset_index(drop=True)


def build_surface_tools(
    df: pd.DataFrame,
) -> tuple[LabelEncoder, dict[str, float] | None]:
    surface = df["Surface"].fillna(df["Surface"].mode(dropna=True)[0])
    encoder = LabelEncoder()
    encoder.fit(surface)

    mapping: dict[str, float] | None = None
    if "Surface_Encoded" in df.columns:
        encoded = pd.to_numeric(df["Surface_Encoded"], errors="coerce")
        mapping = (
            pd.DataFrame({"surface": surface, "encoded": encoded})
            .dropna()
            .groupby("surface")["encoded"]
            .agg(lambda s: s.value_counts().idxmax())
            .astype(float)
            .to_dict()
        )
        if not mapping:
            mapping = None
    return encoder, mapping


def encode_surface(
    surface_name: str,
    *,
    surface_encoder: LabelEncoder,
    surface_mapping: dict[str, float] | None,
) -> float:
    if surface_mapping and surface_name in surface_mapping:
        return float(surface_mapping[surface_name])
    return float(surface_encoder.transform([surface_name])[0])


def build_player_stats(
    df: pd.DataFrame,
    *,
    form_window: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Build latest rank/form lookup tables for the prediction form."""
    df = sort_matches(df)
    history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=form_window))
    latest_rank: dict[str, float] = {}
    players: set[str] = set()

    for _, row in df.iterrows():
        player_1 = str(row["Player_1"])
        player_2 = str(row["Player_2"])
        winner = str(row["Winner"])
        players.update([player_1, player_2])

        rank_1 = pd.to_numeric(row["Rank_1"], errors="coerce")
        rank_2 = pd.to_numeric(row["Rank_2"], errors="coerce")
        if pd.notna(rank_1):
            latest_rank[player_1] = float(rank_1)
        if pd.notna(rank_2):
            latest_rank[player_2] = float(rank_2)

        if winner == player_1:
            history[player_1].append(1)
            history[player_2].append(0)
        elif winner == player_2:
            history[player_2].append(1)
            history[player_1].append(0)

    form_rate: dict[str, float] = {}
    for player in players:
        hist = history[player]
        form_rate[player] = (
            FORM_DEFAULT_RATE if len(hist) == 0 else sum(hist) / len(hist)
        )

    return sorted(players), latest_rank, form_rate


def compute_form_rates(
    df: pd.DataFrame,
    *,
    window: int,
    default_rate: float = FORM_DEFAULT_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling win rates using only matches played before each row."""
    history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))
    form_p1 = np.zeros(len(df), dtype=np.float32)
    form_p2 = np.zeros(len(df), dtype=np.float32)

    for idx, row in df.iterrows():
        player_1 = str(row["Player_1"])
        player_2 = str(row["Player_2"])
        winner = str(row["Winner"])

        p1_hist = history[player_1]
        p2_hist = history[player_2]

        form_p1[idx] = default_rate if len(p1_hist) == 0 else sum(p1_hist) / len(p1_hist)
        form_p2[idx] = default_rate if len(p2_hist) == 0 else sum(p2_hist) / len(p2_hist)

        if winner == player_1:
            p1_hist.append(1)
            p2_hist.append(0)
        elif winner == player_2:
            p2_hist.append(1)
            p1_hist.append(0)

    return form_p1, form_p2


def build_features(
    df: pd.DataFrame,
    *,
    form_window: int,
    surface_encoder: LabelEncoder | None = None,
    surface_mapping: dict[str, float] | None = None,
    target_col: str | None = TARGET_COL,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create the exact feature matrix expected by the saved scaler/model."""
    ensure_columns(df, REQUIRED_COLUMNS)
    df = df.dropna(subset=REQUIRED_COLUMNS)

    rank_1 = pd.to_numeric(df["Rank_1"], errors="coerce")
    rank_2 = pd.to_numeric(df["Rank_2"], errors="coerce")
    valid = rank_1.notna() & rank_2.notna() & (rank_1 != rank_2)
    df = df[valid].copy()
    if df.empty:
        raise ValueError("No rows with valid ranks after filtering.")

    df = sort_matches(df)

    rank_1 = pd.to_numeric(df["Rank_1"], errors="coerce").to_numpy()
    rank_2 = pd.to_numeric(df["Rank_2"], errors="coerce").to_numpy()
    favorite_is_p1 = rank_1 < rank_2

    favorite_rank = np.where(favorite_is_p1, rank_1, rank_2)
    underdog_rank = np.where(favorite_is_p1, rank_2, rank_1)
    rank_diff = favorite_rank - underdog_rank

    player_1 = df["Player_1"].astype(str).to_numpy()
    player_2 = df["Player_2"].astype(str).to_numpy()
    winner = df["Winner"].astype(str).to_numpy()
    favorite_player = np.where(favorite_is_p1, player_1, player_2)

    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        y = (winner == favorite_player).astype(int)

    if surface_encoder is None:
        surface_encoder, surface_mapping = build_surface_tools(df)

    surface = df["Surface"].fillna(df["Surface"].mode(dropna=True)[0])
    surface_text = surface_encoder.transform(surface).astype(float)
    if "Surface_Encoded" in df.columns:
        surface_encoded = pd.to_numeric(df["Surface_Encoded"], errors="coerce").to_numpy()
        missing_mask = np.isnan(surface_encoded)
        if np.any(missing_mask):
            surface_encoded = surface_encoded.copy()
            surface_encoded[missing_mask] = surface_text[missing_mask]
    else:
        surface_encoded = surface_text

    form_p1, form_p2 = compute_form_rates(df, window=form_window)
    favorite_form = np.where(favorite_is_p1, form_p1, form_p2)
    underdog_form = np.where(favorite_is_p1, form_p2, form_p1)

    features = pd.DataFrame(
        {
            "Rank_Diff": rank_diff,
            "Surface_Encoded": surface_encoded,
            "Favorite_Form": favorite_form,
            "Underdog_Form": underdog_form,
        }
    )

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].fillna(features[col].median())

    x = features.to_numpy(dtype=np.float32)
    return x, y, FEATURE_COLUMNS.copy()

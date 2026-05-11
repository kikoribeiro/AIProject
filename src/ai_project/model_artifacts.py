"""Persistence helpers for trained ATP prediction models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import joblib

DEFAULT_MODEL_PATH = "outputs/atp_model.keras"
DEFAULT_SCALER_PATH = "outputs/atp_scaler.joblib"
DEFAULT_METADATA_PATH = "outputs/atp_model_metadata.json"


@dataclass(frozen=True)
class SavedModelArtifacts:
    model_path: str
    scaler_path: str
    metadata_path: str


@dataclass(frozen=True)
class LoadedModelArtifacts:
    model: Any
    scaler: Any
    metadata: dict[str, Any]


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_model_artifacts(
    model: Any,
    scaler: Any,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    metadata: dict[str, Any] | None = None,
) -> SavedModelArtifacts:
    """Save the Keras model, fitted scaler, and metadata needed by Streamlit."""
    for path in (model_path, scaler_path, metadata_path):
        _ensure_parent_dir(path)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata or {}, fh, indent=2, sort_keys=True)

    return SavedModelArtifacts(
        model_path=model_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )


def load_model_artifacts(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    keras_module: Any | None = None,
) -> LoadedModelArtifacts:
    """Load saved artifacts for inference."""
    missing = [
        path
        for path in (model_path, scaler_path, metadata_path)
        if not os.path.exists(path)
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing model artifacts: {missing_list}. Train the model first with "
            "`python atp_mlp_keras.py atp_tennis.csv`, or update the artifact paths."
        )

    if keras_module is None:
        from tensorflow import keras as keras_module

    model = keras_module.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return LoadedModelArtifacts(model=model, scaler=scaler, metadata=metadata)

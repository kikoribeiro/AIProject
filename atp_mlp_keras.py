"""Train a simple MLP to predict if the favorite wins (binary classification)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is required. Install with: pip install tensorflow"
    ) from exc

try:
    from ai_project import atp_features
    from ai_project.model_artifacts import (
        DEFAULT_METADATA_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_SCALER_PATH,
        save_model_artifacts,
    )
    from ai_project.metrics.baselines import favorite_wins_baseline
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from ai_project import atp_features
    from ai_project.model_artifacts import (
        DEFAULT_METADATA_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_SCALER_PATH,
        save_model_artifacts,
    )
    from ai_project.metrics.baselines import favorite_wins_baseline

RANDOM_SEED = 42
DATA_PATH = "atp_tennis.csv"
START_YEAR = 2018
END_YEAR = 2026
LABELS = [0, 1]
LABEL_NAMES = ["Upset (0)", "Favorite Wins (1)"]
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_train_val_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
    )
    train_idx, temp_idx = next(sss.split(np.zeros(len(y)), y))

    y_temp = y[temp_idx]
    val_share = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    sss_temp = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1.0 - val_share),
        random_state=RANDOM_SEED,
    )
    val_rel, test_rel = next(sss_temp.split(np.zeros(len(y_temp)), y_temp))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def load_dataset(
    path: str,
    *,
    form_window: int = atp_features.FORM_WINDOW_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load ATP data and build the feature matrix used by both training paths."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found at '{path}'. Update DATA_PATH or pass --data-path."
        )

    df = pd.read_csv(path)
    df = atp_features.filter_years(df, START_YEAR, END_YEAR)
    df, summary = atp_features.validate_and_clean(df)

    missing_counts = summary["missing_counts"]
    if summary["missing_rows"] > 0:
        print("Missing values in required columns:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"- {col}: {int(count)}")
    if summary["duplicate_rows"] > 0:
        print(f"Duplicate rows detected: {summary['duplicate_rows']}")

    return atp_features.build_features(df, form_window=form_window)


def _plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cm_array = np.asarray(cm)
    max_value = float(np.max(cm_array)) if cm_array.size else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color="white" if cm[i][j] > (max_value * 0.5) else "black",
            )

    fig.tight_layout()
    return fig


def export_confusion_matrix_pdf(cm: list[list[int]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig = _plot_confusion_matrix(cm, LABEL_NAMES, title="Confusion Matrix")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(12, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_final_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[keras.Model, StandardScaler]:
    """Train one final model on all prepared rows for Streamlit inference."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = build_model(input_dim=x_scaled.shape[1])
    model.fit(
        x_scaled,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return model, scaler


def train_with_stratified_kfold(
    x: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    export_pdf: str | None = None,
) -> None:
    """Run cross-validation for report metrics without reusing fold scalers."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_true: list[int] = []
    all_pred: list[int] = []
    weight_importances: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_model(input_dim=x_train.shape[1])
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        y_prob = model.predict(x_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        first_layer_weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        weight_importances.append(importance)

        print(f"Fold {fold_idx} confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=LABELS))

    cm_all = confusion_matrix(all_true, all_pred, labels=LABELS)
    print("\nConfusion matrix (all folds):")
    print(cm_all)
    print("\nClassification report (all folds):")
    print(
        classification_report(
            all_true,
            all_pred,
            labels=LABELS,
            target_names=LABEL_NAMES,
            digits=3,
        )
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        average="binary",
        zero_division=0,
    )
    f1_manual = 0.0
    if precision + recall > 0:
        f1_manual = 2.0 * (precision * recall) / (precision + recall)
    print(f"Manual F1: {f1_manual:.4f}")

    baseline = favorite_wins_baseline(all_true, labels=LABELS)
    baseline_metrics = baseline["metrics"]
    baseline_positive = next(m for m in baseline_metrics["per_class"] if m.label == 1)
    model_accuracy = float(np.mean(np.asarray(all_true) == np.asarray(all_pred)))

    print("\nBaseline comparison:")
    print(f"{baseline['name']} confusion matrix:")
    print(np.asarray(baseline["confusion_matrix"]))
    print(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline F1 (favorite wins): {baseline_positive.f1:.4f}")
    print(f"MLP accuracy lift vs baseline: {model_accuracy - baseline_metrics['accuracy']:+.4f}")
    print(f"MLP F1 lift vs baseline: {f1 - baseline_positive.f1:+.4f}")

    avg_importance = np.mean(weight_importances, axis=0)
    ranked = sorted(zip(feature_cols, avg_importance), key=lambda x: x[1], reverse=True)
    print("\nFeature importance proxy (avg abs first-layer weights):")
    for name, score in ranked:
        print(f"{name}: {score:.4f}")

    if export_pdf:
        export_confusion_matrix_pdf(cm_all, export_pdf)
        print(f"Saved confusion matrix PDF to: {export_pdf}")


def main() -> None:
    set_seeds()

    parser = argparse.ArgumentParser(
        description="Train an MLP to predict ATP favorite wins (2018-2026)."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=DATA_PATH,
        help="Path to the ATP CSV file.",
    )
    parser.add_argument(
        "--export-pdf",
        dest="export_pdf",
        default=None,
        help="Optional path to save the confusion matrix PDF.",
    )
    parser.add_argument(
        "--form-window",
        type=int,
        default=atp_features.FORM_WINDOW_DEFAULT,
        help="Number of previous matches to compute form (default: 10).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for each fold and final saved model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for each fold and final saved model.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Where to save the final Keras model for Streamlit.",
    )
    parser.add_argument(
        "--scaler-path",
        default=DEFAULT_SCALER_PATH,
        help="Where to save the fitted StandardScaler for Streamlit.",
    )
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
        help="Where to save model metadata for Streamlit.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving final model artifacts after cross-validation.",
    )
    args = parser.parse_args()

    x, y, feature_cols = load_dataset(args.data_path, form_window=args.form_window)
    train_idx, val_idx, test_idx = split_train_val_test(y)
    print(
        f"Split sizes -> train: {len(train_idx)} ({len(train_idx) / len(y):.2%}), "
        f"val: {len(val_idx)} ({len(val_idx) / len(y):.2%}), "
        f"test: {len(test_idx)} ({len(test_idx) / len(y):.2%})"
    )
    train_with_stratified_kfold(
        x,
        y,
        feature_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        export_pdf=args.export_pdf,
    )

    if not args.no_save_model:
        final_model, final_scaler = train_final_model(
            x,
            y,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        artifacts = save_model_artifacts(
            final_model,
            final_scaler,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            metadata_path=args.metadata_path,
            metadata={
                "data_path": args.data_path,
                "feature_cols": feature_cols,
                "form_window": args.form_window,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "start_year": START_YEAR,
                "end_year": END_YEAR,
                "target": atp_features.TARGET_COL,
            },
        )
        print(f"\nSaved model to: {artifacts.model_path}")
        print(f"Saved scaler to: {artifacts.scaler_path}")
        print(f"Saved metadata to: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()

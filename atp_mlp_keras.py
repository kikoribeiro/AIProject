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
    log_loss,
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
    from ai_project.metrics.plots import CONFUSION_MATRIX_CMAP, CONFUSION_MATRIX_TEXT_COLOR
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
    from ai_project.metrics.plots import CONFUSION_MATRIX_CMAP, CONFUSION_MATRIX_TEXT_COLOR

RANDOM_SEED = 42
DATA_PATH = "atp_tennis.csv"
DEFAULT_DECISION_THRESHOLD = 0.60
LABELS = [0, 1]
LABEL_NAMES = ["Upset (0)", "Favorite Wins (1)"]
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
DEFAULT_OVERFITTING_GRAPH_PATH = "outputs/overfitting_check.png"
OVERFITTING_ACCURACY_GAP_THRESHOLD = 0.10
OVERFITTING_LOSS_GAP_THRESHOLD = 0.15


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_train_val_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a balanced 60/20/20 split for reporting dataset sizes."""
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

    # Training uses the full CSV; cleaning and feature creation live in shared helpers.
    df = pd.read_csv(path)
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
    im = ax.imshow(cm, interpolation="nearest", cmap=CONFUSION_MATRIX_CMAP)
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

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color=CONFUSION_MATRIX_TEXT_COLOR,
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
    """Small MLP: enough for a course demo without hiding the feature logic."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(12, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(8, activation="relu", kernel_initializer="he_normal"),
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


def _last_history_value(history: dict[str, list[float]], key: str) -> float:
    values = history.get(key, [])
    if not values:
        return float("nan")
    return float(values[-1])


def _accuracy_from_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_pred = (y_prob >= DEFAULT_DECISION_THRESHOLD).astype(int)
    return float(np.mean(y_true == y_pred))


def _plot_training_history(
    history: dict[str, list[float]],
    output_path: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    losses = history.get("loss", [])
    epochs = range(1, len(losses) + 1)

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4))
    loss_ax.plot(epochs, losses, marker="o", label="Train loss")
    if "val_loss" in history:
        loss_ax.plot(
            epochs,
            history["val_loss"],
            marker="o",
            label="Validation loss",
        )
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Binary crossentropy")
    loss_ax.legend()

    accuracy = history.get("accuracy", [])
    accuracy_epochs = range(1, len(accuracy) + 1)
    accuracy_ax.plot(
        accuracy_epochs,
        accuracy,
        marker="o",
        label="Train accuracy",
    )
    if "val_accuracy" in history:
        accuracy_ax.plot(
            accuracy_epochs,
            history["val_accuracy"],
            marker="o",
            label="Validation accuracy",
        )
    accuracy_ax.set_title("Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


def run_overfitting_check(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    graph_path: str = DEFAULT_OVERFITTING_GRAPH_PATH,
) -> dict[str, float | str | bool]:
    """Train a diagnostic holdout model and save train-vs-validation curves."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_val = scaler.transform(x[val_idx])
    x_test = scaler.transform(x[test_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    model = build_model(input_dim=x_train.shape[1])
    fit_result = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    history = getattr(fit_result, "history", {})
    _plot_training_history(history, graph_path)

    train_accuracy = _last_history_value(history, "accuracy")
    val_accuracy = _last_history_value(history, "val_accuracy")
    train_loss = _last_history_value(history, "loss")
    val_loss = _last_history_value(history, "val_loss")
    accuracy_gap = train_accuracy - val_accuracy
    loss_gap = val_loss - train_loss

    test_prob = model.predict(x_test, verbose=0).ravel()
    test_accuracy = _accuracy_from_probabilities(y_test, test_prob)
    test_log_loss = float(log_loss(y_test, test_prob, labels=LABELS))

    possible_overfitting = bool(
        accuracy_gap > OVERFITTING_ACCURACY_GAP_THRESHOLD
        or loss_gap > OVERFITTING_LOSS_GAP_THRESHOLD
    )

    print("\nOverfitting check:")
    print(f"Final train accuracy: {train_accuracy:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Train/validation accuracy gap: {accuracy_gap:+.4f}")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Validation/train loss gap: {loss_gap:+.4f}")
    print(f"Holdout test accuracy: {test_accuracy:.4f}")
    print(f"Holdout test log loss: {test_log_loss:.4f}")
    if possible_overfitting:
        print(
            "Possible overfitting: training performance is noticeably better "
            "than validation performance."
        )
    else:
        print("No strong overfitting signal from the train/validation gap.")
    print(f"Saved overfitting graph to: {graph_path}")

    return {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "accuracy_gap": accuracy_gap,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "loss_gap": loss_gap,
        "test_accuracy": test_accuracy,
        "test_log_loss": test_log_loss,
        "possible_overfitting": possible_overfitting,
        "graph_path": graph_path,
    }


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
    all_prob: list[float] = []
    weight_importances: list[np.ndarray] = []

    print(f"Decision threshold: {DEFAULT_DECISION_THRESHOLD:.2f}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        # Fit a new scaler per fold so test rows never influence training scaling.
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
        y_pred = (y_prob >= DEFAULT_DECISION_THRESHOLD).astype(int)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(y_prob.tolist())

        first_layer_weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        weight_importances.append(importance)

        print(f"Fold {fold_idx} confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=LABELS))

    cm_all = confusion_matrix(all_true, all_pred, labels=LABELS)
    print("\nConfusion matrix (all folds):")
    print(cm_all)
    print(f"\nLog loss (all folds): {log_loss(all_true, all_prob, labels=LABELS):.4f}")
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

    # Main reality check: the MLP should beat "favorite always wins" to be useful.
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
        description="Train an MLP to predict ATP favorite wins from the full CSV."
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
        help=f"Number of previous matches to compute form (default: {atp_features.FORM_WINDOW_DEFAULT}).",
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
        "--overfitting-graph",
        default=DEFAULT_OVERFITTING_GRAPH_PATH,
        help="Path to save the train-vs-validation overfitting check PNG.",
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
        # Streamlit must load both artifacts because the model expects scaled features.
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
                "data_scope": "full_csv",
                "target": atp_features.TARGET_COL,
            },
        )
        print(f"\nSaved model to: {artifacts.model_path}")
        print(f"Saved scaler to: {artifacts.scaler_path}")
        print(f"Saved metadata to: {artifacts.metadata_path}")

    run_overfitting_check(
        x,
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        graph_path=args.overfitting_graph,
    )


if __name__ == "__main__":
    main()

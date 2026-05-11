"""Streamlit app for evaluating and using a saved ATP favorite-wins model."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import io
import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from ai_project import atp_features
from ai_project.model_artifacts import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    load_model_artifacts,
)
from ai_project.metrics.baselines import favorite_wins_baseline
from ai_project.metrics.classification_metrics import classification_metrics
from ai_project.metrics.confusion_matrix import confusion_matrix, normalize_confusion_matrix
from ai_project.metrics.plots import plot_confusion_matrix

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

DEFAULT_PATH = "atp_tennis.csv"
LABELS = [0, 1]
LABEL_NAMES = ["Upset (0)", "Favorite Wins (1)"]


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_loaded_model(
    model: Any,
    scaler: StandardScaler,
    x: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    feature_count: int,
) -> tuple[list[list[int]], dict, np.ndarray]:
    """Evaluate the saved model on the currently filtered dataset."""
    x_scaled = scaler.transform(x)
    y_prob = model.predict(x_scaled, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y.tolist(), y_pred.tolist(), labels=LABELS)
    metrics = classification_metrics(cm, LABELS)

    try:
        first_layer_weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(first_layer_weights), axis=1)
    except (AttributeError, IndexError, ValueError):
        importance = np.zeros(feature_count)

    return cm, metrics, importance


def main() -> None:
    st.set_page_config(page_title="ATP Tennis MLP (2018-2026)", layout="wide")
    st.title("ATP Tennis MLP - Favorite Wins (2018-2026)")
    st.write(
        "Load a saved Keras MLP to predict if the favorite player wins. "
        "Train the model first in Jupyter or with the Python script, then use it here "
        "for evaluation and player-vs-player predictions."
    )

    if not TF_AVAILABLE:
        st.error("TensorFlow is required. Install with: pip install tensorflow")
        st.stop()

    with st.sidebar:
        st.header("Data")
        data_path = st.text_input("CSV path", value=DEFAULT_PATH)
        start_year = st.number_input("Start year", value=2018, step=1)
        end_year = st.number_input("End year", value=2026, step=1)
        show_preview = st.checkbox("Show data preview", value=False)

        st.header("Saved model")
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        scaler_path = st.text_input("Scaler path", value=DEFAULT_SCALER_PATH)
        metadata_path = st.text_input("Metadata path", value=DEFAULT_METADATA_PATH)

        st.header("Evaluation")
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
        form_window = st.slider(
            "Form window (previous matches)",
            3,
            30,
            atp_features.FORM_WINDOW_DEFAULT,
            1,
        )
        normalize_mode = st.selectbox(
            "CM normalization",
            options=["none", "true", "pred"],
            help=(
                "none = raw counts; true = row-normalized (recall view); "
                "pred = column-normalized (precision view)."
            ),
        )
        load_saved_model = st.button("Load saved model")

    if not os.path.exists(data_path):
        st.error(f"CSV not found at '{data_path}'. Update the path in the sidebar.")
        st.stop()

    df = load_csv(data_path)

    try:
        df_filtered = atp_features.filter_years(df, int(start_year), int(end_year))
        df_filtered, validation_summary = atp_features.validate_and_clean(df_filtered)
        surface_encoder, surface_mapping = atp_features.build_surface_tools(df_filtered)
        player_list, latest_rank, player_form = atp_features.build_player_stats(
            df_filtered,
            form_window=int(form_window),
        )
        x, y, feature_cols = atp_features.build_features(
            df_filtered,
            form_window=int(form_window),
            surface_encoder=surface_encoder,
            surface_mapping=surface_mapping,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    favorite_rate = float(np.mean(y)) if len(y) > 0 else 0.0
    baseline = favorite_wins_baseline(y.tolist(), labels=LABELS)
    st.write(f"Rows after filtering: {len(df_filtered):,}")
    st.write(f"Favorite win rate: {favorite_rate:.3f}")

    st.subheader("Data validation")
    if validation_summary["missing_rows"] == 0:
        st.write("Missing rows in required columns: 0")
    else:
        st.write(
            f"Missing rows in required columns: {validation_summary['missing_rows']}"
        )
        missing_counts = validation_summary["missing_counts"]
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            st.dataframe(missing_counts.to_frame("missing_count"))
    st.write(f"Duplicate rows removed: {validation_summary['duplicate_rows']}")

    st.subheader("Evaluation dataset")
    st.write(f"Rows used for saved-model evaluation: {len(y):,}")
    st.write(f"Features: {', '.join(feature_cols)}")

    if show_preview:
        st.dataframe(df_filtered.head(50), use_container_width=True)

    # Any change to data/model inputs invalidates the cached model results.
    artifact_signature = (
        data_path,
        int(start_year),
        int(end_year),
        int(form_window),
        float(threshold),
        model_path,
        scaler_path,
        metadata_path,
    )

    if load_saved_model:
        try:
            with st.spinner("Loading saved model artifacts..."):
                loaded = load_model_artifacts(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    metadata_path=metadata_path,
                )
                cm, metrics, avg_importance = evaluate_loaded_model(
                    loaded.model,
                    loaded.scaler,
                    x,
                    y,
                    threshold=float(threshold),
                    feature_count=len(feature_cols),
                )
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(str(exc))
            return

        st.session_state["artifact_signature"] = artifact_signature
        st.session_state["model_results"] = {
            "cm": cm,
            "metrics": metrics,
            "avg_importance": avg_importance,
            "feature_cols": feature_cols,
            "final_model": loaded.model,
            "final_scaler": loaded.scaler,
            "metadata": loaded.metadata,
        }
    elif st.session_state.get("artifact_signature") != artifact_signature:
        st.info(
            "Train the model in Jupyter or with `python atp_mlp_keras.py atp_tennis.csv`, "
            "then click 'Load saved model'."
        )
        return

    # From here down, the page only renders after a saved model has been loaded.
    results = st.session_state["model_results"]
    cm = results["cm"]
    metrics = results["metrics"]
    avg_importance = results["avg_importance"]
    feature_cols = results["feature_cols"]
    final_model = results["final_model"]
    final_scaler = results["final_scaler"]
    metadata = results["metadata"]

    if metadata:
        trained_form_window = metadata.get("form_window")
        if trained_form_window is not None and int(trained_form_window) != int(form_window):
            st.warning(
                f"This model was saved with form_window={trained_form_window}, "
                f"but the app is evaluating with form_window={form_window}."
            )

    cm_show = normalize_confusion_matrix(cm, mode=normalize_mode)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion matrix")
        fig = plot_confusion_matrix(
            cm_show,
            labels=LABEL_NAMES,
            title=f"Confusion matrix (normalize={normalize_mode})",
        )
        st.pyplot(fig)
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        st.markdown("**How to read this matrix**")
        st.markdown(
            "- Rows are the true outcomes (Upset=0, Favorite Wins=1)."
            " Columns are the model predictions."
        )
        st.markdown(
            f"- TN (true 0 predicted 0): {tn} - correct upsets."
            f" FP (true 0 predicted 1): {fp} - missed upsets."
        )
        st.markdown(
            f"- FN (true 1 predicted 0): {fn} - missed favorites."
            f" TP (true 1 predicted 1): {tp} - correct favorites."
        )
        st.markdown(
            "**Interpretation:** high TP/FN means the model is good at favorites; "
            "high FP means it misses upsets; high TN means it catches upsets."
        )
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
        pdf_buffer.seek(0)
        st.download_button(
            "Download confusion matrix (PDF)",
            data=pdf_buffer,
            file_name="confusion_matrix.pdf",
            mime="application/pdf",
        )

    with col2:
        st.subheader("Summary metrics")
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        positive_metrics = next(m for m in metrics["per_class"] if m.label == 1)
        baseline_metrics = baseline["metrics"]
        baseline_positive_metrics = next(
            m for m in baseline_metrics["per_class"] if m.label == 1
        )
        st.metric("Precision (favorite wins)", f"{positive_metrics.precision:.3f}")
        st.metric("Recall (favorite wins)", f"{positive_metrics.recall:.3f}")
        st.metric("F1 (favorite wins)", f"{positive_metrics.f1:.3f}")

        st.subheader("Baseline comparison")
        comparison_rows = [
            {
                "model": "MLP",
                "accuracy": metrics["accuracy"],
                "precision_favorite": positive_metrics.precision,
                "recall_favorite": positive_metrics.recall,
                "f1_favorite": positive_metrics.f1,
            },
            {
                "model": baseline["name"],
                "accuracy": baseline_metrics["accuracy"],
                "precision_favorite": baseline_positive_metrics.precision,
                "recall_favorite": baseline_positive_metrics.recall,
                "f1_favorite": baseline_positive_metrics.f1,
            },
        ]
        st.dataframe(comparison_rows, use_container_width=True)
        st.caption(
            "The baseline predicts Favorite Wins (1) for every match. "
            "The MLP should beat this to show it learned more than the obvious shortcut."
        )

        st.write("Macro avg:")
        st.json(metrics["macro_avg"])
        st.write("Weighted avg:")
        st.json(metrics["weighted_avg"])

    st.subheader("Per-class metrics")
    per_class_rows = []
    for m in metrics["per_class"]:
        row = asdict(m)
        row["class_name"] = LABEL_NAMES[m.label]
        per_class_rows.append(row)
    st.dataframe(per_class_rows, use_container_width=True)

    st.subheader("Feature importance (weight magnitude)")
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": avg_importance}
    ).sort_values("importance", ascending=False)
    st.dataframe(importance_df, use_container_width=True)
    st.bar_chart(importance_df.set_index("feature"))

    st.subheader("Player vs Player prediction")
    if len(player_list) < 2:
        st.info("Not enough players in the filtered data to run a matchup prediction.")
    else:
        surface_options = list(surface_encoder.classes_)
        with st.form("player_prediction"):
            player_a = st.selectbox("Player A", player_list, index=0, key="player_a")
            default_index = 1 if len(player_list) > 1 else 0
            player_b = st.selectbox("Player B", player_list, index=default_index, key="player_b")
            surface_choice = st.selectbox("Surface", surface_options, key="surface_choice")
            predict_match = st.form_submit_button("Predict match")

        if predict_match:
            if player_a == player_b:
                st.warning("Choose two different players.")
            else:
                rank_a = latest_rank.get(player_a)
                rank_b = latest_rank.get(player_b)
                if rank_a is None or rank_b is None:
                    st.error("Missing rank data for one or both players.")
                else:
                    favorite_is_a = rank_a < rank_b
                    favorite_player = player_a if favorite_is_a else player_b
                    underdog_player = player_b if favorite_is_a else player_a

                    favorite_rank = min(rank_a, rank_b)
                    underdog_rank = max(rank_a, rank_b)
                    rank_diff = favorite_rank - underdog_rank

                    favorite_form = player_form.get(
                        favorite_player,
                        atp_features.FORM_DEFAULT_RATE,
                    )
                    underdog_form = player_form.get(
                        underdog_player,
                        atp_features.FORM_DEFAULT_RATE,
                    )

                    surface_value = atp_features.encode_surface(
                        surface_choice,
                        surface_encoder=surface_encoder,
                        surface_mapping=surface_mapping,
                    )

                    # The saved scaler expects the same four-feature order used in training.
                    match_features = np.array(
                        [[rank_diff, surface_value, favorite_form, underdog_form]],
                        dtype=np.float32,
                    )
                    match_scaled = final_scaler.transform(match_features)
                    prob_favorite = float(final_model.predict(match_scaled, verbose=0).ravel()[0])

                    if favorite_is_a:
                        prob_a = prob_favorite
                        prob_b = 1.0 - prob_favorite
                    else:
                        prob_b = prob_favorite
                        prob_a = 1.0 - prob_favorite

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(f"{player_a} win probability", f"{prob_a:.3f}")
                    with col_b:
                        st.metric(f"{player_b} win probability", f"{prob_b:.3f}")

                    prediction = (
                        "Favorite wins" if prob_favorite >= float(threshold) else "Upset"
                    )
                    st.write(
                        f"Favorite (lower rank): {favorite_player} ({favorite_rank:.0f})"
                    )
                    st.write(
                        f"Underdog: {underdog_player} ({underdog_rank:.0f})"
                    )
                    st.write(f"Model prediction at threshold {threshold:.2f}: {prediction}")

    with st.expander("Saved model metadata"):
        st.json(metadata)


if __name__ == "__main__":
    main()

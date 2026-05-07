"""Streamlit app: explore confusion matrix + metrics on the Iris dataset."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ai_project.metrics.classification_metrics import classification_metrics
from ai_project.metrics.confusion_matrix import confusion_matrix, normalize_confusion_matrix
from ai_project.metrics.plots import plot_confusion_matrix


def main() -> None:
    st.set_page_config(page_title="Confusion Matrix Explorer (Iris)", layout="wide")

    st.title("Confusion Matrix Explorer — Iris dataset")
    st.write(
        "This app trains a simple classifier on the Iris dataset and shows the confusion matrix "
        "and metrics derived from it (accuracy, precision, recall, F1)."
    )

    with st.sidebar:
        st.header("Settings")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        seed = st.number_input("Random seed", value=42, step=1)
        normalize_mode = st.selectbox(
            "CM normalization",
            options=["none", "true", "pred"],
            help=(
                "none = raw counts; true = row-normalized (recall view); "
                "pred = column-normalized (precision view)."
            ),
        )
        c_value = st.slider("LogisticRegression C (regularization)", 0.01, 10.0, 1.0)

    iris = load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    model = LogisticRegression(max_iter=500, C=float(c_value))
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = list(sorted(set(y)))
    cm = confusion_matrix(list(y_test), list(y_pred), labels=labels)
    cm_show = normalize_confusion_matrix(cm, mode=normalize_mode)

    metrics = classification_metrics(cm, labels)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion matrix")
        fig = plot_confusion_matrix(
            cm_show,
            labels=[str(iris.target_names[i]) for i in labels],
            title=f"Confusion matrix (normalize={normalize_mode})",
        )
        st.pyplot(fig)

    with col2:
        st.subheader("Summary metrics")
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.write("Macro avg (unweighted across classes):")
        st.json(metrics["macro_avg"])
        st.write("Weighted avg (weighted by class support):")
        st.json(metrics["weighted_avg"])

    st.subheader("Per-class metrics")
    per_class_rows = []
    for m in metrics["per_class"]:
        row = asdict(m)
        row["class_name"] = iris.target_names[m.label]
        per_class_rows.append(row)

    st.dataframe(per_class_rows, use_container_width=True)


if __name__ == "__main__":
    main()

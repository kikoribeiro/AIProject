# AIProject — Confusion Matrix + Streamlit (Project Guide)

This repository is a starter base for an **Introduction to AI / Machine Learning** project in **Python**.

This guide proposes a concrete project focus aligned with the course topics you listed:

- **Confusion Matrix (CM)** for classification (and how to adapt the idea for regression)
- Consolidation of **metrics** derived from CM (accuracy, precision, recall, F1, …)
- A small **interactive Streamlit app** to explore predictions, errors and metrics
- An ANN-based example (optional): a small neural network model, evaluated with CM

> Note: you asked to **ignore the provided ZIPs/PDFs**. This repo will be self-contained.

---

## What is Streamlit (and why use it here)?

[Streamlit](https://streamlit.io/) is a Python framework for building **interactive data/ML web apps** with minimal boilerplate.

- You write a normal Python script (e.g. `streamlit run app.py`).
- Widgets (sliders, dropdowns, file upload) are created via Python functions.
- It’s ideal for ML demos: you can show a model’s prediction + confusion matrix + plots in a browser.

In this project, Streamlit will be used to:

- Load a dataset (or generate synthetic data)
- Train or load a model
- Show predictions and a **confusion matrix**
- Compute and explain metrics

---

## Recommended project focus (so you don’t get stuck choosing)

If you’re unsure of the focus, this is a safe, coherent scope:

### **Focus**: Confusion Matrix + Metrics + Explainability (Classification)

1. Implement a reusable CM + metrics module.
2. Use a simple classifier (baseline) and show how metrics change.
3. Build a Streamlit app to explore:
   - the confusion matrix
   - per-class errors
   - threshold effects (binary)

### Optional extensions (only if time allows)

- ANN example (PyTorch): evaluate an ANN with the same CM/metrics utilities.
- “Regression confusion matrix”: explain limitations and implement a **binned confusion matrix** for regression targets.

---

## Deliverables (what you should be able to show)

1. **A documented metrics module**
   - Confusion matrix computation
   - Accuracy/precision/recall/F1
   - Macro/micro/weighted averages (multi-class)
   - Normalized CM (row-normalized recommended)

2. **A Streamlit app**
   - Controls to select dataset/model
   - Visual confusion matrix (matplotlib)
   - Metric cards + short explanations
   - Error inspection: “most confused classes”

3. **A short report section in README**
   - What CM is and what it measures
   - Why accuracy can be misleading
   - What precision/recall trade off means

---

## Proposed repository structure

```
AIProject/
  README.md
  START_HERE.md
  requirements.txt
  main.py

  src/ai_project/
    __init__.py
    neural_network.py
    torch_model.py

  # New (proposed)
  src/ai_project/metrics/
    confusion_matrix.py
    classification_metrics.py
    plots.py

  apps/
    confusion_matrix_app.py

  docs/
    PROJECT_PLAN.md

  tests/
    test_confusion_matrix.py
    test_metrics.py
```

---

## Plan / guide (step-by-step)

### Step 1 — Define the “dataset + task”
Pick one of these (in increasing complexity):

- **Synthetic 2D classification** (recommended first):
  - easy to visualize
  - quick to train
  - great to demonstrate CM behavior
- Iris dataset (scikit-learn)
- MNIST/Fashion-MNIST (heavier; needs extra deps and training time)

Recommendation: start with **synthetic 2D** + one baseline model.

### Step 2 — Implement confusion matrix from scratch
Implement:

- `confusion_matrix(y_true, y_pred, labels=None)`
- normalization modes:
  - none
  - row-normalized (recall-like)
  - column-normalized (precision-like)

Also implement:

- extract TP/FP/FN/TN (binary)
- per-class TP/FP/FN for multi-class

### Step 3 — Implement metrics derived from CM
Metrics to implement and explain:

- Accuracy
- Precision, Recall
- F1-score
- Macro average vs micro average vs weighted average

### Step 4 — Add “explainability” views
Even without SHAP/LIME, you can add explainability via analysis of errors:

- Top confusions (pairs of classes with most mistakes)
- Per-class recall (which classes the model fails to detect)
- Per-class precision (which predicted classes are often wrong)

### Step 5 — Build the Streamlit app
App sections:

- Dataset selector (synthetic / iris)
- Model selector:
  - baseline: `sklearn.linear_model.LogisticRegression`
  - optional: simple MLP
- Training controls:
  - train/test split
  - random seed
  - threshold (binary)
- Outputs:
  - confusion matrix plot
  - classification report-like metrics table
  - top confusions

### Step 6 — Optional: regression CM adaptation
Explain (in README/docs) that CM is fundamentally for **discrete labels**.

For regression you can:

- **bin** y into intervals (e.g., quantiles or fixed-width bins)
- treat bins as classes
- compute a confusion matrix on those binned labels

Deliverable:

- `binned_confusion_matrix(y_true, y_pred, bins)`
- note tradeoffs: information loss, dependence on bin definition

---

## How you’ll run it (target UX)

### Install
```bash
pip install -r requirements.txt
```

### Run the Streamlit app (to be added)
```bash
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

### Run tests
```bash
PYTHONPATH=src python -m unittest -q
```

---

## Next action I can take in this repo

If you want, I can now **apply the structure above** by committing:

- add Streamlit to `requirements.txt`
- add `docs/PROJECT_PLAN.md` (detailed checklist)
- add a first `apps/confusion_matrix_app.py` skeleton
- add CM + metrics modules + starter tests

Tell me if you prefer:

1) **minimal** (only docs + dependencies), or
2) **full starter implementation** (docs + code + app + tests).

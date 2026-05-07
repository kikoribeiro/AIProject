# AIProject

Starter base for an Introduction to AI project in **Python**.

## What this project is now (recommended focus)

This repo is structured to support a course-aligned mini-project around:

- **Confusion Matrix** computation (from scratch)
- **Classification metrics** derived from the CM (accuracy, precision, recall, F1)
- A small **Streamlit** app to explore those metrics interactively on the **Iris** dataset

Project guide:
- `docs/PROJECT_GUIDE.md`

## Run the Streamlit app

What is Streamlit? It’s a Python framework that turns a script into an interactive web app.

```bash
pip install -r requirements.txt
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

## Included base
- Minimal pure-Python neural network in `src/ai_project/neural_network.py`
- PyTorch neural network in `src/ai_project/torch_model.py`
- Runnable example script in `main.py`
- Streamlit app in `apps/confusion_matrix_app.py`
- Metrics utilities in `src/ai_project/metrics/`
- Starter tests in `tests/`

## Test
```bash
PYTHONPATH=src python -m unittest -q
```

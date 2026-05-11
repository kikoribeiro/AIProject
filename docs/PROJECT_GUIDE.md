# AIProject — ATP Tennis Favorite Wins (Project Guide)

This repository is a starter base for an **Introduction to AI / Machine Learning** project in **Python**.

The project focus is a **binary classifier (ANN/MLP)** that predicts whether the **favorite player wins** an ATP match.
The emphasis is not just accuracy, but **confusion matrix evaluation** and **explainability**.

---

## Objective

- **Target ($y$):** 1 if the favorite (higher-ranked player) wins, 0 if the underdog wins.
- **Model:** MLP (Keras/TensorFlow) with two hidden layers.
- **Evaluation:** confusion matrix + precision/recall/F1.
- **Explainability:** feature-importance proxy from learned weights.

---

## Dataset

Source: Kaggle **"ATP Tennis Dataset (2000–2025)"**.

For this project, use matches from **2018–2026**.

Required columns (present in your file):

- `Date`
- `Rank_1`, `Rank_2`
- `Player_1`, `Player_2`, `Winner`
- `Surface` (or `Surface_Encoded`)

---

## Feature engineering

Define the favorite as the **lower rank number**:

- `Rank_Diff` = rank(favorite) - rank(underdog)
- `Surface_Encoded` = label-encoded surface (Clay/Grass/Hard)
- `Favorite_Form` = rolling win rate of the favorite (previous matches only)
- `Underdog_Form` = rolling win rate of the underdog (previous matches only)

Handle NaNs by filling with the column median.

---

## Model architecture (Keras)

- Input layer: 4 features
- Hidden layer 1: 12 neurons, ReLU
- Hidden layer 2: 8 neurons, ReLU
- Output: 1 neuron, Sigmoid

Loss: `binary_crossentropy`, optimizer: Adam.

---

## Training strategy

- **Stratified K-Fold Cross-Validation** to handle class imbalance.
- Standardize features per fold (fit on train, transform on test).

---

## Evaluation

- Favorite-always-wins baseline comparison
- Confusion matrix (raw + normalized)
- Precision, Recall, and F1:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

The baseline predicts `1` for every match. This gives a simple benchmark for
checking whether the MLP is better than always guessing that the higher-ranked
favorite wins.

---

## Explainability

Report a simple feature-importance proxy:

- Average absolute weight magnitude from the first dense layer

This mirrors the idea of “weight influence” shown in the fw/bw visualization lectures.

---

## Streamlit app

The app at `apps/confusion_matrix_app.py`:

- Loads the ATP CSV
- Filters to 2018–2026
- Loads the saved model/scaler artifacts created by the notebook or CLI
- Displays confusion matrix + metrics + feature importance
- Exports the confusion matrix as a PDF
- Lets you pick two players and predict win probabilities

---

## Setup tutorial (start to finish)

1. **Install Python 3.10+** if needed.
2. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
```

3. **Activate the environment**:

- Windows PowerShell:
  ```bash
  .venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```bash
  .venv\Scripts\activate.bat
  ```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

5. **Download the dataset** from Kaggle and place it in the repo root as `atp_tennis.csv`.
6. **Verify columns** (from your file):

- `Tournament`, `Date`, `Series`, `Court`, `Surface`, `Round`, `Best of`,
  `Player_1`, `Player_2`, `Winner`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`,
  `Odd_1`, `Odd_2`, `Score`.

7. **Train and save the model** (CLI or from a notebook cell):

```bash
python atp_mlp_keras.py atp_tennis.csv
```

This creates the files Streamlit will load:

- `outputs/atp_model.keras`
- `outputs/atp_scaler.joblib`
- `outputs/atp_model_metadata.json`

8. **(Optional) Export confusion matrix to PDF**:

```bash
python atp_mlp_keras.py atp_tennis.csv --export-pdf outputs/confusion_matrix.pdf
```

9. **Run the Streamlit app**:

```bash
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

10. **In the app**, click "Load saved model", use "Download confusion matrix (PDF)",
    and try the **Player vs Player** prediction section.

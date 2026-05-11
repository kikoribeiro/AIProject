# AIProject

ATP tennis binary classification project for **Introduction to AI**. The goal is to predict whether the **favorite player** in a match wins.

The favorite is defined as the player with the lower ATP ranking number. For example, rank 2 is favored over rank 20.

## What The AI Does

This project trains a neural network to answer one question:

```text
Will the favorite player win this ATP match?
```

The model receives four numeric features:

- `Rank_Diff`: favorite rank minus underdog rank.
- `Surface_Encoded`: numeric encoding of the court surface.
- `Favorite_Form`: favorite player's recent win rate before the match.
- `Underdog_Form`: underdog player's recent win rate before the match.

The target is:

- `1`: the favorite wins.
- `0`: the underdog wins.

The AI learns patterns from historical ATP matches between 2018 and 2026. After training, the saved model can be loaded in Streamlit to evaluate performance and make player-vs-player predictions.

## Technologies Used

- **Python**: main programming language.
- **TensorFlow / Keras**: builds and trains the neural network.
- **Pandas**: loads and cleans the ATP CSV dataset.
- **NumPy**: handles numeric arrays and feature calculations.
- **scikit-learn**: provides `StandardScaler`, label encoding, stratified splits, and evaluation helpers.
- **Matplotlib**: creates confusion matrix plots and PDF exports.
- **Streamlit**: provides the interactive web app.
- **joblib**: saves and loads the fitted scaler.
- **Jupyter Notebook**: optional interactive training/exploration workflow.

## Model

The model is a Keras MLP, which means **multi-layer perceptron**. Keras is used because this project trains a neural network, specifically an MLP, or multi-layer perceptron.

Keras is useful here because:

- It makes neural networks easy to define with readable code.
- It works directly with TensorFlow.
- It supports dense layers, ReLU, sigmoid output, binary classification loss, and model saving.
- It lets us save the trained model as .keras and reload it later in Streamlit.
- It is appropriate for a course AI project because the model architecture is clear and explainable.

Architecture:

- Input: 4 features.
- Hidden layer 1: 12 neurons, ReLU activation.
- Hidden layer 2: 8 neurons, ReLU activation.
- Output: 1 neuron, sigmoid activation.

The sigmoid output is a probability from 0 to 1. A value near 1 means the model thinks the favorite is likely to win. A value near 0 means the model thinks an upset is more likely.

## AI Workflow

1. Load `atp_tennis.csv`.
2. Filter matches to 2018-2026.
3. Clean missing and duplicate rows.
4. Sort matches by date.
5. Create rolling form features using only previous matches.
6. Build the final feature matrix.
7. Standardize features with `StandardScaler`.
8. Train the Keras MLP.
9. Evaluate using stratified K-fold cross-validation.
10. Save the final model, scaler, and metadata.
11. Load those saved artifacts in Streamlit.

The training script trains and saves the model. The Streamlit app loads the saved model and uses it.

## Evaluation

The project also compares the MLP against a simple baseline:

- **Favorite always wins baseline**: predicts `1` for every match.

This is useful because favorites win more often than underdogs. If the neural
network does not beat this baseline, then it may not be learning much beyond
the obvious ranking shortcut.

Normalize=true helps answer:

Of all real upsets, how many did the model catch?
Of all real favorite wins, how many did the model catch?
Is the model ignoring upsets and mostly predicting favorite wins?
That is especially useful because favorites probably win more often than underdogs, so raw counts can make the model look better than it really is.

Use them like this:
none: best for showing actual counts.
true: best for judging class-by-class recall. Recommended.
pred: best for checking prediction reliability, like “when the model predicts favorite wins, how often is that correct?”

Accuracy alone can hide problems. For example, if favorites win most matches, a model could predict “favorite wins” almost every time and still look decent. The confusion matrix shows whether the model is actually catching upsets or just mostly predicting the majority class.

From the confusion matrix we calculate:

- **Accuracy**: how many total predictions were correct.
- **Precision**: when the model says “favorite wins,” how often is it right?
- **Recall**: out of all real favorite wins, how many did it catch?
- **F1 score**: balance between precision and recall.

The confusion matrix uses:

- **TN**, **True Negative**: the model predicted upset, and the match was actually an upset.
- **FP**, **False Positive**: the model predicted favorite wins, but the underdog actually won.
- **FN**, **False Negative**: the model predicted upset, but the favorite actually won.
- **TP**, **True Positive**: the model predicted favorite wins, and the favorite actually won.

## Dataset

Dataset source: Kaggle, **ATP Tennis 2000 - 2026 Daily Update**.

Use a CSV named:

```text
atp_tennis.csv
```

Expected columns include:

- `Date`
- `Rank_1`
- `Rank_2`
- `Player_1`
- `Player_2`
- `Winner`
- `Surface`
- optional: `Surface_Encoded`

## Project Structure

```text
AIProject/
  atp_mlp_keras.py                  # CLI training/evaluation script
  atp_run.ipynb                     # optional notebook workflow
  atp_tennis.csv                    # local dataset, not needed in git
  requirements.txt                  # Python dependencies

  apps/
    confusion_matrix_app.py         # Streamlit app that loads the saved model

  src/ai_project/
    atp_features.py                 # shared ATP preprocessing and feature engineering
    model_artifacts.py              # save/load model, scaler, and metadata
    metrics/
      confusion_matrix.py           # confusion matrix implementation
      classification_metrics.py     # accuracy, precision, recall, F1
      plots.py                      # confusion matrix plot

  docs/
    PROJECT_GUIDE.md                # step-by-step project guide

  tests/
    test_atp_features.py
    test_model_artifacts.py
    test_streamlit_app_structure.py
```

## Install

```bash
pip install -r requirements.txt
```

On Windows PowerShell, if you use a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train And Save The Model

Run:

```bash
python atp_mlp_keras.py atp_tennis.csv
```

This trains/evaluates the MLP and saves:

```text
outputs/atp_model.keras
outputs/atp_scaler.joblib
outputs/atp_model_metadata.json
```

Optional PDF export:

```bash
python atp_mlp_keras.py atp_tennis.csv --export-pdf outputs/confusion_matrix.pdf
```

Useful options:

```bash
python atp_mlp_keras.py atp_tennis.csv --epochs 100 --batch-size 64 --form-window 10
```

## Run The Streamlit App

PowerShell:

```bash
$env:PYTHONPATH="src"
streamlit run apps/confusion_matrix_app.py
```

macOS/Linux:

```bash
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

In the app:

1. Confirm the CSV path.
2. Confirm the saved model/scaler/metadata paths.
3. Click **Load saved model**.
4. Review the confusion matrix and metrics.
5. Use **Player vs Player prediction** to estimate matchup probabilities.

## Saved Artifacts

The model cannot be used alone. Streamlit needs all three files:

- `atp_model.keras`: trained neural network.
- `atp_scaler.joblib`: fitted `StandardScaler` used to transform features.
- `atp_model_metadata.json`: training settings such as feature names and form window.

The scaler matters because the model was trained on standardized features. If predictions are made without the same scaler, the model input values will be inconsistent.

## Tests

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

On Windows PowerShell:

```bash
$env:PYTHONPATH="src"
python -m unittest discover -s tests -q
```

## Step-By-Step Guide

For the full walkthrough, see:

```text
docs/PROJECT_GUIDE.md
```

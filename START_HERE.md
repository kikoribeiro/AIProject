# AIProject starter (Python neural network)

This repository now has a simple base to start your AI project.

## What was prepared
- Minimal Python project structure (`src/`, `tests/`, `main.py`)
- Simple neural network implementation from scratch — **no dependencies needed to run this**
- **PyTorch** neural network (`src/ai_project/torch_model.py`) — ready for real data
- Example dataset inspired by sports/F1-style performance features
- Unit tests for both implementations

## Packages included (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| **torch** (PyTorch) | Build and train neural networks using tensors, autograd, and GPU support |
| **numpy** | Fast numerical arrays; essential for data pre-processing |
| **pandas** | Load CSV/Excel datasets, clean data, and inspect features |
| **scikit-learn** | Train/test splits, label encoders, accuracy and classification metrics |
| **matplotlib** | Plot training loss curves, feature correlations, confusion matrices |

Install all at once:

```bash
pip install -r requirements.txt
```

## Quick start (pure Python — no install needed)
```bash
PYTHONPATH=src python main.py
```

## Quick start (PyTorch model)
```python
from ai_project.torch_model import train_model, predict

x_data = [[0.90, 0.90], [0.80, 0.70], [0.20, 0.30], [0.30, 0.20]]
y_data = [1, 1, 0, 0]

model = train_model(x_data, y_data, epochs=50)
print(predict(model, [0.88, 0.82]))  # probability → closer to 1 = likely podium
```

## Run tests
```bash
PYTHONPATH=src python -m unittest -q
```

## Next prep ideas
- Replace sample data with your real topic data (F1 or another sport)
- Load a CSV with `pandas.read_csv()` and feed it to `train_model()`
- Add more features (weather, pit strategy, track type, driver stats…)
- Use `sklearn.model_selection.train_test_split` to get validation accuracy
- Use `sklearn.metrics.classification_report` to see precision/recall
- Plot the training loss with `matplotlib.pyplot.plot()`
- Save/load the trained model with `torch.save()` / `torch.load()`

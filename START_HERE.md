# AIProject starter (Python neural network)

This repository now has a simple base to start your AI project.

## What was prepared
- Minimal Python project structure (`src/`, `tests/`, `main.py`)
- Simple neural network implementation from scratch (no external dependencies)
- Example dataset inspired by sports/F1-style performance features
- Unit test to validate prediction range and basic training flow

## Quick start
From the repository root:

```bash
PYTHONPATH=src python main.py
```

Expected output is a probability between `0.0` and `1.0`.

## Run tests

```bash
PYTHONPATH=src python -m unittest -q
```

## Next prep ideas
- Replace sample data with your real topic data (F1 or another sport)
- Add more features (weather, pit strategy, track type, etc.)
- Split data into train/validation sets
- Track model metrics (accuracy, precision/recall)
- Later, consider moving to PyTorch or TensorFlow

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def import_training_module():
    fake_tf = ModuleType("tensorflow")
    fake_keras = ModuleType("tensorflow.keras")
    fake_layers = ModuleType("tensorflow.keras.layers")

    fake_tf.random = SimpleNamespace(set_seed=lambda seed: None)
    fake_tf.keras = fake_keras
    fake_keras.layers = fake_layers
    fake_keras.Sequential = lambda *args, **kwargs: None
    fake_keras.optimizers = SimpleNamespace(Adam=lambda learning_rate: None)
    fake_layers.Input = lambda *args, **kwargs: None
    fake_layers.Dense = lambda *args, **kwargs: None

    fake_modules = {
        "tensorflow": fake_tf,
        "tensorflow.keras": fake_keras,
        "tensorflow.keras.layers": fake_layers,
    }
    sys.modules.pop("atp_mlp_keras", None)
    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module("atp_mlp_keras")


class TrainingDatasetLoadingTest(unittest.TestCase):
    def test_load_dataset_keeps_all_valid_csv_rows(self) -> None:
        training = import_training_module()
        csv_text = "\n".join(
            [
                "Tournament,Date,Series,Court,Surface,Round,Best of,Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score",
                "Old Open,2017-01-01,ATP250,Outdoor,Hard,1st Round,3,A,B,A,1,10,1000,100,1.2,4.0,6-0 6-0",
                "New Open,2025-01-01,ATP250,Outdoor,Clay,1st Round,3,C,D,D,8,2,200,900,3.0,1.4,6-0 6-0",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "matches.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            _, y, _ = training.load_dataset(str(csv_path))

        self.assertEqual(y.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from sklearn.preprocessing import StandardScaler

from ai_project.model_artifacts import load_model_artifacts, save_model_artifacts


class FakeModel:
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("fake model")


class ModelArtifactsTest(unittest.TestCase):
    def test_save_and_load_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            scaler = StandardScaler()
            metadata = {"feature_cols": ["Rank_Diff"], "form_window": 10}

            artifacts = save_model_artifacts(
                FakeModel(),
                scaler,
                model_path=str(tmp_path / "model.keras"),
                scaler_path=str(tmp_path / "scaler.joblib"),
                metadata_path=str(tmp_path / "metadata.json"),
                metadata=metadata,
            )

            fake_keras = SimpleNamespace(
                models=SimpleNamespace(load_model=lambda path: f"loaded:{path}")
            )
            loaded = load_model_artifacts(
                model_path=artifacts.model_path,
                scaler_path=artifacts.scaler_path,
                metadata_path=artifacts.metadata_path,
                keras_module=fake_keras,
            )

        self.assertEqual(loaded.model, f"loaded:{artifacts.model_path}")
        self.assertIsInstance(loaded.scaler, StandardScaler)
        self.assertEqual(loaded.metadata, metadata)

    def test_missing_artifact_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Train the model first"):
            load_model_artifacts(
                model_path="missing/model.keras",
                scaler_path="missing/scaler.joblib",
                metadata_path="missing/metadata.json",
                keras_module=SimpleNamespace(models=SimpleNamespace(load_model=lambda _: None)),
            )


if __name__ == "__main__":
    unittest.main()

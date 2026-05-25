from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class StreamlitImportPathTest(unittest.TestCase):
    def test_app_imports_without_external_pythonpath(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        site_paths = [
            entry
            for entry in sys.path
            if entry and "site-packages" in Path(entry).parts
        ]

        code = textwrap.dedent(
            f"""
            import json
            import sys
            from pathlib import Path

            repo_root = Path.cwd().resolve()
            src_path = repo_root / "src"
            apps_path = repo_root / "apps"
            site_paths = json.loads({json.dumps(json.dumps(site_paths))})
            sys.path = [
                entry
                for entry in sys.path
                if not entry or Path(entry).resolve() != src_path
            ]
            for site_path in reversed(site_paths):
                sys.path.insert(0, site_path)
            sys.path.insert(0, str(apps_path))

            import confusion_matrix_app

            print(confusion_matrix_app.atp_features.__name__)
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )
        self.assertIn("ai_project.atp_features", result.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import setup_avit_model


class SetupAViTModelTests(unittest.TestCase):
    def test_headless_patch_removes_only_unused_turtle_imports_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            for relative_path in setup_avit_model.UNUSED_GUI_IMPORTS:
                path = source / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "import torch\nfrom turtle import forward\nVALUE = 'kept'\n",
                    encoding="utf-8",
                )

            setup_avit_model.patch_unused_gui_imports(source)
            setup_avit_model.patch_unused_gui_imports(source)

            for relative_path in setup_avit_model.UNUSED_GUI_IMPORTS:
                text = (source / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("from turtle import forward", text)
                self.assertIn("import torch", text)
                self.assertIn("VALUE = 'kept'", text)


if __name__ == "__main__":
    unittest.main()

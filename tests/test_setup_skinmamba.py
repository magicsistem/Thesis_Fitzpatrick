from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]


class SkinMambaSetupTests(unittest.TestCase):
    def test_checkpoint_archive_is_extracted_and_verified(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "setup_skinmamba_under_test", REPO_ROOT / "scripts/setup_skinmamba.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_archive = root / "source.zip"
            payloads = {"isic17.pth": b"checkpoint-17", "isic18.pth": b"checkpoint-18"}
            with zipfile.ZipFile(source_archive, "w") as archive:
                for name, payload in payloads.items():
                    archive.writestr(f"weight/{name}", payload)

            module.CHECKPOINT_DIR = root / "checkpoints"
            module.ARCHIVE_PATH = module.CHECKPOINT_DIR / "published_weights.zip"
            module.ARCHIVE_SHA256 = module.sha256(source_archive)
            module.CHECKPOINTS = {
                name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
            }

            def fake_download(command, *, cwd=REPO_ROOT):
                del command, cwd
                shutil.copyfile(source_archive, module.ARCHIVE_PATH)

            module.run = fake_download
            module.install_checkpoints()

            self.assertFalse(module.ARCHIVE_PATH.exists())
            for name, payload in payloads.items():
                self.assertEqual((module.CHECKPOINT_DIR / name).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()

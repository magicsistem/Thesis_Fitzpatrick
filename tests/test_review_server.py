from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from serve_segmentation_review import ReviewHandler, ThreadingHTTPServer


class ReviewServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_state_lists_eight_models(self) -> None:
        with urlopen(f"{self.base_url}/api/state") as response:
            payload = json.load(response)
        self.assertEqual(len(payload["models"]), 8)
        self.assertEqual(sum(model["recommended"] for model in payload["models"]), 5)

    def test_index_is_served(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("Comparación de máscaras", page)
        self.assertIn("id=\"carousel\"", page)


if __name__ == "__main__":
    unittest.main()

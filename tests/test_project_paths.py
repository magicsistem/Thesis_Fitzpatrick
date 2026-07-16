from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _project_paths import resolve_data_output, resolve_report_output, resolve_result_output


class ProjectPathTests(unittest.TestCase):
    def test_known_roots_are_accepted(self) -> None:
        self.assertEqual(resolve_data_output("data/raw/example").name, "example")
        self.assertEqual(resolve_report_output("reports/tables/example").name, "example")
        self.assertEqual(resolve_result_output("results/example").name, "example")

    def test_arbitrary_directory_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_result_output("unexpected-folder/result")

    def test_protected_root_itself_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_data_output("data/raw")
        with self.assertRaises(SystemExit):
            resolve_result_output("results")


if __name__ == "__main__":
    unittest.main()

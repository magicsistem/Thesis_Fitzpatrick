from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.disjoint import IDENTITY_FIELDS, build_disjoint_grouped_folds, derive_disjoint_isic2018


def manifest(split, items):
    return {"schema_version": 1, "dataset_id": "isic2018_task1", "split": split, "source": "official", "expected_count": len(items), "complete": True, "integrity_errors": [], "items": items}


def item(image_id, **identities):
    return {"image_id": image_id, "image_path": f"{image_id}.jpg", "mask_paths": [], **identities}


class DisjointDerivationTests(unittest.TestCase):
    def test_test_is_preserved_and_four_identity_fields_connect_components(self):
        train = manifest("train", [
            item("t_hash", image_sha256="same-hash"),
            item("t_patient", patient_id="p-val"),
            item("t_lesion", lesion_id="l-test"),
            item("t_duplicate", duplicate_group_id="d-test"),
            item("t_safe", patient_id="safe"),
        ])
        validation = manifest("validation", [item("v_hash", image_sha256="same-hash"), item("v_patient", patient_id="p-val")])
        test = manifest("test", [item("x_lesion", lesion_id="l-test"), item("x_duplicate", duplicate_group_id="d-test")])
        derived_train, derived_validation, report = derive_disjoint_isic2018(train, validation, test)
        self.assertEqual([entry["image_id"] for entry in derived_train["items"]], ["t_safe"])
        self.assertEqual([entry["image_id"] for entry in derived_validation["items"]], ["v_hash", "v_patient"])
        self.assertEqual(derived_train["expected_count"], 1)
        self.assertEqual(derived_validation["expected_count"], 2)
        self.assertEqual(report["policy"]["identity_fields"], list(IDENTITY_FIELDS))

    def test_transitive_component_connected_to_test_is_removed(self):
        train = manifest("train", [item("a", patient_id="p1", lesion_id="l1"), item("b", lesion_id="l1", duplicate_group_id="d1")])
        validation = manifest("validation", [item("c", patient_id="p1")])
        test = manifest("test", [item("d", duplicate_group_id="d1")])
        derived_train, derived_validation, _ = derive_disjoint_isic2018(train, validation, test)
        self.assertEqual(derived_train["items"], [])
        self.assertEqual(derived_validation["items"], [])

    def test_folds_keep_sha_connected_images_together(self):
        train = manifest("train", [item("a", image_sha256="same"), item("b", image_sha256="same"), item("c")])
        folds = build_disjoint_grouped_folds(train, folds=2, seed=7)
        assigned = {image_id: fold["fold"] for fold in folds["folds"] for image_id in fold["validation_ids"]}
        self.assertEqual(assigned["a"], assigned["b"])
        self.assertEqual(folds["identity_fields"], list(IDENTITY_FIELDS))


if __name__ == "__main__":
    unittest.main()

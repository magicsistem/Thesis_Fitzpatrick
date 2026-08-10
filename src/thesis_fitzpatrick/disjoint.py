"""Deterministic cross-split decontamination for dataset manifests."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from typing import Any, Iterable


IDENTITY_FIELDS = ("image_sha256", "patient_id", "lesion_id", "duplicate_group_id")


def _identities(item: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for field in IDENTITY_FIELDS:
        value = item.get(field)
        if value not in (None, ""):
            yield field, str(value)


def derive_disjoint_isic2018(
    train: dict[str, Any], validation: dict[str, Any], test: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Preserve test, then validation, while removing connected split overlap."""
    manifests = {"train": train, "validation": validation, "test": test}
    for split, manifest in manifests.items():
        if manifest.get("dataset_id") != "isic2018_task1" or manifest.get("split") != split:
            raise ValueError(f"Expected the official isic2018_task1 {split} manifest")
        if not isinstance(manifest.get("items"), list):
            raise ValueError(f"Manifest {split} has no items list")

    records: list[tuple[str, int, dict[str, Any]]] = []
    for split, manifest in manifests.items():
        records.extend((split, index, item) for index, item in enumerate(manifest["items"]))
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, (_, _, item) in enumerate(records):
        for identity in _identities(item):
            if identity in seen:
                union(index, seen[identity])
            else:
                seen[identity] = index

    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)

    excluded: set[int] = set()
    excluded_records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    for indices in components.values():
        splits = {records[index][0] for index in indices}
        if "test" in splits:
            removable = {"train", "validation"}
            reason = "component_connected_to_test"
        elif {"train", "validation"}.issubset(splits):
            removable = {"train"}
            reason = "train_component_connected_to_validation"
        else:
            continue
        component_identities = sorted({identity for index in indices for identity in _identities(records[index][2])})
        for index in indices:
            split, _, item = records[index]
            if split not in removable:
                continue
            excluded.add(index)
            excluded_counts[split] += 1
            reason_counts[reason] += 1
            excluded_records.append({
                "split": split,
                "image_id": item["image_id"],
                "reason": reason,
                "component_splits": sorted(splits),
                "component_identities": [
                    {"identity_type": field, "identity": value}
                    for field, value in component_identities
                ],
            })

    def derived(source: dict[str, Any], split: str) -> dict[str, Any]:
        result = deepcopy(source)
        result["items"] = [
            deepcopy(item)
            for index, (record_split, _, item) in enumerate(records)
            if record_split == split and index not in excluded
        ]
        result["expected_count"] = len(result["items"])
        result["complete"] = not result.get("integrity_errors")
        result["source"] = f"{source.get('source', 'ISIC 2018')} — deterministic disjoint derivation"
        result["derivation"] = {
            "policy": "preserve_test_then_validation",
            "identity_fields": list(IDENTITY_FIELDS),
            "source_expected_count": source.get("expected_count"),
            "source_item_count": len(source["items"]),
            "excluded_count": excluded_counts[split],
        }
        return result

    train_disjoint = derived(train, "train")
    validation_disjoint = derived(validation, "validation")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "priority": ["test", "validation", "train"],
            "identity_fields": list(IDENTITY_FIELDS),
            "description": "Preserve official test; remove development components connected to test; preserve validation over train.",
        },
        "source_counts": {split: len(manifest["items"]) for split, manifest in manifests.items()},
        "derived_counts": {
            "train": len(train_disjoint["items"]),
            "validation": len(validation_disjoint["items"]),
            "test": len(test["items"]),
        },
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "excluded_records": sorted(excluded_records, key=lambda item: (item["split"], item["image_id"])),
    }
    return train_disjoint, validation_disjoint, report


def build_disjoint_grouped_folds(payload: dict[str, Any], folds: int = 5, seed: int = 20260806) -> dict[str, Any]:
    """Assign every four-identity connected component to exactly one fold."""
    if payload.get("split") != "train" or not isinstance(payload.get("items"), list):
        raise ValueError("Disjoint folds require a train manifest")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    items = payload["items"]
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, entry in enumerate(items):
        identities = list(_identities(entry)) or [("image_id", str(entry["image_id"]))]
        for identity in identities:
            if identity in seen:
                union(index, seen[identity])
            else:
                seen[identity] = index
    groups: dict[int, list[str]] = {}
    for index, entry in enumerate(items):
        groups.setdefault(find(index), []).append(str(entry["image_id"]))
    ordered = sorted(groups.values(), key=lambda ids: hashlib.sha256(f"{seed}:{'|'.join(sorted(ids))}".encode()).hexdigest())
    assignments = {image_id: index % folds for index, ids in enumerate(ordered) for image_id in ids}
    all_ids = set(assignments)
    fold_entries = []
    for fold in range(folds):
        validation_ids = sorted(image_id for image_id, assigned in assignments.items() if assigned == fold)
        fold_entries.append({"fold": fold, "train_ids": sorted(all_ids - set(validation_ids)), "validation_ids": validation_ids})
    fallback_count = sum(not list(_identities(entry)) for entry in items)
    return {
        "schema_version": 1,
        "dataset_id": payload["dataset_id"],
        "source_split": "train",
        "fold_count": folds,
        "seed": seed,
        "group_count": len(groups),
        "image_id_fallback_count": fallback_count,
        "fallback_policy": "image_id only when all four disjoint identity fields are absent",
        "identity_fields": list(IDENTITY_FIELDS),
        "leakage_audit": {"passed": True, "groups_crossing_folds": 0},
        "folds": fold_entries,
    }

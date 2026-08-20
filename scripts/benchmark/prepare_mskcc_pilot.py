#!/usr/bin/env python3
"""Download/import, audit, and deterministically select the 100-image MSKCC pilot.

No MSKCC URL is hard-coded because distribution endpoints/licensing can change.
The exact source URL and optional checksum are explicit CLI inputs and are saved
in the download audit.  Sampling is deterministic and refuses to fabricate
missing lesion/patient identifiers unless the user explicitly relaxes a gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file  # noqa: E402
from thesis_fitzpatrick.datasets import download_resumable, safe_extract_zip  # noqa: E402
from thesis_fitzpatrick.post05 import (  # noqa: E402
    infer_column_map,
    parse_optional_float,
    parse_optional_int,
    read_metadata_table,
    stable_tiebreak,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _download(args: argparse.Namespace) -> None:
    if not args.confirm_download:
        raise SystemExit("Review the dataset license/source and repeat with --confirm-download")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = download_resumable(args.url, args.output, args.expected_sha256)
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "output": str(args.output.resolve()),
        "expected_sha256": args.expected_sha256,
        **result,
    }
    atomic_write_json(args.audit or args.output.with_suffix(args.output.suffix + ".download.json"), audit)
    print(json.dumps(audit, indent=2))


def _extract(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Extraction target is not empty: {args.output}")
    staging = args.output.with_name(args.output.name + ".extracting")
    if staging.exists():
        raise SystemExit(f"Incomplete staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        members = safe_extract_zip(args.archive, staging)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(args.output)
    except BaseException:
        raise
    print(json.dumps({"archive": str(args.archive), "output": str(args.output), "members": len(members)}, indent=2))


def _find_images(root: Path) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_stem[path.stem].append(path.resolve())
    return by_stem


def _value(row: dict[str, Any], mapping: dict[str, str], canonical: str) -> Any:
    source = mapping.get(canonical)
    return row.get(source) if source else None


def _registry(args: argparse.Namespace) -> None:
    rows, fields = read_metadata_table(args.metadata)
    explicit = json.loads(args.column_map) if args.column_map else None
    mapping = infer_column_map(fields, explicit)
    if "image_id" not in mapping or "mst" not in mapping:
        raise SystemExit(f"Metadata must expose image_id and MST. Detected mapping: {mapping}")
    images = _find_images(args.images_root)
    normalized: list[dict[str, Any]] = []
    ambiguous_images: list[str] = []
    missing_images: list[str] = []

    for source_row in rows:
        image_id = str(_value(source_row, mapping, "image_id") or "").strip()
        if not image_id:
            continue
        candidates = images.get(Path(image_id).stem, [])
        if len(candidates) > 1:
            ambiguous_images.append(image_id)
            image_path = None
        elif not candidates:
            missing_images.append(image_id)
            image_path = None
        else:
            image_path = candidates[0]
        mst = parse_optional_int(_value(source_row, mapping, "mst"))
        fst = parse_optional_int(_value(source_row, mapping, "fst"))
        record = {
            "image_id": image_id,
            "patient_id": str(_value(source_row, mapping, "patient_id") or "").strip() or None,
            "lesion_id": str(_value(source_row, mapping, "lesion_id") or "").strip() or None,
            "mst": mst if mst is not None and 1 <= mst <= 10 else None,
            "fst": fst if fst is not None and 1 <= fst <= 6 else None,
            "l_star": parse_optional_float(_value(source_row, mapping, "l_star")),
            "a_star": parse_optional_float(_value(source_row, mapping, "a_star")),
            "b_star": parse_optional_float(_value(source_row, mapping, "b_star")),
            "ita": parse_optional_float(_value(source_row, mapping, "ita")),
            "acquisition_mode": str(_value(source_row, mapping, "acquisition_mode") or "").strip() or None,
            "anatomical_site": str(_value(source_row, mapping, "anatomical_site") or "").strip() or None,
            "image_path": str(image_path) if image_path else None,
            "image_sha256": sha256_file(image_path) if image_path else None,
        }
        record["tone_variable_completeness"] = sum(record[key] is not None for key in ("fst", "mst", "l_star", "b_star", "ita"))
        normalized.append(record)

    if not normalized:
        raise SystemExit("No metadata rows could be normalized")
    duplicate_ids = [key for key, count in Counter(row["image_id"] for row in normalized).items() if count > 1]
    if duplicate_ids:
        raise SystemExit(f"Duplicate image_id values in metadata: {duplicate_ids[:10]}")
    distributions = {
        "n_metadata_rows": len(rows),
        "n_registry_rows": len(normalized),
        "n_images_found": sum(row["image_path"] is not None for row in normalized),
        "n_missing_images": len(set(missing_images)),
        "n_ambiguous_images": len(set(ambiguous_images)),
        "mst": dict(sorted(Counter(str(row["mst"]) for row in normalized if row["mst"] is not None).items())),
        "fst": dict(sorted(Counter(str(row["fst"]) for row in normalized if row["fst"] is not None).items())),
        "acquisition_mode": dict(sorted(Counter(str(row["acquisition_mode"] or "UNKNOWN") for row in normalized).items())),
        "anatomical_site": dict(sorted(Counter(str(row["anatomical_site"] or "UNKNOWN") for row in normalized).items())),
        "availability": {
            key: sum(row[key] is not None for row in normalized)
            for key in ("patient_id", "lesion_id", "mst", "fst", "l_star", "a_star", "b_star", "ita", "acquisition_mode", "anatomical_site")
        },
    }
    payload = {
        "schema_version": 1,
        "dataset_role": "MSKCC_skin_tone_segmentation_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "images_root": str(args.images_root.resolve()),
        "column_map": mapping,
        "distributions": distributions,
        "rows": normalized,
        "warnings": {
            "missing_image_ids": sorted(set(missing_images)),
            "ambiguous_image_ids": sorted(set(ambiguous_images)),
        },
    }
    payload["identity_hash"] = content_hash(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "mskcc_registry.json", payload)
    atomic_write_json(args.output / "mskcc_distribution_report.json", distributions)
    columns = [
        "image_id", "patient_id", "lesion_id", "mst", "fst", "l_star", "a_star", "b_star", "ita",
        "acquisition_mode", "anatomical_site", "tone_variable_completeness", "image_path", "image_sha256",
    ]
    with (args.output / "mskcc_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows([{key: row.get(key) for key in columns} for row in normalized])
    print(json.dumps({"output": str(args.output), "column_map": mapping, "distributions": distributions}, indent=2))


def _candidate_score(row: dict[str, Any], seed: int) -> tuple[int, str]:
    # Completeness is primary.  The cryptographic tie break makes selection
    # deterministic without favoring metadata ordering.
    completeness = sum(row.get(key) is not None for key in ("fst", "l_star", "b_star", "ita"))
    return completeness, stable_tiebreak(seed, str(row["image_id"]))


def _greedy_sample(
    rows: list[dict[str, Any]],
    groups: dict[str, set[int]],
    target_per_group: int,
    seed: int,
    max_patient: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group = {name: [row for row in rows if row.get("mst") in tones] for name, tones in groups.items()}
    # Scarce strata are filled first to reduce avoidable patient-cap conflicts.
    order = sorted(groups, key=lambda name: (len(by_group[name]), name))
    selected: list[dict[str, Any]] = []
    used_lesions: set[str] = set()
    patient_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    for group in order:
        candidates = sorted(by_group[group], key=lambda row: (-_candidate_score(row, seed)[0], _candidate_score(row, seed)[1]))
        for row in candidates:
            if group_counts[group] >= target_per_group:
                break
            lesion = str(row["lesion_id"])
            patient = str(row["patient_id"])
            if lesion in used_lesions:
                rejections["duplicate_lesion"] += 1
                continue
            if patient_counts[patient] >= max_patient:
                rejections["patient_cap"] += 1
                continue
            selected.append({**row, "sampling_group": group})
            used_lesions.add(lesion)
            patient_counts[patient] += 1
            group_counts[group] += 1
    success = all(group_counts[name] == target_per_group for name in groups)
    audit = {
        "success": success,
        "target_per_group": target_per_group,
        "group_order_scarcity_first": order,
        "candidate_counts": {name: len(by_group[name]) for name in groups},
        "selected_counts": dict(group_counts),
        "rejections": dict(rejections),
        "unique_lesions": len(used_lesions),
        "unique_patients": len(patient_counts),
        "max_observed_lesions_per_patient": max(patient_counts.values(), default=0),
    }
    return selected, audit


def _sample(args: argparse.Namespace) -> None:
    registry = load_json(args.registry)
    identity = registry.pop("identity_hash", None)
    if content_hash(registry) != identity:
        raise SystemExit("MSKCC registry identity hash is invalid")
    registry["identity_hash"] = identity
    rows = [row for row in registry["rows"] if row.get("image_path") and row.get("mst") in range(1, 11)]
    if not args.allow_missing_lesion_id:
        rows = [row for row in rows if row.get("lesion_id")]
    else:
        for row in rows:
            row["lesion_id"] = row.get("lesion_id") or f"IMAGE::{row['image_id']}"
    if not args.allow_missing_patient_id:
        rows = [row for row in rows if row.get("patient_id")]
    else:
        for row in rows:
            row["patient_id"] = row.get("patient_id") or f"UNKNOWN::{row['lesion_id']}"
    if not rows:
        raise SystemExit("No eligible rows after lesion/patient/MST/image gates")

    known_modes = Counter(str(row["acquisition_mode"]) for row in rows if row.get("acquisition_mode"))
    if args.acquisition_mode:
        chosen_mode = args.acquisition_mode
        rows = [row for row in rows if row.get("acquisition_mode") == chosen_mode]
    elif known_modes:
        chosen_mode = sorted(known_modes.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
        rows = [row for row in rows if row.get("acquisition_mode") == chosen_mode]
    else:
        chosen_mode = None
        if args.require_known_acquisition_mode:
            raise SystemExit("No acquisition_mode metadata is available; cannot enforce the frozen single-mode sampling contract")

    primary_groups = {str(tone): {tone} for tone in range(1, 11)}
    selected, primary_audit = _greedy_sample(rows, primary_groups, 10, args.seed, args.max_lesions_per_patient)
    strategy = "MST_10x10"
    fallback_audit = None
    if not primary_audit["success"]:
        fallback_groups = {"1-2": {1, 2}, "3-4": {3, 4}, "5-6": {5, 6}, "7-8": {7, 8}, "9-10": {9, 10}}
        selected, fallback_audit = _greedy_sample(rows, fallback_groups, 20, args.seed, args.max_lesions_per_patient)
        strategy = "MST_pairs_5x20"
        if not fallback_audit["success"]:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.output / "mskcc_sampling_failure.json", {
                "primary": primary_audit,
                "fallback": fallback_audit,
                "eligible_rows": len(rows),
                "chosen_acquisition_mode": chosen_mode,
            })
            raise SystemExit("Neither 10x10 MST nor 5x20 paired-MST sampling is feasible under the frozen constraints")

    if len(selected) != 100 or len({row["lesion_id"] for row in selected}) != 100:
        raise SystemExit("Internal sampling error: expected exactly 100 unique lesions")
    patient_counts = Counter(row["patient_id"] for row in selected)
    if max(patient_counts.values(), default=0) > args.max_lesions_per_patient:
        raise SystemExit("Internal sampling error: patient cap violated")

    selected = sorted(selected, key=lambda row: (row["sampling_group"], stable_tiebreak(args.seed, row["image_id"])))
    distributions = {
        "mst": dict(sorted(Counter(str(row["mst"]) for row in selected).items())),
        "fst": dict(sorted(Counter(str(row["fst"]) for row in selected if row.get("fst") is not None).items())),
        "acquisition_mode": dict(sorted(Counter(str(row["acquisition_mode"] or "UNKNOWN") for row in selected).items())),
        "anatomical_site": dict(sorted(Counter(str(row["anatomical_site"] or "UNKNOWN") for row in selected).items())),
        "complete_FST_MST_Lstar_bstar_ITA": sum(all(row.get(key) is not None for key in ("fst", "mst", "l_star", "b_star", "ita")) for row in selected),
        "unique_patients": len(patient_counts),
        "unique_lesions": len({row["lesion_id"] for row in selected}),
    }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "target_n": 100,
        "seed": args.seed,
        "registry_path": str(args.registry.resolve()),
        "registry_identity_hash": identity,
        "chosen_acquisition_mode": chosen_mode,
        "single_acquisition_mode_enforced": chosen_mode is not None,
        "one_image_per_lesion": True,
        "max_lesions_per_patient": args.max_lesions_per_patient,
        "primary_audit": primary_audit,
        "fallback_audit": fallback_audit,
        "distributions": distributions,
        "items": selected,
    }
    payload["identity_hash"] = content_hash(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "mskcc_sample_100.json", payload)
    atomic_write_json(args.output / "mskcc_sample_distribution.json", distributions)
    columns = [
        "sampling_group", "image_id", "patient_id", "lesion_id", "mst", "fst", "l_star", "a_star", "b_star", "ita",
        "acquisition_mode", "anatomical_site", "tone_variable_completeness", "image_path", "image_sha256",
    ]
    with (args.output / "mskcc_sample_100.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows([{key: row.get(key) for key in columns} for row in selected])
    print(json.dumps({"strategy": strategy, "n": len(selected), "chosen_acquisition_mode": chosen_mode, "distributions": distributions, "output": str(args.output)}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    download = sub.add_parser("download")
    download.add_argument("--url", required=True)
    download.add_argument("--output", required=True, type=Path)
    download.add_argument("--expected-sha256")
    download.add_argument("--audit", type=Path)
    download.add_argument("--confirm-download", action="store_true")
    extract = sub.add_parser("extract-zip")
    extract.add_argument("--archive", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    registry = sub.add_parser("build-registry")
    registry.add_argument("--metadata", required=True, type=Path)
    registry.add_argument("--images-root", required=True, type=Path)
    registry.add_argument("--column-map", help='JSON object, e.g. {"mst":"Monk Skin Tone","image_id":"image_name"}')
    registry.add_argument("--output", required=True, type=Path)
    sample = sub.add_parser("sample")
    sample.add_argument("--registry", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--seed", type=int, default=20260806)
    sample.add_argument("--max-lesions-per-patient", type=int, default=2)
    sample.add_argument("--acquisition-mode")
    sample.add_argument("--require-known-acquisition-mode", action="store_true")
    sample.add_argument("--allow-missing-lesion-id", action="store_true")
    sample.add_argument("--allow-missing-patient-id", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "download":
        _download(args)
    elif args.action == "extract-zip":
        _extract(args)
    elif args.action == "build-registry":
        _registry(args)
    else:
        _sample(args)


if __name__ == "__main__":
    main()

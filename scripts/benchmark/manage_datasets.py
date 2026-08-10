#!/usr/bin/env python3
"""Download metadata, import datasets, build manifests, and verify integrity."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, load_json, sha256_file  # noqa: E402
from thesis_fitzpatrick.datasets import (  # noqa: E402
    OFFICIAL_IMAPP_FILES,
    OFFICIAL_ISIC2018_ARCHIVES,
    audit_manifest_overlap,
    build_fitzpatrick_manifest,
    build_isic2018_manifest,
    build_novice_manifest,
    download_resumable,
    import_imapp_metadata,
    safe_extract_zip,
    verify_manifest_files,
    write_manifest,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="action", required=True)
    download = commands.add_parser("download-imapp-metadata", help="Download official IMA++ Zenodo v1.1 metadata and segmentations archive")
    download.add_argument("--output", required=True, type=Path)
    download.add_argument("--include-segmentations", action="store_true")
    download.add_argument("--confirm-download", action="store_true")
    download_isic = commands.add_parser("download-isic2018", help="Resume the six official Task 1 archives; test masks stay sealed")
    download_isic.add_argument("--output", required=True, type=Path); download_isic.add_argument("--confirm-download", action="store_true")
    extract_isic = commands.add_parser("extract-isic2018", help="Atomically extract train/validation and test images, never sealed test masks")
    extract_isic.add_argument("--data-root", required=True, type=Path)
    isic = commands.add_parser("import-isic2018", help="Index an already downloaded official Task 1 split")
    isic.add_argument("--data-root", required=True, type=Path)
    isic.add_argument("--split", required=True, choices=("train", "validation", "test"))
    isic.add_argument("--output", required=True, type=Path)
    isic.add_argument("--metadata-root", type=Path, help="Official ISIC API JSON records used for patient/lesion grouping")
    isic.add_argument("--skip-checksums", action="store_true")
    imapp = commands.add_parser("import-imapp", help="Verify official IMA++ files and create manifests")
    imapp.add_argument("--source-root", required=True, type=Path)
    imapp.add_argument("--data-root", required=True, type=Path)
    imapp.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="Decode and checksum every file in a manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--data-root", required=True, type=Path)
    fitz = commands.add_parser("register-fitzpatrick", help="Register existing images and audit SHA-256 overlap")
    fitz.add_argument("--images-root", required=True, type=Path); fitz.add_argument("--metadata", required=True, type=Path); fitz.add_argument("--data-root", required=True, type=Path); fitz.add_argument("--output", required=True, type=Path)
    novice = commands.add_parser("register-novice", help="Register novice masks as disabled auxiliary data only")
    novice.add_argument("--masks-root", required=True, type=Path); novice.add_argument("--data-root", required=True, type=Path); novice.add_argument("--mapping", type=Path); novice.add_argument("--metadata", type=Path); novice.add_argument("--output", required=True, type=Path)
    images = commands.add_parser("download-isic-images", help="Download manifest IDs from the official ISIC Archive API")
    images.add_argument("--manifest", required=True, type=Path); images.add_argument("--output", required=True, type=Path); images.add_argument("--metadata-output", type=Path); images.add_argument("--workers", type=int, default=8); images.add_argument("--confirm-download", action="store_true")
    metadata = commands.add_parser("download-isic-metadata", help="Download official ISIC Archive records without downloading images")
    metadata.add_argument("--manifest", required=True, type=Path); metadata.add_argument("--output", required=True, type=Path); metadata.add_argument("--workers", type=int, default=8); metadata.add_argument("--confirm-download", action="store_true")
    overlap = commands.add_parser("audit-overlap", help="Reject patient/lesion/duplicate/SHA overlap across dataset splits")
    overlap.add_argument("--manifest", required=True, action="append", type=Path); overlap.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.action == "extract-isic2018":
        extracted = []
        for name, (_, relative, _) in OFFICIAL_ISIC2018_ARCHIVES.items():
            if name == "sealed_test_masks": continue
            archive = args.data_root / relative; target = args.data_root / archive.stem
            if target.is_dir(): extracted.append({"name": name, "target": str(target), "reused": True}); continue
            if not archive.is_file(): raise SystemExit(f"Falta archivo oficial: {archive}")
            staging = args.data_root / f".{archive.stem}.extracting"
            if staging.exists(): raise SystemExit(f"Extracción incompleta previa: {staging}; audítela antes de retirarla y reanudar")
            members = safe_extract_zip(archive, staging); staged_target = staging / archive.stem
            expected_files = sum(not member.endswith("/") for member in members)
            actual_files = sum(path.is_file() for path in staged_target.rglob("*"))
            if not staged_target.is_dir() or actual_files != expected_files: raise SystemExit(f"Extracción incompleta de {archive.name}: {actual_files}/{expected_files}")
            staged_target.replace(target); staging.rmdir()
            extracted.append({"name": name, "target": str(target), "files": actual_files, "reused": False})
        print(json.dumps({"extracted": extracted, "sealed_test_masks_extracted": False}, indent=2))
    elif args.action == "download-isic2018":
        if not args.confirm_download: raise SystemExit("Revise espacio/licencia y repita con --confirm-download")
        results = []
        for name, (url, relative, expected_bytes) in OFFICIAL_ISIC2018_ARCHIVES.items():
            result = download_resumable(url, args.output / relative)
            if expected_bytes is not None and result["bytes"] != expected_bytes: raise SystemExit(f"Tamaño oficial inesperado para {name}: {result['bytes']} != {expected_bytes}")
            results.append({"name": name, "relative_path": relative, **result})
        atomic_write_json(args.output / "download_manifest.json", {"schema_version": 1, "source": "ISIC 2018 Challenge Task 1", "sealed_test_masks_extracted": False, "archives": results})
        print(json.dumps(results, indent=2))
    elif args.action == "download-imapp-metadata":
        names = list(OFFICIAL_IMAPP_FILES)
        if not args.include_segmentations:
            names.remove("segs.zip")
        if not args.confirm_download:
            raise SystemExit("Revise la licencia CC BY-NC-ND 4.0 y repita con --confirm-download; use --include-segmentations para segs.zip.")
        results = []
        for name in names:
            results.append(download_resumable(
                f"https://zenodo.org/records/14201693/files/{name}?download=1",
                args.output / name,
                OFFICIAL_IMAPP_FILES[name],
                "md5",
            ))
        request = urllib.request.Request("https://zenodo.org/api/records/14201693", headers={"User-Agent": "Thesis-Fitzpatrick-reproducible-downloader/1.0"})
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed official registry
            zenodo_record = json.load(response)
        license_id = zenodo_record.get("metadata", {}).get("license", {}).get("id")
        if license_id != "cc-by-nc-nd-4.0": raise SystemExit(f"Licencia IMA++ inesperada en Zenodo: {license_id}")
        atomic_write_json(args.output / "zenodo_record_14201693.json", zenodo_record)
        print(json.dumps(results, indent=2))
    elif args.action == "import-isic2018":
        manifest = build_isic2018_manifest(args.data_root, args.split, include_checksums=not args.skip_checksums, metadata_root=args.metadata_root)
        write_manifest(args.output, manifest)
        print(json.dumps({"manifest": str(args.output), "items": len(manifest["items"]), "complete": manifest["complete"], "errors": len(manifest["integrity_errors"])}, indent=2))
    elif args.action == "import-imapp":
        imported = import_imapp_metadata(args.source_root, args.data_root)
        args.output.mkdir(parents=True, exist_ok=True)
        for split, manifest in imported["manifests"].items():
            write_manifest(args.output / f"imapp_{split}.json", manifest)
        atomic_write_json(args.output / "imapp_import_audit.json", {"verified_md5": imported["verified_md5"], "segmentation_masks_verified": imported["segmentation_masks_verified"]})
        print(json.dumps({"output": str(args.output), "verified": list(imported["verified_md5"]), "splits": {key: len(value["items"]) for key, value in imported["manifests"].items()}}, indent=2))
    elif args.action == "register-fitzpatrick":
        result = build_fitzpatrick_manifest(args.images_root, args.metadata, args.data_root); write_manifest(args.output, result); print(json.dumps({"items": len(result["items"]), "overlap_audit": result["overlap_audit"]}, indent=2))
    elif args.action == "register-novice":
        result = build_novice_manifest(args.masks_root, args.data_root, mapping_csv=args.mapping, metadata_csv=args.metadata); write_manifest(args.output, result); print(json.dumps({"items": len(result["items"]), "complete": result["complete"], "enabled": False, "warning": result["warning"]}, indent=2))
    elif args.action in {"download-isic-images", "download-isic-metadata"}:
        manifest = load_json(args.manifest); ids = [item["image_id"] for item in manifest["items"]]
        if not args.confirm_download: raise SystemExit(f"Se consultarían hasta {len(ids)} registros oficiales. Revise espacio/red y repita con --confirm-download.")
        if args.workers < 1: raise SystemExit("--workers debe ser al menos 1")
        args.output.mkdir(parents=True, exist_ok=True)
        metadata_output = (args.metadata_output or args.output.parent / "metadata") if args.action == "download-isic-images" else args.output
        metadata_output.mkdir(parents=True, exist_ok=True)
        def fetch(image_id: str) -> dict[str, str]:
            metadata_path = metadata_output / f"{image_id}.json"
            existing = [path for path in args.output.glob(f"{image_id}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if args.action == "download-isic-metadata" and metadata_path.is_file(): return {"image_id": image_id, "metadata": str(metadata_path), "reused": True}
            if args.action == "download-isic-images" and existing and metadata_path.is_file(): return {"image_id": image_id, "path": str(existing[0]), "sha256": sha256_file(existing[0]), "reused": True}
            endpoint = f"https://api.isic-archive.com/api/v2/images/{image_id}/"
            request = urllib.request.Request(endpoint, headers={"User-Agent": "Thesis-Fitzpatrick-reproducible-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed official API
                record = json.load(response)
            atomic_write_json(metadata_path, record)
            if args.action == "download-isic-metadata": return {"image_id": image_id, "metadata": str(metadata_path)}
            full = record.get("files", {}).get("full", {})
            url = full.get("url")
            if not url: raise ValueError(f"La API oficial no publicó files.full.url para {image_id}")
            suffix = Path(url.split("?", 1)[0]).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}: suffix = ".jpg"
            destination = args.output / f"{image_id}{suffix}"
            result = {"sha256": sha256_file(destination)} if destination.is_file() else download_resumable(url, destination)
            return {"image_id": image_id, "path": str(destination), "sha256": result["sha256"]}
        failures = []; completed = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch, image_id): image_id for image_id in ids}
            for future in as_completed(futures):
                try: completed.append(future.result())
                except Exception as exc: failures.append({"image_id": futures[future], "error": str(exc)})
        print(json.dumps({"requested": len(ids), "downloaded_or_verified": len(completed), "failures": failures}, indent=2))
        if failures: raise SystemExit(2)
    elif args.action == "audit-overlap":
        report = audit_manifest_overlap([load_json(path) for path in args.manifest])
        if args.output: atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
        if not report["passed"]: raise SystemExit(2)
    else:
        report = verify_manifest_files(load_json(args.manifest), args.data_root)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

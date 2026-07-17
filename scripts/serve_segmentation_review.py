"""Serve the local segmentation comparison and mask-review interface."""

from __future__ import annotations

import argparse
import csv
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import resource
import random
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

from PIL import Image

from _project_paths import REPO_ROOT, require_within

sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.masks import (  # noqa: E402
    build_clean_skin_mask,
    measure_skin_colour,
    normalize_binary_mask,
    save_binary_mask,
)


WEB_DIR = REPO_ROOT / "web"
MODEL_CONFIG = REPO_ROOT / "configs" / "segmentation_models.json"
PILOT_INPUT_DIR = REPO_ROOT / "data" / "processed" / "isic_segmentation_pilot_resized_inference"
FITZPATRICK_INPUT_DIR = REPO_ROOT / "data" / "raw" / "isic_fitzpatrick_images"
INPUT_ROOTS = {
    "pilot": PILOT_INPUT_DIR,
    "fitzpatrick": FITZPATRICK_INPUT_DIR,
}
METADATA_CSV = REPO_ROOT / "data" / "raw" / "isic_fitzpatrick_metadata_full.csv"
RESULTS_DIR = REPO_ROOT / "results" / "segmentation_benchmark"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FITZPATRICK_TYPES = ("I", "II", "III", "IV", "V", "VI")


def load_models() -> list[dict]:
    payload = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    return sorted(payload["models"], key=lambda model: model["priority"])


def load_image_metadata() -> dict[str, dict]:
    if not METADATA_CSV.is_file():
        return {}
    metadata = {}
    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = str(row.get("isic_id", "")).strip()
            if not image_id or image_id in metadata:
                continue
            metadata[image_id] = {
                "fitzpatrick_skin_type": str(row.get("fitzpatrick_skin_type", "")).strip(),
                "diagnosis_1": str(row.get("diagnosis_1", "")).strip(),
                "image_type": str(row.get("image_type", "")).strip(),
                "copyright_license": str(row.get("copyright_license", "")).strip(),
            }
    return metadata


def list_images() -> list[dict]:
    metadata = load_image_metadata()
    images = []
    seen = set()
    for source, root in INPUT_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_id = path.stem
            if image_id in seen:
                continue
            seen.add(image_id)
            details = metadata.get(image_id, {})
            images.append({
                "id": image_id,
                "filename": path.name,
                "source": source,
                "url": f"/files/input/{source}/{path.name}",
                "fitzpatrick_skin_type": details.get("fitzpatrick_skin_type", ""),
                "diagnosis_1": details.get("diagnosis_1", ""),
                "image_type": details.get("image_type", ""),
                "copyright_license": details.get("copyright_license", ""),
            })
    return images


def image_path(record: dict) -> Path:
    source = record.get("source", "pilot")
    if source not in INPUT_ROOTS:
        raise ValueError(f"Origen de imagen desconocido: {source}")
    return require_within(
        INPUT_ROOTS[source] / record["filename"],
        (INPUT_ROOTS[source],),
        "Input image",
    )


def pool_summary(images: list[dict]) -> dict:
    counts = {
        fitzpatrick_type: sum(
            image.get("fitzpatrick_skin_type") == fitzpatrick_type for image in images
        )
        for fitzpatrick_type in FITZPATRICK_TYPES
    }
    return {
        "total_local_images": len(images),
        "eligible_fitzpatrick_images": sum(counts.values()),
        "fitzpatrick_counts": counts,
        "metadata_available": METADATA_CSV.is_file(),
        "full_image_directory": str(FITZPATRICK_INPUT_DIR.relative_to(REPO_ROOT)),
    }


def stratified_sample(images: list[dict], total: int, seed: int = 42) -> list[dict]:
    if total < len(FITZPATRICK_TYPES) or total % len(FITZPATRICK_TYPES) != 0:
        raise ValueError("La cantidad debe ser un múltiplo de 6 y al menos 6.")
    per_type = total // len(FITZPATRICK_TYPES)
    selected_by_type = {}
    for fitzpatrick_type in FITZPATRICK_TYPES:
        candidates = sorted(
            (
                image for image in images
                if image.get("fitzpatrick_skin_type") == fitzpatrick_type
            ),
            key=lambda image: image["id"],
        )
        if len(candidates) < per_type:
            raise ValueError(
                f"Fitzpatrick {fitzpatrick_type} solo tiene {len(candidates)} imágenes "
                f"locales; se necesitan {per_type}."
            )
        selected_by_type[fitzpatrick_type] = random.Random(
            f"{seed}:{fitzpatrick_type}"
        ).sample(candidates, per_type)
    return [
        selected_by_type[fitzpatrick_type][index]
        for index in range(per_type)
        for fitzpatrick_type in FITZPATRICK_TYPES
    ]


def initial_images(images: list[dict]) -> list[dict]:
    try:
        return stratified_sample(images, 12, 42)
    except ValueError:
        return images[:10]


def result_payload(
    model_id: str,
    image_id: str,
    image_path: Path,
    runtime: dict | None = None,
    *,
    original_url: str | None = None,
    image_metadata: dict | None = None,
) -> dict:
    output_dir = RESULTS_DIR / model_id / image_id
    lesion_path = output_dir / "lesion_mask.png"
    skin_path = output_dir / "clean_skin_mask.png"
    stats_path = output_dir / "colour_stats.json"
    runtime_path = output_dir / "runtime_metrics.json"
    if not lesion_path.is_file():
        return {
            "model_id": model_id,
            "image_id": image_id,
            "status": "mask_missing",
            "message": "El adaptador todavía no produjo lesion_mask.png.",
            "original_url": original_url or f"/files/input/{image_path.name}",
            "image_metadata": image_metadata or {},
        }

    image = Image.open(image_path).convert("RGB")
    lesion = normalize_binary_mask(Image.open(lesion_path), image.size)
    clean_skin, metadata = build_clean_skin_mask(image, lesion)
    save_binary_mask(lesion, lesion_path)
    save_binary_mask(clean_skin, skin_path)
    try:
        stats = measure_skin_colour(image, clean_skin)
        skin_colour = stats.to_dict()
        skin_colour_error = None
    except ValueError:
        skin_colour = None
        skin_colour_error = (
            "La predicción no dejó píxeles válidos de piel limpia; "
            "las estadísticas de color no están disponibles para este resultado."
        )
    stats_payload = {
        "postprocessing": metadata,
        "skin_colour": skin_colour,
        "skin_colour_available": skin_colour is not None,
        "skin_colour_error": skin_colour_error,
        "warning": "Fitzpatrick no se deduce únicamente a partir del color de la imagen.",
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if runtime is not None:
        runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    base = f"/files/result/{model_id}/{image_id}"
    return {
        "model_id": model_id,
        "image_id": image_id,
        "status": "ready",
        "original_url": original_url or f"/files/input/{image_path.name}",
        "lesion_mask_url": f"{base}/lesion_mask.png",
        "clean_skin_mask_url": f"{base}/clean_skin_mask.png",
        "stats": stats_payload,
        "runtime": runtime,
        "image_metadata": image_metadata or {},
    }


def _process_tree_rss_kb(root_pid: int) -> int:
    """Read aggregate resident memory for a Linux process and descendants."""
    processes: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        ppid = 0
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        processes[int(entry.name)] = (ppid, rss_kb)

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in processes.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _runtime_metrics(start: float, peak_rss_kb: int, before: resource.struct_rusage) -> dict:
    wall_seconds = max(time.perf_counter() - start, 1e-9)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(
        0.0,
        (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
    )
    logical_cpus = os.cpu_count() or 1
    effective_cores = min(cpu_seconds / wall_seconds, float(logical_cpus))
    single_core_percent = effective_cores * 100.0
    system_capacity_percent = min(effective_cores / logical_cpus * 100.0, 100.0)
    return {
        "wall_seconds": round(wall_seconds, 4),
        "cpu_seconds": round(cpu_seconds, 4),
        "effective_cpu_cores": round(effective_cores, 4),
        "cpu_percent_single_core_equivalent": round(single_core_percent, 2),
        "cpu_percent_system_capacity": round(system_capacity_percent, 2),
        "logical_cpu_count": logical_cpus,
        "peak_ram_mb": round(peak_rss_kb / 1024.0, 2),
        "sampling_interval_seconds": 0.05,
    }


def benchmark_summaries(results: list[dict]) -> list[dict]:
    summaries = []
    model_ids = sorted({result["model_id"] for result in results})
    for model_id in model_ids:
        runtimes = [
            result["runtime"]
            for result in results
            if result["model_id"] == model_id
            and result.get("status") == "ready"
            and result.get("runtime")
        ]
        if not runtimes:
            continue
        summaries.append({
            "model_id": model_id,
            "images_timed": len(runtimes),
            "total_wall_seconds": round(sum(item["wall_seconds"] for item in runtimes), 4),
            "average_wall_seconds": round(
                sum(item["wall_seconds"] for item in runtimes) / len(runtimes), 4
            ),
            "average_cpu_percent_single_core_equivalent": round(
                sum(item["cpu_percent_single_core_equivalent"] for item in runtimes) / len(runtimes), 2
            ),
            "average_cpu_percent_system_capacity": round(
                sum(item["cpu_percent_system_capacity"] for item in runtimes) / len(runtimes), 2
            ),
            "average_effective_cpu_cores": round(
                sum(
                    item.get(
                        "effective_cpu_cores",
                        item["cpu_percent_single_core_equivalent"] / 100.0,
                    )
                    for item in runtimes
                ) / len(runtimes),
                4,
            ),
            "peak_ram_mb_max": round(max(item["peak_ram_mb"] for item in runtimes), 2),
        })
    return summaries


def run_adapter(model: dict, image_path: Path, lesion_path: Path) -> tuple[bool, str, dict | None]:
    """Run a configured adapter without invoking a shell."""
    template = model.get("adapter_command")
    if not template:
        return False, "El modelo está catalogado, pero todavía no tiene adaptador configurado.", None
    if not isinstance(template, list) or not all(isinstance(token, str) for token in template):
        return False, "adapter_command debe ser una lista JSON de argumentos.", None
    lesion_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "{image}": str(image_path),
        "{lesion_mask}": str(lesion_path),
        "{repo_root}": str(REPO_ROOT),
        "{python}": sys.executable,
        "{conda}": os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda",
    }
    command = []
    for token in template:
        for placeholder, value in replacements.items():
            token = token.replace(placeholder, value)
        command.append(token)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    peak_rss_kb = 0
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while True:
            peak_rss_kb = max(peak_rss_kb, _process_tree_rss_kb(process.pid))
            try:
                stdout, stderr = process.communicate(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter() - start > 900:
                    process.kill()
                    stdout, stderr = process.communicate()
                    metrics = _runtime_metrics(start, peak_rss_kb, before)
                    lesion_path.unlink(missing_ok=True)
                    return False, "El adaptador superó el límite de 900 segundos.", metrics
    except OSError as exc:
        return False, str(exc), None
    metrics = _runtime_metrics(start, peak_rss_kb, before)
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "sin detalle"
        lesion_path.unlink(missing_ok=True)
        return False, f"El adaptador terminó con código {process.returncode}: {detail[-800:]}", metrics
    if not lesion_path.is_file():
        return False, "El adaptador terminó, pero no creó lesion_mask.png.", metrics
    try:
        with Image.open(lesion_path) as candidate:
            candidate.verify()
    except (OSError, ValueError) as exc:
        lesion_path.unlink(missing_ok=True)
        return False, f"El adaptador creó una máscara inválida: {exc}", metrics
    return True, "ok", metrics


class ReviewHandler(SimpleHTTPRequestHandler):
    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, root: Path) -> None:
        try:
            safe_path = require_within(path, (root,), "Requested file")
        except SystemExit:
            self.send_error(403)
            return
        if not safe_path.is_file():
            self.send_error(404)
            return
        content = safe_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(safe_path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/state":
            images = list_images()
            self._json({
                "models": load_models(),
                "images": initial_images(images),
                "pool": pool_summary(images),
            })
            return
        if path.startswith("/files/input/"):
            relative = path.removeprefix("/files/input/")
            source, separator, filename = relative.partition("/")
            if separator and source in INPUT_ROOTS:
                self._serve_file(INPUT_ROOTS[source] / filename, INPUT_ROOTS[source])
            else:
                self._serve_file(PILOT_INPUT_DIR / relative, PILOT_INPUT_DIR)
            return
        if path.startswith("/files/result/"):
            self._serve_file(RESULTS_DIR / path.removeprefix("/files/result/"), RESULTS_DIR)
            return
        if path == "/":
            path = "/index.html"
        self._serve_file(WEB_DIR / path.lstrip("/"), WEB_DIR)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/infer", "/api/sample"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/api/sample":
                images = list_images()
                total = int(request.get("total", 12))
                seed = int(request.get("seed", 42))
                self._json({
                    "images": stratified_sample(images, total, seed),
                    "pool": pool_summary(images),
                    "total": total,
                    "per_type": total // len(FITZPATRICK_TYPES),
                    "seed": seed,
                })
                return
            requested_models = request.get("model_ids", [])
            requested_images = request.get("image_ids", [])
            models = {model["id"]: model for model in load_models()}
            images = {image["id"]: image for image in list_images()}
            if not requested_models or not requested_images:
                raise ValueError("Selecciona al menos un modelo y una imagen.")
            if len(requested_models) != len(set(requested_models)):
                raise ValueError("La selección contiene modelos duplicados.")

            results = []
            for model_id in requested_models:
                if model_id not in models:
                    raise ValueError(f"Modelo desconocido: {model_id}")
                for image_id in requested_images:
                    if image_id not in images:
                        raise ValueError(f"Imagen desconocida: {image_id}")
                    image_record = images[image_id]
                    current_image_path = image_path(image_record)
                    lesion_path = RESULTS_DIR / model_id / image_id / "lesion_mask.png"
                    runtime = None
                    if not lesion_path.is_file():
                        success, message, runtime = run_adapter(
                            models[model_id], current_image_path, lesion_path
                        )
                        if not success:
                            results.append({
                                "model_id": model_id,
                                "image_id": image_id,
                                "status": "adapter_unavailable",
                                "message": message,
                                "original_url": image_record["url"],
                                "runtime": runtime,
                                "image_metadata": image_record,
                            })
                            continue
                    results.append(result_payload(
                        model_id,
                        image_id,
                        current_image_path,
                        runtime,
                        original_url=image_record["url"],
                        image_metadata=image_record,
                    ))
            self._json({"results": results, "summaries": benchmark_summaries(results)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PILOT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FITZPATRICK_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Segmentation review: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

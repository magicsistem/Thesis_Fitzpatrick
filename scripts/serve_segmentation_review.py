"""Serve the local segmentation comparison and mask-review interface."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
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
INPUT_DIR = REPO_ROOT / "data" / "processed" / "isic_segmentation_pilot_resized_inference"
RESULTS_DIR = REPO_ROOT / "results" / "segmentation_benchmark"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_models() -> list[dict]:
    payload = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    return sorted(payload["models"], key=lambda model: model["priority"])


def list_images() -> list[dict]:
    if not INPUT_DIR.exists():
        return []
    return [
        {"id": path.stem, "filename": path.name, "url": f"/files/input/{path.name}"}
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def result_payload(model_id: str, image_id: str, image_path: Path) -> dict:
    output_dir = RESULTS_DIR / model_id / image_id
    lesion_path = output_dir / "lesion_mask.png"
    skin_path = output_dir / "clean_skin_mask.png"
    stats_path = output_dir / "colour_stats.json"
    if not lesion_path.is_file():
        return {
            "model_id": model_id,
            "image_id": image_id,
            "status": "mask_missing",
            "message": "El adaptador todavía no produjo lesion_mask.png.",
            "original_url": f"/files/input/{image_path.name}",
        }

    image = Image.open(image_path).convert("RGB")
    lesion = normalize_binary_mask(Image.open(lesion_path), image.size)
    clean_skin, metadata = build_clean_skin_mask(image, lesion)
    stats = measure_skin_colour(image, clean_skin)
    save_binary_mask(lesion, lesion_path)
    save_binary_mask(clean_skin, skin_path)
    stats_payload = {
        "postprocessing": metadata,
        "skin_colour": stats.to_dict(),
        "warning": "Fitzpatrick no se deduce únicamente a partir del color de la imagen.",
    }


def run_adapter(model: dict, image_path: Path, lesion_path: Path) -> tuple[bool, str]:
    """Run a configured adapter without invoking a shell."""
    template = model.get("adapter_command")
    if not template:
        return False, "El modelo está catalogado, pero todavía no tiene adaptador configurado."
    if not isinstance(template, list) or not all(isinstance(token, str) for token in template):
        return False, "adapter_command debe ser una lista JSON de argumentos."
    lesion_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "{image}": str(image_path),
        "{lesion_mask}": str(lesion_path),
        "{repo_root}": str(REPO_ROOT),
    }
    command = [replacements.get(token, token) for token in template]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "sin detalle"
        return False, f"El adaptador terminó con código {completed.returncode}: {detail[-800:]}"
    if not lesion_path.is_file():
        return False, "El adaptador terminó, pero no creó lesion_mask.png."
    return True, "ok"
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    base = f"/files/result/{model_id}/{image_id}"
    return {
        "model_id": model_id,
        "image_id": image_id,
        "status": "ready",
        "original_url": f"/files/input/{image_path.name}",
        "lesion_mask_url": f"{base}/lesion_mask.png",
        "clean_skin_mask_url": f"{base}/clean_skin_mask.png",
        "stats": stats_payload,
    }


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
            self._json({"models": load_models(), "images": list_images()})
            return
        if path.startswith("/files/input/"):
            self._serve_file(INPUT_DIR / path.removeprefix("/files/input/"), INPUT_DIR)
            return
        if path.startswith("/files/result/"):
            self._serve_file(RESULTS_DIR / path.removeprefix("/files/result/"), RESULTS_DIR)
            return
        if path == "/":
            path = "/index.html"
        self._serve_file(WEB_DIR / path.lstrip("/"), WEB_DIR)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/infer":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            requested_models = request.get("model_ids", [])
            requested_images = request.get("image_ids", [])
            models = {model["id"]: model for model in load_models()}
            images = {image["id"]: image for image in list_images()}
            if not requested_models or not requested_images:
                raise ValueError("Selecciona al menos un modelo y una imagen.")
            if len(requested_models) > 10:
                raise ValueError("El máximo es 10 modelos por ejecución.")

            results = []
            for model_id in requested_models:
                if model_id not in models:
                    raise ValueError(f"Modelo desconocido: {model_id}")
                for image_id in requested_images:
                    if image_id not in images:
                        raise ValueError(f"Imagen desconocida: {image_id}")
                    image_path = INPUT_DIR / images[image_id]["filename"]
                    lesion_path = RESULTS_DIR / model_id / image_id / "lesion_mask.png"
                    if not lesion_path.is_file():
                        success, message = run_adapter(models[model_id], image_path, lesion_path)
                        if not success:
                            results.append({
                                "model_id": model_id,
                                "image_id": image_id,
                                "status": "adapter_unavailable",
                                "message": message,
                                "original_url": f"/files/input/{image_path.name}",
                            })
                            continue
                    results.append(result_payload(model_id, image_id, image_path))
            self._json({"results": results})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Segmentation review: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

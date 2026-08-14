#!/usr/bin/env python3
"""Fine-tune one existing S01-S15 backend on a common P0 fold, resumably."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, load_benchmark_methods, load_json, sha256_file  # noqa: E402


def arguments_from_command(method: dict) -> tuple[Path, dict[str, str]]:
    command = method["adapter_command"]
    script = Path((command[command.index("python") + 1] if "python" in command else next(value for value in command if value.endswith(".py"))).replace("{repo_root}", str(REPO_ROOT)))
    values = {}
    index = 0
    while index < len(command) - 1:
        if command[index].startswith("--") and not command[index + 1].startswith("--"):
            values[command[index][2:].replace("-", "_")] = command[index + 1].replace("{repo_root}", str(REPO_ROOT))
            index += 2
        else: index += 1
    return script if script.is_absolute() else REPO_ROOT / script, values


def load_adapter(method: dict, device: str):
    script, values = arguments_from_command(method)
    spec = importlib.util.spec_from_file_location(f"b2_{method['backend_id'].replace('-', '_')}", script)
    module = importlib.util.module_from_spec(spec)
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    spec.loader.exec_module(module)
    signature = inspect.signature(module.load_model)
    kwargs = {}
    for name in signature.parameters:
        if name == "device": kwargs[name] = device
        elif name in values: kwargs[name] = Path(values[name]) if name in {"source", "checkpoint"} else values[name]
        else: raise ValueError(f"No se puede resolver load_model({name}) para {method['method_id']}")
    torch, model = module.load_model(**kwargs)
    model.to(device)
    return torch, model, module, values


def forward_logits(model, tensor, module_name: str):
    if module_name.endswith("avit"): output = model(tensor, d="0")["seg"]
    elif module_name.endswith("delightsam"): output = model(x=tensor, domain_seq=0)[0]
    else: output = model(tensor)
    if isinstance(output, dict): output = output["seg"] if "seg" in output else output["out"] if "out" in output else next(iter(output.values()))
    if isinstance(output, (tuple, list)): output = output[0]
    return output


def preprocess(module, image: Image.Image, values: dict[str, str]) -> np.ndarray:
    parameters = inspect.signature(module.preprocess).parameters
    kwargs = {}
    if "dataset" in parameters: kwargs["dataset"] = values["dataset"]
    if "imagenet_normalized" in parameters: kwargs["imagenet_normalized"] = False
    return module.preprocess(image, **kwargs)


def save_inference_checkpoint(torch, model, module_name: str, output: Path) -> None:
    state = model.state_dict()
    if module_name.endswith("avit"): payload = {"model_weights": state}
    elif module_name.endswith("unixio_isic2018"): payload = {"model_state": state}
    else: payload = state
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True); parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path); parser.add_argument("--folds", required=True, type=Path); parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--p0-root", required=True, type=Path, help="Precomputed common-P0 artifact root; raw images are never substituted")
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_training: raise SystemExit("Entrenamiento B2 bloqueado; revise --help y repita con --confirm-training.")
    methods = load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json")
    method = next((value for value in methods if value["method_id"] == args.method and value["kind"] == "neural"), None)
    if method is None: raise SystemExit("--method debe ser S01-S15")
    manifest, folds = load_json(args.manifest), load_json(args.folds)
    if manifest.get("split") != "train": raise SystemExit("B2 solo entrena desde split=train")
    fold = next((value for value in folds["folds"] if value["fold"] == args.fold), None)
    if fold is None: raise SystemExit("Fold inexistente")
    p0_manifest_path = args.p0_root / "oof_manifest.json"
    p0_manifest = load_json(p0_manifest_path)
    if p0_manifest.get("status") != "completed" or p0_manifest.get("source_manifest_sha256") != sha256_file(args.manifest) or p0_manifest.get("folds_sha256") != sha256_file(args.folds):
        raise SystemExit("El manifest P0 OOF no corresponde al manifest/folds de este entrenamiento B2")
    _script, configured_values = arguments_from_command(method)
    native_checkpoint = Path(configured_values["checkpoint"])
    contract = {
        "method_id": method["method_id"], "backend_id": method["backend_id"], "fold": args.fold,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed,
        "source_manifest_sha256": sha256_file(args.manifest), "folds_sha256": sha256_file(args.folds),
        "p0_manifest_sha256": sha256_file(p0_manifest_path), "parent_native_checkpoint_sha256": sha256_file(native_checkpoint),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "training_state.pt"
    checkpoint = args.output / f"{method['method_id']}.fold{args.fold}.b2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
    if checkpoint.is_file() and metadata_path.is_file():
        metadata = load_json(metadata_path)
        if metadata.get("status") == "completed" and metadata.get("training_contract") == contract and metadata.get("checkpoint_sha256") == sha256_file(checkpoint):
            print(json.dumps({**metadata, "reused": True}, indent=2)); return
    if state_path.exists() and not args.resume:
        raise SystemExit("Existe training_state.pt; solicite --resume para validarlo, no se sobrescribirá")
    by_id = {item["image_id"]: item for item in manifest["items"]}
    def collect(ids):
        collected = []
        for image_id in ids:
            item = by_id[image_id]
            candidates = list(args.p0_root.glob(f"*/{image_id}/*/roi_input.png"))
            if len(candidates) != 1: raise SystemExit(f"Se esperaba exactamente un P0 congelado para {image_id}; encontrados {len(candidates)}")
            if not item.get("mask_paths"): raise SystemExit(f"Falta máscara autorizada de training: {image_id}")
            collected.append((image_id, candidates[0], args.data_root / item["mask_paths"][0]))
        return collected
    samples = collect(fold["train_ids"])
    validation_samples = collect(fold["validation_ids"])
    random.seed(args.seed); np.random.seed(args.seed)
    torch, model, module, values = load_adapter(method, args.device)
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("--device cuda solicitado, pero CUDA no está disponible; use cpu")
    torch.manual_seed(args.seed)
    if args.device == "cuda": torch.cuda.manual_seed_all(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0; best_validation_loss = float("inf"); best_epoch = None
    if args.resume:
        if not state_path.is_file(): raise SystemExit("--resume solicitado, pero training_state.pt no existe")
        state = torch.load(state_path, map_location=args.device)
        if state.get("schema_version") != 2 or state.get("training_contract") != contract:
            raise SystemExit("training_state.pt no pertenece a este método/fold/configuración")
        if not 0 <= int(state.get("epoch", -1)) < args.epochs:
            raise SystemExit("training_state.pt contiene una época inválida")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start_epoch = int(state["epoch"]) + 1; best_validation_loss = float(state.get("best_validation_loss", best_validation_loss)); best_epoch = state.get("best_epoch")
    history = list(state.get("history", [])) if args.resume else []

    def tensors(image_path, mask_path):
        with Image.open(image_path) as opened: values_np = preprocess(module, ImageOps.exif_transpose(opened).convert("RGB"), values)
        with Image.open(mask_path) as opened: mask = np.asarray(opened.convert("L"), dtype=np.uint8)
        p0_manifest = load_json(image_path.parent / "preprocessing_manifest.json")
        x0, y0, x1, y1 = p0_manifest["expanded_bbox_original"]
        mask = mask[y0:y1, x0:x1]
        tensor = torch.from_numpy(values_np).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
        return tensor, mask

    def loss_for(logits, mask):
        target_np = np.asarray(Image.fromarray(mask).resize((logits.shape[-1], logits.shape[-2]), Image.Resampling.NEAREST), dtype=np.float32) / 255
        target = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).to(args.device)
        if logits.shape[1] == 2: return torch.nn.functional.cross_entropy(logits, target[:, 0].long())
        if bool((logits.detach().min() >= 0) & (logits.detach().max() <= 1)): return torch.nn.functional.binary_cross_entropy(logits, target)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        random.Random(args.seed + epoch).shuffle(samples)
        losses = []
        for _, image_path, mask_path in samples:
            tensor, mask = tensors(image_path, mask_path)
            generator = random.Random(args.seed + epoch + sum(map(ord, image_path.name)))
            if generator.random() < 0.5: tensor = torch.flip(tensor, dims=(-1,)); mask = np.flip(mask, axis=1).copy()
            if generator.random() < 0.5: tensor = torch.flip(tensor, dims=(-2,)); mask = np.flip(mask, axis=0).copy()
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, tensor, module.__name__)
            loss = loss_for(logits, mask)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); validation_losses = []
        with torch.inference_mode():
            for _, image_path, mask_path in validation_samples:
                tensor, mask = tensors(image_path, mask_path); validation_losses.append(float(loss_for(forward_logits(model, tensor, module.__name__), mask).cpu()))
        model.train(); validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_loss": validation_loss, "training_samples": len(losses), "validation_samples": len(validation_losses)})
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch = validation_loss, epoch; save_inference_checkpoint(torch, model, module.__name__, checkpoint)
        state_tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save({"schema_version": 2, "training_contract": contract, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best_validation_loss": best_validation_loss, "best_epoch": best_epoch}, state_tmp)
        os.replace(state_tmp, state_path)
        atomic_write_json(args.output / "training_history.json", history)
    metadata = {"schema_version": 2, "status": "completed", "protocol": "B2", "backend_id": method["backend_id"], "method_id": method["method_id"], "fold": args.fold, "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate, "device": args.device, "best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "augmentations": ["horizontal_flip_p0.5", "vertical_flip_p0.5"], "checkpoint_sha256": sha256_file(checkpoint), "parent_native_checkpoint_sha256": sha256_file(Path(values["checkpoint"])), "source_manifest": str(args.manifest), "source_manifest_sha256": sha256_file(args.manifest), "folds": str(args.folds), "folds_sha256": sha256_file(args.folds), "p0_root": str(args.p0_root), "p0_cache_keys": sorted(path.parent.name for _, path, _ in samples + validation_samples), "training_contract": contract, "completed_epochs": len(history), "completed_utc": datetime.now(timezone.utc).isoformat(), "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    atomic_write_json(checkpoint.with_suffix(checkpoint.suffix + ".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()

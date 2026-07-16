# Segmentation benchmark and review interface

## Scientific objective

The benchmark compares lesion-segmentation models on the same images. Every
successful model/image pair must produce exactly one canonical binary lesion
mask named `lesion_mask.png`.

The project then derives `clean_skin_mask.png`. This second mask is not simply
the inverse of the lesion. It restricts sampling to nearby visible skin and
excludes a lesion safety margin, black field-of-view pixels, likely specular
highlights, and narrow local dark structures compatible with hair.

Both masks require visual quality control. A clean-skin mask can still contain
markers, rulers, gel, colour shifts, or transparent dermatoscope-tip artifacts.

## Model shortlist

The first five models to integrate and test are:

1. AViT.
2. ISCF.
3. SkinMamba.
4. UltraLight VM-UNet.
5. UCM-Net.

EGE-UNet, MALUNet, MHorUNet, HSH-UNet, and MobileSAM are secondary
comparators. MobileSAM remains a prompt-based baseline and must not be mixed
conceptually with the prompt-free supervised segmenters. TMUNet, Attention
DeepLabv3+, and BCDU-Net were removed from the live catalogue after GitHub
reported their original repositories as unavailable on 2026-07-16.

The catalogue at `configs/segmentation_models.json` distinguishes public
checkpoint availability from local integration. A model is not runnable until
its adapter has been reproduced and validated.

## Adapter contract

Each adapter receives one image path and must write a binary mask at the exact
path represented by `{lesion_mask}`. Commands are JSON argument lists and are
executed without a shell. Supported placeholders are:

- `{image}`: absolute input-image path.
- `{lesion_mask}`: required output path.
- `{repo_root}`: absolute project root.
- `{python}`: interpreter that started the review server.
- `{conda}`: Conda executable, resolved without invoking a shell.

Example after an adapter has been implemented:

```json
"adapter_command": [
  "conda", "run", "-n", "seg-model", "python",
  "scripts/adapters/run_model.py",
  "--image", "{image}",
  "--output", "{lesion_mask}"
]
```

Do not configure a command until the official checkpoint inference has been
reproduced on its original test data and on a small local pilot.

## Start the interface

From the repository root with the `tesis-sam` environment active:

```bash
python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The page selects 1 to all downloaded images and
up to 10 models. It displays a carousel with:

1. original image;
2. lesion mask;
3. clean-skin mask.

If an adapter is not configured, the interface reports that state instead of
fabricating an inference.

## AViT on CPU

AViT runs in a separate environment so that `tesis-sam` is not modified. The
official source is pinned to commit
`b77b01af263727b2b2cf2555a5ed9f1f48a2b4a2`; its public ISIC 2018 checkpoint
is verified with SHA-256 before every inference.

```bash
mamba env create -f configs/environments/avit-cpu.yml
conda run --no-capture-output -n thesis-avit python scripts/setup_avit_model.py
```

The first inference loads a 398 MB checkpoint and is slower than a warm model
would be. Each web request currently starts an isolated adapter process, so the
reported time intentionally includes model loading. Inference uses the official
224 × 224 ImageNet-normalized evaluation preprocessing; the binary mask is then
resized to the original image dimensions with nearest-neighbour interpolation.

## UltraLight VM-UNet on CPU

The second runnable model reuses the `thesis-avit` environment. Its official
MIT-licensed source is pinned to commit
`27e44181b3cd5b0b2ab3ad1e3ddb8cad67368fbd`. The 229 KB checkpoint linked by
the author is verified before use. The publication repository depends on a
CUDA Mamba kernel, so this project evaluates the same Mamba-1 state recurrence
with a plain PyTorch CPU reference implementation and unchanged parameter
names.

```bash
conda run --no-capture-output -n thesis-avit \
  python scripts/setup_ultralight_vm_unet.py
```

The author describes the downloadable file only as weights trained on skin
lesions and does not identify ISIC 2017 versus ISIC 2018 in the download issue.
Results from that checkpoint must therefore retain `dataset unspecified` as a
provenance limitation.

## Runtime measurements

For every adapter invocation, the server records:

- wall time for the individual image;
- child-process CPU seconds;
- CPU percentage normalized to the machine's full logical-core capacity
  (0–100%);
- effective logical cores used on average, which may be greater than one;
- approximate peak resident memory for the adapter process tree.

The Linux `/proc` filesystem is sampled every 0.05 seconds for memory. RAM is
therefore an approximation and very short peaks can be missed. CPU comes from
the operating system's child-process resource accounting. The interface shows
per-image measurements and a per-model summary with total time, average time,
average CPU, and maximum observed RAM.

Metrics are saved beside each mask in `runtime_metrics.json`. They describe the
local machine and environment used for that run and must not be compared with
published GPU results as if the hardware were equivalent.

## Post-process an existing lesion mask

```bash
python scripts/postprocess_segmentation_masks.py \
  --image data/processed/isic_segmentation_pilot_resized_inference/ISIC_ID.jpg \
  --lesion-mask results/segmentation_benchmark/MODEL/ISIC_ID/source_mask.png \
  --output-dir results/segmentation_benchmark/MODEL/ISIC_ID
```

The output also includes `colour_stats.json` with pixel count, coverage, robust
RGB, CIELAB, and individual typology angle (ITA) summaries.

## Fitzpatrick interpretation

Fitzpatrick type describes a person's tendency to burn and tan after sun
exposure. It is not a direct skin-colour scale. RGB, CIELAB, or ITA values can be
compared with an existing Fitzpatrick metadata label as an exploratory
concordance analysis, but they must not be reported as a replacement
Fitzpatrick diagnosis.

For quantitative work, keep acquisition modality fixed, use colour calibration
when available, report whether the clean-skin fallback was used, and prefer
median or trimmed-mean colour over an unrestricted arithmetic mean.

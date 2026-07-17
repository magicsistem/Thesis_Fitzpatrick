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

The live catalogue contains 15 runnable, checkpoint-verified variants:

1. the original six: AViT, UltraLight VM-UNet, BA-Transformer,
   U-Net/ResNet34 and SkinMamba ISIC 2017/2018;
2. Unixio/BertinAm U-Net, U-Net++ and Attention U-Net;
3. Theodore Ioannidis U-Net, Inception and SegFormer-B0;
4. VM-UNet ISIC 2017 and ISIC 2018;
5. De-LightSAM Dermoscopy.

Variants that share an architecture intentionally retain their different
training domains. Ten are preselected, matching the maximum per run; every
other verified model remains selectable after unchecking another one. Models
without a retrievable public checkpoint were removed. MobileSAM remains a
separate prompt-based pilot and is not mixed with these automatic supervised
segmenters. Every candidate, download attempt, hash, exclusion and decision is recorded in
[`SEGMENTATION_CHECKPOINT_AUDIT.md`](SEGMENTATION_CHECKPOINT_AUDIT.md).

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

Open <http://127.0.0.1:8000>. The page can create an exact Fitzpatrick-balanced
sample from the locally downloaded full cohort. The requested total must be a
multiple of six; 12 therefore means two images from every type I–VI. The seed
makes the selection reproducible. Up to 10 models can be selected. In the
results, first select one model and then move image by image within that model.
Every slide displays:

1. original image;
2. lesion mask;
3. clean-skin mask.

If an adapter is not configured, the interface reports that state instead of
fabricating an inference.

## Full Fitzpatrick image pool

The interface joins local image filenames to
`data/raw/isic_fitzpatrick_metadata_full.csv`. It reads images from both the
93-image pilot and `data/raw/isic_fitzpatrick_images`, de-duplicating by ISIC
identifier. Only local images with a valid I–VI metadata label are eligible for
balanced sampling.

To download every image listed by the audited metadata, use the `isic-api`
environment, which contains the ISIC CLI. This can require substantial disk
space and network time; the downloader is batch-based and skips files that are
already complete.

```bash
conda run --no-capture-output -n isic-api \
  python scripts/download_segmentation_pilot_images.py \
  --manifest data/raw/isic_fitzpatrick_metadata_full.csv \
  --output-dir data/raw/isic_fitzpatrick_images \
  --batch-size 50 \
  --sleep-seconds 0.5
```

The pool summary in the page reports how many local images are available for
each Fitzpatrick type. If a requested stratum is too small, the server rejects
the sample and states which type is missing images; it never fills the deficit
from another type.

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

## BA-Transformer on CPU

The third runnable model reuses `thesis-avit`. Its official source is pinned to
commit `d1ccdff68beac82d18c2cd64bb800e51d7afc5e1`. The published archive is
verified before extracting `isic2016/.../best.pkl`, and the extracted checkpoint
is verified again with SHA-256. The repository has no declared software license.
After successful extraction and verification, the setup script removes the
downloaded ZIP so that only the checkpoint remains on disk.

```bash
conda run --no-capture-output -n thesis-avit \
  python scripts/setup_ba_transformer.py
```

The official README labels the Google Drive download as PH2, but the archive
itself contains two directories named `isic2016` and the integrated checkpoint
comes from one of them. The catalogue therefore records the checkpoint as ISIC
2016 and preserves the README inconsistency as a provenance note. Inference
matches the repository's 352 × 352 BGR, 0–1 preprocessing and restores the
binary mask to the source dimensions.

The official ResNet-50 constructor contains the author's absolute cache path,
`/home/wjc/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`. The adapter
supplies a shape-compatible blank ResNet state only while constructing the
network and then restores `torch.load`. Immediately afterwards it loads the
verified full BA-Transformer checkpoint with `strict=True`, so no blank
bootstrap parameters remain in the inference model and no extra ResNet download
is required. Before that strict load, the adapter also removes the `module.`
prefix produced by the original multi-GPU training, matching the official
checkpoint loader.

## U-Net/ResNet34 on CPU

The fourth runnable comparator is a conventional U-Net with a ResNet34 encoder,
trained by a community project on ISIC 2018 Task 1 at 256 × 256. Its MIT-licensed
training repository reports Dice 0.8982 and IoU 0.8293 on an 80/20 split. These
are author-reported validation values, not an independent evaluation and not a
new peer-reviewed architecture; the project uses it as a transparent classical
baseline. The 94 MB checkpoint is fetched from an immutable Hugging Face model
revision and verified with SHA-256.

Install the single additional runtime dependency into the existing CPU
environment, then download the checkpoint:

```bash
conda run --no-capture-output -n thesis-avit \
  python -m pip install --no-deps segmentation-models-pytorch==0.5.0
conda run --no-capture-output -n thesis-avit \
  python scripts/setup_unet_resnet34.py
```

`--no-deps` deliberately preserves the already verified CPU builds of PyTorch,
Torchvision, timm, Hugging Face Hub, and safetensors in `thesis-avit`.

## SkinMamba on CPU

SkinMamba is exposed as separate ISIC 2017 and ISIC 2018 variants. The setup
script pins the Apache-2.0 source to commit
`131a85da14da4d3bf135b52912d274191c05f3b8`, downloads the authors' ZIP once,
and verifies the archive plus both 54 MB checkpoints. The official model imports
a CUDA selective-scan extension; the adapter uses the repository's equivalent
chunked PyTorch recurrence, which keeps inference available on this CPU-only
machine without changing learned parameters.

```bash
conda run --no-capture-output -n thesis-avit \
  python scripts/setup_skinmamba.py
```

CPU inference can be substantially slower than the CUDA kernel. Runtime and RAM
remain measured per image by the same benchmark process.

## Additional verified ISIC 2018 families

Unixio/BertinAm publishes U-Net, U-Net++ and Attention U-Net checkpoints from
one ISIC 2018 experiment. Theodore Ioannidis publishes U-Net, Inception and
SegFormer-B0 from another. All six checkpoints are downloaded from immutable
Hugging Face revisions and checked with SHA-256:

```bash
mamba install -n thesis-avit -c conda-forge transformers
conda run --no-capture-output -n thesis-avit \
  python -m pip install --no-deps segmentation-models-pytorch==0.5.0
conda run --no-capture-output -n thesis-avit python scripts/setup_unixio_isic2018.py
conda run --no-capture-output -n thesis-avit python scripts/setup_theodore_isic2018.py
```

Theodore's U-Net and Inception use RGB 0–1 at 128 × 128; SegFormer additionally
uses ImageNet normalization. Unixio models use ImageNet normalization at
256 × 256 and resize the probability map before thresholding. No variant uses
automatic morphological cleanup, so the comparison remains attributable to
the checkpoint.

## VM-UNet on CPU

VM-UNet ISIC 2017 and ISIC 2018 use official Google Drive checkpoints and
source commit `b87827eb5a5faff00bd4b7b6505e56d3ca813ca2`. The adapter replaces
the CUDA selective-scan kernel with an equivalent plain-PyTorch recurrence.
This makes both variants executable without an NVIDIA GPU, but inference may
take minutes depending on the processor:

```bash
conda run --no-capture-output -n thesis-avit python scripts/setup_vmunet.py
```

## De-LightSAM Dermoscopy on CPU

De-LightSAM uses the official dermoscopy member of its six-modality model zoo,
source commit `d2260d75e4ad1f5da8e053711cd9c36128088a7b`, and modality index zero
for ISIC. It is automatic: the modality selector is not a spatial prompt. The
1024 × 1024 TinyViT path is expected to be the slowest CPU candidate, so it is
available but not preselected:

```bash
conda run --no-capture-output -n thesis-avit python scripts/setup_delightsam.py
```

No field-of-view correction or border removal is applied in this phase. The
original image, lesion prediction and derived clean-skin mask therefore expose
each model's raw behaviour around dermatoscope circles for manual comparison.

## Empty clean-skin masks

Some failed segmenters label the entire field of view as lesion. In that case
there are legitimately no candidate clean-skin pixels. The server still saves
and displays the original image, lesion mask, empty clean-skin mask, and runtime
metrics. It records colour statistics as unavailable instead of aborting the
whole multi-model request. It never fabricates RGB, Lab, ITA, or Fitzpatrick
values from an empty selection.

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

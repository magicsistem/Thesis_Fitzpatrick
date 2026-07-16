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

1. TMUNet.
2. Attention DeepLabv3+.
3. BCDU-Net.
4. AViT.
5. ISCF.

BA-Transformer, SkinMamba, and MobileSAM are secondary comparators. MobileSAM
remains a prompt-based baseline and must not be mixed conceptually with the
prompt-free supervised segmenters.

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

Example after an adapter has been implemented:

```json
"adapter_command": [
  "conda", "run", "-n", "seg-tmunet", "python",
  "scripts/adapters/run_tmunet.py",
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

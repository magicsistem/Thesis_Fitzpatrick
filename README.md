# Thesis_Fitzpatrick

Repository for thesis experiments on Fitzpatrick-aware skin lesion analysis using ISIC Archive metadata, lesion segmentation, and multiclasse classification.

## Main objective

Evaluate whether Fitzpatrick skin type information can be used to audit and improve skin-lesion classification workflows, especially for darker skin tones.

## Initial research direction

1. Audit ISIC Archive metadata filtered by `fitzpatrick_skin_type`.
2. Download a controlled subset of images after metadata validation.
3. Segment lesions using SAM/SAMed/MedSAM-style models.
4. Validate segmentation quality before using masks for classification.
5. Train and evaluate multiclasse classifiers with stratified metrics by Fitzpatrick skin type.
6. Compare baseline images versus segmented/cropped lesion images.

## Important external reference

ISIC Archive API documentation:

<https://api.isic-archive.com/api/docs/swagger/>

## Environment strategy

This project will use separated environments:

- `tesis-ml`: general analysis, notebooks, classical ML, lightweight experiments.
- `isic-api`: ISIC CLI/API access and metadata/image download.
- `segmentation`: PyTorch-based segmentation experiments.

Do not mix all dependencies into a single environment unless explicitly justified.

## Segmentation review interface

The local review interface compares up to 10 segmentation models over downloaded
images joined to the complete Fitzpatrick metadata. It can build reproducible,
balanced samples in multiples of six: for example, 12 selects exactly two local
images of each Fitzpatrick type I–VI. Each result displays the original image, a
lesion mask, and a derived clean-skin mask for colour measurement. Results are
selected by model first and then reviewed image by image. The model cards expose
year, declared license, GitHub source, and description. Once an adapter is
connected, results also report per-image and average execution time, CPU use,
and peak RAM. A model that predicts the whole image as lesion no longer aborts
multi-model review: its masks remain visible and its unavailable skin-colour
statistics are reported explicitly.

```bash
python scripts/serve_segmentation_review.py
```

Open <http://127.0.0.1:8000>. See
[`docs/SEGMENTATION_REVIEW_UI.md`](docs/SEGMENTATION_REVIEW_UI.md) for the model
shortlist, adapter contract, post-processing rules, and Fitzpatrick caveat. The
complete checkpoint search—including every rejected model, download attempt,
fixed commit and SHA-256—is in
[`docs/SEGMENTATION_CHECKPOINT_AUDIT.md`](docs/SEGMENTATION_CHECKPOINT_AUDIT.md).

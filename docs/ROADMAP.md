# Project Roadmap

## Phase 0 — Repository and environment setup

- Create reproducible project structure.
- Keep datasets, images, masks, checkpoints, and large outputs out of Git.
- Define separate environments for analysis, ISIC access, and segmentation.

## Phase 1 — ISIC-Fitzpatrick metadata audit

Goal: determine whether the ISIC Archive subset filtered by `fitzpatrick_skin_type` is scientifically usable.

Checks:

- Number of images by Fitzpatrick type I-VI.
- Diagnosis availability and class distribution.
- Image modality.
- Source collections.
- Licenses.
- Patient and lesion identifiers.
- Duplicates or repeated lesions.
- Balance of darker skin types IV-VI.

## Phase 2 — Controlled image download

Download only a small pilot subset first.

Suggested pilot:

- 10-20 images per Fitzpatrick group.
- Include diagnosis and collection metadata.
- Avoid downloading the full dataset before auditing metadata.

## Phase 3 — Segmentation pilot

Compare candidate segmentation approaches:

- SAM with prompts.
- MedSAM or SAM-Med-style models.
- SAMed only as an experimental branch due to older dependency stack and non-dermoscopy original target.

Segmentation outputs must be visually checked before being used for classification.

## Phase 4 — Segmentation validation

Validate masks using one or more of:

- Existing ISIC segmentation masks if available for matching images.
- Manual review on a small stratified subset.
- Dice/IoU where ground-truth masks exist.
- Failure analysis by Fitzpatrick group.

## Phase 5 — Classification and fairness evaluation

Compare:

- Original images.
- Cropped lesion images.
- Masked lesion images.
- Optional metadata-aware models.

Metrics:

- Global AUROC/AUPRC/F1/balanced accuracy.
- Per-class metrics.
- Per-Fitzpatrick metrics.
- Performance gap between lighter and darker skin groups.
- Calibration by Fitzpatrick group.

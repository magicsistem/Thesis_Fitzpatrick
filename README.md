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

# ISIC Access Plan

## Reference

Official ISIC Archive API documentation:

<https://api.isic-archive.com/api/docs/swagger/>

Official ISIC CLI:

<https://github.com/ImageMarkup/isic-cli>

## Core principle

The ISIC Archive subset filtered by `fitzpatrick_skin_type` must be treated as a metadata-filtered cohort, not as a single homogeneous dataset.

Before downloading images or training models, the metadata must be audited.

## Phase 1: Metadata audit

Questions to answer:

1. How many images have `fitzpatrick_skin_type` I, II, III, IV, V, or VI?
2. Which diagnoses are available?
3. Are the labels suitable for multiclass classification?
4. Which collections contribute the images?
5. Are there repeated patients or repeated lesions?
6. What image modalities are present?
7. What licenses apply?
8. Are skin types IV-VI sufficiently represented?

## Phase 2: Controlled pilot download

Download only a small pilot subset first.

Suggested initial subset:

- 10 to 20 images per Fitzpatrick group.
- Include metadata with diagnosis, collection, license, patient ID, and lesion ID where available.
- Do not download the full cohort until the metadata audit is complete.

## Phase 3: Segmentation pilot

Candidate segmentation approaches:

- SAM with prompts.
- MedSAM or SAM-Med-style models.
- SAMed as an experimental branch only.

SAMed must not be assumed correct for dermoscopy or clinical skin images without validation.

## Phase 4: Mask validation

Segmentation masks should be evaluated before being used for classification.

Possible validation routes:

- Compare against available ISIC ground-truth masks where possible.
- Manual review of a stratified subset.
- Dice/IoU if reference masks exist.
- Failure analysis by Fitzpatrick skin type.

## Phase 5: Downstream classification

Compare at least:

1. Original image classifier.
2. Cropped lesion classifier.
3. Masked lesion classifier.
4. Optional metadata-aware classifier.

Report metrics globally and stratified by Fitzpatrick group.

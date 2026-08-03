# Thesis_Fitzpatrick

Repository for Fitzpatrick-aware skin-lesion research using ISIC Archive
metadata, dermoscopic lesion segmentation, clean-skin colour measurements, and
later classification experiments.

## Scientific objective

Evaluate whether Fitzpatrick skin-type metadata can be used to audit and improve
skin-lesion analysis workflows, with particular attention to performance gaps
for types IV–VI. Fitzpatrick describes response to sun exposure; it is not a
direct skin-colour scale and is never inferred solely from image RGB, CIELAB, or
ITA measurements.

The current implementation covers the segmentation benchmark. It provides 15
checkpoint-verified model variants, a reproducible Fitzpatrick-balanced sampler,
binary lesion masks, derived clean-skin masks, runtime measurements, and a local
review interface. Generated masks remain predictions and require visual and
quantitative validation before downstream classification.

## Local environments

The project deliberately separates incompatible workloads:

- `tesis-sam`: review server, tests, mask post-processing, and the earlier
  MobileSAM prompt pilot.
- `thesis-avit`: the 15 supervised segmentation variants and their CPU adapters.
- `isic-api`: official ISIC CLI/API metadata and image downloads.
- `tesis-ml`: general analysis and later classification experiments.

Do not reinstall or merge these environments without first checking the
versions recorded in the phase handoff.

## Segmentation review interface

From the repository root in `fish`:

```fish
conda activate tesis-sam
python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The interface can select any subset or all 15
models and can build balanced image samples in multiples of six. Every result
shows the original image, `lesion_mask.png`, `clean_skin_mask.png`, available
metadata, execution time, CPU use, and peak RAM. A failed model/image pair does
not discard successful results from the same request.

The lesion predictions are currently raw checkpoint outputs: no field-of-view
correction is applied to them. The derived clean-skin mask does exclude nearly
black background pixels, highlights, a lesion margin, and likely hair so those
pixels are not used for colour statistics.

## Documentation map

- [`docs/phases/`](docs/phases/): one independent handoff for each completed
  project phase. The filename contains the phase number, closure date, and
  document version.
- [`docs/ISIC_FITZPATRICK_METADATA_SUMMARY.md`](docs/ISIC_FITZPATRICK_METADATA_SUMMARY.md):
  audited metadata counts and scientific interpretation.
- [`docs/MOBILE_SAM_PROMPT_PILOT_FINDINGS.md`](docs/MOBILE_SAM_PROMPT_PILOT_FINDINGS.md):
  evidence and decision from the prompt-based MobileSAM pilot.
- [`docs/DERMOSCOPY_SEGMENTATION_MODEL_SEARCH_2026.md`](docs/DERMOSCOPY_SEGMENTATION_MODEL_SEARCH_2026.md):
  literature and checkpoint search, including future FOV evaluation criteria.
- [`docs/SEGMENTATION_CHECKPOINT_AUDIT.md`](docs/SEGMENTATION_CHECKPOINT_AUDIT.md):
  fixed source commits, checkpoint hashes, licenses, exclusions, and setup
  commands.
- [`docs/SEGMENTATION_REVIEW_UI.md`](docs/SEGMENTATION_REVIEW_UI.md): interface,
  adapter, post-processing, and runtime-measurement details.

## Repository policy

Git tracks source code, tests, configuration, phase handoffs, and small
scientific summaries. ISIC images, raw/interim datasets, checkpoints, generated
masks, runtime outputs, caches, and other large artifacts remain local and are
excluded by `.gitignore`.

The executable model catalogue is `configs/segmentation_models.json`. The study
design and fixed paths are in `configs/isic_fitzpatrick_study.toml`.

## Validation

Run the canonical checks from `tesis-sam`:

```fish
python -m unittest discover -s tests -v

python -m py_compile \
    scripts/*.py \
    scripts/adapters/*.py \
    src/thesis_fitzpatrick/*.py

python -m json.tool configs/segmentation_models.json >/dev/null

git diff --check
```

The detailed, date-specific result belongs in the handoff for the corresponding
phase rather than in this README.

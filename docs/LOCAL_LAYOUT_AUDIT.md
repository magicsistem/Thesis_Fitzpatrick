# Local directory audit

The supplied local tree contains no random directory names. The extra folders
are meaningful outputs from earlier pilot runs, empty scaffolding, or Python
caches. Git tracks files, not directories, so empty local folders can remain
even when they do not appear on GitHub.

## Keep

- `data/raw/isic_segmentation_pilot_images`: downloaded pilot originals.
- `data/processed/isic_segmentation_pilot_resized_sensor`: sensor-size derivatives.
- `data/processed/isic_segmentation_pilot_resized_inference`: model inputs.
- `models/mobile_sam`: local MobileSAM checkpoint baseline.
- `reports/tables/*`: reproducibility summaries.
- `results/mobile_sam_prompt_pilot`: documented pilot evidence.
- `configs`, `docs`, `scripts`, `src`, `tests`, and `web`: project source.

## Safe to remove when empty or regenerated

- `scripts/__pycache__`.
- empty `notebooks` and `references` directories.
- empty `data/external` and `reports/figures` placeholders.

## Review before deleting

- `data/raw/isic_download_test`.
- `data/raw/isic_download_test_multi`.
- `results/mobile_sam_prompt_test`.
- `results/mobile_sam_single_test`.

These appear to be superseded tests, but they may contain the only record of an
earlier check. Inspect file counts and sizes before removal:

```bash
du -sh \
  data/raw/isic_download_test \
  data/raw/isic_download_test_multi \
  results/mobile_sam_prompt_test \
  results/mobile_sam_single_test

find \
  data/raw/isic_download_test \
  data/raw/isic_download_test_multi \
  results/mobile_sam_prompt_test \
  results/mobile_sam_single_test \
  -maxdepth 2 -type f -printf '%p\n' | sort
```

No script should create an output outside `data`, `reports/tables`, or
`results`. `_project_paths.py` now resolves relative paths from the repository
root and rejects unexpected output roots before creating a directory.

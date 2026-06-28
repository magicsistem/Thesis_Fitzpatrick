# Technical Notes

## ISIC Archive

Reference API documentation:

<https://api.isic-archive.com/api/docs/swagger/>

For bulk or filtered access, prefer the official ISIC CLI or documented API usage rather than ad-hoc scraping.

## SAMed / SAM-Med-style segmentation

SAMed is not automatically guaranteed to work well on dermoscopy or clinical skin images. It must be validated before using its masks for downstream classification.

Important risks:

- Incorrect masks may introduce bias.
- Segmentation may perform worse on darker skin.
- Automatic masks should not be treated as ground truth.
- Dependency stack may require a separate environment.

## Repository policy

Do not commit:

- ISIC images.
- Downloaded datasets.
- Masks.
- Model checkpoints.
- Large result files.

Commit only:

- Code.
- Configuration.
- Documentation.
- Small metadata summaries.
- Reproducible scripts.

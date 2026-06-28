# Project Workflow

## General workflow

This repository will be developed with a step-by-step verification workflow.

For technical tasks:

1. Define the objective clearly.
2. Make one small change or run one small command block.
3. Inspect the output.
4. Commit only validated changes.
5. Push changes to GitHub.
6. Review the pushed code before continuing.

## Codex-assisted development

When code is needed, it can be generated or modified with Codex inside VS Code.

Workflow:

1. Ask Codex for a specific, limited change.
2. Do not let Codex modify unrelated files.
3. Review local changes with `git status` and `git diff`.
4. Run the smallest relevant validation.
5. Commit and push only after validation.
6. Use GitHub as the review point before continuing.

ChatGPT can review the pushed repository state through GitHub before recommending the next step.

## Repository safety

Do not commit:

- ISIC images.
- Downloaded datasets.
- Generated masks.
- Model checkpoints.
- Large result files.
- API tokens or credentials.

Commit:

- Source code.
- Configuration files.
- Documentation.
- Small metadata summaries.
- Reproducibility scripts.

## Contributing

Thank you for your interest in contributing! This project provides a reproducible object detection pipeline for document images using DVC.

### Workflow
- Create a feature branch from `master`.
- Keep changes focused and small; open draft PRs early.
- Add/update docs and tests where relevant.

### Code style
- Python 3.10+.
- Follow clear naming and early-return patterns.
- Add docstrings for modules and functions.
- Avoid deep nesting and silent exception handling.

### DVC/Git hygiene
- Track large data with DVC (`python -m dvc add ...`), keep Git clean.
- Commit `params.yaml`, `dvc.yaml`, and code with meaningful messages.
- Use `python -m dvc exp run --queue` for experiments; document noteworthy findings in PRs.

### Reporting issues
- Include OS/Python versions, steps to reproduce, and expected vs actual behavior.






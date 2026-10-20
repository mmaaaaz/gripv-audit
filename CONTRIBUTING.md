# Contributing

Thanks for helping make benchmark keys auditable.

## Ground rules

1. **No dataset content.** Never commit questions, answer keys, `params`,
   videos, frames, or a populated `cache/`/`results/` directory. The dataset is
   private and access is granted by its owner.
2. **Compute, don't copy.** A contract must derive the answer from `params`.
   Hard-coding a key is not a test; it hides the bug you are looking for.
3. **Calibrate before flagging.** A contract must reproduce at least two anchor
   clips of its category before a full run. A `mismatch` is a hypothesis until
   it is checked against more clips.
4. **Fail loudly.** If a template is not handled, return
   `Answer(None, confidence="unsupported")`. Do not return a plausible guess.
5. **Keep the diff small.** One category per pull request, plus its tests and
   registry entry.

## Pull request checklist

- [ ] `uv run pytest` passes.
- [ ] `uv run python -m gripv.run --list` shows the new category.
- [ ] A full run writes `results/<category>.csv` with no unexplained
      `mismatch`.
- [ ] The PR body states the category, the templates handled, the anchor clips
      used for calibration (clip ids only, no keys), and the residual
      unsupported templates.
- [ ] No dataset content, no secrets, no generated caches in the diff.

## Code style

- Python 3.10+, type hints on public functions.
- Standard library first; the only required third-party packages are pandas,
  pyarrow and duckdb.
- Comments explain *why* a convention is used, with the observed evidence.
- Avoid clever one-liners in event math; readable beats short.

## Reporting a security issue

See [`SECURITY.md`](SECURITY.md). Do not open a public issue for a leaked token
or a dataset exposure.

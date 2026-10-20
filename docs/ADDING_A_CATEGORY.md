# Adding a category

## 1. Learn the recipe

Fetch a few clips' `params` and print the structure. Do not guess field names.
Then read the category's questions and group them by template (`qid` is a good
first cut).

```bash
python -m gripv.run --category <category> --limit 20
```

The first run downloads params into `cache/<category>.params.json`. Keep that
file out of git.

## 2. Write the contract

Create `gripv/contracts/<category>.py`:

```python
from ..engine import Answer


class MyCategory:
    def __init__(self, params: dict):
        self.p = params

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        if "how many" in q:
            return Answer(str(self.count()), confidence="exact")
        return Answer(None, confidence="unsupported", note="template not handled")
```

Rules:

- `params` is the only source of truth. Do not read the answer key inside the
  contract.
- Return `unsupported` rather than a guess.
- Return `uncertain` when the answer depends on an unstated convention.
- `Answer.strict` is required for a finding; `Answer.loose` is optional.
- Keep event helpers small and testable (`fall_time`, `torque`, `gate_frame`).

## 3. Register it

```python
# gripv/contracts/__init__.py
from .my_category import MyCategory

CONTRACTS = {..., "my_category": MyCategory}
```

## 4. Calibrate on anchors

Pick at least two clips:

- one ordinary clip;
- one clip with the hardest template you intend to support.

Run only those:

```bash
python -m gripv.run --category my_category --limit 20
python -m gripv.run --category my_category --qid q07
```

The contract must reproduce the key for the templates it claims to handle. If it
does not, fix the contract before running the full category. Do not “fix” the
key.

## 5. Run the category

```bash
python -m gripv.run --category my_category --out results/my_category.csv
```

Inspect every `mismatch`. Before calling one an error, check at least two more
clips of the same template and confirm the convention. If the template is not
consistent, mark it `uncertain` and move on.

## 6. Add a regression anchor

Add a synthetic unit test for the helper logic. Do **not** commit a real clip's
params or key as a fixture. If the helper cannot be tested without real data,
that is a design smell: split the pure computation out.

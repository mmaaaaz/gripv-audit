# GRIP-V audit — deterministic params replay

[![CI](https://github.com/mmaaaaz/gripv-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/mmaaaaz/gripv-audit/actions/workflows/ci.yml)

A small, auditable harness for checking the **answer keys** of the GRIP-V
video-geometric-reasoning benchmark.

Every clip ships the generator's full recipe in a `params` field. Instead of
asking a model to watch the video and guess, this harness **rebuilds the event
timeline from the recipe in pure Python** and evaluates each question against
the literal wording. A key that disagrees is reproducible from the data alone:
no language model, no sampling, no opinion.

This is the Layer-1 engine behind the audit. It is deliberately boring: a
contract per category, a table of statuses, and a CSV you can inspect.

> **Private data.** GRIP-V is not public. This repository contains **code only** —
> no questions, answer keys, params, videos or frames. See
> [Privacy](#privacy).

## Quick start

```bash
git clone https://github.com/mmaaaaz/gripv-audit.git
cd gripv-audit
cp .env.example .env          # add HF_TOKEN; .env is gitignored

uv run --with duckdb --with pandas --with pyarrow \
  python -m gripv.run --category glasses_fill --out results/glasses_fill.csv
```

Prefer a plain environment? `python -m pip install -e ".[dev]"` and run
`python -m gripv.run ...` from the repository root.

### Dataset access

The dataset is private. Request read access from the dataset owner, create a
Hugging Face token with that access, and put it in `.env`:

```dotenv
HF_TOKEN=hf_...
GRIPV_REPO=hamzahanif/GRIP-V
```

If `data/` and `cache/` are empty, the harness reads only the
`sample_id, params` columns over HTTPS with DuckDB. First run downloads the
params column once; later runs are offline.

## Commands

```bash
# list categories that have a contract
python -m gripv.run --list

# one category, all questions
python -m gripv.run --category robot_long_jump

# one qid, or a question-template substring
python -m gripv.run --category glasses_fill --qid q01
python -m gripv.run --category glasses_fill --contains "first to pour"

# smoke test: first 50 rows
python -m gripv.run --category balance_scale_weighing --limit 50

# re-fetch params instead of using cache/
python -m gripv.run --category conveyor_belt_sort --refresh
```

Outputs:

- `results/<category>.csv` — one row per question with key, strict answer,
  loose answer, status and note.
- `results/<category>.summary.json` — counts, per-qid counts, and the flag list.

## Statuses

| status | meaning | action |
|---|---|---|
| `match` | strict answer equals the key | ok |
| `loose_only` | key matches only the alternative reading | **flag** |
| `mismatch` | key matches neither reading | **flag** |
| `unmapped` | answer computed, but no option matched it | inspect matcher |
| `uncertain` | convention-dependent answer | counted, never a finding |
| `unsupported` | no handler for this template yet | add handler |

`uncertain` exists on purpose: an answer that depends on an unstated convention
must never be promoted to an accusation.

## Coverage

Measured on the released private dataset (25,000 questions per category).

| category | match | unsupported | unmapped | uncertain | mismatch |
|---|---:|---:|---:|---:|---:|
| `glasses_fill` | 18,665 | 6,206 | 0 | 0 | 0 (129 flags) |
| `balance_scale_weighing` | 15,354 | 6,743 | 2,312 | 589 | 2 (edge cases) |
| `conveyor_belt_sort` | 15,308 | 9,678 | 5 | 0 | 9 (unresolved) |
| `robot_long_jump` | 17,037 | 7,963 | 0 | 0 | 0 |
| `bouncing_ball_restitution` | 3,535 | 18,822 | 2,520 | 0 | 123 (in calibration) |

`bouncing_ball_restitution` is **not** counted as verified: the `pads`
scenario does not store restitution, and wall-bounce totals are off by one.
It stays in the repository as a development contract.

The `glasses_fill` flags split into **66 confirmed** answer errors on
“Which jug is the first to pour?” (the key names the jug that tilts but pours
nothing; q01 38, q02 21, q03 7), **38** same-signature flags on the follow-up
template, and **25** rows whose question has no valid option. The original
manual catch was **Inam's** (`glasses_fill_000007`); the replay and
quantification are Maaz's.

## Adding a category

1. Read several clips' `params` and questions. Write down the convention the
   generator actually uses — not the one you expect.
2. Create `gripv/contracts/<category>.py` with a class that takes `params` and
   exposes `answer(question, options) -> Answer`.
3. Register it in `gripv/contracts/__init__.py`.
4. **Calibrate before flagging.** The contract must reproduce the keys of at
   least two anchor clips of that category, including the hard templates. Only
   then run the full category.
5. Investigate every `mismatch` against more clips before calling it an error.
   A convention you have not seen yet is more likely than a dataset bug.

Rules that keep findings honest:

- Return `Answer(None, confidence="unsupported", note=...)` when the recipe
  does not contain what the question needs.
- Return `confidence="uncertain"` when the answer depends on an unstated
  convention.
- Never hard-code an expected answer from the key. Compute it from `params`.
- A category is not “verified” because it runs; it is verified when its
  contract reproduces that category's own keys.

More detail: [`docs/METHOD.md`](docs/METHOD.md) and
[`docs/ADDING_A_CATEGORY.md`](docs/ADDING_A_CATEGORY.md).

## Layout

```
gripv/
  env.py          env vars / .env loading, paths
  store.py        local parquet or DuckDB-over-HTTPS, params cache
  engine.py       Answer, normalization, option matching
  run.py          CLI and CSV/summary writer
  contracts/
    glasses_fill.py
    balance_scale_weighing.py
    conveyor_belt_sort.py
    robot_long_jump.py
    bouncing_ball_restitution.py
tests/            engine unit tests (no dataset needed)
docs/             method and contributor notes
```

## Privacy

This repository intentionally contains **no dataset content**:

- no questions, options or answer keys;
- no `params` or cached JSON;
- no videos or extracted frames.

`.env`, `data/`, `cache/`, `results/`, `frames/` and `*.parquet` are gitignored.
The private audit dashboard is a separate, password-gated deployment; publishing
frames requires the dataset owner's written permission.

If you add a fixture, make it synthetic. If you need real data to debug, keep it
outside the repository and never commit it.

## License

MIT — see [`LICENSE`](LICENSE).

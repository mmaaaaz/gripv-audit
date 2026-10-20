# Method

## The idea

A generated benchmark clip is not a black box. Its `params` field records the
recipe: event times, object properties, counts, thresholds. If the recipe is
sufficient to regenerate the video, it is sufficient to answer most questions
about the video.

The harness therefore does not watch pixels. It:

1. loads `params` for a clip;
2. reconstructs the event timeline (pours, impacts, hops, gate crossings, …);
3. evaluates the question under the **strict** literal reading;
4. optionally evaluates an alternative **loose** reading where the wording is
   genuinely ambiguous;
5. matches both answers against the published options;
6. compares with the key.

A flag is a disagreement between a key and a computation that is reproducible
from the same file. That is why the output is a CSV, not a paragraph.

## Strict vs loose

The strict reading is the one the question literally states. The loose reading
is the charitable interpretation a human might use.

Example: “Which jug is the first to pour?”

- strict: liquid must leave the jug (`got_ml > 0`);
- loose: any tilt counts, even a dry one.

If the key matches strict, it is a match. If it matches only loose, the key is
using a different convention from the question — a `loose_only` flag. If it
matches neither, it is a `mismatch`.

The two-reading design is what catches the original `glasses_fill` bug: a dry
jug tilts first, the key calls it “first to pour”, and the strict computation
disagrees while the loose one agrees.

## Confidence

`Answer.confidence` controls whether a row may become a finding:

| confidence | meaning | can be a finding? |
|---|---|---|
| `exact` | computed from stored values | yes |
| `derived` | computed through a documented derivation | yes, with note |
| `uncertain` | depends on an unstated convention | **no** |
| `unsupported` | recipe lacks the needed field | **no** |

`uncertain` is not a failure. It is the harness refusing to accuse the dataset
of a convention it cannot prove.

## Calibration rule

A contract is trusted only after it reproduces the keys of at least two anchor
clips of its own category, including the templates that are easy to get wrong.
Calibration is done before a full-category run. A full run with mismatches is a
signal to investigate the contract first, not to publish a finding.

## What counts as a confirmed error

1. The key contradicts the strict computation.
2. The computation uses only fields present in `params`.
3. The template is understood well enough that comparable rows in the same
   template use the intended convention.
4. The row is reproducible with `python -m gripv.run`.

Anything weaker stays `uncertain`, `unresolved` or `unsupported`.

## Limits

- Pixels are not read. A contract can only be as good as the recipe; if a
  question depends on a renderer detail absent from `params` (for example a
  camera azimuth), it must be marked unsupported.
- Some categories use conventions that are not documented; those need anchor
  calibration against several clips.
- A flag is evidence about the **key**, not proof that the video is wrong.

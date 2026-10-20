"""Run a Layer-1 (deterministic params replay) audit for one category.

Usage:
  uv run --with duckdb --with pandas --with pyarrow python -m gripv.run \
      --category glasses_fill --out results/glasses_fill.csv

Statuses:
  match        computed strict answer == key
  loose_only   key matches only the loose reading of the question  <-- FLAG
  mismatch     key matches neither reading                          <-- FLAG
  unmapped     answer computed but no option matched it
  uncertain    convention-dependent answer; counted, never a finding
  unsupported  no handler for this template yet (not a finding)

Exit code is 0 even when flags exist: a flag is a result, not a crash. Pipe the
summary through your own threshold check in CI if you need one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

from . import contracts, env
from .engine import letter, match_option
from .store import Store


def run(category: str, out: Path, qid: str | None, contains: str | None,
        limit: int, refresh: bool) -> dict:
    store = Store(category)
    questions = store.questions()
    params = store.params(refresh=refresh)

    if qid:
        questions = questions[questions["qid"] == qid]
    if contains:
        questions = questions[questions["question"].str.contains(contains, case=False, regex=True)]
    if limit:
        questions = questions.head(limit)

    cls = contracts.get(category)
    rows, counts = [], collections.Counter()
    by_qid = collections.defaultdict(collections.Counter)

    for r in questions.itertuples(index=False):
        sid = str(r.sample_id)
        p = params.get(sid)
        if p is None:
            counts["no_params"] += 1
            continue

        options = list(r.options)
        ans = cls(p).answer(str(r.question), options)
        key_i = "ABCD".find(str(r.answer).strip())
        ql = str(r.question).lower()
        tol = 0.35 if ("what time" in ql or "clock time" in ql) else (0.25 if "how many seconds" in ql else 0.051)
        strict_i = match_option(ans.strict, options, tol=tol)
        loose_i = match_option(ans.loose, options, tol=tol)

        if ans.confidence == "unsupported":
            status = "unsupported"
        elif ans.confidence == "uncertain":
            status = "uncertain"
        elif strict_i == key_i:
            status = "match"
        elif loose_i == key_i:
            status = "loose_only"
        elif strict_i is None:
            status = "unmapped"
        else:
            status = "mismatch"

        counts[status] += 1
        by_qid[str(r.qid)][status] += 1
        rows.append({
            "category": category,
            "sample_id": sid,
            "qid": r.qid,
            "question": r.question,
            "key_letter": r.answer,
            "key_text": r.answer_text,
            "strict": ans.strict,
            "strict_letter": letter(strict_i),
            "loose": ans.loose,
            "loose_letter": letter(loose_i),
            "status": status,
            "confidence": ans.confidence,
            "note": ans.note,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["status"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "category": category,
        "rows": len(rows),
        "counts": dict(counts),
        "by_qid": {k: dict(v) for k, v in sorted(by_qid.items())},
        "flags": [r for r in rows if r["status"] in ("loose_only", "mismatch")],
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", help="category name (see --list)")
    ap.add_argument("--out", type=Path, default=None, help="CSV path (default: results/<category>.csv)")
    ap.add_argument("--qid", default=None, help="e.g. q01")
    ap.add_argument("--contains", default=None, help="regex filter on question text")
    ap.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch params instead of using cache")
    ap.add_argument("--list", action="store_true", help="list categories with a contract")
    a = ap.parse_args()

    if a.list:
        for name in sorted(contracts.CONTRACTS):
            print(name)
        return
    if not a.category:
        ap.error("--category is required (or use --list)")

    out = a.out or env.RESULTS_DIR / f"{a.category}.csv"
    s = run(a.category, out, a.qid, a.contains, a.limit, a.refresh)
    print(f"{s['category']}: {s['rows']} rows -> {out}")
    for k, v in sorted(s["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v}")
    if s["flags"]:
        print(f"\nFLAGS ({len(s['flags'])}):")
        for f in s["flags"][:40]:
            print(f"  {f['sample_id']} {f['qid']} key={f['key_text']!r} strict={f['strict']!r} loose={f['loose']!r}")
        if len(s["flags"]) > 40:
            print(f"  ... {len(s['flags']) - 40} more in {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()

"""Answer/option matching shared by every category contract."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Answer:
    """A computed answer under the strict reading, plus an optional loose reading.

    strict: the literal reading of the question (e.g. "pour" = liquid left the jug)
    loose:  the charitable/alternative reading (e.g. any tilt counts)
    confidence: exact | derived | uncertain | unsupported
    """

    strict: str | None
    loose: str | None = None
    confidence: str = "exact"
    note: str = ""
    extra: dict = field(default_factory=dict)


def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[^a-z0-9.\- ]+", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


def match_option(answer: str | None, options: list[str], tol: float = 0.051) -> int | None:
    """Return the index of the option that matches the computed answer."""
    if answer is None:
        return None
    a = norm(answer)
    opts = [norm(o) for o in options]
    for i, o in enumerate(opts):
        if a == o:
            return i
    na = number(answer)
    if na is not None:
        best_i, best_d = None, None
        for i, o in enumerate(opts):
            no = number(o)
            if no is None or abs(no - na) > tol:
                continue
            if best_d is None or abs(no - na) < best_d:
                best_i, best_d = i, abs(no - na)
        if best_i is not None:
            o = opts[best_i]
            if (
                re.search(r"\d\s*(ml|s|seconds?|times?)\b", a)
                or re.search(r"\d\s*(ml|s|seconds?|times?)\b", o)
                or a == str(int(na))
                or o == str(int(na))
            ):
                return best_i
    at = set(a.split())
    for i, o in enumerate(opts):
        ot = set(o.split())
        if ot and ot <= at:
            return i
    return None


LETTERS = "ABCD"
ALPHABET = "ABCDEFGH"


def letter(index: int | None) -> str | None:
    if index is None or index >= len(ALPHABET):
        return None
    return ALPHABET[index]

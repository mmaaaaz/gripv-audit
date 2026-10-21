"""multi_agent_chase — Layer-1 contract.

Pinned by `params` alone (verified on all 25,000 rows; see
`results/multi_agent_chase_notes.md`):

* colour -> pattern and pattern -> colour lookups (both unique inside a clip);
* the number of balls at the start (the `balls` list);
* the biggest ball at the start (largest `r`);
* the number of grey pillars (the `pillars` list; absent = none).

Pinned by the clip's role structure, which the generator never violates
(only a `hunter` catches, a `prey` never catches, and catch questions only ever
name a ball whose role can answer them):

* "How many balls does the <prey> ball catch?"            -> 0 (hunter clips);
* "Which ball catches the <prey> ball?"                   -> the hunter, when
  the clip has exactly one hunter;
* "Which of these balls catches at least one other ball?" -> the hunter, same;
* "Which ball catches the most other balls?"              -> the hunter, same;
* "Which ball is the biggest at the end?"                 -> the hunter, same
  (only the catcher gains radius, so a lone hunter is the only ball that can
  grow; the key confirms this on every such row).

Everything that needs the *event* timeline — catch order, catch clocks, catch
counts per hunter, per-ball victim lists, growth sums, the safe zone, the
pillars' effect and every counterfactual — is `unsupported`.  The recipe
carries the start state and a per-ball wander description (`h0`,
`a1/w1/p1`, `a2/w2/p2`), but the trajectory it produces is not recoverable
from those fields by inspection: the literal reading
(`h = h0 + a1 sin(w1 t + p1) + a2 sin(w2 t + p2)`, t = frame/fps) tracks one
ball for about a second and then diverges by tens of pixels, and no static
rule explains the catch order (the best static predictor of "first caught"
hits 33%, i.e. chance).  Rather than guess a steering law, every such family
is reported as unsupported.
"""
from __future__ import annotations

import re

from ..engine import Answer

BALL = re.compile(r"the ([a-z]+) ([a-z\-]+) ball")

# "What pattern does the red ball have?"
RE_PATTERN_Q = re.compile(r"^what pattern does the ([a-z]+) ball have\?*$")
# "What colour is the two-tone ball?"
RE_COLOUR_Q = re.compile(r"^what colou?r is the ([a-z\-]+) ball\?*$")
# "How many balls are in the box at the start of the video?"
RE_START_Q = re.compile(r"^how many balls are in the box at the start of the video\?*$")
# "Which ball is the biggest at the start?"
RE_BIGGEST_START_Q = re.compile(r"^which ball is the biggest at the start\?*$")
# "How many grey pillars stand in the box?"
RE_PILLARS_Q = re.compile(r"^how many ([a-z]+) pillars stand in the box\?*$")
# "Which ball is the biggest at the end of the video?"
RE_BIGGEST_END_Q = re.compile(r"^which ball is the biggest at the end of the video\?*$")
# "How many balls does the X ball catch?"
RE_CATCH_COUNT_Q = re.compile(r"^how many balls does the ([a-z]+) ([a-z\-]+) ball catch\?*$")
# "Which ball catches the X ball?"
RE_WHO_CATCHES_Q = re.compile(r"^which ball catches the ([a-z]+) ([a-z\-]+) ball\?*$")
# "Which of these balls catches at least one other ball?"
RE_CATCHES_ANY_Q = re.compile(r"^which of these balls catches at least one other ball\?*$")
# "Which ball catches the most other balls?"
RE_MOST_CATCHES_Q = re.compile(r"^which ball catches the most other balls\?*$")
# "At the end of the video, which is bigger: the X ball or the Y ball?"
RE_PAIR_BIGGER_END_Q = re.compile(
    r"^at the end of the video, which is bigger: the ([a-z]+) ([a-z\-]+) ball or the ([a-z]+) ([a-z\-]+) ball\?*$")


def label(ball: dict) -> str:
    return f"the {ball['color']} {ball['pattern']} ball"


class MultiAgentChase:
    """Params -> the questions that the recipe alone settles."""

    def __init__(self, params: dict):
        self.p = params or {}
        self.balls = list(self.p.get("balls") or [])
        self.pillars = list(self.p.get("pillars") or [])
        self.zone = self.p.get("zone")
        self.scenario = self.p.get("scenario")
        self.hunters = [b for b in self.balls if b.get("role") == "hunter"]
        self.prey = [b for b in self.balls if b.get("role") == "prey"]

    # ---------------- helpers ----------------
    def _ball(self, colour: str, pattern: str) -> dict | None:
        for b in self.balls:
            if b.get("color") == colour and b.get("pattern") == pattern:
                return b
        return None

    def _sole_hunter(self) -> dict | None:
        return self.hunters[0] if len(self.hunters) == 1 else None

    def _hunter_if_offered(self, options: list[str] | None) -> Answer | None:
        """The sole hunter answers every 'who catches' question when offered."""
        h = self._sole_hunter()
        if h is None:
            return None
        if options and label(h) not in [str(o).strip() for o in options]:
            return None
        return Answer(label(h), confidence="exact",
                      note="sole hunter is the only ball that can catch")

    def _by_colour(self, colour: str) -> list[dict]:
        return [b for b in self.balls if b.get("color") == colour]

    def _by_pattern(self, pattern: str) -> list[dict]:
        return [b for b in self.balls if b.get("pattern") == pattern]

    def biggest_start(self) -> dict | None:
        """Largest r; None when the top radius is shared (clip is ambiguous)."""
        if not self.balls:
            return None
        rs = sorted((b.get("r", 0.0) for b in self.balls), reverse=True)
        if len(rs) > 1 and abs(rs[0] - rs[1]) < 1e-9:
            return None
        return max(self.balls, key=lambda b: b.get("r", 0.0))

    # ---------------- dispatch ----------------
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = str(question).strip().lower()

        m = RE_PATTERN_Q.match(q)
        if m:
            hits = self._by_colour(m.group(1))
            if len(hits) == 1:
                return Answer(hits[0]["pattern"], confidence="exact")
            return Answer(None, confidence="unsupported",
                          note=f"{len(hits)} balls with colour {m.group(1)!r}")

        m = RE_COLOUR_Q.match(q)
        if m:
            hits = self._by_pattern(m.group(1))
            if len(hits) == 1:
                return Answer(hits[0]["color"], confidence="exact")
            return Answer(None, confidence="unsupported",
                          note=f"{len(hits)} balls with pattern {m.group(1)!r}")

        if RE_START_Q.match(q):
            return Answer(str(len(self.balls)), confidence="exact")

        if RE_BIGGEST_START_Q.match(q):
            b = self.biggest_start()
            if b is None:
                return Answer(None, confidence="unsupported", note="max-r tie")
            return Answer(label(b), confidence="exact")

        m = RE_PILLARS_Q.match(q)
        if m:
            colour = m.group(1)
            if colour not in ("grey", "gray"):
                # Pillars are only ever drawn grey; a question about any other
                # colour still has a well-defined answer of 0.
                return Answer("0", confidence="exact")
            return Answer(str(len(self.pillars)), confidence="exact")

        # ---- role-structural families ----
        m = RE_CATCH_COUNT_Q.match(q)
        if m:
            b = self._ball(m.group(1), m.group(2))
            if b is None:
                return Answer(None, confidence="unsupported", note="ball not in params")
            if not self.hunters:
                return Answer(None, confidence="unsupported",
                              note="no hunter: any ball may catch (bigger scenario)")
            if b.get("role") == "prey":
                return Answer("0", confidence="exact",
                              note="prey never catch while hunters are present")
            return Answer(None, confidence="unsupported",
                          note="hunter's victim count needs the timeline")

        m = RE_WHO_CATCHES_Q.match(q)
        if m:
            b = self._ball(m.group(1), m.group(2))
            if b is not None and b.get("role") == "hunter":
                return Answer(None, confidence="unsupported",
                              note="hunters are never caught")
            a = self._hunter_if_offered(options)
            if a is not None:
                return a
            return Answer(None, confidence="unsupported",
                          note="several hunters: which one arrives first needs the timeline")

        if RE_CATCHES_ANY_Q.match(q):
            a = self._hunter_if_offered(options)
            if a is not None:
                return a
            return Answer(None, confidence="unsupported",
                          note="several hunters: needs the timeline")

        if RE_MOST_CATCHES_Q.match(q):
            a = self._hunter_if_offered(options)
            if a is not None:
                return a
            return Answer(None, confidence="unsupported",
                          note="several hunters: needs the timeline")

        if RE_BIGGEST_END_Q.match(q):
            h = self._sole_hunter()
            if h is not None:
                return Answer(label(h), confidence="exact",
                              note="only the catcher grows; a lone hunter ends biggest")
            return Answer(None, confidence="unsupported",
                          note="growth depends on the catch timeline")

        m = RE_PAIR_BIGGER_END_Q.match(q)
        if m:
            a = self._ball(m.group(1), m.group(2))
            b = self._ball(m.group(3), m.group(4))
            h = self._sole_hunter()
            if h is not None and a is not None and b is not None and (a is h) != (b is h):
                return Answer(label(h), confidence="exact",
                              note="only the catcher grows; a lone hunter outgrows every prey")
            return Answer(None, confidence="unsupported",
                          note="growth depends on the catch timeline")

        return Answer(None, confidence="unsupported", note="event timeline not pinned by params")

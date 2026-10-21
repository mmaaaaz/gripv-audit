"""tangled_braid — replay of the braid's crossing recipe.

The recipe pins the whole animation:

* ``strands``  — the roster, ``name`` = "<colour> <style>" (colour is the
  first word and is unique inside a clip).
* ``start``    — the strand names in loom-bar order at frame 0; index 0 is the
  **leftmost** place on a vertical braid (``layout.orient == "v"``) and the
  **top** place on a sideways one (``orient == "h"``).  "Nth from the left"
  and "Nth from the top" are the same index, which is why both phrasings
  appear for every scenario.
* ``steps``    — consecutive crossing events.  Each step spans frames
  ``a`` .. ``m`` and carries one to three ``moves``; every move swaps the
  neighbouring places ``i`` and ``i+1``.  Steps with several moves cross at
  the *same moment* (one ``m`` for the whole step).
* ``moves[].over`` — ``"high"`` means the strand standing in the **higher
  place index** (i+1, the right-hand / lower one) passes over; ``"low"`` means
  the strand in place ``i`` does.
* clock — 24 fps, ``frame / 24`` seconds, and the on-screen timer is that value
  **truncated** to one decimal (calibrated on 1,049 "what does the clock read"
  keys; rounding gets 96.7 %, truncation 100 %).
* the end-of-video order is the start order after every swap has been applied,
  which is also exactly what the "which strand ends up Nth" keys use.

Counterfactuals keep every other crossing "in the same two places" — the move
list is re-run with the suppressed swap deleted, so later swaps stay at the
same place indices but involve different strands.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

def _colour(name: str) -> str:
    return str(name).split()[0]


def _ordinal(n: int) -> str:
    suffix = "th" if n > 3 else ("st", "nd", "rd")[n - 1]
    return f"{n}{suffix}"


def _clock(frame: float) -> float:
    """On-screen timer: frame/24 s truncated to one decimal."""
    return math.floor(frame / FPS * 10.0 + 1e-9) / 10.0


def _clock_text(frame: float) -> str:
    return f"{_clock(frame):.1f} s"


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


class _Event:
    __slots__ = ("step", "i", "over", "frame", "a", "left", "right")

    def __init__(self, step, i, over, frame, a, left, right):
        self.step = step
        self.i = i
        self.over = over
        self.frame = frame
        self.a = a
        self.left = left          # strand standing in place i
        self.right = right        # strand standing in place i+1

    # ``over`` names the side the crossing strand comes from: "high" = the
    # strand that was in the higher place (i+1).
    @property
    def over_strand(self) -> str:
        return self.right if self.over == "high" else self.left

    @property
    def under_strand(self) -> str:
        return self.left if self.over == "high" else self.right

    @property
    def pair(self) -> frozenset:
        """The two colours that cross, order-free."""
        return frozenset((_colour(self.left), _colour(self.right)))


class TangledBraid:
    """Replay one ``tangled_braid`` clip from its generator recipe."""

    def __init__(self, params: dict):
        self.p = params
        self.names = [s["name"] for s in params["strands"]]
        self.start = list(params["start"])
        self.steps = params["steps"]
        # (step index, place index, over) for every crossing, in video order
        self.moves = [
            (si, m["i"], m["over"])
            for si, st in enumerate(self.steps)
            for m in st["moves"]
        ]
        self.events = self._sim(self.moves)[0]
        self.end = self._sim(self.moves)[1]

    # ------------------------------------------------------------------ sim
    def _sim(self, moves, start=None):
        pos = list(self.start if start is None else start)
        events = []
        for si, i, over in moves:
            st = self.steps[si]
            left, right = pos[i], pos[i + 1]
            events.append(_Event(si, i, over, st["m"], st["a"], left, right))
            pos[i], pos[i + 1] = right, left
        return events, pos

    def _position_at(self, clock: float):
        """Places at the moment the on-screen timer reads ``clock``."""
        pos = list(self.start)
        for ev in self.events:
            if ev.frame / FPS <= clock + 1e-9:
                pos[ev.i], pos[ev.i + 1] = pos[ev.i + 1], pos[ev.i]
        return pos

    # --------------------------------------------------------------- helpers
    def _pick(self, candidates, options):
        """First candidate that is literally one of the offered options."""
        if options:
            for cand in candidates:
                if cand in options:
                    return cand
        return candidates[0] if candidates else None

    def _idx(self, colour: str, order=None) -> int:
        return [_colour(s) for s in (order or self.start)].index(colour)

    def _events_of(self, colour: str, events=None):
        events = self.events if events is None else events
        return [e for e in events
                if _colour(e.left) == colour or _colour(e.right) == colour]

    def _partner(self, event, colour: str) -> str:
        return _colour(event.right) if _colour(event.left) == colour else _colour(event.left)

    def _pair_events(self, c1: str, c2: str, events=None):
        events = self.events if events is None else events
        want = frozenset((c1, c2))
        return [e for e in events if e.pair == want]

    # ---------------------------------------------------------------- replay
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = " ".join(str(question).split())

        for pattern, handler in self._handlers:
            m = pattern.search(q)
            if m:
                try:
                    value = handler(self, m, options or [])
                except (IndexError, ValueError, KeyError):
                    return Answer(None, confidence="unsupported",
                                  note=f"could not evaluate {pattern.pattern[:60]}")
                if value is None:
                    return Answer(None, confidence="unsupported",
                                  note=f"no value for {pattern.pattern[:60]}")
                return Answer(str(value))

        return Answer(None, confidence="unsupported",
                      note="question family not pinned by params")

    # -------------------------------------------------------------- families
    def _h_strand_count(self, m, options):
        return len(self.names)

    def _h_crossing_count(self, m, options):
        return len(self.events)

    def _h_extreme_place(self, m, options):
        when, side = m.group(1), m.group(2)
        order = self.start if when == "start" else self.end
        if side in ("leftmost", "at the top"):
            return _colour(order[0])
        return _colour(order[-1])

    def _h_moved_places(self, m, options):
        colour = m.group(1)
        return abs(self._idx(colour) - self._idx(colour, self.end))

    def _h_same_place_count(self, m, options):
        return sum(1 for a, b in zip(self.start, self.end) if a == b)

    def _h_ends_where_started(self, m, options):
        return _colour(self.end[self._idx(m.group(1))])

    def _h_never_crosses(self, m, options):
        crossed = {_colour(n) for e in self.events for n in (e.left, e.right)}
        silent = [_colour(s) for s in self.start if _colour(s) not in crossed]
        return self._pick(silent, options)

    def _h_most_crossings(self, m, options):
        counts = {s: 0 for s in self.start}
        for e in self.events:
            counts[e.left] += 1
            counts[e.right] += 1
        best = max(counts.values()) if counts else 0
        cands = [_colour(s) for s in self.start if counts[s] == best]
        return self._pick(cands, options)

    def _h_most_over(self, m, options):
        counts = {s: 0 for s in self.start}
        for e in self.events:
            counts[e.over_strand] += 1
        best = max(counts.values()) if counts else 0
        cands = [_colour(s) for s in self.start if counts[s] == best]
        return self._pick(cands, options)

    def _h_strand_crossings(self, m, options):
        return len(self._events_of(m.group(1)))

    def _h_strand_overs(self, m, options):
        colour = m.group(1)
        return sum(1 for e in self.events if _colour(e.over_strand) == colour)

    def _h_pair_crossings(self, m, options):
        return len(self._pair_events(m.group(1), m.group(2)))

    def _h_nth_over(self, m, options):
        n = ORDINALS[m.group(1)]
        return _colour(self.events[n - 1].over_strand)

    def _h_first_pair(self, m, options):
        n = ORDINALS[m.group(1)]
        e = self.events[n - 1]
        return " and ".join(sorted((_colour(e.left), _colour(e.right))))

    def _h_same_moment(self, m, options):
        c1, c2, n = m.group(1), m.group(2), ORDINALS[m.group(3)]
        hits = self._pair_events(c1, c2)
        step = hits[n - 1].step
        others = [
            " and ".join(sorted((_colour(e.left), _colour(e.right))))
            for e in self.events if e.step == step and e.pair != frozenset((c1, c2))
        ]
        return self._pick(others, options)

    def _h_which_nth(self, m, options):
        n = ORDINALS[m.group(1)]
        offered = [x.strip() for x in m.group(2).split(",") if x.strip()]
        first = {}
        for k, e in enumerate(self.events):
            for name in (e.left, e.right):
                first.setdefault(_colour(name), k)
        ranked = sorted((first[c], c) for c in offered if c in first)
        if len(ranked) < n:
            return None
        return ranked[n - 1][1]

    def _h_nth_what_happens(self, m, options):
        colour, n = m.group(2), ORDINALS[m.group(1)]
        e = self._events_of(colour)[n - 1]
        if _colour(e.over_strand) == colour:
            return f"it passes over the {self._partner(e, colour)} strand"
        return f"it passes under the {self._partner(e, colour)} strand"

    def _h_nth_partner(self, m, options):
        colour, n = m.group(1), ORDINALS[m.group(2)]
        return self._partner(self._events_of(colour)[n - 1], colour)

    def _h_cross_order(self, m, options):
        colour = m.group(1)
        seq = [self._partner(e, colour) for e in self._events_of(colour)]
        return ", then ".join(seq)

    def _h_after_pair(self, m, options):
        colour, other, n = m.group(1), m.group(3), ORDINALS[m.group(2)]
        hits = self._pair_events(colour, other)
        start = self.events.index(hits[n - 1]) + 1
        for e in self.events[start:]:
            if _colour(e.left) == colour or _colour(e.right) == colour:
                return self._partner(e, colour)
        return None

    # ------------------------------------------------------------- clocks
    def _h_clock_at_meet(self, m, options):
        c1, c2 = m.group(1), m.group(2)
        n = ORDINALS[m.group(3)] if m.re.groups >= 3 else 1
        hits = self._pair_events(c1, c2)
        return _clock_text(hits[n - 1].frame)

    def _h_crossings_between(self, m, options):
        t1, t2 = float(m.group(1)), float(m.group(2))
        return sum(1 for e in self.events if t1 - 1e-9 <= e.frame / FPS <= t2 + 1e-9)

    def _h_strand_by_clock(self, m, options):
        clock, colour = float(m.group(1)), m.group(2)
        return sum(1 for e in self.events
                   if e.frame / FPS <= clock + 1e-9
                   and colour in (_colour(e.left), _colour(e.right)))

    def _h_place_at_clock(self, m, options):
        clock, colour, axis = float(m.group(1)), m.group(2), m.group(3)
        places = [_colour(s) for s in self._position_at(clock)]
        return f"{_ordinal(places.index(colour) + 1)} from the {axis}"

    def _h_neighbour_at_clock(self, m, options):
        clock, side, colour = float(m.group(1)), m.group(2), m.group(3)
        places = [_colour(s) for s in self._position_at(clock)]
        j = places.index(colour) + (-1 if side in ("left", "above") else 1)
        if not 0 <= j < len(places):
            return None
        return places[j]

    # --------------------------------------------------------------- sums
    def _h_signed_sum(self, m, options):
        c1, c2, step, counted = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        total = 0
        for e in self._pair_events(c1, c2):
            total += step if _colour(e.over_strand) == counted else -step
        return _signed(total)

    def _h_flip_all_overs(self, m, options):
        colour = m.group(1)
        return sum(1 for e in self._events_of(colour) if _colour(e.over_strand) != colour)

    def _h_flip_nth_sum(self, m, options):
        n = ORDINALS[m.group(1)]
        flip_a, flip_b = m.group(2), m.group(3)
        c1, c2, step, counted = m.group(4), m.group(5), int(m.group(6)), m.group(7)
        want = frozenset((flip_a, flip_b))
        total = 0
        for k, e in enumerate(self.events):
            over = _colour(e.over_strand)
            if k == n - 1 and e.pair == want:
                over = flip_b if over == flip_a else flip_a
            if e.pair == frozenset((c1, c2)):
                total += step if over == counted else -step
        return _signed(total)

    def _h_flip_pair_nth_sum(self, m, options):
        n = ORDINALS[m.group(1)]
        pair = (m.group(2), m.group(3))
        flip_a, flip_b = m.group(4), m.group(5)
        c1, c2, step, counted = m.group(6), m.group(7), int(m.group(8)), m.group(9)
        target = self.events.index(self._pair_events(*pair)[n - 1])
        total = 0
        for k, e in enumerate(self.events):
            over = _colour(e.over_strand)
            if k == target:
                over = flip_b if over == flip_a else flip_a
            if e.pair == frozenset((c1, c2)):
                total += step if over == counted else -step
        return _signed(total)

    # ------------------------------------------------------- counterfactuals
    def _without(self, drop):
        moves = [mv for k, mv in enumerate(self.moves) if k not in drop]
        return self._sim(moves)

    def _h_drop_nth_count(self, m, options):
        n = ORDINALS[m.group(1)]
        c1, c2 = m.group(2), m.group(3)
        events, _ = self._without({n - 1})
        return len(self._pair_events(c1, c2, events))

    def _h_drop_nth_place(self, m, options):
        n = ORDINALS[m.group(1)]
        k, axis = int(m.group(2)), m.group(3)
        _, pos = self._without({n - 1})
        return _colour(pos[k - 1])

    def _h_drop_meet_count(self, m, options):
        c1, c2, n = m.group(1), m.group(2), ORDINALS[m.group(3)]
        want_c, want_d = m.group(4), m.group(5)
        hits = self._pair_events(c1, c2)
        events, _ = self._without({self.events.index(hits[n - 1])})
        return len(self._pair_events(want_c, want_d, events))

    def _h_drop_meet_place(self, m, options):
        c1, c2, n = m.group(1), m.group(2), ORDINALS[m.group(3)]
        k, axis = int(m.group(4)), m.group(5)
        hits = self._pair_events(c1, c2)
        _, pos = self._without({self.events.index(hits[n - 1])})
        return _colour(pos[k - 1])

    def _h_drop_pair_count(self, m, options):
        c1, c2, want_c, want_d = m.group(1), m.group(2), m.group(3), m.group(4)
        drop = {self.events.index(e) for e in self._pair_events(c1, c2)}
        events, _ = self._without(drop)
        return len(self._pair_events(want_c, want_d, events))

    def _h_drop_pair_place(self, m, options):
        c1, c2, k, axis = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        drop = {self.events.index(e) for e in self._pair_events(c1, c2)}
        _, pos = self._without(drop)
        return _colour(pos[k - 1])

    def _h_swapped_starts_place(self, m, options):
        c1, c2, k = m.group(1), m.group(2), int(m.group(3))
        i1, i2 = self._idx(c1), self._idx(c2)
        start = list(self.start)
        start[i1], start[i2] = start[i2], start[i1]
        _, pos = self._sim(self.moves, start=start)
        return _colour(pos[k - 1])

    # ---------------------------------------------------------- registries
    _handlers = [
        # ---- census
        (re.compile(r"^How many strands are there in the braid\?$"), _h_strand_count),
        (re.compile(r"^How many crossings \(two neighbouring strands swapping places.*happen in the whole video\?$"), _h_crossing_count),
        (re.compile(r"^At the (start|end) of the video, which strand is (leftmost|rightmost|at the top|at the bottom) at the loom bar"), _h_extreme_place),
        (re.compile(r"^By how many places has the (\w+) strand moved at the loom bar"), _h_moved_places),
        (re.compile(r"^How many strands end at the loom bar .*?in the same place they started\?$"), _h_same_place_count),
        (re.compile(r"^Which strand ends at the loom bar .*?in the place where the (\w+) strand started\?$"), _h_ends_where_started),
        (re.compile(r"^Which strand never crosses another strand\?$"), _h_never_crosses),
        (re.compile(r"^Which strand takes part in the most crossings\?$"), _h_most_crossings),
        (re.compile(r"^Which strand passes over other strands the most times\?$"), _h_most_over),
        # ---- per-strand counts
        (re.compile(r"^How many crossings does the (\w+) strand take part in\?$"), _h_strand_crossings),
        (re.compile(r"^How many times does the (\w+) strand pass over another strand\?$"), _h_strand_overs),
        (re.compile(r"^How many times do the (\w+) and (\w+) strands cross each other\?$"), _h_pair_crossings),
        # ---- ordering
        (re.compile(r"^At the (\w+) crossing of the video, which strand passes over the other one\?$"), _h_nth_over),
        (re.compile(r"^Which two strands cross (\w+)\?$"), _h_first_pair),
        (re.compile(r"^Which two strands cross at the same moment as the (\w+) and (\w+) strands cross for the (\w+) time\?$"), _h_same_moment),
        (re.compile(r"^Which of these strands is the (\w+) to cross another strand: (.*)\?$"), _h_which_nth),
        (re.compile(r"^What happens the (\w+) time the (\w+) strand crosses another strand\?$"), _h_nth_what_happens),
        (re.compile(r"^Which strand does the (\w+) strand cross (\w+)\?$"), _h_nth_partner),
        (re.compile(r"^Which strands does the (\w+) strand cross, in order\?$"), _h_cross_order),
        (re.compile(r"^After the (\w+) strand (\w+) crosses the (\w+) strand, which strand does the \w+ strand cross next\?$"), _h_after_pair),
        # ---- clock
        (re.compile(r"^What does the clock read at the moment the (\w+) and (\w+) strands meet, the (\w+) time they cross\?$"), _h_clock_at_meet),
        (re.compile(r"^What does the clock read at the moment the (\w+) and (\w+) strands meet, when they cross each other\?$"), _h_clock_at_meet),
        (re.compile(r"^How many crossings happen between the moment the clock reads ([\d.]+) s and the moment it reads ([\d.]+) s\?$"), _h_crossings_between),
        (re.compile(r"^By the time the clock reads ([\d.]+) s, how many crossings has the (\w+) strand taken part in\?$"), _h_strand_by_clock),
        (re.compile(r"^When the clock reads ([\d.]+) s, in which place is the (\w+) strand at the loom bar .*?counting from the (left|top)\?$"), _h_place_at_clock),
        (re.compile(r"^When the clock reads ([\d.]+) s, which strand is "
                    r"directly (?:to the )?(left|right|above|below) (?:of )?the "
                    r"(\w+) strand"),
         _h_neighbour_at_clock),
        # ---- sums
        (re.compile(r"^Go through every crossing between the (\w+) and (\w+) "
                    r"strands: count \+(\d+) each time the (\w+) strand passes "
                    r"over, -\d+ each time it passes under\. What is the total\?$"),
         _h_signed_sum),
        (re.compile(r"^Suppose that at every crossing the other strand had "
                    r"passed over .*?How many times would the (\w+) strand pass "
                    r"over another strand\?$"),
         _h_flip_all_overs),
        (re.compile(r"^Suppose that at the (\w+) crossing the (\w+) strand had "
                    r"passed over the (\w+) strand instead \(everything else the "
                    r"same\)\. Going through every crossing between the (\w+) and "
                    r"(\w+) strands, count \+(\d+) each time the (\w+) strand "
                    r"passes over.*?What would the total be\?$"),
         _h_flip_nth_sum),
        (re.compile(r"^Suppose that at the (\w+) time the (\w+) and (\w+) "
                    r"strands cross the (\w+) strand had passed over the (\w+) "
                    r"strand instead \(everything else the same\)\. Going through "
                    r"every crossing between the (\w+) and (\w+) strands, count "
                    r"\+(\d+) each time the (\w+) strand passes over.*?"
                    r"What would the total be\?$"),
         _h_flip_pair_nth_sum),
        # ---- counterfactuals
        (re.compile(r"^Suppose the (\w+) crossing had not happened,.*?How many times would the (\w+) and (\w+) strands cross each other\?$"), _h_drop_nth_count),
        (re.compile(r"^Suppose the (\w+) crossing had not happened,.*?Which strand would end up (\d+)(?:st|nd|rd|th) from the (left|top)"), _h_drop_nth_place),
        (re.compile(r"^Suppose the (\w+) and (\w+) strands had not crossed the (\w+) time they meet,.*?How many times would the (\w+) and (\w+) strands cross each other\?$"), _h_drop_meet_count),
        (re.compile(r"^Suppose the (\w+) and (\w+) strands had not crossed the (\w+) time they meet,.*?Which strand would end up (\d+)(?:st|nd|rd|th) from the (left|top)"), _h_drop_meet_place),
        (re.compile(r"^Suppose the (\w+) and (\w+) strands had not crossed, and every other.*?How many times would the (\w+) and (\w+) strands cross each other\?$"), _h_drop_pair_count),
        (re.compile(r"^Suppose the (\w+) and (\w+) strands had not crossed, and every other.*?Which strand would end up (\d+)(?:st|nd|rd|th) from the (left|top)"), _h_drop_pair_place),
        (re.compile(r"^Suppose the (\w+) and (\w+) strands had swapped starting places,.*?Which strand would end up (\d+)(?:st|nd|rd|th) from the (left|top)"), _h_swapped_starts_place),
    ]

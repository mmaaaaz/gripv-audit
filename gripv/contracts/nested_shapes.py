"""Deterministic replay for nested_shapes (ring order / visibility / spin / colour).

Semantics pinned against anchors (see nested_anchors.py):

* ``rings`` is stored outside-in: index 0 is the largest outline.  "the Nth
  outline from the outside" is the Nth *visible* ring in that order at the
  queried moment.
* A ring is on screen at frame ``f`` when ``appear <= f < vanish`` (``vanish``
  null means it stays to the end).  "At the end of the video" is frame
  ``n_frames - 1``.
* ``spin`` entries are ``[frame, degrees_per_second]``; positive is clockwise.
  A ring is "turning" when it is on screen and the current rate is non-zero.
* ``recolor`` entries are ``[frame, colour]`` applied from that frame onward.
  Every entry is a real change (the generator emits no no-op recolours); the
  number of colour changes is ``len(recolor)``.
* Timer readings in question text are truncated to one decimal (a recolour on
  frame 95 displays as 3.9 s, not 4.0 s).
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0

COLOURS = ("black", "blue", "green", "orange", "pink", "purple", "red", "teal", "white", "yellow")
SHAPES = ("circle", "cross", "diamond", "ellipse", "hexagon", "octagon", "pentagon",
          "rectangle", "square", "star", "trapezoid", "triangle")

_MENTION = re.compile(r"(?:the )?((?:%s) )?(%s)\b" % ("|".join(COLOURS), "|".join(SHAPES)), re.I)


def _sign(w: float) -> int:
    return 0 if abs(w) < 1e-12 else (1 if w > 0 else -1)


def _time_str(frame: float) -> str:
    """Timer text: one decimal, truncated (frame 95 -> 3.9 s)."""
    tenths = math.floor(frame / FPS * 10 + 1e-9)
    return f"{tenths / 10:.1f} s"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class NestedShapes:
    def __init__(self, params: dict):
        self.p = params
        self.rings = params.get("rings", [])
        self.marks = params.get("marks", [])
        self.n_frames = int(params.get("n_frames") or 0)
        self._idx = {id(r): i for i, r in enumerate(self.rings)}

    # ------------------------------------------------------------------ model
    @property
    def end_frame(self) -> int:
        return max(0, self.n_frames - 1)

    def visible(self, r: dict, f: float) -> bool:
        if f < (r.get("appear") or 0):
            return False
        v = r.get("vanish")
        return v is None or f < v

    def colour(self, r: dict, f: float) -> str:
        c = r.get("color")
        for fr, col in r.get("recolor", []):
            if fr <= f + 1e-9:
                c = col
        return c

    def rate(self, r: dict, f: float) -> float:
        if not self.visible(r, f):
            return 0.0
        w = 0.0
        for fr, val in r.get("spin", []):
            if fr <= f + 1e-9:
                w = float(val)
        return w

    def end_order(self) -> list[dict]:
        vis = [r for r in self.rings if self.visible(r, self.end_frame)]
        return sorted(vis, key=lambda r: -float(r.get("size", 0)))

    def mention(self, text: str) -> dict | None:
        """The unique ring named as "the [colour] shape" in *text*."""
        for m in _MENTION.finditer(text):
            col = (m.group(1) or "").strip().lower()
            shape = m.group(2).lower()
            cand = [r for r in self.rings if r["shape"] == shape and (not col or r["color"] == col)]
            if len(cand) == 1:
                return cand[0]
        return None

    def option_ring(self, option: str) -> dict | None:
        return self.mention(option)

    def pick_option(self, options, predicate) -> str | None:
        """The single option whose ring satisfies *predicate*."""
        hits = []
        for o in options:
            r = self.option_ring(o)
            if r is not None and predicate(r):
                hits.append(o)
        return hits[0] if len(hits) == 1 else None

    def changes(self, r: dict) -> list[tuple[float, str]]:
        return [(float(fr), c) for fr, c in r.get("recolor", [])]

    def phase_schedule(self, r: dict, on_screen: bool = False) -> list[tuple[float, int]]:
        """Compressed sign phases (1 cw, -1 ccw, 0 still)."""
        entries = [(float(fr), _sign(float(w))) for fr, w in r.get("spin", [])]
        if not entries:
            return [(0.0, 0)]
        if on_screen:
            start = float(r.get("appear") or 0)
            cur = 0
            for fr, sym in entries:
                if fr <= start:
                    cur = sym
            phases = [(start, cur)]
            for fr, sym in entries:
                if fr <= start:
                    continue
                if r.get("vanish") is not None and fr >= r["vanish"]:
                    break
                if sym != cur:
                    phases.append((fr, sym))
                    cur = sym
            return phases
        phases = [(0.0, 0)] if entries[0][0] > 0 else [(0.0, entries[0][1])]
        cur = phases[0][1]
        for fr, sym in entries:
            if sym != cur:
                phases.append((fr, sym))
                cur = sym
        return phases

    @staticmethod
    def _phase_text(phases: list[int]) -> str:
        def word(sym: int, first: bool) -> str:
            if sym > 0:
                return "turns clockwise"
            if sym < 0:
                return "turns counterclockwise"
            return "stays still" if first else "stops"

        if len(phases) == 1:
            sym = phases[0]
            if sym > 0:
                return "turns clockwise the whole time"
            if sym < 0:
                return "turns counterclockwise the whole time"
            return "stays still the whole time"
        return ", then ".join(word(s, i == 0) for i, s in enumerate(phases))

    def _parse_phase_option(self, option: str) -> list[int] | None:
        text = option.lower().strip()
        if text.endswith("the whole time"):
            if text.startswith("turns clockwise"):
                return [1]
            if text.startswith("turns counterclockwise"):
                return [-1]
            if text.startswith("stays still"):
                return [0]
            return None
        parts = text.split(", then ")
        syms = []
        for part in parts:
            part = part.strip()
            if part.startswith("turns clockwise"):
                syms.append(1)
            elif part.startswith("turns counterclockwise"):
                syms.append(-1)
            elif part.startswith("stops") or part.startswith("stays still"):
                syms.append(0)
            else:
                return None
        return syms

    def describe(self, r: dict, options, on_screen: bool = False):
        phases = [s for _, s in self.phase_schedule(r, on_screen)]
        for o in options:
            if self._parse_phase_option(o) == phases:
                return o
        return self._phase_text(phases)

    # --------------------------------------------------------------- marks
    def mark_adds(self, r: dict) -> int:
        i = self._idx[id(r)]
        return sum(1 for m in self.marks if m.get("ring") == i and m.get("add") is not None)

    def marks_carried(self, r: dict, f: float) -> int:
        i = self._idx[id(r)]
        n = 0
        for m in self.marks:
            if m.get("ring") != i:
                continue
            a = m.get("add")
            if a is None or a > f:
                continue
            rem = m.get("remove")
            if rem is None or f < rem:
                n += 1
        return n

    # -------------------------------------------------------------- handlers
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        options = list(options or [])

        # ---- ring order / geometry -------------------------------------
        m = re.search(r"what (shape|colour) is (?:the border of )?the (\d+)(?:st|nd|rd|th) outline from the outside", q)
        if m:
            return self._nth_outside(m.group(2), m.group(1) == "colour")

        m = re.search(r"what (shape|colour) is (?:the border of )?the (innermost|outermost) outline", q)
        if m:
            order = self.end_order()
            if not order:
                return Answer(None, confidence="unsupported")
            r = order[-1] if m.group(2) == "innermost" else order[0]
            val = self.colour(r, self.end_frame) if m.group(1) == "colour" else r["shape"]
            return Answer(val, confidence="exact")

        if "where in the finished nest is the outline that appears last" in q:
            order = self.end_order()
            if not order:
                return Answer(None, confidence="unsupported")
            last = max(self.rings, key=lambda r: float(r.get("appear") or 0))
            if not any(r is last for r in order):
                return Answer(None, confidence="unsupported")
            i = next(k for k, r in enumerate(order) if r is last)
            if i == 0:
                return Answer("outermost outline", confidence="exact")
            if i == len(order) - 1:
                return Answer("innermost outline", confidence="exact")
            return Answer(f"{_ordinal(i + 1)} outline from the outside", confidence="exact")

        m = re.search(r"if the (.+?) were taken out of the nest.*?which outline would be directly inside the (.+?) at the end", q)
        if m:
            removed = self.mention("the " + m.group(1))
            target = self.mention("the " + m.group(2))
            if removed is None or target is None:
                return Answer(None, confidence="unsupported")
            seq = [r for r in self.end_order() if r is not removed]
            if not any(r is target for r in seq):
                return Answer(None, confidence="unsupported")
            i = next(k for k, r in enumerate(seq) if r is target)
            return Answer(self._option_for(seq[i + 1], options), confidence="exact") if i + 1 < len(seq) else Answer(None, confidence="unsupported")

        m = re.search(r"which outline is directly inside the (.+?)(?:\?|$)", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            seq = self.end_order()
            if not any(r is target for r in seq):
                return Answer(None, confidence="unsupported")
            i = next(k for k, r in enumerate(seq) if r is target)
            return Answer(self._option_for(seq[i + 1], options), confidence="exact") if i + 1 < len(seq) else Answer(None, confidence="unsupported")

        m = re.search(r"how many outlines are inside the (.+?)(?:\?|$)", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            n = sum(1 for r in self.end_order() if float(r.get("size", 0)) < float(target.get("size", 0)))
            return Answer(str(n), confidence="exact")

        # ---- visibility counts -----------------------------------------
        if "nested outlines are on screen at the end" in q:
            return Answer(str(len(self.end_order())), confidence="exact")

        if "largest number of outlines on screen at the same time" in q:
            events: dict[int, int] = {}
            for r in self.rings:
                a = int(r.get("appear") or 0)
                events[a] = events.get(a, 0) + 1
                if r.get("vanish") is not None:
                    v = int(r["vanish"])
                    events[v] = events.get(v, 0) - 1
            cur = best = 0
            for f in sorted(events):
                if f > self.end_frame:
                    break
                cur += events[f]
                best = max(best, cur)
            return Answer(str(best), confidence="exact")

        m = re.search(r"if the video had stopped the moment the timer first read ([\d.]+) s, how many outlines would have been on screen", q)
        if m:
            f = float(m.group(1)) * FPS
            return Answer(str(sum(1 for r in self.rings if self.visible(r, f))), confidence="exact")

        m = re.search(r"how many outlines are on screen when the timer reads ([\d.]+) s", q)
        if m:
            f = float(m.group(1)) * FPS
            return Answer(str(sum(1 for r in self.rings if self.visible(r, f))), confidence="exact")

        # ---- appearance order ------------------------------------------
        m = re.search(r"how many outlines appear after the (.+?)(?:\?|$)", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            a = target.get("appear") or 0
            return Answer(str(sum(1 for r in self.rings if r is not target and (r.get("appear") or 0) > a)), confidence="exact")

        m = re.search(r"what does the timer read when the (.+?) appears", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            return Answer(_time_str(target.get("appear") or 0), confidence="exact")

        if "in which order do the outlines appear" in q:
            asp = [(r.get("appear") or 0, float(r.get("size", 0))) for r in self.rings]
            if len(set(a for a, _ in asp)) == 1:
                return Answer("all at the same moment", confidence="exact")
            by_appear = sorted(asp)
            if all(by_appear[i][1] >= by_appear[i + 1][1] for i in range(len(by_appear) - 1)):
                return Answer("from the outside in", confidence="exact")
            if all(by_appear[i][1] <= by_appear[i + 1][1] for i in range(len(by_appear) - 1)):
                return Answer("from the inside out", confidence="exact")
            return Answer("in a mixed order, neither outside-in nor inside-out", confidence="exact")

        if "which outline appears first" in q or "which one appears earliest" in q:
            opts = [r for r in (self.option_ring(o) for o in options) if r is not None]
            pool = opts if opts else self.rings
            earliest = min((r.get("appear") or 0) for r in pool)
            hits = [r for r in pool if (r.get("appear") or 0) == earliest]
            return Answer(self._option_for(hits[0], options), confidence="exact") if len(hits) == 1 else Answer(None, confidence="unsupported")

        if "disappears before the end" in q:
            return self._pick(options, lambda r: r.get("vanish") is not None and r["vanish"] <= self.end_frame)

        # ---- colour ------------------------------------------------------
        m = re.search(r"how many outlines have a ([\w]+) border", q)
        if m:
            col = m.group(1)
            n = sum(1 for r in self.end_order() if self.colour(r, self.end_frame) == col)
            return Answer(str(n), confidence="exact")

        m = re.search(r"what colour is the border of the (.+?) at the (end|start) of the video", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            f = self.end_frame if m.group(2) == "end" else 0
            return Answer(self.colour(target, f), confidence="exact")

        m = re.search(r"what colour is the border of the (.+?) when the timer reads ([\d.]+) s", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.colour(target, float(m.group(2)) * FPS), confidence="exact")

        m = re.search(r"if the video had stopped the moment the timer first read ([\d.]+) s, what colour would the border of the (.+?) have been", q)
        if m:
            target = self.mention("the " + m.group(2))
            if target is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.colour(target, float(m.group(1)) * FPS - 1e-6), confidence="exact")

        if "how many times does the" in q and "change colour" in q:
            target = self.mention(q)
            if target is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.changes(target))), confidence="exact")

        if "counting every outline, how many colour changes" in q:
            return Answer(str(sum(len(self.changes(r)) for r in self.rings)), confidence="exact")

        if "which border colours does the" in q and "show, in order" in q:
            target = self.mention(q)
            if target is None:
                return Answer(None, confidence="unsupported")
            cols = [target.get("color")] + [c for _, c in self.changes(target)]
            return Answer(", then ".join(cols), confidence="exact")

        m = re.search(r"what does the timer read when the (.+?) first changes colour", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            ch = self.changes(target)
            return Answer(_time_str(ch[0][0]), confidence="exact") if ch else Answer(None, confidence="unsupported")

        if "which of these outlines changes colour" in q:
            return self._pick(options, lambda r: bool(self.changes(r)))

        if "if the last colour change of the video had never happened" in q:
            m = re.search(r"would have a ([\w]+) border", q)
            col = m.group(1) if m else None
            events = [(fr, i, c) for i, r in enumerate(self.rings) for fr, c in self.changes(r)]
            if not events or col is None:
                return Answer(None, confidence="unsupported")
            last = max(events, key=lambda e: e[0])
            n = 0
            for i, r in enumerate(self.rings):
                if not self.visible(r, self.end_frame):
                    continue
                c = r.get("color")
                for fr, cc in self.changes(r):
                    if fr == last[0] and i == last[1] and cc == last[2]:
                        break
                    c = cc
                if c == col:
                    n += 1
            return Answer(str(n), confidence="exact")

        # ---- spin: counts ----------------------------------------------
        m = re.search(r"how many outlines (?:are|would have been) turning( clockwise)?", q)
        if m:
            cw = bool(m.group(1))
            if "at the end of the video" in q:
                f = self.end_frame
            else:
                tm = re.search(r"(?:timer reads|first read) ([\d.]+) s", q)
                if not tm:
                    return Answer(None, confidence="unsupported")
                f = float(tm.group(1)) * FPS
            if cw:
                return Answer(str(sum(1 for r in self.rings if self.rate(r, f) > 0)), confidence="exact")
            return Answer(str(sum(1 for r in self.rings if abs(self.rate(r, f)) > 1e-12)), confidence="exact")

        # pairwise
        m = re.search(r"when the timer reads ([\d.]+) s, how are the (.+?) and the (.+?) turning", q)
        if m:
            f = float(m.group(1)) * FPS
            a = self.mention("the " + m.group(2))
            b = self.mention("the " + m.group(3))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            sa, sb = _sign(self.rate(a, f)), _sign(self.rate(b, f))
            if sa and sb:
                return Answer("both turn the same way" if sa == sb else "they turn opposite ways", confidence="exact")
            if sa:
                return Answer(f"only the {m.group(2)} turns", confidence="exact")
            if sb:
                return Answer(f"only the {m.group(3)} turns", confidence="exact")
            return Answer("neither of them turns", confidence="exact")

        m = re.search(r"when the timer reads ([\d.]+) s, which of these outlines is turning the same way as the (.+?)(?:\?|$)", q)
        if m:
            f = float(m.group(1)) * FPS
            target = self.mention("the " + m.group(2))
            if target is None:
                return Answer(None, confidence="unsupported")
            st = _sign(self.rate(target, f))
            return self._pick(options, lambda r: _sign(self.rate(r, f)) == st)

        if "which describes how the" in q and "turns" in q:
            target = self.mention(q)
            if target is None:
                return Answer(None, confidence="unsupported")
            on_screen = "while it is on screen" in q
            return Answer(self.describe(target, options, on_screen), confidence="exact")

        if "how many times does the" in q and "change how it turns" in q:
            target = self.mention(q)
            if target is None:
                return Answer(None, confidence="unsupported")
            phases = self.phase_schedule(target)
            return Answer(str(sum(1 for i in range(1, len(phases)) if phases[i][1] != phases[i - 1][1])), confidence="exact")

        m = re.search(r"what does the timer read when the (.+?) first changes how it turns", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            phases = self.phase_schedule(target)
            return Answer(_time_str(phases[1][0]), confidence="exact") if len(phases) > 1 else Answer(None, confidence="unsupported")

        m = re.search(r"if the outlines had appeared in the reverse order.*which one would have been the (\w+) to appear", q)
        if m:
            words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7}
            k = words.get(m.group(1))
            if k is None:
                return Answer(None, confidence="unsupported")
            order = sorted(self.rings, key=lambda r: float(r.get("appear") or 0))
            if k > len(order):
                return Answer(None, confidence="unsupported")
            pos = len(order) - k
            tgt = order[pos]
            # ambiguous when the reversed slot sits inside a tie on the appear frame
            if sum(1 for r in order if (r.get("appear") or 0) == (tgt.get("appear") or 0)) > 1:
                return Answer(None, confidence="unsupported")
            return Answer(self._option_for(tgt, options), confidence="exact")

        if "never turns" in q:
            return self._pick(options, lambda r: all(abs(float(w)) < 1e-12 for _, w in r.get("spin", [])))

        if "changes how it turns" in q and "the most times" in q:
            def transitions(r):
                ph = self.phase_schedule(r)
                return sum(1 for i in range(1, len(ph)) if ph[i][1] != ph[i - 1][1])
            vals = {id(r): transitions(r) for r in self.rings}
            best = max(vals.values()) if vals else 0
            return self._pick(options, lambda r: vals[id(r)] == best)

        if "is the first to change how it turns" in q:
            def first_change(r):
                ph = self.phase_schedule(r)
                return ph[1][0] if len(ph) > 1 else None
            vals = {id(r): first_change(r) for r in self.rings}
            cand = [v for v in vals.values() if v is not None]
            if not cand:
                return Answer(None, confidence="unsupported")
            best = min(cand)
            return self._pick(options, lambda r: vals[id(r)] == best)

        if "turns the fastest" in q:
            opts = [r for r in (self.option_ring(o) for o in options) if r is not None]
            if not opts:
                return Answer(None, confidence="unsupported")
            best = max(abs(self.rate(r, self.end_frame)) for r in opts)
            return self._pick(options, lambda r: abs(self.rate(r, self.end_frame)) == best)

        if "had always turned the opposite way" in q:
            target = self.mention(q)
            if target is None:
                return Answer(None, confidence="unsupported")
            st = _sign(-self.rate(target, self.end_frame))
            n = sum(1 for r in self.rings
                    if r is not target and _sign(self.rate(r, self.end_frame)) == st)
            return Answer(str(n), confidence="exact")

        if "kept turning the way it turned at the start" in q:
            n = 0
            for r in self.rings:
                if not self.visible(r, self.end_frame):
                    continue
                first = next((float(w) for _, w in r.get("spin", [])), 0.0)
                if first > 0:
                    n += 1
            return Answer(str(n), confidence="exact")

        # ---- marks -------------------------------------------------------
        if "counting every outline, how many marks are placed" in q:
            return Answer(str(sum(1 for m in self.marks if m.get("add") is not None)), confidence="exact")

        if "counting every outline, how many marks are taken off" in q:
            n = sum(1 for m in self.marks if m.get("remove") is not None and m["remove"] <= self.end_frame)
            return Answer(str(n), confidence="exact")

        m = re.search(r"how many marks are placed between ([\d.]+) s and ([\d.]+) s", q)
        if m:
            a, b = float(m.group(1)) * FPS, float(m.group(2)) * FPS
            return Answer(str(sum(1 for mm in self.marks if mm.get("add") is not None and a <= mm["add"] < b)), confidence="exact")

        m = re.search(r"what does the timer read when the (.+?) gets its first mark", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            i = self._idx[id(target)]
            adds = [mm["add"] for mm in self.marks if mm.get("ring") == i and mm.get("add") is not None]
            return Answer(_time_str(min(adds)), confidence="exact") if adds else Answer(None, confidence="unsupported")

        m = re.search(r"how many more marks does the (.+?) carry than the (.+?)(?:\?|$)", q)
        if m:
            a = self.mention("the " + m.group(1))
            b = self.mention("the " + m.group(2))
            tm = re.search(r"when the timer reads ([\d.]+) s", q)
            if a is None or b is None or not tm:
                return Answer(None, confidence="unsupported")
            f = float(tm.group(1)) * FPS
            return Answer(str(self.marks_carried(a, f) - self.marks_carried(b, f)), confidence="exact")

        m = re.search(r"how many more marks are placed on the (.+?) than on the (.+?)(?:\?|$)", q)
        if m:
            a = self.mention("the " + m.group(1))
            b = self.mention("the " + m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.mark_adds(a) - self.mark_adds(b)), confidence="exact")

        m = re.search(r"if no mark had ever been taken off, how many marks would the (.+?) carry at the end", q)
        if m:
            target = self.mention("the " + m.group(1))
            if target is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.mark_adds(target)), confidence="exact")

        if "which outline gets a mark first" in q:
            adds = {}
            for i, r in enumerate(self.rings):
                v = [mm["add"] for mm in self.marks if mm.get("ring") == i and mm.get("add") is not None]
                adds[i] = min(v) if v else None
            cand = [v for v in adds.values() if v is not None]
            if not cand:
                return Answer(None, confidence="unsupported")
            best = min(cand)
            return self._pick(options, lambda r: adds.get(self._idx[id(r)]) == best)

        if "loses a mark during the video" in q:
            return self._pick(options, lambda r: any(mm.get("ring") == self._idx[id(r)] and mm.get("remove") is not None for mm in self.marks))

        if "which outline has the most marks placed on it" in q:
            counts = {i: sum(1 for mm in self.marks if mm.get("ring") == i) for i in range(len(self.rings))}
            best = max(counts.values()) if counts else 0
            return self._pick(options, lambda r: counts.get(self._idx[id(r)]) == best)

        m = re.search(r"if the video had stopped the moment the timer first read ([\d.]+) s, how many marks would have been on the outlines", q)
        if m:
            f = float(m.group(1)) * FPS
            n = sum(1 for mm in self.marks if mm.get("add") is not None and mm["add"] <= f and (mm.get("remove") is None or f < mm["remove"]))
            return Answer(str(n), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

    # ------------------------------------------------------------- helpers
    def _name(self, r: dict) -> str:
        return f"{r.get('color')} {r.get('shape')}"

    def _option_for(self, r: dict, options) -> str:
        for o in options:
            if self.option_ring(o) is r:
                return o
        return self._name(r)

    def _pick(self, options, predicate) -> Answer:
        hit = self.pick_option(options, predicate)
        if hit is None:
            return Answer(None, confidence="unsupported")
        return Answer(hit, confidence="exact")

    def _nth_outside(self, nth: str, want_colour: bool) -> Answer:
        order = self.end_order()
        i = int(nth) - 1
        if not 0 <= i < len(order):
            return Answer(None, confidence="unsupported")
        r = order[i]
        return Answer(self.colour(r, self.end_frame) if want_colour else r["shape"], confidence="exact")

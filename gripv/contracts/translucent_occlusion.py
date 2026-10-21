"""Translucent occlusion — deterministic replay of the generator's recipe.

Conventions pinned from the six anchor clips (all 10 keys reproduced) and
calibrated over the full 2,500-clip params set:

  * an actor is **completely hidden** when its bounding disc (centre + r) is
    fully inside a panel: for a rect ``x0+r <= x <= x1-r`` and the same in y,
    for a circle ``hypot(x-cx, y-cy) <= R-r``.  Positions are rounded to two
    decimals first — the generator's own arithmetic lands on that grid (one
    clip sits 0.0035 px outside the edge at a frame boundary and its key
    treats that frame as hidden).
  * actor keyframes are linear in frame; a hold repeats the position, and a
    trailing hold past the last keyframe keeps the final position.
  * ``changes`` are ``[frame, field, value]`` triples applied *at* the frame;
    the appearance "when it went in" is the state one frame before the first
    fully-hidden frame, the appearance "when it comes back out" is the state
    at the first frame after the last fully-hidden frame.
  * clocks truncate to 0.1 s.  ``for how long`` is the difference of the two
    displayed clock readings (entry clock, exit clock), not a raw duration.
  * "becomes completely hidden" is the first **transition** into hiding; an
    actor that is already hidden at frame 0 has not "become" hidden there, so
    its first later entry is used (it still counts for "for how long" and for
    "comes out", which use the first hidden stretch).
  * "first starts to come out" is the first frame after a fully-hidden
    stretch.  Left/right sides follow the panel's horizontal order.
  * counterfactual "had started moving 1 s later": the first keyframe keeps
    its frame and every later keyframe moves +24 frames, so the actor holds
    its initial position for one extra second before following the same path.
    Its colour/shape changes keep their original frames.  "Which shape would
    be the first to start coming out" ignores actors whose colour or shape
    changes at some point (their name is ambiguous across the reveal); a pure
    radius change does not exclude them.
  * "which of these shapes ..." answers and order answers are option text.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0
EPS = 1e-6

COLORS = r"(red|orange|yellow|green|blue|purple|pink|cyan|white)"
SHAPES = r"(triangle|diamond|pentagon|hexagon|circle|square|cross|star)"

PANEL_SHAPE = {"rect": "rectangle", "circle": "circle"}


# ---------------------------------------------------------------- geometry
def _pos(kf: list, f: float) -> tuple[float, float]:
    if f <= kf[0][0]:
        return kf[0][1], kf[0][2]
    for i in range(len(kf) - 1):
        f0, x0, y0 = kf[i]
        f1, x1, y1 = kf[i + 1]
        if f0 <= f <= f1:
            t = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
            return round(x0 + (x1 - x0) * t, 2), round(y0 + (y1 - y0) * t, 2)
    return round(kf[-1][1], 2), round(kf[-1][2], 2)


def _bbox(panel: dict) -> tuple[float, float, float, float]:
    if panel["kind"] == "rect":
        return panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    return (panel["cx"] - panel["R"], panel["cy"] - panel["R"],
            panel["cx"] + panel["R"], panel["cy"] + panel["R"])


def _contains(panel: dict, x: float, y: float, r: float) -> bool:
    if panel["kind"] == "rect":
        return (panel["x0"] + r - EPS <= x <= panel["x1"] - r + EPS
                and panel["y0"] + r - EPS <= y <= panel["y1"] - r + EPS)
    return math.hypot(x - panel["cx"], y - panel["cy"]) <= panel["R"] - r + EPS


def _overlaps(panel: dict, x: float, y: float, r: float) -> bool:
    if panel["kind"] == "rect":
        dx = max(panel["x0"] - x, 0.0, x - panel["x1"])
        dy = max(panel["y0"] - y, 0.0, y - panel["y1"])
        return dx * dx + dy * dy < r * r
    return math.hypot(x - panel["cx"], y - panel["cy"]) < panel["R"] + r


def clock(frames: float) -> float:
    """Clock reading in seconds, truncated to 0.1 s."""
    return math.floor((frames / FPS) * 10.0 + 1e-9) / 10.0


def fmt_clock(frames: float) -> str:
    return f"{clock(frames):.1f} s"


# ---------------------------------------------------------------- contract
class TranslucentOcclusion:
    def __init__(self, params: dict):
        self.p = params
        self.panels = params["panels"]
        self.actors = params["actors"]
        self.n = int(params["frames"])
        self.fps = FPS

    # ----------------------------- actor state
    def _attrs(self, i: int, f: float) -> tuple[str, str, float]:
        a = self.actors[i]
        color, shape, r = a["color"], a["shape"], a["r"]
        for fr, field, val in a.get("changes", []):
            if fr <= f:
                if field == "color":
                    color = val
                elif field == "shape":
                    shape = val
                elif field == "r":
                    r = val
        return color, shape, r

    def _name(self, i: int, f: float = 0.0) -> str:
        color, shape, _ = self._attrs(i, f)
        return f"{color} {shape}"

    def _find(self, color: str, shape: str) -> int | None:
        for i in range(len(self.actors)):
            c, s, _ = self._attrs(i, 0)
            if c == color and s == shape:
                return i
        return None

    # ----------------------------- hidden intervals
    def _hidden_row(self, i: int) -> list[bool]:
        cache = self.p.setdefault("_hidden_cache", {})
        if i in cache:
            return cache[i]
        a = self.actors[i]
        row = []
        for f in range(self.n):
            x, y = _pos(a["kf"], f)
            _, _, r = self._attrs(i, f)
            row.append(any(_contains(p, x, y, r) for p in self.panels))
        cache[i] = row
        return row

    def _segments(self, i: int) -> list[tuple[int, int]]:
        row = self._hidden_row(i)
        segs, start = [], None
        for f, h in enumerate(row):
            if h and start is None:
                start = f
            elif not h and start is not None:
                segs.append((start, f - 1))
                start = None
        if start is not None:
            segs.append((start, self.n - 1))
        return segs

    def _become_segments(self, i: int) -> list[tuple[int, int]]:
        """Stretches that begin after frame 0 (transitions into hiding)."""
        return [s for s in self._segments(i) if s[0] > 0]

    def _entry(self, i: int) -> int | None:
        segs = self._become_segments(i)
        return segs[0][0] if segs else None

    def _exit(self, i: int) -> int | None:
        segs = self._segments(i)
        return segs[0][1] + 1 if segs else None

    def _panel_index(self, which: str) -> int:
        if len(self.panels) == 1:
            return 0
        if which == "left":
            return min(range(len(self.panels)), key=lambda j: _bbox(self.panels[j])[0])
        return max(range(len(self.panels)), key=lambda j: _bbox(self.panels[j])[0])

    def _entry_panel(self, i: int) -> int | None:
        segs = self._become_segments(i)
        if not segs:
            return None
        f = segs[0][0]
        x, y = _pos(self.actors[i]["kf"], f)
        _, _, r = self._attrs(i, f)
        for j, p in enumerate(self.panels):
            if _contains(p, x, y, r):
                return j
        return None

    def _exit_side(self, i: int, j: int) -> str | None:
        e = self._exit(i)
        if e is None:
            return None
        f = min(e, self.n - 1)
        x, y = _pos(self.actors[i]["kf"], f)
        _, _, r = self._attrs(i, f)
        panel = self.panels[j]
        if panel["kind"] == "rect":
            dl = x - (panel["x0"] + r)
            dr = (panel["x1"] - r) - x
            dt = y - (panel["y0"] + r)
            db = (panel["y1"] - r) - y
            best = min(dl, dr, dt, db)
            if best == dl:
                return "left"
            if best == dr:
                return "right"
            if best == dt:
                return "top"
            return "bottom"
        x0, y0, x1, y1 = _bbox(panel)
        dx, dy = x - (x0 + x1) / 2, y - (y0 + y1) / 2
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "bottom" if dy > 0 else "top"

    def _count_hidden(self, f: int) -> int:
        return sum(1 for i in range(len(self.actors)) if self._hidden_row(i)[f])

    def _max_hidden(self) -> int:
        return max(self._count_hidden(f) for f in range(self.n))

    # ----------------------------- counterfactual
    def _delayed(self, i: int, frames: int = 24) -> "TranslucentOcclusion":
        """This actor starts moving 1 s later; later keyframes shift, the first stays."""
        p = dict(self.p)
        p["actors"] = [dict(a) for a in self.actors]
        p["actors"][i] = dict(p["actors"][i])
        kf = p["actors"][i]["kf"]
        p["actors"][i]["kf"] = [[kf[0][0] if k == 0 else fr + frames, x, y]
                                for k, (fr, x, y) in enumerate(kf)]
        p["actors"][i]["changes"] = [list(c) for c in p["actors"][i].get("changes", [])]
        p.pop("_hidden_cache", None)
        return TranslucentOcclusion(p)

    def _static_name_actors(self) -> list[int]:
        return [i for i, a in enumerate(self.actors)
                if not any(k in ("color", "shape") for _, k, _ in a.get("changes", []))]

    # ----------------------------- dispatch
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = question.strip()
        ql = q.lower()
        a = self._dispatch(q, ql, options or [])
        if a is None:
            return Answer(None, confidence="unsupported", note=f"no handler: {q[:80]}")
        return a

    def _dispatch(self, q: str, ql: str, options: list[str]) -> Answer | None:
        refs = re.findall(COLORS + r" " + SHAPES, ql)

        # ---- static description ------------------------------------------------
        if ql.startswith("what shape is the panel"):
            j = 0
            return Answer(PANEL_SHAPE[self.panels[j]["kind"]])
        if ql.startswith("what shape is the left panel"):
            j = self._panel_index("left")
            return Answer(PANEL_SHAPE[self.panels[j]["kind"]])
        if ql.startswith("what shape is the right panel"):
            j = self._panel_index("right")
            return Answer(PANEL_SHAPE[self.panels[j]["kind"]])
        if ql.startswith("how many translucent panels"):
            return Answer(str(len(self.panels)))
        if ql.startswith("how many shapes appear"):
            return Answer(str(len(self.actors)))

        m = re.match(r"what shape is the " + COLORS + r" shape", ql)
        if m:
            for i in range(len(self.actors)):
                c, s, _ = self._attrs(i, 0)
                if c == m.group(1):
                    return Answer(s)
            return None

        m = re.match(r"what colour is the " + SHAPES, ql)
        if m:
            for i in range(len(self.actors)):
                c, s, _ = self._attrs(i, 0)
                if s == m.group(1):
                    return Answer(c)
            return None

        if "completely outside the panels" in ql:
            n = 0
            for i in range(len(self.actors)):
                x, y = _pos(self.actors[i]["kf"], 0)
                _, _, r = self._attrs(i, 0)
                if not any(_overlaps(p, x, y, r) for p in self.panels):
                    n += 1
            return Answer(str(n))

        # ---- panel membership --------------------------------------------------
        if ql.startswith("behind which panel"):
            i = self._find(*refs[0])
            if i is None:
                return None
            j = self._entry_panel(i)
            if j is None:
                return Answer("neither panel")
            return Answer("the left panel" if j == 0 else "the right panel")

        if "on which side of the" in ql:
            i = self._find(*refs[0])
            if i is None:
                return None
            j = (self._panel_index("left") if "left panel" in ql
                 else self._panel_index("right") if "right panel" in ql else 0)
            return Answer(self._exit_side(i, j))

        # ---- ordering between two named actors ---------------------------------
        if ql.startswith("which goes completely behind a panel first"):
            ia, ib = self._find(*refs[0]), self._find(*refs[1])
            return self._first(ia, ib, self._entry, hidden=True)
        if ql.startswith("which starts to come out"):
            ia, ib = self._find(*refs[0]), self._find(*refs[1])
            return self._first(ia, ib, self._exit, hidden=False)

        # ---- option-set membership --------------------------------------------
        if "which of these shapes" in ql:
            want_hidden = "never" not in ql
            ever = {i for i in range(len(self.actors)) if self._segments(i)}
            target = ever if want_hidden else set(range(len(self.actors))) - ever
            for o in options or []:
                hit = re.findall(COLORS + r" " + SHAPES, str(o).lower())
                if not hit:
                    continue
                i = self._find(*hit[0])
                if i in target:
                    return Answer(o)
            return None

        # ---- counterfactuals (before the generic clock/count branches) ---------
        if "suppose the panel" in ql:
            return Answer(str(len(self.actors)))

        if "had started moving" in ql:
            i = self._find(*refs[0])
            if i is None:
                return None
            cf = self._delayed(i)
            if "which shape would be the first" in ql:
                cand = [(cf._exit(k), k) for k in cf._static_name_actors()
                        if cf._exit(k) is not None]
                if not cand:
                    return None
                cand.sort()
                return Answer("the " + cf._name(cand[0][1]))
            if "at the end of the video" in ql:
                return Answer(str(cf._count_hidden(cf.n - 1)))
            mm = re.search(r"when the clock shows ([\d.]+) s", ql)
            if mm:
                t = float(mm.group(1))
                return Answer(str(cf._count_hidden(
                    max(0, min(cf.n - 1, int(round(t * 24)))))))
            return None

        # ---- clocks and counts ------------------------------------------------
        if "for how long" in ql:
            i = self._find(*refs[0])
            if i is None:
                return None
            segs = self._segments(i)
            if not segs:
                return Answer("it never goes behind a panel")
            return Answer(f"{clock(segs[0][1] + 1) - clock(segs[0][0]):.1f} s")

        if "becomes completely hidden" in ql:
            i = self._find(*refs[0])
            e = self._entry(i) if i is not None else None
            return Answer(fmt_clock(e)) if e is not None else None

        if "first starts to come out" in ql:
            i = self._find(*refs[0])
            e = self._exit(i) if i is not None else None
            return Answer(fmt_clock(e)) if e is not None else None

        m = re.search(r"when the clock shows ([\d.]+) s", ql)
        if m:
            t = float(m.group(1))
            return Answer(str(self._count_hidden(max(0, min(self.n - 1, int(round(t * 24)))))))

        if ql.startswith("at the end of the video"):
            return Answer(str(self._count_hidden(self.n - 1)))
        if "largest number" in ql:
            return Answer(str(self._max_hidden()))
        if "how many different shapes" in ql:
            return Answer(str(sum(1 for i in range(len(self.actors)) if self._segments(i))))

        # ---- order lists -------------------------------------------------------
        if "in what order" in ql:
            idxs = [self._find(*r) for r in refs]
            fn = self._exit if "start to come out" in ql else self._entry
            order = sorted(idxs, key=lambda i: (fn(i) is None, fn(i) or 0))
            return Answer(", then ".join("the " + self._name(i) for i in order))

        # ---- appearance / size across the panel -------------------------------
        if "what did it look like when it went in" in ql:
            want = refs[0]
            i = None
            for k in range(len(self.actors)):
                x = self._exit(k)
                if x is None:
                    continue
                c, s, _ = self._attrs(k, min(self.n - 1, x))
                if (c, s) == want:
                    i = k
                    break
            if i is None:
                return None
            e = self._entry(i)
            if e is None:
                e = self._segments(i)[0][0]
            c, s, _ = self._attrs(i, max(0, e - 1))
            return Answer(f"a {c} {s}")

        if "what does it look like when it comes back out" in ql:
            i = self._find(*refs[0])
            x = self._exit(i) if i is not None else None
            if x is None:
                return None
            c, s, _ = self._attrs(i, min(self.n - 1, x))
            return Answer(f"a {c} {s}")

        if "how has its size changed" in ql:
            i = self._find(*refs[0])
            if i is None:
                return None
            segs = self._segments(i)
            if not segs:
                return Answer("it never comes out")
            e, x = segs[0]
            r0 = self._attrs(i, max(0, e - 1))[2]
            r1 = self._attrs(i, min(self.n - 1, x + 1))[2]
            if abs(r1 - r0) < 1e-9:
                return Answer("the same size")
            return Answer("bigger" if r1 > r0 else "smaller")

        return None

    def _first(self, ia: int | None, ib: int | None, fn, hidden: bool) -> Answer | None:
        if ia is None or ib is None:
            return None
        fa, fb = fn(ia), fn(ib)
        if fa is None and fb is None:
            return Answer("neither goes completely behind a panel" if hidden else "neither comes out")
        if fa is None:
            return Answer("the " + self._name(ib))
        if fb is None:
            return Answer("the " + self._name(ia))
        if fa == fb:
            return Answer("both at the same moment")
        return Answer("the " + self._name(ia if fa < fb else ib))

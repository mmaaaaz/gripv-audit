"""Occluded pattern counting — deterministic replay of the generator's recipe.

Conventions pinned from the six anchor clips (all 10 keys reproduced):

  * occluder keyframe (x, y) is the **top-left corner** of the w x h block —
    not its centre.  Calibrated on 000000/000001/000002/000005 where the
    centre reading gives 5/1/0/0 hidden instead of 2/3/4/2.
  * an object counts as hidden when its **centre** lies inside the block.
    Partly covered objects stay "visible" (the option list carries a separate
    "partly hidden"/"only partly" distractor that never becomes a key).
  * clock = frame / 24 fps; a hold [a, b] keeps the block still through frame
    b, so "starts to move" and "stay still at the start" both read b.
  * rest positions are the hold starts, in order.

Everything else is plain set algebra over the per-frame hidden set.
"""
from __future__ import annotations

import collections
import math
import re

from ..engine import Answer

FPS = 24.0

ARRANGEMENT = {
    "grid": "a rectangular grid",
    "ring": "a ring",
    "triangle": "a triangle",
    "honeycomb": "a honeycomb (offset rows)",
}

def _pos(occ: dict, f: float) -> tuple[float, float]:
    kf = occ["kf"]
    if f <= kf[0][0]:
        return kf[0][1], kf[0][2]
    for i in range(len(kf) - 1):
        f0, x0, y0 = kf[i]
        f1, x1, y1 = kf[i + 1]
        if f0 <= f <= f1:
            t = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
            return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    return kf[-1][1], kf[-1][2]


def _rect(occ: dict, f: float) -> tuple[float, float, float, float]:
    x, y = _pos(occ, f)
    return x, y, x + occ["w"], y + occ["h"]


def _inside(obj: dict, r) -> bool:
    x0, y0, x1, y1 = r
    return x0 <= obj["x"] <= x1 and y0 <= obj["y"] <= y1


def _touches(obj: dict, r) -> bool:
    x0, y0, x1, y1 = r
    dx = max(x0 - obj["x"], 0.0, obj["x"] - x1)
    dy = max(y0 - obj["y"], 0.0, obj["y"] - y1)
    return dx * dx + dy * dy < obj["r"] ** 2


def _secs_round(f: float) -> str:
    """Clock reading for the start-of-video hold: nearest 0.1 s, .5 rounds down."""
    v = math.floor((f / FPS) * 10.0 + 0.5 - 1e-9) / 10.0
    return f"{v:.1f} s"


def _secs_floor(f: float) -> str:
    """Clock reading when a move ends: truncated to 0.1 s."""
    v = math.floor((f / FPS) * 10.0) / 10.0
    return f"{v:.1f} s"


class OccludedPattern:
    def __init__(self, params: dict):
        self.p = params
        self.objects = params["objects"]
        self.occluders = params["occluders"]
        self.holds = params["holds"]
        self.frames = int(params["frames"])
        self.last = self.frames - 1
        self.n = len(self.objects)
        self._cache: dict[tuple, set[int]] = {}
        self._occ_cache: dict[tuple, list[set[int]]] = {}

    # ---------------- geometry ----------------
    def rest_frames(self) -> list[int]:
        return [int(h[0]) for h in self.holds]

    def hidden(self, f: float, skip: int | None = None) -> set[int]:
        key = (f, skip)
        if key in self._cache:
            return self._cache[key]
        out: set[int] = set()
        for i, oc in enumerate(self.occluders):
            if skip is not None and i == skip:
                continue
            r = _rect(oc, f)
            for j, o in enumerate(self.objects):
                if _inside(o, r):
                    out.add(j)
        self._cache[key] = out
        return out

    def hidden_by(self, f: float) -> list[set[int]]:
        key = (f, "by")
        if key in self._occ_cache:
            return self._occ_cache[key]
        res = []
        for oc in self.occluders:
            r = _rect(oc, f)
            res.append({j for j, o in enumerate(self.objects) if _inside(o, r)})
        self._occ_cache[key] = res
        return res

    def state(self, obj: dict, f: float) -> str:
        """hidden | partly | visible"""
        for oc in self.occluders:
            r = _rect(oc, f)
            if _inside(obj, r):
                return "hidden"
        for oc in self.occluders:
            if _touches(obj, _rect(oc, f)):
                return "partly"
        return "visible"

    # ---------------- timeline ----------------
    def first_move_end(self) -> int:
        return int(self.occluders[0]["kf"][1][0])

    def last_stop(self) -> int:
        return int(max(oc["kf"][-1][0] for oc in self.occluders))

    def second_last_frame(self) -> int | None:
        kf = self.occluders[0]["kf"]
        if len(kf) < 4:
            return None
        return int(kf[-2][0])

    def direction(self, occ: dict) -> str:
        kf = occ["kf"]
        # skip the duplicate keys that open the first hold
        i = 0
        while i + 1 < len(kf) and kf[i + 1][1] == kf[0][1] and kf[i + 1][2] == kf[0][2]:
            i += 1
        dx = kf[i + 1][1] - kf[0][1]
        dy = kf[i + 1][2] - kf[0][2]
        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"

    def part_hidden(self, f: float) -> str:
        xs = [o["x"] for o in self.objects]
        ys = [o["y"] for o in self.objects]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        rects = [_rect(oc, f) for oc in self.occluders]
        rx = sum((r[0] + r[2]) / 2.0 for r in rects) / len(rects)
        ry = sum((r[1] + r[3]) / 2.0 for r in rects) / len(rects)
        # normalise by the pattern extent so wide patterns stay comparable
        spanx = max(max(xs) - min(xs), 1.0)
        spany = max(max(ys) - min(ys), 1.0)
        dx = (rx - cx) / spanx
        dy = (ry - cy) / spany
        if abs(dy) >= abs(dx):
            return "the bottom part" if dy > 0 else "the top part"
        return "the right part" if dx > 0 else "the left part"

    # ---------------- answers ----------------
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = question.strip()
        ql = q.lower()
        a = self._dispatch(q, ql)
        if a is None:
            return Answer(None, confidence="unsupported", note=f"no handler: {q[:80]}")
        return a

    def _dispatch(self, q: str, ql: str) -> Answer | None:
        p = self.p
        opts = {}

        # ---- static description ----
        if ql.startswith("what shape are the objects"):
            return Answer(p["shape"])
        if ql.startswith("how are the") and "arranged" in ql:
            return Answer(ARRANGEMENT.get(p["family"], p["family"]))
        if ql.startswith("what colour are most"):
            cnt = collections.Counter(o["color"] for o in self.objects)
            top = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
            return Answer(top[0])
        if "are there in total" in ql and ("counting the ones" in ql or ql.startswith("how many")):
            colour = self._colour_in(ql)
            if colour:
                return Answer(str(sum(1 for o in self.objects if o["color"] == colour)))
            return Answer(str(self.n))
        if ql.startswith("how many occluders"):
            return Answer(str(len(self.occluders)))
        if "empty places" in ql:
            return Answer(str(len(p.get("gaps") or [])))

        # ---- visibility at a moment ----
        if "at the start of the video, how many" in ql:
            return Answer(str(self.n - len(self.hidden(0))))
        if "at the end of the video, how many" in ql and "can you see" in ql:
            return Answer(str(self.n - len(self.hidden(self.last))))

        if "when the clock shows" in ql:
            t = self._time_in(q)
            f = round(t * FPS)
            h = len(self.hidden(min(max(f, 0), self.last)))
            if "visible" in ql:
                return Answer(str(self.n - h))
            return Answer(str(h))

        # ---- part of the pattern ----
        if ql.startswith("which part of the pattern is hidden"):
            f = 0 if "at the start" in ql else self.last
            return Answer(self.part_hidden(f))

        # ---- motion ----
        if "which way does" in ql and "move the first time" in ql:
            colour = self._colour_in(ql)
            if colour:
                for oc in self.occluders:
                    if oc["color"] == colour:
                        return Answer(self.direction(oc))
                return Answer(None, confidence="unsupported", note="occluder colour not found")
            return Answer(self.direction(self.occluders[0]))

        if "stay still at the start" in ql:
            return Answer(_secs_round(self.first_move_end()))
        if "first starts to move" in ql or "first start to move" in ql:
            return Answer(_secs_round(self.first_move_end()))
        if "stops moving for the last time" in ql or "stop moving for the last time" in ql:
            return Answer(_secs_floor(self.last_stop()))

        # ---- transitions ----
        S = self.hidden(0)
        E = self.hidden(self.last)
        if "hidden at the start but visible at the end" in ql:
            return Answer(str(len(S - E)))
        if "visible at the start but hidden at the end" in ql:
            return Answer(str(len(E - S)))
        if ql.startswith("are more") and "visible at the start or at the end" in ql:
            if not S and not E:
                return Answer("no objects are visible at either time")
            vs, ve = self.n - len(S), self.n - len(E)
            if vs > ve:
                return Answer("more at the start")
            if ve > vs:
                return Answer("more at the end")
            return Answer("the same number")

        # ---- single-object state ----
        if "visible or hidden" in ql and "is the" in ql:
            colour = self._colour_in(ql)
            if not colour:
                return Answer(None, confidence="unsupported", note="colour not parsed")
            cands = [o for o in self.objects if o["color"] == colour]
            if not cands:
                return Answer("there is no such object")
            states = {self.state(o, self.last) for o in cands}
            st = "hidden" if "hidden" in states else ("partly" if "partly" in states else "visible")
            return Answer("hidden" if st == "hidden" else ("partly hidden" if st == "partly" else "visible"))
        if "is the" in ql and "hidden?" in ql:
            colour = self._colour_in(ql)
            if not colour:
                return Answer(None, confidence="unsupported", note="colour not parsed")
            cands = [o for o in self.objects if o["color"] == colour]
            if not cands:
                return Answer("there is no such object")
            f = 0 if "at the start" in ql else self.last
            states = {self.state(o, f) for o in cands}
            st = "hidden" if "hidden" in states else ("partly" if "partly" in states else "visible")
            if "hidden?" in ql:
                return Answer("yes" if st == "hidden" else ("only partly" if st == "partly" else "no"))
            return Answer("hidden" if st == "hidden" else ("partly hidden" if st == "partly" else "visible"))

        # ---- resting-position families ----
        if "standing still" in ql:
            rests = [self.hidden(f) for f in self.rest_frames()]
            if "how many different" in ql:
                u: set[int] = set()
                for s in rests:
                    u |= s
                return Answer(str(len(u)))
            if "largest number" in ql:
                return Answer(str(max((len(s) for s in rests), default=0)))
        if "at which of its resting positions" in ql:
            rests = [self.hidden(f) for f in self.rest_frames()]
            counts = [len(s) for s in rests]
            best = max(counts) if counts else 0
            if counts.count(best) > 1:
                return Answer("it hides the same number everywhere")
            idx = counts.index(best)
            return Answer(["its first position", "its second position",
                           "its third position", "its fourth position"][idx])

        # ---- first move ----
        if "during the first move" in ql:
            kf1 = int(self.occluders[0]["kf"][2][0])
            after = self.hidden(kf1)
            before = self.hidden(0)
            if after == before and kf1 == 0:
                return Answer("it does not move")
            newly = len(after - before)
            uncovered = len(before - after)
            if newly == uncovered:
                return Answer("the same number")
            return Answer("it covers more" if newly > uncovered else "it uncovers more")
        if "after the first move" in ql:
            kf1 = int(self.occluders[0]["kf"][2][0])
            after, before = len(self.hidden(kf1)), len(self.hidden(0))
            if after == before == 0:
                return Answer("none are hidden")
            if after > before:
                return Answer("more")
            if after < before:
                return Answer("fewer")
            return Answer("the same number")

        # ---- which occluder hides more ----
        if "which occluder hides more" in ql:
            by = self.hidden_by(self.last)
            counts = [len(s) for s in by]
            if len(counts) < 2:
                return Answer(None, confidence="unsupported", note="single occluder")
            if counts[0] == counts[1]:
                if counts[0] == 0:
                    return Answer("neither hides any object")
                return Answer("they hide the same number")
            idx = 0 if counts[0] > counts[1] else 1
            return Answer("the %s occluder" % self.occluders[idx]["color"])

        # ---- counterfactuals ----
        if "never moved from where" in ql:
            frozen = self.hidden(0)
            if "never be visible" in ql:
                return Answer(str(len(frozen)))
            if "would be hidden" in ql:
                return Answer(str(len(frozen)))
            if "would be visible" in ql:
                return Answer(str(self.n - len(frozen)))
        if "were taken away" in ql:
            colour = self._colour_in(ql)
            idx = None
            for i, oc in enumerate(self.occluders):
                if oc["color"] == colour:
                    idx = i
            if idx is None:
                return Answer(None, confidence="unsupported", note="occluder colour not found")
            h = len(self.hidden(self.last, skip=idx))
            if "would be hidden" in ql:
                return Answer(str(h))
            return Answer(str(self.n - h))
        if "stopped one move early" in ql:
            f = self.second_last_frame()
            if f is None:
                return Answer(None, confidence="unsupported", note="no earlier position")
            h = len(self.hidden(f))
            if "would be hidden" in ql:
                return Answer(str(h))
            return Answer(str(self.n - h))

        return None

    # ---------------- helpers ----------------
    def _colour_in(self, ql: str) -> str | None:
        toks = set(re.findall(r"[a-z]+", ql))
        for t in ("pink", "cyan", "orange", "yellow", "purple", "blue", "red", "green",
                  "teal", "plum", "slate", "olive", "brown", "gray", "grey", "gold",
                  "navy", "maroon", "lime", "magenta", "violet", "indigo", "beige",
                  "tan", "salmon", "coral", "khaki", "amber", "rose", "mint", "sage",
                  "rust", "sand", "sky", "forest", "black", "white"):
            if t in toks:
                return t
        return None

    def _time_in(self, q: str) -> float:
        m = re.search(r"clock shows\s+(\d+(?:\.\d+)?)\s*s", q)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)\s*s", q)
        return float(m.group(1)) if m else 0.0

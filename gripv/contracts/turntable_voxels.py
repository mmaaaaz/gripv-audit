"""Deterministic replay for ``turntable_voxels``: voxel structure, turntable, cameras.

Everything is rebuilt from the generator's own recipe in ``params``: the
``heights[y][x]`` column grid, the ``legs`` keyframes (``t0, t1, a0, a1`` in
degrees), the ``events`` list (``drop`` / ``lift`` at ``x, y, z``), the
``marked`` cubes (``colour, x, y, z``) and ``elev``.

Pinned conventions (each verified against keyed rows; see
``results/turntable_voxels_notes.md``):

* A structure rotation of ``+a`` degrees is counter-clockwise seen from above
  and moves the near edge to the **right**; ``-a`` moves it left.  The camera
  always looks from the front (``-y``) side.  The side facing the camera after
  a net turn ``theta`` is the original side with normal ``270 - theta``:
  ``theta % 360 == 0`` front, ``90`` left, ``180`` back, ``270`` right.
* The renderer is an orthographic shear: screen ``v = cos(elev) * z +
  sin(elev) * y`` (up) and screen ``u = x``.  So a cube shows its front face
  (height ``cos``) and its top face (height ``sin``) and nothing else; a
  column at depth ``y`` is drawn shifted **up** by ``sin(elev) * y``.
* A cube counts as visible when its front face or (for the top cube) its top
  face is not covered by nearer columns at the same screen column.  A nearer
  column covers ``v <= cos(elev) * h + sin(elev) * (y + 1)``.  A residue
  thinner than ``TAU`` world units is swallowed by the 3 px outlines, so it is
  not visible (any ``TAU`` in 0.03 .. 0.15 reproduces every keyed row).
* "Completely hidden" is ``total - visible``.  A marked cube is visible iff
  its ``(x, y, z)`` is in the visible set after rotating it with the grid.
* Events use their landing/removal layer: a ``drop`` at ``z`` sets the column
  height to ``z + 1``; a ``lift`` at ``z`` sets it to ``z``.  "At the end of
  the video" applies every event.
* Clocks ("answer in whole seconds") are the keyframe times rounded to the
  nearest second; "how far" answers the first leg's |delta| in quarter-turn
  words; "quarter turns during the video" sums |delta| over legs.
* The "look straight ... no tilt" silhouette is the per-screen-column maximum
  of the (rotated) grid, read left to right.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

TAU = 0.05
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink")

SIDE_BY_Q = {
    0: "the side that faced you at the start",
    1: "the side that was on your left at the start",
    2: "the side that was at the back at the start",
    3: "the side that was on your right at the start",
}

TURN_WORDS = {
    1: ("a quarter turn", 1),
    2: ("half a turn", 2),
    3: ("three quarters of a turn", 3),
    4: ("one full turn", 4),
    5: ("one and a quarter turns", 5),
    6: ("one and a half turns", 6),
    7: ("one and three quarter turns", 7),
    8: ("two full turns", 8),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("\u2013", "-")).strip()


def _pick(options: list[str], *patterns: str) -> str | None:
    """Return option text matching the first pattern (exact before substring)."""
    texts = [_norm(o) for o in options]
    for pat in patterns:
        for opt, text in zip(options, texts):
            if re.fullmatch(pat, text):
                return opt
    for pat in patterns:
        for opt, text in zip(options, texts):
            if re.search(pat, text):
                return opt
    return None


def _sig(x: float) -> int:
    return 1 if x > 1e-9 else (-1 if x < -1e-9 else 0)


def _quarter_words(quarters: int) -> str | None:
    word = TURN_WORDS.get(quarters % 8)
    return word[0] if word else None


class TurntableVoxels:
    def __init__(self, params: dict):
        self.p = params
        self.gx = int(params.get("gx") or 0)
        self.gy = int(params.get("gy") or 0)
        self.h0 = [list(row) for row in (params.get("heights") or [])]
        self.legs = list(params.get("legs") or [])
        self.events = sorted(params.get("events") or [], key=lambda e: e["t0"])
        self.marked = list(params.get("marked") or [])
        self.elev = float(params.get("elev") or 35.0)
        self.e = math.radians(self.elev)
        self.c = math.cos(self.e)
        self.s = math.sin(self.e)

    # ---------------------------------------------------------------- heights
    def heights_at(self, t: float | None = None, drop: bool = True, lift: bool = True) -> list[list[int]]:
        h = [row[:] for row in self.h0]
        for e in self.events:
            if t is not None and e["t1"] > t + 1e-9:
                continue
            if e["kind"] == "drop" and not drop:
                continue
            if e["kind"] == "lift" and not lift:
                continue
            if e["kind"] == "drop":
                h[e["y"]][e["x"]] = max(h[e["y"]][e["x"]], e["z"] + 1)
            else:
                h[e["y"]][e["x"]] = min(h[e["y"]][e["x"]], e["z"])
        return h

    @staticmethod
    def total(grid: list[list[int]]) -> int:
        return sum(sum(row) for row in grid)

    # --------------------------------------------------------------- rotation
    @staticmethod
    def rot(grid: list[list[int]], gx: int, gy: int, q: int):
        g = [row[:] for row in grid]
        for _ in range(q % 4):
            ngx, ngy = gy, gx
            out = [[0] * ngx for _ in range(ngy)]
            for y in range(gy):
                for x in range(gx):
                    out[x][gy - 1 - y] = g[y][x]
            g = out
            gx, gy = ngx, ngy
        return g, gx, gy

    @staticmethod
    def rot_pt(x: int, y: int, gx: int, gy: int, q: int) -> tuple[int, int]:
        for _ in range(q % 4):
            x, y = gy - 1 - y, x
            gx, gy = gy, gx
        return x, y

    # ------------------------------------------------------------------ angle
    def net_angle(self) -> float:
        return sum((lg["a1"] - lg["a0"]) for lg in self.legs)

    def total_path(self) -> float:
        return sum(abs(lg["a1"] - lg["a0"]) for lg in self.legs)

    def angle_at(self, t: float) -> float:
        if not self.legs:
            return 0.0
        ang = self.legs[0]["a0"]
        for lg in self.legs:
            if t < lg["t0"]:
                break
            if t >= lg["t1"]:
                ang = lg["a1"]
                continue
            frac = (t - lg["t0"]) / max(lg["t1"] - lg["t0"], 1e-9)
            return lg["a0"] + (lg["a1"] - lg["a0"]) * frac
        return ang

    def q_at(self, angle: float) -> int:
        return int(round(angle / 90.0)) % 4

    # -------------------------------------------------------------- visibility
    def visible(self, grid: list[list[int]], q: int = 0) -> set[tuple[int, int, int]]:
        g, gx, gy = self.rot(grid, self.gx, self.gy, q)
        c, s = self.c, self.s
        vis: set[tuple[int, int, int]] = set()
        for x in range(gx):
            cover = -1e9
            for y in range(gy):
                h = g[y][x]
                if h <= 0:
                    continue
                for z in range(h):
                    lo = c * z + s * y
                    hi = lo + c
                    front = hi - max(cover, lo)
                    top = (hi + s) - max(cover, hi) if z == h - 1 else 0.0
                    if max(front, top) > TAU:
                        vis.add((x, y, z))
                cover = max(cover, c * h + s * (y + 1))
        return vis

    def visible_count(self, grid: list[list[int]], q: int = 0) -> int:
        return len(self.visible(grid, q))

    # -------------------------------------------------------------- silhoutte
    def silhouette(self, grid: list[list[int]], q: int = 0) -> list[int]:
        g, gx, gy = self.rot(grid, self.gx, self.gy, q)
        return [max((g[y][x] for y in range(gy)), default=0) for x in range(gx)]

    # ------------------------------------------------------------------ marks
    def mark(self, colour: str) -> dict | None:
        for m in self.marked:
            if m["colour"] == colour:
                return m
        return None

    def marked_visible(self, grid, q, t=None) -> set[str]:
        vis = self.visible(grid, q)
        out = set()
        for m in self.marked:
            x, y = self.rot_pt(m["x"], m["y"], self.gx, self.gy, q)
            if (x, y, m["z"]) in vis:
                out.add(m["colour"])
        return out

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, options=None) -> Answer:
        q = _norm(question)
        opts = list(options or [])
        out = self._answer(q, opts)
        return out if out is not None else Answer(
            None, confidence="unsupported", note="template family not pinned"
        )

    def _answer(self, q: str, opts: list[str]) -> Answer | None:
        # ---- near-edge direction -------------------------------------------
        if "which way does its near edge" in q:
            deltas = [lg["a1"] - lg["a0"] for lg in self.legs]
            nonzero = [d for d in deltas if _sig(d)]
            if not nonzero:
                return Answer(None, confidence="unsupported", note="turntable never turns")
            signs = {_sig(d) for d in nonzero}
            if len(signs) == 1:
                pats = (r"always to the right",) if nonzero[0] > 0 else (r"always to the left",)
            elif _sig(nonzero[0]) > 0:
                pats = (r"first to the right, later to the left",)
            else:
                pats = (r"first to the left, later to the right",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="near-edge option wording changed")

        # ---- how far before it stops / before a change ----------------------
        if "how far does the turntable turn before it stops" in q or \
                "before the turntable first stops or changes direction" in q:
            if not self.legs:
                return Answer(None, confidence="unsupported", note="no legs")
            delta = self.legs[0]["a1"] - self.legs[0]["a0"]
            quarters = int(round(abs(delta) / 90.0))
            words = _quarter_words(quarters)
            if not words:
                return Answer(None, confidence="unsupported", note="turn length outside table")
            side = "right" if delta > 0 else "left"
            pick = _pick(opts, rf"{re.escape(words)}, with its near edge moving {side}")
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="turn-length option wording changed")

        # ---- quarter turns over the whole clip ------------------------------
        if "through how many quarter turns" in q:
            return Answer(str(int(round(self.total_path() / 90.0))), confidence="exact")

        # ---- grid -----------------------------------------------------------
        if "how many squares does the grid" in q:
            return Answer(str(self.gx * self.gy), confidence="exact")

        # ---- counts of cubes in the structure -------------------------------
        end = ("at the end of the video" in q)
        if "how many cubes are in the structure altogether" in q:
            grid = self.heights_at() if end else self.h0
            return Answer(str(self.total(grid)), confidence="exact")
        if "how many squares of the grid does at least one cube stand" in q:
            return Answer(str(sum(1 for row in self.h0 for h in row if h > 0)), confidence="exact")
        if "how many cubes drop onto the structure" in q:
            return Answer(str(sum(1 for e in self.events if e["kind"] == "drop")), confidence="exact")
        if "how many cubes rise off the structure" in q:
            return Answer(str(sum(1 for e in self.events if e["kind"] == "lift")), confidence="exact")

        # ---- tallest stack ---------------------------------------------------
        if "how many cubes tall is the tallest stack" in q:
            grid = self.heights_at() if end else self.h0
            return Answer(str(max((h for row in grid for h in row), default=0)), confidence="exact")

        # ---- one more cube on every stack ------------------------------------
        if "one more cube had been placed on top of every stack" in q:
            extra = sum(1 for row in self.h0 for h in row if h > 0)
            return Answer(str(self.total(self.h0) + extra), confidence="exact")

        # ---- one of these colours --------------------------------------------
        if "one of these colours is on a cube" in q:
            present = {m["colour"] for m in self.marked}
            pick = next((o for o in opts if _norm(o) in present), None)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="no marked colour in options")

        # ---- highest marked cube ---------------------------------------------
        if "sits highest above the turntable" in q:
            if not self.marked:
                return Answer(None, confidence="unsupported", note="no marked cubes")
            top = max(m["z"] for m in self.marked)
            leaders = [m for m in self.marked if m["z"] == top]
            if len(leaders) > 1:
                pick = _pick(opts, r"all .* at the same height")
                return Answer(pick, confidence="exact") if pick else Answer(
                    None, confidence="unsupported", note="tie option wording changed")
            pick = _pick(opts, rf"^{re.escape(leaders[0]['colour'])}$")
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="colour option missing")

        # ---- marked-cube stack relations -------------------------------------
        m_under = re.search(r"how many cubes are under the (\w+) cube in its stack", q)
        if m_under:
            m = self.mark(m_under.group(1))
            if not m:
                return Answer(None, confidence="unsupported", note="marked colour not found")
            return Answer(str(m["z"]), confidence="exact")
        m_above = re.search(r"how many cubes are stacked on top of the (\w+) cube", q)
        if m_above:
            m = self.mark(m_above.group(1))
            if not m:
                return Answer(None, confidence="unsupported", note="marked colour not found")
            h = self.h0[m["y"]][m["x"]]
            return Answer(str(max(0, h - m["z"] - 1)), confidence="exact")
        m_layers = re.search(r"how many layers higher than the (\w+) cube is the (\w+) cube", q)
        if m_layers:
            a = self.mark(m_layers.group(1))
            b = self.mark(m_layers.group(2))
            if not a or not b:
                return Answer(None, confidence="unsupported", note="marked colour not found")
            return Answer(str(max(0, b["z"] - a["z"])), confidence="exact")

        # ---- visible / hidden counts -----------------------------------------
        if "how many cubes can you see (even partly)" in q:
            grid, qq = self._state_for(q)
            return Answer(str(self.visible_count(grid, qq)), confidence="exact")
        if "how many cubes are completely hidden" in q:
            grid, qq = self._state_for(q)
            return Answer(str(self.total(grid) - self.visible_count(grid, qq)), confidence="exact")

        # ---- visible marked colours ------------------------------------------
        if "which of the coloured cubes can you see" in q:
            grid, qq = self._state_for(q)
            seen = self.marked_visible(grid, qq)
            pick = next((o for o in opts if self._colour_opt(o) == seen), None)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="colour-set option not listed")

        # ---- visible & top-of-stack ------------------------------------------
        m_see = re.search(r"can you see the (\w+) cube \(even partly\), and is it the top cube", q)
        if m_see:
            m = self.mark(m_see.group(1))
            if not m:
                return Answer(None, confidence="unsupported", note="marked colour not found")
            vis = self.visible(self.h0, 0)
            visible = (m["x"], m["y"], m["z"]) in vis
            top = m["z"] == self.h0[m["y"]][m["x"]] - 1
            pat = {
                (True, True): r"visible, and it is the top cube",
                (True, False): r"visible, but it is not the top cube",
                (False, True): r"hidden, and it is the top cube",
                (False, False): r"hidden, and it is not the top cube",
            }[(visible, top)]
            pick = _pick(opts, pat)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="visibility option wording changed")

        # ---- which side faces you --------------------------------------------
        if "which side of the structure faces you" in q:
            if "first time the turntable stops partway" in q:
                if len(self.legs) < 2:
                    return Answer(None, confidence="unsupported", note="no partway stop")
                angle = self.legs[0]["a1"]
            else:
                angle = self.net_angle()
            pick = _pick(opts, re.escape(SIDE_BY_Q[self.q_at(angle)]))
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="side option wording changed")
        if "same turns, but each one in the opposite direction" in q:
            pick = _pick(opts, re.escape(SIDE_BY_Q[self.q_at(-self.net_angle())]))
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="side option wording changed")

        # ---- silhouette -------------------------------------------------------
        if "how many cubes tall" in q and "each column" in q and "from left to right" in q:
            if "whole stack holding the" in q:
                m = re.search(r"whole stack holding the (\w+) cube", q)
                mark = self.mark(m.group(1)) if m else None
                if not mark:
                    return Answer(None, confidence="unsupported", note="marked colour not found")
                grid = [row[:] for row in self.h0]
                grid[mark["y"]][mark["x"]] = 0
                sil = self.silhouette(grid, 0)
            else:
                grid, qq = self._state_for(q)
                sil = self.silhouette(grid, qq)
            return Answer(", ".join(str(v) for v in sil), confidence="exact")

        # ---- stack removed -> hidden before turning ---------------------------
        if "whole stack holding the" in q and "had been taken away" in q and "completely hidden" in q:
            m = re.search(r"whole stack holding the (\w+) cube", q)
            mark = self.mark(m.group(1)) if m else None
            if not mark:
                return Answer(None, confidence="unsupported", note="marked colour not found")
            grid = [row[:] for row in self.h0]
            grid[mark["y"]][mark["x"]] = 0
            return Answer(str(self.total(grid) - self.visible_count(grid, 0)), confidence="exact")

        # ---- counterfactual visibility ---------------------------------------
        if "last turn had ended a quarter turn sooner" in q:
            grid = self.heights_at()
            last = self.legs[-1] if self.legs else None
            if not last:
                return Answer(None, confidence="unsupported", note="no legs")
            step = 90.0 if (last["a1"] - last["a0"]) > 0 else -90.0
            qq = self.q_at(last["a1"] - step)
            return Answer(str(self.visible_count(grid, qq)), confidence="exact")
        if "no cubes had been dropped" in q:
            grid = self.heights_at(drop=False)
            return Answer(str(self.visible_count(grid, self.q_at(self.net_angle()))), confidence="exact")
        if "no cubes had been lifted off" in q:
            grid = self.heights_at(lift=False)
            return Answer(str(self.visible_count(grid, self.q_at(self.net_angle()))), confidence="exact")

        # ---- clocks -----------------------------------------------------------
        if "at about which clock reading" in q:
            t = self._clock_time(q)
            if t is None:
                return Answer(None, confidence="unsupported", note="event/leg not present")
            return Answer(str(int(round(t))), confidence="exact")
        if "for about how many seconds does it stand still" in q:
            if len(self.legs) < 2:
                return Answer(None, confidence="unsupported", note="no partway stop")
            return Answer(str(int(round(self.legs[1]["t0"] - self.legs[0]["t1"]))), confidence="exact")

        return None

    # ------------------------------------------------------------------ utils
    def _state_for(self, q: str) -> tuple[list[list[int]], int]:
        if "before the turntable starts" in q or "faces you at the start" in q:
            return self.h0, 0
        if "first time the turntable stops partway" in q:
            t = self.legs[0]["t1"] if self.legs else 0.0
            return self.heights_at(t), self.q_at(self.legs[0]["a1"] if self.legs else 0.0)
        return self.heights_at(), self.q_at(self.net_angle())

    def _colour_opt(self, opt: str) -> frozenset[str]:
        o = _norm(opt)
        if "none" in o:
            return frozenset()
        return frozenset(c for c in COLOURS if re.search(rf"\b{c}\b", o))

    def _clock_time(self, q: str) -> float | None:
        if "start to turn" in q:
            return self.legs[0]["t0"] if self.legs else None
        if "change direction" in q:
            prev = None
            for lg in self.legs:
                d = lg["a1"] - lg["a0"]
                if prev is not None and _sig(d) and _sig(prev) and _sig(d) != _sig(prev):
                    return lg["t0"]
                if _sig(d):
                    prev = d
            return None
        if "first stop partway" in q:
            return self.legs[0]["t1"] if len(self.legs) > 1 else None
        m = re.search(r"(first|second) (dropped cube land|dropped cube lands|cube start to rise)", q)
        if m:
            nth = 0 if m.group(1) == "first" else 1
            want = "drop" if "dropped" in m.group(2) else "lift"
            evs = [e for e in self.events if e["kind"] == want]
            if len(evs) <= nth:
                return None
            return evs[nth]["t1"] if want == "drop" else evs[nth]["t0"]
        return None

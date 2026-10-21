"""Deterministic replay for spinning_polyhedra: solid geometry, spin motion, timing.

Everything is rebuilt from the generator's own recipe in ``params``:
``bodies[].solid``, ``bodies[].motion[]`` (frames, axis, signed deg/s) and
``marks[]``. No pixels, no key lookups.

Pinned conventions (each verified against all 25,000 keyed rows; see
``results/spinning_polyhedra_notes.md``):

* Frame ``f`` is at ``f / 24`` s. A motion segment ``[f0, f1]`` integrates over
  ``(f1 - f0) / 24`` s with the speed linear from ``w0`` to ``w1``. Questions
  that cut a segment short (timer readings) use the interpolated speed at the
  cut, not the segment average.
* ``w`` is signed deg/s about the named axis. For ``y`` a positive ``w`` moves
  the near-side edges **right**, negative **left**; for ``x`` positive moves
  them **down**, negative **up**. ``z`` clips are never asked about edge
  direction (the near face rotates in the image plane).
* A stop is a segment with ``w0 == w1 == 0``. Its clock reads ``f0 / 24``
  truncated to 0.1 s; the restart clock reads ``(f1 + 1) / 24`` truncated.
* "Complete turns" and "turn further" count rotation distance
  (integral of ``|w|``), not net signed angle. Reverse clips therefore keep
  accumulating turns in both directions.
* "Count the complete turns in each stretch between stops, then add them up"
  floors each stretch separately; "during the clip" floors the whole path.

Marker families (``marks[].v`` indexes the renderer's vertex table, which the
params do not carry) stay ``unsupported``: left/right and near/far of a marker
cannot be derived from a bare vertex index. ``How many edges meet at the corner
with the <marker>`` is answered whenever every corner has the same degree
(Platonic/Archimedean solids, prisms, antiprisms) and left unsupported for
pyramids, bipyramids and the rhombic dodecahedron, which mix corner degrees.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..engine import Answer

FPS = 24.0

# face-shape vocabulary used by "What shape are most of the faces?"
TRI, QUAD, PENT, HEX = "tri", "quad", "pent", "hex"
SHAPE_OPTION = {TRI: "triangles", QUAD: "four-sided", PENT: "pentagons", HEX: "hexagons"}
FACE_SIDES = {TRI: 3, QUAD: 4, PENT: 5, HEX: 6}


@dataclass(frozen=True)
class Solid:
    """One entry of the closed table of solids the generator draws."""

    faces: int
    edges: int
    corners: int
    shapes: tuple[tuple[str, int], ...] = ()
    family: str = ""          # prism | antiprism | pyramid | bipyramid
    n: int = 0                # ring size for the family counterfactuals
    vertex_transitive: bool = False
    uniform_corners: bool = False   # every corner has the same degree, but not transitive

    def variety(self) -> int:
        """How many distinct face shapes the solid has."""
        return len(self.shapes)

    def majority(self) -> str | None:
        """Most common face shape, or None on a tie."""
        if not self.shapes:
            return None
        ranked = sorted(self.shapes, key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def degree(self) -> int | None:
        """Edges meeting at a corner, only when every corner is equivalent.

        Average degree is 2E/V; for a uniform solid every corner hits it. Prism
        corners are all 3-valent (two ring edges + one upright) and antiprism
        corners all 4-valent, so both are uniform even though their corners are
        not all the same distance from the centre. Pyramids, bipyramids and the
        rhombic dodecahedron mix degrees, so a bare vertex index cannot pick one.
        """
        if not (self.vertex_transitive or self.uniform_corners):
            return None
        return (2 * self.edges) // self.corners


# Faces, edges, corners, face-shape census. Euler holds for all of them
# (V - E + F == 2); validate() also checks 2E == sum of face sides.
SOLIDS: dict[str, Solid] = {
    # Platonic
    "tetrahedron": Solid(4, 6, 4, ((TRI, 4),), vertex_transitive=True),
    "cube": Solid(6, 12, 8, ((QUAD, 6),), vertex_transitive=True),
    "hexahedron": Solid(6, 12, 8, ((QUAD, 6),), vertex_transitive=True),
    "octahedron": Solid(8, 12, 6, ((TRI, 8),), vertex_transitive=True),
    "dodecahedron": Solid(12, 30, 20, ((PENT, 12),), vertex_transitive=True),
    "icosahedron": Solid(20, 30, 12, ((TRI, 20),), vertex_transitive=True),
    # pyramids
    "triangular pyramid": Solid(4, 6, 4, ((TRI, 4),), "pyramid", 3, True),
    "square pyramid": Solid(5, 8, 5, ((TRI, 4), (QUAD, 1)), "pyramid", 4),
    "pentagonal pyramid": Solid(6, 10, 6, ((TRI, 5), (PENT, 1)), "pyramid", 5),
    "hexagonal pyramid": Solid(7, 12, 7, ((TRI, 6), (HEX, 1)), "pyramid", 6),
    # prisms
    "triangular prism": Solid(5, 9, 6, ((TRI, 2), (QUAD, 3)), "prism", 3, False, True),
    "pentagonal prism": Solid(7, 15, 10, ((PENT, 2), (QUAD, 5)), "prism", 5, False, True),
    "hexagonal prism": Solid(8, 18, 12, ((HEX, 2), (QUAD, 6)), "prism", 6, False, True),
    # bipyramids
    "triangular bipyramid": Solid(6, 9, 5, ((TRI, 6),), "bipyramid", 3),
    "pentagonal bipyramid": Solid(10, 15, 7, ((TRI, 10),), "bipyramid", 5),
    "hexagonal bipyramid": Solid(12, 18, 8, ((TRI, 12),), "bipyramid", 6),
    # antiprisms
    "square antiprism": Solid(10, 16, 8, ((TRI, 8), (QUAD, 2)), "antiprism", 4, False, True),
    "pentagonal antiprism": Solid(12, 20, 10, ((TRI, 10), (PENT, 2)), "antiprism", 5, False, True),
    "hexagonal antiprism": Solid(14, 24, 12, ((TRI, 12), (HEX, 2)), "antiprism", 6, False, True),
    # Archimedean / Catalan
    "cuboctahedron": Solid(14, 24, 12, ((TRI, 8), (QUAD, 6)), vertex_transitive=True),
    "truncated tetrahedron": Solid(8, 18, 12, ((TRI, 4), (HEX, 4)), vertex_transitive=True),
    "truncated octahedron": Solid(14, 36, 24, ((QUAD, 6), (HEX, 8)), vertex_transitive=True),
    "rhombic dodecahedron": Solid(12, 24, 14, ((QUAD, 12),)),
}

# The key always spells the solid the way the params do; these aliases only
# exist so a re-bake that swaps the naming still resolves.
ALIASES = {"hexahedron": "cube", "triangular pyramid": "tetrahedron"}

AXIS_LONG = {
    "y": "an upright axis (it turns like a turntable)",
    "x": "a left-right axis (it tumbles forward or backward)",
    "z": "a front-back axis (it turns like a clock hand)",
}
AXIS_SHORT = {"y": "an upright axis", "x": "a left-right axis", "z": "a front-back axis"}
AXIS_PAT = {"y": r"upright", "x": r"left-right", "z": r"front-back"}

MARK_RE = re.compile(r"\b(red|green|blue|yellow|cyan|magenta|orange) (dot|square|diamond)\b")
MARKER_FAMILIES = (
    "cross in front of the spin axis",
    "sweeps around the widest circle",
    "which marker is the first",
    "where is the ",
    "where would the ",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("\u2013", "-")).strip()


def _pick(options: list[str], *patterns: str) -> str | None:
    """Return the option text matching the first pattern (verbatim, for the key).

    Exact wording wins before substring wording: "slowing down" must not pick
    the option "speeding up, then slowing down" just because it contains it.
    """
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


def _int(value: float) -> int:
    """Floor toward -inf; every turn count handed to this is non-negative."""
    return int(math.floor(value + 1e-9))


def _clock(seconds: float) -> str:
    """0.1 s clock reading, truncated the way the generator prints it."""
    return f"{math.floor(seconds * 10.0 + 1e-9) / 10.0:.1f} s"


def _sign(w: float) -> int:
    return 1 if w > 1e-9 else (-1 if w < -1e-9 else 0)


class SpinningPolyhedra:
    def __init__(self, params: dict):
        self.p = params
        self.bodies = params.get("bodies", []) or []
        self.marks = params.get("marks", []) or []
        self.n_frames = int(params.get("n_frames") or 0)

    # ------------------------------------------------------------------ bodies
    def _pair(self) -> tuple[dict, dict] | None:
        if len(self.bodies) < 2:
            return None
        return (min(self.bodies, key=lambda b: b["cx"]), max(self.bodies, key=lambda b: b["cx"]))

    def _mark(self, q: str) -> tuple[dict, dict] | None:
        m = MARK_RE.search(q)
        if not m:
            return None
        for mark in self.marks:
            if mark.get("colour") == m.group(1) and mark.get("shape") == m.group(2):
                i = int(mark.get("body", 0))
                if 0 <= i < len(self.bodies):
                    return mark, self.bodies[i]
        return None

    def _target(self, q: str) -> dict | None:
        if "on the left" in q:
            return min(self.bodies, key=lambda b: b["cx"]) if self.bodies else None
        if "on the right" in q:
            return max(self.bodies, key=lambda b: b["cx"]) if self.bodies else None
        hit = self._mark(q)
        if hit:
            return hit[1]
        return self.bodies[0] if len(self.bodies) == 1 else None

    # ------------------------------------------------------------------ motion
    @staticmethod
    def _stop_segments(b: dict) -> list[dict]:
        return [m for m in b.get("motion", []) if _sign(m["w0"]) == 0 and _sign(m["w1"]) == 0]

    @staticmethod
    def _speed_at(m: dict, t: float) -> float:
        f0, f1 = m["f0"] / FPS, m["f1"] / FPS
        if t <= f0:
            return m["w0"]
        if t >= f1:
            return m["w1"]
        return m["w0"] + (m["w1"] - m["w0"]) * (t - f0) / (f1 - f0)

    def _path(self, b: dict, upto: float | None = None, start: float = 0.0) -> float:
        """Rotation distance in degrees: integral of |w| over the window."""
        total = 0.0
        for m in b.get("motion", []):
            f0, f1 = m["f0"] / FPS, m["f1"] / FPS
            if f1 <= start:
                continue
            lo = max(f0, start)
            hi = min(f1, upto) if upto is not None else f1
            if hi <= lo:
                continue
            total += (hi - lo) * abs((m["w0"] + self._speed_at(m, hi)) / 2.0)
        return total

    @staticmethod
    def _duration(b: dict) -> float:
        motion = b.get("motion", [])
        if not motion:
            return 0.0
        return (motion[-1]["f1"] - motion[0]["f0"]) / FPS

    def _first_stop_time(self, b: dict) -> float | None:
        stops = self._stop_segments(b)
        return stops[0]["f0"] / FPS if stops else None

    @staticmethod
    def _speed_trend(b: dict) -> str:
        """Shape of |w| over the clip, in the four option wordings."""
        samples: list[float] = []
        for m in b.get("motion", []):
            for w in (m["w0"], m["w1"]):
                if not samples or abs(abs(w) - samples[-1]) > 1e-6:
                    samples.append(abs(w))
        rises = any(samples[i + 1] > samples[i] + 1e-6 for i in range(len(samples) - 1))
        falls = any(samples[i + 1] < samples[i] - 1e-6 for i in range(len(samples) - 1))
        if not rises and not falls:
            return "keeping a steady speed"
        if rises and not falls:
            return "speeding up"
        if falls and not rises:
            return "slowing down"
        peak = max(range(len(samples)), key=lambda i: samples[i])
        up_then_down = all(samples[i] <= samples[i + 1] + 1e-6 for i in range(peak)) and all(
            samples[i] >= samples[i + 1] - 1e-6 for i in range(peak, len(samples) - 1)
        )
        return "speeding up, then slowing down" if up_then_down else "slowing down, then speeding up"

    # ------------------------------------------------------------------ geometry
    def _solid(self, b: dict) -> Solid | None:
        name = str(b.get("solid", "")).lower().strip()
        return SOLIDS.get(name) or SOLIDS.get(ALIASES.get(name, ""))

    # ------------------------------------------------------------------ answers
    def answer(self, question: str, options=None) -> Answer:
        q = _norm(question)
        opts = list(options or [])
        out = self._answer(q, opts)
        return out if out is not None else Answer(
            None, confidence="unsupported", note="template family not pinned"
        )

    def _answer(self, q: str, opts: list[str]) -> Answer | None:
        # ---- marker families: need the renderer's vertex table -----------------
        if any(k in q for k in MARKER_FAMILIES):
            return Answer(None, confidence="unsupported",
                          note="marker kinematics need per-vertex coordinates (not in params)")

        # ---- identity and counts ---------------------------------------------
        if "compared with its number of corners" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            solid = self._solid(b)
            if not solid:
                return Answer(None, confidence="unsupported", note=f"unknown solid {b.get('solid')!r}")
            if solid.faces == 2 * solid.corners:
                pats = (r"twice as many faces as corners",)
            elif solid.faces == solid.corners:
                pats = (r"same number of each",)
            elif solid.faces > solid.corners:
                pats = (r"more faces than corners",)
            else:
                pats = (r"more corners than faces",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="comparison option wording changed")

        if "which solid is the solid" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            name = ALIASES.get(str(b["solid"]).lower(), str(b["solid"]).lower())
            pick = _pick(opts, rf"^{re.escape(name)}$")
            return Answer(pick or name, confidence="exact" if pick else "derived")

        m = re.search(r"how many (faces|edges|corners) does the solid(?: on the (left|right))? have", q)
        if m:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            solid = self._solid(b)
            if not solid:
                return Answer(None, confidence="unsupported", note=f"unknown solid {b.get('solid')!r}")
            return Answer(str({"faces": solid.faces, "edges": solid.edges,
                               "corners": solid.corners}[m.group(1)]), confidence="exact")

        m = re.search(r"how many (faces|edges|corners) do the two solids have altogether", q)
        if m:
            pair = self._pair()
            if not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")
            total = 0
            for b in pair:
                solid = self._solid(b)
                if not solid:
                    return Answer(None, confidence="unsupported", note=f"unknown solid {b.get('solid')!r}")
                total += {"faces": solid.faces, "edges": solid.edges,
                          "corners": solid.corners}[m.group(1)]
            return Answer(str(total), confidence="exact")

        if "which is true of the faces" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid:
                return Answer(None, confidence="unsupported", note="unknown/ambiguous body")
            pats = {1: (r"all the same shape",), 2: (r"two different shapes",),
                    3: (r"three different shapes",), 4: (r"four different shapes",)}
            pick = _pick(opts, *pats[solid.variety()])
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="face-variety option wording changed")

        if "what shape are most of the faces" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid:
                return Answer(None, confidence="unsupported", note="unknown/ambiguous body")
            shape = solid.majority()
            if shape is None:
                return Answer(None, confidence="unsupported", note="tied face shapes")
            pick = _pick(opts, SHAPE_OPTION[shape])
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="face-shape option wording changed")

        if "which solid has more corners, and which has more faces" in q:
            pair = self._pair()
            if not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")
            left, right = pair
            ls, rs = self._solid(left), self._solid(right)
            if not ls or not rs:
                return Answer(None, confidence="unsupported", note="unknown solid")
            lc, rc = ls.corners > rs.corners, rs.corners > ls.corners
            lf, rf = ls.faces > rs.faces, rs.faces > ls.faces
            if (lc or rc) and (lf or rf):
                if lc and lf:
                    pats = (r"left one has more of both",)
                elif rc and rf:
                    pats = (r"right one has more of both",)
                elif lc and rf:
                    pats = (r"left one has more corners",)
                else:
                    pats = (r"right one has more corners",)
            else:
                pats = (r"same number",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="corner/face comparison option wording changed")

        if "how many edges meet at the corner with the" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid:
                return Answer(None, confidence="unsupported", note="unknown/ambiguous body")
            deg = solid.degree()
            if deg is None:
                return Answer(None, confidence="unsupported",
                              note=f"{b['solid']} has several corner types; vertex index alone is ambiguous")
            return Answer(str(deg), confidence="exact")

        # ---- geometric counterfactuals ---------------------------------------
        if "sliced off flat" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid:
                return Answer(None, confidence="unsupported", note="unknown/ambiguous body")
            if "new face" in q:                     # one new face per corner
                return Answer(str(solid.faces + solid.corners), confidence="exact")
            return Answer(str(2 * solid.edges), confidence="exact")  # each edge -> two corners

        if "putting one corner at the centre of each face" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid:
                return Answer(None, confidence="unsupported", note="unknown/ambiguous body")
            return Answer(str(solid.faces), confidence="exact")       # dual: V* = F

        if "two matching end faces" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid or solid.family not in ("prism", "antiprism"):
                return Answer(None, confidence="unsupported", note="not a prism/antiprism")
            n = solid.n + 1
            return Answer(str(3 * n if solid.family == "prism" else 4 * n), confidence="exact")

        if "middle ring of corners" in q:
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid or solid.family != "bipyramid":
                return Answer(None, confidence="unsupported", note="not a bipyramid")
            return Answer(str(3 * (solid.n + 1)), confidence="exact")

        if re.search(r"one face of the solid(?: on the (?:left|right))? that is not a triangle", q):
            b = self._target(q)
            solid = self._solid(b) if b else None
            if not solid or solid.family != "pyramid":
                return Answer(None, confidence="unsupported", note="not a pyramid with a non-triangular base")
            return Answer(str(2 * (solid.n + 1)), confidence="exact")

        # ---- axis ------------------------------------------------------------
        if "about which kind of axis" in q:
            b = self._target(q)
            if not b or not b.get("motion"):
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            seg = b["motion"][0] if "at the very start" in q else b["motion"][-1]
            pick = _pick(opts, AXIS_PAT[seg["axis"]])
            return Answer(pick or AXIS_LONG[seg["axis"]], confidence="exact" if pick else "derived")

        if "which describes how the solid turns" in q:
            b = self._target(q)
            if not b or len(b.get("motion", [])) < 2:
                return Answer(None, confidence="unsupported", note="single-axis clip")
            a0, a1 = b["motion"][0]["axis"], b["motion"][-1]["axis"]
            want = f"first about {AXIS_SHORT[a0]}, then about {AXIS_SHORT[a1]}"
            pick = _pick(opts, re.escape(want))
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="axis-sequence option wording changed")

        # ---- edge direction --------------------------------------------------
        if "do the edges" in q and "compared with the very start" in q:
            b = self._target(q)
            if not b or not b.get("motion"):
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            first, last = b["motion"][0], b["motion"][-1]
            if first["axis"] == last["axis"]:
                same = _sign(first["w0"]) == _sign(last["w1"])
                pats = (r"the same way",) if same else (r"opposite way",)
            elif first["axis"] == "y" and last["axis"] == "x":
                pats = (r"up or down",)
            elif first["axis"] == "y" and last["axis"] == "z":
                pats = (r"clock hand",)
            else:
                return Answer(None, confidence="unsupported",
                              note=f"unpinned edge-direction change {first['axis']}->{last['axis']}")
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="edge-change option wording changed")

        if "which way do the edges" in q:
            b = self._target(q)
            if not b or not b.get("motion"):
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            seg = b["motion"][0] if "at the very start" in q else b["motion"][-1]
            w = seg["w0"] if "at the very start" in q else seg["w1"]
            axis, s = seg["axis"], _sign(w)
            if axis == "y" and s:
                pats = (r"to the right",) if s > 0 else (r"to the left",)
            elif axis == "x" and s:
                pats = (r"downward",) if s > 0 else (r"upward",)
            else:
                return Answer(None, confidence="unsupported",
                              note=f"edge direction undefined for axis={axis} w={w}")
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="edge-direction option wording changed")

        # ---- stops and speeds ------------------------------------------------
        if "stop turning for a moment" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            motion = b.get("motion", [])
            count = 0
            for i, seg in enumerate(motion):
                if _sign(seg["w0"]) or _sign(seg["w1"]):
                    continue
                if any(_sign(x["w0"]) or _sign(x["w1"]) for x in motion[i + 1:]):
                    count += 1
            return Answer(str(count), confidence="exact")

        if "at about what time does the solid stop turning" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stops = self._stop_segments(b)
            idx = 1 if "second time" in q else 0
            if len(stops) <= idx:
                return Answer(None, confidence="unsupported", note="fewer stops than asked")
            return Answer(_clock(stops[idx]["f0"] / FPS), confidence="exact")

        if "at about what time does the solid start turning again" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stops = self._stop_segments(b)
            idx = 1 if "second stop" in q else 0
            if len(stops) <= idx:
                return Answer(None, confidence="unsupported", note="fewer stops than asked")
            return Answer(_clock((stops[idx]["f1"] + 1) / FPS), confidence="exact")

        if "how does the turning speed" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            pick = _pick(opts, re.escape(self._speed_trend(b)))
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="speed-trend option wording changed")

        if "compare how fast the solid turns" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stops = self._stop_segments(b)
            if not stops:
                pats = (r"never stops",)
            else:
                before = [m for m in b["motion"] if m["f1"] <= stops[0]["f0"] and
                          (_sign(m["w0"]) or _sign(m["w1"]))]
                after = [m for m in b["motion"] if m["f0"] >= stops[0]["f1"] and
                         (_sign(m["w0"]) or _sign(m["w1"]))]
                if not before or not after:
                    pats = (r"never stops",)
                else:
                    vb, va = abs(before[-1]["w1"]), abs(after[0]["w0"])
                    if abs(vb - va) < 1e-6:
                        pats = (r"same speed before and after",)
                    else:
                        pats = (r"faster before the stop",) if vb > va else (r"faster after the stop",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="speed-comparison option wording changed")

        if "about how long does the solid" in q and "one full turn" in q:
            b = self._target(q)
            if not b or not b.get("motion") or not abs(b["motion"][0]["w0"]):
                return Answer(None, confidence="unsupported", note="no starting speed")
            return Answer(f"{360.0 / abs(b['motion'][0]['w0']):.1f} s", confidence="exact")

        if "turn further in the first half" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            mid = b["motion"][-1]["f1"] / FPS / 2.0
            first = self._path(b, upto=mid)
            second = self._path(b, start=mid)
            if second < 1e-9:
                pats = (r"does not turn at all in the second half",)
            elif abs(first - second) < 0.5:
                pats = (r"about the same",)
            else:
                pats = (r"further in the first half",) if first > second else (r"further in the second half",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="half-comparison option wording changed")

        # ---- two-solid comparisons -------------------------------------------
        if "which solid spins faster" in q:
            pair = self._pair()
            if not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")
            left, right = pair
            sl = self._path(left) / max(self._duration(left), 1e-9)
            sr = self._path(right) / max(self._duration(right), 1e-9)
            if abs(sl - sr) < 0.5:
                pats = (r"same speed",)
            else:
                pats = (r"solid on the left",) if sl > sr else (r"solid on the right",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="speed-ranking option wording changed")

        if "which way do the two solids spin" in q:
            pair = self._pair()
            if not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")

            def near_side(b: dict) -> str:
                w = b["motion"][0]["w0"]
                if not _sign(w):
                    w = b["motion"][-1]["w1"]
                return "left" if w < 0 else "right"

            ls, rs = near_side(pair[0]), near_side(pair[1])
            if ls != rs:
                pats = (rf"left solid's near side moves {ls}, the right solid's moves {rs}",)
            else:
                pats = (rf"both with their near side moving {ls}",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="two-solid direction option wording changed")

        if "over the whole clip, which way has the solid turned further in total" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            turning = [m for m in b["motion"] if _sign(m["w0"]) or _sign(m["w1"])]
            signs = {_sign(m["w0"] + m["w1"]) for m in turning}
            if len(signs) <= 1:
                pats = (r"only ever turns one way",)
            else:
                fwd = sum(self._path(b, upto=m["f1"] / FPS, start=m["f0"] / FPS)
                          for m in turning if m["w0"] + m["w1"] > 0)
                back = sum(self._path(b, upto=m["f1"] / FPS, start=m["f0"] / FPS)
                           for m in turning if m["w0"] + m["w1"] < 0)
                if abs(fwd - back) < 1.0:
                    pats = (r"same amount each way",)
                else:
                    pats = (r"near side moving right",) if fwd > back else (r"near side moving left",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="turn-direction option wording changed")

        # ---- counterfactual turn counts (before the generic turn handlers) ---
        if ("had turned half as fast until its first stop" in q
                or "had turned twice as fast until its first stop" in q):
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stop = self._first_stop_time(b)
            if stop is None:
                return Answer(None, confidence="unsupported", note="clip never stops")
            factor = 0.5 if "half as fast" in q else 2.0
            return Answer(str(_int(self._path(b, upto=stop) * factor / 360.0)), confidence="exact")

        if "had never stopped" in q:
            b = self._target(q)
            if not b or not b.get("motion"):
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            w0 = abs(b["motion"][0]["w0"])
            return Answer(str(_int(w0 * self._duration(b) / 360.0)), confidence="exact")

        if "gone back to turning about the upright axis" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stops = self._stop_segments(b)
            if not stops:
                return Answer(None, confidence="unsupported", note="clip never stops")
            before = self._path(b, upto=stops[0]["f0"] / FPS)
            after = sum(self._path(b, upto=m["f1"] / FPS, start=m["f0"] / FPS)
                        for m in b["motion"] if m["f0"] >= stops[0]["f1"])
            return Answer(str(_int((before + after) / 360.0)), confidence="exact")

        if "turned at the same speed as the other solid" in q:
            b = self._target(q)
            pair = self._pair()
            if not b or not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")
            other = pair[1] if b is pair[0] else pair[0]
            speed = abs(other["motion"][0]["w0"])
            return Answer(str(_int(speed * self._duration(b) / 360.0)), confidence="exact")

        if "had turned half as fast the whole time" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            return Answer(str(_int(self._path(b) * 0.5 / 360.0)), confidence="exact")

        if "had turned twice as fast the whole time" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            return Answer(str(_int(self._path(b) * 2.0 / 360.0)), confidence="exact")

        # ---- complete-turn accounting ----------------------------------------
        if re.search(r"how many complete turns has the solid(?: on the (?:left|right))? made "
                     r"by the time the timer reads", q):
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            t = re.search(r"timer reads (\d+(?:\.\d+)?) s", q)
            if not t:
                return Answer(None, confidence="unsupported", note="no clock reading in question")
            return Answer(str(_int(self._path(b, upto=float(t.group(1))) / 360.0)), confidence="exact")

        if re.match(r"count the complete turns", q):
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            total = 0
            for m in b["motion"]:
                if _sign(m["w0"]) or _sign(m["w1"]):
                    total += _int(self._path(b, upto=m["f1"] / FPS, start=m["f0"] / FPS) / 360.0)
            return Answer(str(total), confidence="exact")

        if re.search(r"how many complete turns does the solid(?: on the (?:left|right))? "
                     r"make before it first stops", q):
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            stop = self._first_stop_time(b)
            return Answer(str(_int(self._path(b, upto=stop) / 360.0)), confidence="exact")

        if re.search(r"how many complete turns does the solid(?: on the (?:left|right))? "
                     r"make during the clip", q):
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            return Answer(str(_int(self._path(b) / 360.0)), confidence="exact")

        if "after the last complete turn" in q:
            b = self._target(q)
            if not b:
                return Answer(None, confidence="unsupported", note="ambiguous body reference")
            rest = self._path(b) % 360.0
            if rest < 90.0:
                pats = (r"less than a quarter turn",)
            elif rest < 180.0:
                pats = (r"between a quarter and a half turn",)
            elif rest < 270.0:
                pats = (r"between a half and three quarters",)
            else:
                pats = (r"more than three quarters",)
            pick = _pick(opts, *pats)
            return Answer(pick, confidence="exact") if pick else Answer(
                None, confidence="unsupported", note="remainder option wording changed")

        if "how many more complete turns" in q:
            pair = self._pair()
            if not pair:
                return Answer(None, confidence="unsupported", note="not a two-solid clip")
            a = _int(self._path(pair[0]) / 360.0)
            b = _int(self._path(pair[1]) / 360.0)
            return Answer(str(abs(a - b)), confidence="exact")

        return None


def validate() -> list[str]:
    """Table self-check: Euler, face-side sum, and family edge formulas."""
    problems: list[str] = []
    for name, s in SOLIDS.items():
        if s.corners - s.edges + s.faces != 2:
            problems.append(f"{name}: V-E+F != 2")
        sides = sum(v * FACE_SIDES[k] for k, v in s.shapes)
        if sides != 2 * s.edges:
            problems.append(f"{name}: face sides {sides} != 2E {2 * s.edges}")
        if sum(v for _, v in s.shapes) != s.faces:
            problems.append(f"{name}: shape census {sum(v for _, v in s.shapes)} != F {s.faces}")
        if (s.vertex_transitive or s.uniform_corners) and (2 * s.edges) % s.corners:
            problems.append(f"{name}: 2E/V not integral")
        want = {"prism": 3 * s.n, "antiprism": 4 * s.n,
                "pyramid": 2 * s.n, "bipyramid": 3 * s.n}.get(s.family)
        if want is not None and s.edges != want:
            problems.append(f"{name}: {s.family} E {s.edges} != {want}")
    return problems

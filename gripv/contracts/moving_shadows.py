"""moving_shadows — deterministic params replay.

Scene model (calibrated against rendered frames):

* The sun (`light.kind == "arc"`) is a directional light.  Its screen position is
  ``(cx + Rx cos phi, cy - Ry sin phi)`` with ``phi`` linear from ``phi0`` to
  ``phi1``; shadows are parallel projections along ``phi``: a point at height
  ``y`` shifts by ``dx = -(ground - y) * cot(phi)``.
* The lamp (`light.kind == "lamp"`) is a point light at ``(x0..x1, y)``; shadows
  are central projections onto ``ground``.
* A stand is a rectangle ``[x-w/2, x+w/2]`` from ``ground`` up to ``ground-h``
  (``roof`` is decoration only: including it over-extends the shadow and is
  contradicted by the keys).  A float is a shape of half-width ``s`` whose
  centre moves with ``vx``/``vy`` px/s.
* A marker is a point at ``markers[i]``; it is in shadow when any caster's
  ground interval contains it.

Verification status (see results/moving_shadows.summary.json):

Only families reproduced exactly for every row are reported as verified.  The
shadow *extent* convention (per-shape silhouette width, hut roof, lamp
silhouette) could not be pinned: it is right for centres and first-contact
events but off by a few pixels at some sweep edges, which flips duration and
tie-break answers.  Families that depend on it are returned with
``confidence="unsupported"`` (the computed value is still reported in
``strict`` for audit) so the audit never claims a key it cannot derive.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

LIGHT_KINDS = ("arc", "lamp")

OBJ_PAT = (
    r"the (?:dotted |striped |checked |ringed |banded )?"
    r"(?:red|blue|green|lime|teal|purple|pink|brown|orange|yellow|black|white|grey|gray|cyan|magenta) "
    r"(?:post|pillar|hut|circle|square|triangle|rectangle|pentagon|hexagon|octagon|diamond|star|cross|"
    r"crescent|ellipse|semicircle|trapezoid|parallelogram|heart)"
)

# Families whose every one of the 25,000 keys is reproduced exactly.
PINNED = {
    "Which way does the shadow of <OBJ> move along the ground?",
    "How many objects cast a shadow onto the ground in this clip?",
    "Which way does the sun move across the frame?",
    "Which object moves during the clip?",
    "If the sun moved along the same path in the opposite direction (same speed), which marker would be the first to be in shadow?",
    "Which way does <OBJ> move?",
    "Which way does the lamp move across the frame?",
    "If <OBJ> stayed where it starts instead of moving, how many markers would be in shadow at some point?",
    "If <OBJ> stayed where it starts instead of moving, which marker would be the first to be in shadow?",
    "If the sun moved along the same path in the opposite direction (same speed), how many markers would be in shadow at some point?",
}

UNPINNED_NOTE = "family unpinned: shadow-extent convention (shape silhouette / hut roof / lamp silhouette)"


def signature(q: str) -> str:
    s = re.sub(r"\d+(?:\.\d+)?", "<N>", str(q))
    return re.sub(OBJ_PAT, "<OBJ>", s)

DEFAULTS = {
    "hw_scale": 1.0,      # float half-width = s * hw_scale
    "hut_roof": False,    # stand top = ground - (h + roof) when True
    "duration_plus1": True,  # covered frames 10..20 = 11 frames of time
    "sun_rev": "retrace",    # retrace | flip
    "lamp_rev": "retrace",   # retrace | flip
    "eps": 0.0,
    "marker_half": 0.0,   # marker plate half-width counted as covered
    "continuous": True,   # use continuous edge-crossing times for clock/duration families
}


def _objname(c: dict) -> str:
    pat = c.get("pattern", "solid")
    base = f"{c['color']} {c['shape']}"
    return base if pat == "solid" else f"{pat} {base}"


def _norm_name(name: str) -> str:
    name = name.strip().lower()
    return name[4:] if name.startswith("the ") else name


class _Timeline:
    """Per-frame marker coverage for one scene variant."""

    __slots__ = ("cover", "caster_ever", "n_frames", "n_markers")

    def __init__(self, cover, caster_ever, n_frames, n_markers):
        self.cover = cover              # list[int] bitmask per frame
        self.caster_ever = caster_ever  # list[int] bitmask of markers each caster hits
        self.n_frames = n_frames
        self.n_markers = n_markers

    def runs(self, mi: int) -> list[tuple[int, int]]:
        """Contiguous frame runs [start, end] where marker mi is covered."""
        bit = 1 << mi
        out = []
        start = None
        for f, mask in enumerate(self.cover):
            hit = bool(mask & bit)
            if hit and start is None:
                start = f
            elif not hit and start is not None:
                out.append((start, f - 1))
                start = None
        if start is not None:
            out.append((start, len(self.cover) - 1))
        return out

    def first(self, mi: int) -> int | None:
        bit = 1 << mi
        for f, mask in enumerate(self.cover):
            if mask & bit:
                return f
        return None

    def ever(self, mi: int) -> bool:
        bit = 1 << mi
        return any(mask & bit for mask in self.cover)

    def seconds(self, n_frames: int, plus1: bool) -> float:
        return (n_frames + (1 if plus1 else 0)) / 24.0


class MovingShadows:
    def __init__(self, params: dict, opts: dict | None = None):
        self.p = params
        self.o = dict(DEFAULTS)
        if opts:
            self.o.update(opts)
        self.N = int(params["frames"])
        self.g = float(params["ground"])
        self.L = params["light"]
        self.casters = list(params["casters"])
        self.markers = [float(x) for x in params["markers"]]
        self.duration = self.N / 24.0
        self.names = [_objname(c) for c in self.casters]
        self._cache: dict[tuple, _Timeline] = {}

    # ---------- geometry ----------
    def _u(self, f: int) -> float:
        return f / (self.N - 1) if self.N > 1 else 0.0

    def _light(self, f: int, rev: str | None = None):
        L = self.L
        u = self._u(f)
        if L["kind"] == "lamp":
            if rev:
                if self.o["lamp_rev"] == "retrace":
                    x = L["x1"] + (L["x0"] - L["x1"]) * u
                else:
                    x = L["x0"] - (L["x1"] - L["x0"]) * u
            else:
                x = L["x0"] + (L["x1"] - L["x0"]) * u
            return x, float(L["y"]), None
        phi = float(L["phi0"]) + (float(L["phi1"]) - float(L["phi0"])) * u
        if rev:
            if self.o["sun_rev"] == "retrace":
                phi = float(L["phi1"]) + (float(L["phi0"]) - float(L["phi1"])) * u
            else:
                phi = float(L["phi0"]) - (float(L["phi1"]) - float(L["phi0"])) * u
        return None, None, phi

    def _stand_top(self, c: dict) -> float:
        h = float(c["h"])
        if self.o["hut_roof"]:
            h += float(c.get("roof", 0) or 0)
        return self.g - h

    def _interval(self, ci: int, f: int, freeze: int | None, rev: str | None):
        c = self.casters[ci]
        lx, ly, phi = self._light(f, rev)
        if c["kind"] == "stand":
            x = float(c["x"])
            w = float(c["w"])
            top = self._stand_top(c)
            if phi is not None:
                cot = math.cos(math.radians(phi)) / math.sin(math.radians(phi))
                dx = -(self.g - top) * cot
                lo = x - w / 2 + min(0.0, dx)
                hi = x + w / 2 + max(0.0, dx)
                return lo, hi
            k = (self.g - ly) / (top - ly)
            a = lx + (x - w / 2 - lx) * k
            b = lx + (x + w / 2 - lx) * k
            return min(a, x - w / 2), max(b, x + w / 2)
        # float
        if freeze == ci:
            x, y = float(c["x"]), float(c["y"])
        else:
            t = f / 24.0
            x = float(c["x"]) + float(c.get("vx", 0) or 0) * t
            y = float(c["y"]) + float(c.get("vy", 0) or 0) * t
        hw = float(c["s"]) * self.o["hw_scale"]
        if phi is not None:
            cot = math.cos(math.radians(phi)) / math.sin(math.radians(phi))
            dx = -(self.g - y) * cot
            return x - hw + dx, x + hw + dx
        k = (self.g - ly) / (y - ly)
        xc = lx + (x - lx) * k
        return xc - hw * k, xc + hw * k

    # ---------- timeline ----------
    def timeline(self, removed: int | None = None, freeze: int | None = None,
                 rev: str | None = None) -> _Timeline:
        key = (removed, freeze, rev)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        eps = float(self.o["eps"])
        mh = float(self.o["marker_half"])
        cover: list[int] = []
        caster_ever = [0] * len(self.casters)
        for f in range(self.N):
            mask = 0
            for ci in range(len(self.casters)):
                if removed is not None and ci == removed:
                    continue
                lo, hi = self._interval(ci, f, freeze, rev)
                if lo > hi:
                    lo, hi = hi, lo
                for mi, mx in enumerate(self.markers):
                    if lo - eps <= mx + mh and mx - mh <= hi + eps:
                        mask |= 1 << mi
                        caster_ever[ci] |= 1 << mi
            cover.append(mask)
        tl = _Timeline(cover, caster_ever, self.N, len(self.markers))
        self._cache[key] = tl
        return tl

    # ---------- helpers ----------
    def _obj_index(self, name: str) -> int | None:
        name = _norm_name(name)
        for i, n in enumerate(self.names):
            if n == name:
                return i
        return None

    def _markers_from(self, text: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", text)]

    def _fmt_marker(self, mi: int) -> str:
        return f"marker {mi + 1}"

    def _fmt_time(self, f: int) -> str:
        return f"{round(f / 24.0 + 1e-9, 1):.1f} s"

    def _fmt_dur(self, frames: int) -> str:
        d = frames / 24.0
        return f"{math.floor(d * 2 + 0.5) / 2:.1f} s"

    def _fmt_sec(self, seconds: float) -> str:
        return f"{math.floor(seconds * 2 + 0.5) / 2:.1f} s"

    def _fmt_clock(self, seconds: float) -> str:
        return f"{round(seconds + 1e-9, 1):.1f} s"

    def _pick_time(self, seconds: float, options: list[str] | None, window: float = 0.60) -> str:
        """Return the offered time option nearest the computed value.

        Time options are generated on a 0.5 s lattice around the true value, so the
        nearest option is the answer even when the replay is a frame off.
        """
        best, bestd = None, None
        for o in options or []:
            m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*s\s*$", str(o))
            if not m:
                continue
            d = abs(float(m.group(1)) - seconds)
            if bestd is None or d < bestd:
                best, bestd = str(o), d
        if best is not None and bestd is not None and bestd <= window:
            return best
        return self._fmt_clock(seconds)

    def _total_shadow_time(self, tl: _Timeline, mi: int, freeze=None, rev=None) -> float:
        if self.o["continuous"]:
            return sum(b - a for a, b in self._runs_t(tl, mi, freeze, rev))
        return sum(e - s + 1 for s, e in tl.runs(mi)) / 24.0

    def _dur_frames(self, tl: _Timeline, mi: int) -> int:
        runs = tl.runs(mi)
        n = sum(e - s + 1 for s, e in runs)
        return n if self.o["duration_plus1"] else n - len(runs)

    # ---------- continuous time ----------
    def _interval_at(self, ci: int, t: float, freeze: int | None, rev: str | None):
        f = t * 24.0
        lo_f = int(math.floor(f))
        frac = f - lo_f
        if lo_f >= self.N - 1:
            a = b = self._interval(ci, self.N - 1, freeze, rev)
        elif lo_f < 0:
            a = b = self._interval(ci, 0, freeze, rev)
        else:
            a = self._interval(ci, lo_f, freeze, rev)
            b = self._interval(ci, lo_f + 1, freeze, rev)
        lo = a[0] + (b[0] - a[0]) * frac
        hi = a[1] + (b[1] - a[1]) * frac
        return (lo, hi) if lo <= hi else (hi, lo)

    def _covered_t(self, ci_set, mi: int, t: float, freeze: int | None, rev: str | None) -> bool:
        x = self.markers[mi]
        for ci in ci_set:
            lo, hi = self._interval_at(ci, t, freeze, rev)
            if lo <= x <= hi:
                return True
        return False

    def _runs_t(self, tl: _Timeline, mi: int, freeze: int | None = None, rev: str | None = None):
        """Continuous [start, end] shadow intervals for a marker (seconds)."""
        runs = tl.runs(mi)
        if not runs:
            return []
        if not self.o["continuous"]:
            return [(s / 24.0, (e + 1) / 24.0) for s, e in runs]
        casters = [ci for ci in range(len(self.casters))
                   if tl.caster_ever[ci] >> mi & 1]
        out = []
        for s, e in runs:
            t0, t1 = s / 24.0, (e + 1) / 24.0
            if s > 0:
                lo, hi = (s - 1) / 24.0, s / 24.0
                for _ in range(24):
                    mid = (lo + hi) / 2
                    if self._covered_t(casters, mi, mid, freeze, rev):
                        hi = mid
                    else:
                        lo = mid
                t0 = hi
            if e < self.N - 1:
                lo, hi = e / 24.0, (e + 1) / 24.0
                for _ in range(24):
                    mid = (lo + hi) / 2
                    if self._covered_t(casters, mi, mid, freeze, rev):
                        lo = mid
                    else:
                        hi = mid
                t1 = hi
            out.append((t0, t1))
        return out

    def _first_marker(self, tl: _Timeline) -> int | None:
        best, bestf = None, None
        for mi in range(len(self.markers)):
            f = tl.first(mi)
            if f is not None and (bestf is None or f < bestf):
                best, bestf = mi, f
        return best

    # ---------- question dispatch ----------
    def answer(self, question: str, options: list[str]) -> Answer:
        """Answer a question, gated on the family being pinned exactly."""
        ans = self._answer_raw(question, options)
        if signature(question) in PINNED:
            return ans
        note = UNPINNED_NOTE
        if ans.confidence == "unsupported":
            note = f"{note}; handler: {ans.note}"
        return Answer(ans.strict, ans.loose, confidence="unsupported", note=note, extra=ans.extra)

    def _answer_raw(self, question: str, options: list[str]) -> Answer:
        q = str(question).strip()
        ql = q.lower()
        m = re.match(r"^which object moves during the clip\?$", ql)
        if m:
            movers = [i for i, c in enumerate(self.casters)
                      if abs(float(c.get("vx", 0) or 0)) + abs(float(c.get("vy", 0) or 0)) > 0]
            if not movers:
                return Answer("none of them")
            return Answer("the " + self.names[movers[0]])

        if ql == "which way does the sun move across the frame?":
            d = math.cos(math.radians(float(self.L["phi1"]))) - math.cos(math.radians(float(self.L["phi0"])))
            return Answer("left to right" if d > 0 else "right to left")
        if ql == "which way does the lamp move across the frame?":
            return Answer("left to right" if float(self.L["x1"]) > float(self.L["x0"]) else "right to left")

        m = re.match(r"^which way does the shadow of (?:the )?(.+) move along the ground\?$", ql)
        if m:
            ci = self._obj_index(m.group(1))
            if ci is None:
                return Answer(None, confidence="unsupported", note="unknown object")
            vals = []
            for f in range(self.N):
                lo, hi = self._interval(ci, f, None, None)
                vals.append((lo + hi) / 2)
            diffs = [b - a for a, b in zip(vals, vals[1:])]
            if all(d >= -1e-9 for d in diffs):
                return Answer("to the right")
            if all(d <= 1e-9 for d in diffs):
                return Answer("to the left")
            if diffs[0] < 0:
                return Answer("left, then back right")
            return Answer("right, then back left")

        m = re.match(r"^which way does (?:the )?(.+) move\?$", ql)
        if m:
            ci = self._obj_index(m.group(1))
            if ci is None:
                return Answer(None, confidence="unsupported", note="unknown object")
            c = self.casters[ci]
            vx = float(c.get("vx", 0) or 0)
            vy = float(c.get("vy", 0) or 0)
            if abs(vy) > abs(vx):
                return Answer("down" if vy > 0 else "up")
            return Answer("to the right" if vx > 0 else "to the left")

        if ql == "how many objects cast a shadow onto the ground in this clip?":
            return Answer(str(len(self.casters)))

        if ql == "how many of the markers are in shadow at some point in the clip?":
            tl = self.timeline()
            return Answer(str(sum(1 for mi in range(len(self.markers)) if tl.ever(mi))))

        if ql == "counting every marker and every time, how many times in total does a marker go into shadow?":
            tl = self.timeline()
            return Answer(str(sum(len(tl.runs(mi)) for mi in range(len(self.markers)))))

        if ql == "what is the largest number of markers that are in shadow at the same moment?":
            tl = self.timeline()
            return Answer(str(max((bin(mask).count("1") for mask in tl.cover), default=0)))

        m = re.match(r"^how many markers are in shadow when the clock shows ([\d.]+) s\?$", ql)
        if m:
            f = round(float(m.group(1)) * 24)
            tl = self.timeline()
            f = min(max(f, 0), self.N - 1)
            return Answer(str(bin(tl.cover[f]).count("1")))

        m = re.match(r"^when the clock shows ([\d.]+) s, which of these markers is in shadow\?$", ql)
        if m:
            f = min(max(round(float(m.group(1)) * 24), 0), self.N - 1)
            tl = self.timeline()
            mask = tl.cover[f]
            cands = [o for o in options if re.match(r"^marker \d+$", str(o).strip())]
            hit = [o for o in cands if mask >> (int(str(o).split()[1]) - 1) & 1]
            if hit:
                return Answer(str(hit[0]))
            return Answer(None, confidence="unsupported", note="no option in shadow")

        m = re.match(r"^when the clock shows ([\d.]+) s, whose shadow is over marker (\d+)\?$", ql)
        if m:
            f = min(max(round(float(m.group(1)) * 24), 0), self.N - 1)
            mi = int(m.group(2)) - 1
            hits = []
            for ci in range(len(self.casters)):
                lo, hi = self._interval(ci, f, None, None)
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= self.markers[mi] <= hi:
                    hits.append(ci)
            if not hits:
                return Answer("no object's shadow")
            return Answer("the " + self.names[hits[0]])

        if ql == "which marker is the first to be in shadow?":
            tl = self.timeline()
            mi = self._first_marker(tl)
            return Answer(None, confidence="unsupported", note="none") if mi is None else Answer(self._fmt_marker(mi))

        if ql == "of the markers that go into shadow, which one is the last to do so for the first time?":
            tl = self.timeline()
            best, bestf = None, None
            for mi in range(len(self.markers)):
                f = tl.first(mi)
                if f is not None and (bestf is None or f > bestf):
                    best, bestf = mi, f
            return Answer(None, confidence="unsupported", note="none") if best is None else Answer(self._fmt_marker(best))

        if ql == "which marker spends the longest time in shadow in total?":
            tl = self.timeline()
            best, bestd = None, None
            for mi in range(len(self.markers)):
                d = self._dur_frames(tl, mi)
                if bestd is None or d > bestd:
                    best, bestd = mi, d
            return Answer(self._fmt_marker(best))

        if ql == "of the markers that are in shadow at some point, which one spends the shortest time in shadow in total?":
            tl = self.timeline()
            best, bestd = None, None
            for mi in range(len(self.markers)):
                if not tl.ever(mi):
                    continue
                d = self._dur_frames(tl, mi)
                if bestd is None or d < bestd:
                    best, bestd = mi, d
            return Answer(None, confidence="unsupported", note="none") if best is None else Answer(self._fmt_marker(best))

        if ql == "which of these markers is never in shadow?":
            tl = self.timeline()
            cands = [o for o in options if re.match(r"^marker \d+$", str(o).strip())]
            hit = [o for o in cands if not tl.ever(int(str(o).split()[1]) - 1)]
            if hit:
                return Answer(str(hit[0]))
            return Answer(None, confidence="unsupported", note="all in shadow")

        m = re.match(r"^of markers ([\d, and]+), which (spends the most time in shadow in total|goes into shadow first)\?$", ql)
        if m:
            idxs = [x - 1 for x in self._markers_from(m.group(1))]
            tl = self.timeline()
            if "spends" in m.group(2):
                best, bestd = None, None
                for mi in idxs:
                    d = self._dur_frames(tl, mi)
                    if bestd is None or d > bestd:
                        best, bestd = mi, d
                return Answer(self._fmt_marker(best))
            best, bestf = None, None
            for mi in idxs:
                f = tl.first(mi)
                if f is None:
                    continue
                if bestf is None or f < bestf:
                    best, bestf = mi, f
            return Answer(None, confidence="unsupported", note="none") if best is None else Answer(self._fmt_marker(best))

        m = re.match(r"^in what order do markers ([\d, and]+) first go into shadow\?$", ql)
        if m:
            idxs = [x - 1 for x in self._markers_from(m.group(1))]
            tl = self.timeline()
            idxs = [mi for mi in idxs if tl.first(mi) is not None]
            idxs.sort(key=lambda mi: (tl.first(mi), mi))
            return Answer(" → ".join(str(mi + 1) for mi in idxs))

        m = re.match(r"^whose shadow is the first to reach marker (\d+)\?$", ql)
        if m:
            mi = int(m.group(1)) - 1
            best, bestf = None, None
            for ci in range(len(self.casters)):
                for f in range(self.N):
                    lo, hi = self._interval(ci, f, None, None)
                    if lo > hi:
                        lo, hi = hi, lo
                    if lo <= self.markers[mi] <= hi:
                        if bestf is None or f < bestf:
                            best, bestf = ci, f
                        break
            if best is None:
                return Answer("no object's shadow")
            return Answer("the " + self.names[best])

        m = re.match(r"^how many different objects' shadows pass over marker (\d+) during the clip\?$", ql)
        if m:
            mi = int(m.group(1)) - 1
            tl = self.timeline()
            return Answer(str(sum(1 for mask in tl.caster_ever if mask >> mi & 1)))

        m = re.match(r"^how many of the markers does the shadow of (?:the )?(.+) pass over\?$", ql)
        if m:
            ci = self._obj_index(m.group(1))
            if ci is None:
                return Answer(None, confidence="unsupported", note="unknown object")
            tl = self.timeline()
            return Answer(str(bin(tl.caster_ever[ci]).count("1")))

        m = re.match(r"^how many separate times does marker (\d+) go into shadow\?$", ql)
        if m:
            mi = int(m.group(1)) - 1
            return Answer(str(len(self.timeline().runs(mi))))

        m = re.match(r"^for how long in total is marker (\d+) in shadow \(to the nearest 0.5 s\)\?$", ql)
        if m:
            mi = int(m.group(1)) - 1
            return Answer(self._pick_time(self._total_shadow_time(self.timeline(), mi), options))

        m = re.match(r"^how much time passes between marker (\d+) first going into shadow and marker (\d+) first going into shadow \(to the nearest 0.5 s\)\?$", ql)
        if m:
            a, b = int(m.group(1)) - 1, int(m.group(2)) - 1
            tl = self.timeline()
            ra, rb = self._runs_t(tl, a), self._runs_t(tl, b)
            if not ra or not rb:
                return Answer(None, confidence="unsupported", note="never")
            return Answer(self._pick_time(abs(ra[0][0] - rb[0][0]), options))

        m = re.match(r"^on the clock, when does marker (\d+) first (go into shadow|come back out of shadow)\?$", ql)
        if m:
            mi = int(m.group(1)) - 1
            runs = self._runs_t(self.timeline(), mi)
            if not runs:
                return Answer(None, confidence="unsupported", note="never")
            t = runs[0][0] if m.group(2) == "go into shadow" else runs[0][1]
            return Answer(self._pick_time(t, options, window=0.35))

        # ---------- counterfactuals ----------
        m = re.match(r"^if (?:the )?(.+) were taken away \(everything else the same\), (.+)$", ql)
        if m:
            ci = self._obj_index(m.group(1))
            if ci is None:
                return Answer(None, confidence="unsupported", note="unknown object")
            return self._counterfactual(m.group(2), removed=ci, options=options)

        m = re.match(r"^if (?:the )?(.+) stayed where it starts instead of moving, (.+)$", ql)
        if m:
            ci = self._obj_index(m.group(1))
            if ci is None:
                return Answer(None, confidence="unsupported", note="unknown object")
            return self._counterfactual(m.group(2), freeze=ci, options=options)

        m = re.match(r"^if the (sun|lamp) moved along the same path in the opposite direction \(same speed\), (.+)$", ql)
        if m:
            return self._counterfactual(m.group(2), rev=m.group(1), options=options)

        return Answer(None, confidence="unsupported", note="unhandled template")

    def _counterfactual(self, tail: str, removed: int | None = None, freeze: int | None = None,
                        rev: str | None = None, options: list[str] | None = None) -> Answer:
        tl = self.timeline(removed=removed, freeze=freeze, rev=rev)
        if tail == "how many markers would be in shadow at some point?":
            return Answer(str(sum(1 for mi in range(len(self.markers)) if tl.ever(mi))))
        if tail == "which marker would be the first to be in shadow?":
            mi = self._first_marker(tl)
            return Answer(None, confidence="unsupported", note="none") if mi is None else Answer(self._fmt_marker(mi))
        if tail == "which of these markers would never be in shadow?":
            cands = [o for o in (options or []) if re.match(r"^marker \d+$", str(o).strip())]
            hit = [o for o in cands if not tl.ever(int(str(o).split()[1]) - 1)]
            if hit:
                return Answer(str(hit[0]))
            return Answer(None, confidence="unsupported", note="all in shadow")
        m = re.match(r"^for how long in total would marker (\d+) be in shadow \(to the nearest 0.5 s\)\?$", tail)
        if m:
            mi = int(m.group(1)) - 1
            return Answer(self._pick_time(self._total_shadow_time(tl, mi, freeze, rev), options))
        return Answer(None, confidence="unsupported", note="unhandled counterfactual")

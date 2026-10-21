"""Layer-1 contract for `overlapping_shapes`.

The generator draws translucent polygons on a board.  Every answer is a
property of the stored scene: shape polygons (reconstructed from rendered
frames, cached as unit templates), appearance/disappearance events and the
single glide path.  Overlap is strict positive-area polygon intersection.

Conventions pinned by calibration against all 25,000 rows:

* ``params.shapes[i].x/y`` is the shape's ANCHOR, not always its centroid
  (crescent, heart and semicircle are anchored elsewhere).  Templates keep the
  anchor frame, so no re-centring is applied.
* ``size`` is the circumradius / half-extent; ``aspect`` divides the local
  y axis for rectangle and ellipse only.
* ``paper``/``arrive``: events are appearance frames; all shapes survive.
  ``depart``: events are disappearance frames; shapes absent from the event
  list are on the board the whole time.
* Draw order (z): ``paper``/``arrive`` = appearance order; otherwise array
  order.
* Glide paths interpolate linearly from ``(x, y)`` at ``f0`` to
  ``(x1, y1)`` at ``f1``; rotation advances by ``spin`` degrees over the trip.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from shapely.geometry import Polygon

from ..engine import Answer

TAU = 2 * math.pi
HERE = Path(__file__).resolve().parent.parent

TYPES = ["semicircle", "parallelogram", "trapezoid", "rectangle", "triangle",
         "pentagon", "hexagon", "octagon", "diamond", "circle", "square",
         "star", "cross", "heart", "crescent", "ellipse"]
COLORS = ["orange", "purple", "yellow", "brown", "green", "teal", "pink",
          "blue", "gray", "red"]

_TYPE_RE = "|".join(TYPES)
_COLOR_RE = "|".join(COLORS)
_REF = rf"(?:the )?({_COLOR_RE}) ({_TYPE_RE})"
_SPIN_SIGN = 1.0 if os.environ.get("OV_SPIN_SIGN", "+") == "+" else -1.0


def _clean(q: str) -> str:
    q = str(q).strip().replace("\u2019", "'")
    return re.sub(r"\s+", " ", q)


def _n(s: str) -> float:
    return float(re.search(r"-?\d+(?:\.\d+)?", s).group())


def _round_half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2


def _load_templates() -> dict:
    path = HERE.parent / "cache" / "shape_templates.json"
    return json.loads(path.read_text()) if path.exists() else {}


class OverlappingShapes:
    def __init__(self, params: dict):
        self.p = params
        self.shapes = params["shapes"]
        self.n = int(params["n_frames"])
        self.scenario = params["scenario"]
        self.events = {int(e["shape"]): int(e["f"]) for e in params["events"]}
        self.templates = _load_templates()
        self.movers = [i for i, s in enumerate(self.shapes) if "path" in s]
        if self.scenario in ("paper", "arrive"):
            self.order = sorted(self.events, key=lambda i: self.events[i])
        else:
            self.order = list(range(len(self.shapes)))
        # unit polygons (anchor frame, size = 1)
        self._unit = {i: self._unit_poly(i) for i in range(len(self.shapes))}

    # ------------------------------------------------------------------ geometry
    def _unit_poly(self, i: int):
        s = self.shapes[i]
        t = s["type"]
        size = float(s["size"])
        inset = 2.0 / size
        if t == "circle":
            r = max(0.05, 1.0 - inset)
            pts = [(r * math.cos(k * TAU / 96), r * math.sin(k * TAU / 96)) for k in range(96)]
        elif t == "rectangle":
            asp = float(s.get("aspect") or 1.0)
            hu = max(0.05, 1.0 - inset)
            hv = max(0.05, 1.0 / asp - inset)
            pts = [(-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv)]
        elif t == "ellipse":
            asp = float(s.get("aspect") or 1.0)
            ru = max(0.05, 1.0 - inset)
            rv = max(0.05, 1.0 / asp - inset)
            pts = [(ru * math.cos(k * TAU / 96), rv * math.sin(k * TAU / 96)) for k in range(96)]
        else:
            pts = [tuple(p) for p in self.templates[t]]
        return pts

    def _pos(self, i: int, f: float) -> tuple[float, float, float]:
        s = self.shapes[i]
        path = s.get("path")
        if not path:
            return float(s["x"]), float(s["y"]), float(s.get("rot") or 0.0)
        f0, f1 = float(path["f0"]), float(path["f1"])
        prog = 0.0 if f <= f0 else (1.0 if f >= f1 else (f - f0) / max(1.0, f1 - f0))
        return (float(s["x"]) + (float(path["x1"]) - float(s["x"])) * prog,
                float(s["y"]) + (float(path["y1"]) - float(s["y"])) * prog,
                float(s.get("rot") or 0.0) + _SPIN_SIGN * float(path.get("spin") or 0.0) * prog)

    def poly(self, i: int, f: float | None = None, progress: float | None = None) -> Polygon:
        x, y, rot = self._pos(i, f if f is not None else 0)
        s = self.shapes[i]
        if progress is not None:          # counterfactual path fraction
            path = s["path"]
            x = float(s["x"]) + (float(path["x1"]) - float(s["x"])) * progress
            y = float(s["y"]) + (float(path["y1"]) - float(s["y"])) * progress
            rot = float(s.get("rot") or 0.0) + _SPIN_SIGN * float(path.get("spin") or 0.0) * progress
        size = float(s["size"])
        c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        return Polygon([(x + (u * c - v * sn) * size, y + (u * sn + v * c) * size)
                        for u, v in self._unit[i]])

    def progress_at(self, i: int, f: float) -> float:
        path = self.shapes[i]["path"]
        f0, f1 = float(path["f0"]), float(path["f1"])
        return 0.0 if f <= f0 else (1.0 if f >= f1 else (f - f0) / max(1.0, f1 - f0))

    # ------------------------------------------------------------------ presence
    def present(self, f: float) -> list[int]:
        f = max(0.0, min(f, self.n - 1))
        if self.scenario in ("paper", "arrive"):
            return [i for i in range(len(self.shapes)) if self.events.get(i, 10 ** 9) <= f]
        if self.scenario == "depart":
            return [i for i in range(len(self.shapes)) if self.events.get(i, 10 ** 9) > f]
        return list(range(len(self.shapes)))

    def overlap(self, i: int, j: int, f: float | None = None, eps: float = 0.0) -> bool:
        a, b = self.poly(i, f), self.poly(j, f)
        if not a.intersects(b):
            return False
        try:
            return a.intersection(b).area > eps
        except Exception:
            return False

    def pairs(self, f: float | None = None, skip: set[int] | None = None,
              remove: set[int] | None = None, extra: set[int] | None = None) -> set[tuple[int, int]]:
        skip = skip or set()
        remove = remove or set()
        idx = [i for i in self.present(f if f is not None else self.n - 1)
               if i not in skip and i not in remove]
        for i in extra or ():
            if i not in idx and i not in remove:
                idx.append(i)
        out = set()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if self.overlap(i, j, f):
                    out.add((min(i, j), max(i, j)))
        return out

    def degree(self, i: int, f: float | None = None, skip: set[int] | None = None,
               eps: float = 0.0) -> int:
        skip = (skip or set()) | {i}
        return sum(1 for j in self.present(f if f is not None else self.n - 1)
                   if j not in skip and self.overlap(i, j, f, eps=eps))

    def depth(self, f: float | None = None, skip: set[int] | None = None) -> int:
        """Max number of shapes covering one point (rasterised at 1 px)."""
        from PIL import Image, ImageDraw

        x0, y0, x1, y1 = [int(v) for v in self.p["board"]]
        W, H = x1 - x0, y1 - y0
        acc = None
        import numpy as np
        for i in self.present(f if f is not None else self.n - 1):
            if i in (skip or set()):
                continue
            poly = self.poly(i, f)
            poly = poly.buffer(-0.5)
            if poly.is_empty:
                continue
            geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
            for g in geoms:
                im = Image.new("L", (W, H), 0)
                ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in g.exterior.coords], fill=1)
                a = np.array(im, dtype=np.int16)
                acc = a if acc is None else acc + a
        if acc is None:
            return 0
        return int(acc.max())

    def isolated(self, f: float | None = None, skip: set[int] | None = None,
                 remove: set[int] | None = None, extra: set[int] | None = None) -> list[int]:
        prs = self.pairs(f, skip=skip, remove=remove, extra=extra)
        idx = [i for i in self.present(f if f is not None else self.n - 1)
               if i not in (skip or set()) and i not in (remove or set())]
        for i in extra or ():
            if i not in idx and i not in (remove or set()):
                idx.append(i)
        used = {x for p in prs for x in p}
        return [i for i in idx if i not in used]

    # ------------------------------------------------------------------ motion
    def moving_frames(self, i: int):
        path = self.shapes[i]["path"]
        return int(path["f0"]), int(path["f1"])

    def first_overlaps(self, i: int, reverse: bool = False) -> dict[int, float]:
        """j -> first frame at which mover i begins overlapping j."""
        return {j: s for j, (s, e) in self.intervals(i).items()}

    def intervals(self, i: int, step: float = 0.25, fmax: float | None = None) -> dict[int, tuple[float, float]]:
        """j -> (first frame overlapped, last frame overlapped) at sub-frame steps."""
        f0, f1 = self.moving_frames(i)
        if fmax is not None:
            f1 = min(f1, fmax)
        if f1 < f0:
            return {}
        si, sj = self.shapes[i], None
        ri = max(math.hypot(u, v) for u, v in self._unit[i]) * float(si["size"])
        out: dict[int, tuple[float, float]] = {}
        fs = []
        f = float(f0)
        while f <= f1 + 1e-9:
            fs.append(f)
            f += step
        for j in range(len(self.shapes)):
            if j == i:
                continue
            sj = self.shapes[j]
            rj = max(math.hypot(u, v) for u, v in self._unit[j]) * float(sj["size"])
            maxd = ri + rj + 1.0
            entry = exit_ = None
            for f in fs:
                xi, yi, _ = self._pos(i, f)
                xj, yj, _ = self._pos(j, f)
                if math.hypot(xi - xj, yi - yj) > maxd:
                    continue
                if self.overlap(i, j, f, eps=100.0):
                    if entry is None:
                        entry = f
                    exit_ = f
            if entry is not None:
                out[j] = (entry, exit_)
        return out

    def distinct_overlapped(self, i: int, fmax: int | None = None, reverse: bool = False) -> set[int]:
        return set(self.intervals(i, fmax=fmax))

    def max_degree(self, i: int) -> int:
        f0, f1 = self.moving_frames(i)
        best = 0
        f = float(f0)
        while f <= f1 + 1e-9:
            best = max(best, sum(1 for j in range(len(self.shapes))
                                 if j != i and self.overlap(i, j, f, eps=100.0)))
            f += 0.5
        return best

    def stay_frames(self, i: int) -> dict[int, float]:
        """Time (frames) the mover overlaps each shape: 'stays on top of' it."""
        return {j: e - s for j, (s, e) in self.intervals(i).items()}

    # ------------------------------------------------------------------ refs
    def resolve(self, text: str) -> int | None:
        t = str(text).lower()
        hits = []
        for i, s in enumerate(self.shapes):
            typ = s["type"]
            if s["color"].lower() in t and re.search(rf"\b{typ}s?\b", t):
                hits.append(i)
        return hits[0] if len(hits) == 1 else None

    def opt_for(self, options, i: int | None) -> str | None:
        if i is None:
            return None
        for o in options:
            if self.resolve(o) == i:
                return str(o)
        return None

    def order_opt(self, options, order: list[int]) -> str | None:
        for o in options:
            txt = str(o).lower().replace("then the ", "").replace(", ", "|").replace("the ", "")
            parts = [p.strip() for p in txt.split("|")] if "|" in txt else [txt]
            idx = [self.resolve(p) for p in parts]
            if idx == order:
                return str(o)
        return None

    # ------------------------------------------------------------------ helpers
    def _event_time(self, i: int) -> int | None:
        return self.events.get(i)

    def _appear_frame(self, i: int) -> int:
        return self.events.get(i, 0) if self.scenario in ("paper", "arrive") else 0

    def _disappear_frame(self, i: int) -> int | None:
        return self.events.get(i) if self.scenario == "depart" else None

    def _overlap_at_event(self, i: int) -> int:
        f = self._appear_frame(i)
        return sum(1 for j in self.present(f) if j != i and self.overlap(i, j, f))

    def _order_count_none_before(self, order: list[int]) -> int:
        placed: list[int] = [order[0]]
        cnt = 0
        for i in order[1:]:
            if not any(self.overlap(i, j) for j in placed):
                cnt += 1
            placed.append(i)
        return cnt

    def _direction(self, i: int) -> str:
        path = self.shapes[i]["path"]
        dx = float(path["x1"]) - float(self.shapes[i]["x"])
        dy = float(path["y1"]) - float(self.shapes[i]["y"])
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        if abs(dx) > 2.0 * abs(dy):
            return horiz
        if abs(dy) > 2.0 * abs(dx):
            return vert
        return f"{vert} and to the {horiz}"

    def _area(self, i: int) -> float:
        return self.poly(i).area

    # ------------------------------------------------------------------ dispatch
    def answer(self, question: str, options) -> Answer:
        q = _clean(question)
        ql = q.lower()
        opts = [str(o) for o in options]

        # ---- counts on the board -------------------------------------------
        m = re.match(r"^how many shapes are on the board when the clock reads ([\d.]+) s\?$", ql)
        if m:
            return Answer(str(len(self.present(round(float(m.group(1)) * 24)))))
        m = re.match(r"^how many shapes are on the board at the start of the video\?$", ql)
        if m:
            return Answer(str(len(self.present(0))))
        if re.match(r"^how many shapes are on the board at the end of the video\?$", ql):
            return Answer(str(len(self.present(self.n - 1))))
        if re.match(r"^how many shapes are still on the board at the end of the video\?$", ql):
            return Answer(str(len(self.present(self.n - 1))))
        if re.match(r"^how many shapes are on the board, counting the moving ones\?$", ql):
            return Answer(str(len(self.shapes)))

        # ---- event times ----------------------------------------------------
        m = re.match(rf"^to the nearest second, when does {_REF} start to (appear|disappear)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            if m.group(3) == "appear":
                f = self._appear_frame(i)
                if self.scenario not in ("paper", "arrive"):
                    return Answer(None, confidence="unsupported")
            else:
                fr = self._disappear_frame(i)
                if fr is None:
                    return Answer(None, confidence="unsupported")
                f = fr
            return Answer(f"{math.floor(f / 24 + 0.5)} s")

        # ---- pairwise order -------------------------------------------------
        m = re.match(rf"^which (appears|disappears) first: {_REF} or {_REF}\?$", ql)
        if m:
            kind = m.group(1)
            a = self.resolve(f"{m.group(2)} {m.group(3)}")
            b = self.resolve(f"{m.group(4)} {m.group(5)}")
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            if kind == "appears":
                ta, tb = self._appear_frame(a), self._appear_frame(b)
                same, neither = "they appear at the same moment", "neither of them appears"
            else:
                ta, tb = self._disappear_frame(a), self._disappear_frame(b)
                same, neither = "they disappear at the same moment", "neither of them disappears"
            if ta is None and tb is None:
                return Answer(neither)
            if ta is None:
                return Answer(self.opt_for(opts, b))
            if tb is None:
                return Answer(self.opt_for(opts, a))
            if ta == tb:
                return Answer(same)
            return Answer(self.opt_for(opts, a if ta < tb else b))

        m = re.match(r"^of these shapes, which one (appears|disappears) first\?$", ql)
        if m:
            kind = "appear" if m.group(1) == "appears" else "disappear"
            vals = []
            for o in opts:
                i = self.resolve(o)
                if i is None:
                    return Answer(None, confidence="unsupported")
                t = self._appear_frame(i) if kind == "appear" else self._disappear_frame(i)
                if t is None:
                    return Answer(None, confidence="unsupported")
                vals.append((t, i))
            vals.sort()
            return Answer(self.opt_for(opts, vals[0][1]))

        if re.match(r"^which shape appears first\?$", ql):
            i = min(range(len(self.shapes)), key=lambda k: self._appear_frame(k))
            return Answer(self.opt_for(opts, i))
        if re.match(r"^which shape is the last one to appear\?$", ql):
            i = max(range(len(self.shapes)), key=lambda k: self._appear_frame(k))
            return Answer(self.opt_for(opts, i))
        if re.match(r"^which shape is the first to disappear\?$", ql):
            if not self.events:
                return Answer(None, confidence="unsupported")
            i = min(self.events, key=lambda k: self.events[k])
            return Answer(self.opt_for(opts, i))
        m = re.match(rf"^which shape is the next to disappear after {_REF}\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None or self.scenario != "depart":
                return Answer(None, confidence="unsupported")
            seq = sorted(self.events, key=lambda k: self.events[k])
            pos = seq.index(i)
            if pos + 1 >= len(seq):
                return Answer(None, confidence="unsupported")
            return Answer(self.opt_for(opts, seq[pos + 1]))

        # ---- attribute counts ----------------------------------------------
        m = re.match(r"^how many (.+?) shapes appear in the video\?$", ql)
        if m:
            word = m.group(1)
            if word in COLORS:
                return Answer(str(sum(1 for s in self.shapes if s["color"].lower() == word)))
            if word.rstrip("s") in TYPES or word in TYPES:
                stem = word[:-1] if word.endswith("s") else word
                if word == "crosses":
                    stem = "cross"
                return Answer(str(sum(1 for s in self.shapes if s["type"] == stem)))
        m = re.match(r"^how many (\w+) shapes appear in the video\?$", ql)
        if m and m.group(1) in COLORS:
            return Answer(str(sum(1 for s in self.shapes if s["color"].lower() == m.group(1))))
        m = re.match(r"^how many (\w+) appear in the video\?$", ql)
        if m:
            word = m.group(1)
            stem = {"crosses": "cross", "ellipses": "ellipse"}.get(word, word[:-1] if word.endswith("s") else word)
            if stem in TYPES:
                return Answer(str(sum(1 for s in self.shapes if s["type"] == stem)))
        if re.match(r"^how many different kinds of shape appear in the video", ql):
            return Answer(str(len({s["type"] for s in self.shapes})))

        m = re.match(r"^how many shapes have a (solid|double|dotted|dashed)(?:-line)? border\?$", ql)
        if m:
            return Answer(str(sum(1 for s in self.shapes if s["style"] == m.group(1))))
        m = re.match(r"^how many shapes are filled with (.+?)\?$", ql)
        if m:
            word = m.group(1)
            fill = {"crossed lines": "hatch", "dots": "dots",
                    "stripes that run one way": "stripes",
                    "a plain fill with no pattern": "plain"}.get(word)
            if fill:
                return Answer(str(sum(1 for s in self.shapes if s["fill"] == fill)))

        # ---- area ratio -----------------------------------------------------
        m = re.match(r"^(not counting the moving shapes, )?about how many times larger \(by area\) is the biggest shape than the smallest one \(to the nearest ([\d.]+)\)\?$", ql)
        if m:
            # The ratio depends on the renderer's exact polygon areas; our
            # reconstructed outlines differ by ~1-2 px per shape, which flips
            # the nearest-0.5 bin on near-boundary clips. Family left
            # unsupported rather than guessed.
            return Answer(None, confidence="unsupported")

        # ---- overlap counts at a time --------------------------------------
        m = re.match(r"^how many pairs of shapes overlap each other when the clock reads ([\d.]+) s\?$", ql)
        if m:
            return Answer(str(len(self.pairs(round(float(m.group(1)) * 24)))))
        if re.match(r"^how many pairs of shapes overlap each other at the end\?$", ql):
            return Answer(str(len(self.pairs(self.n - 1))))
        if re.match(r"^how many pairs of shapes overlap each other at the start\?$", ql):
            return Answer(str(len(self.pairs(0))))
        if re.match(r"^just before the last shape appears, how many pairs of shapes overlap each other\?$", ql):
            fs = [e["f"] for e in self.p["events"]]
            f = max(0, max(fs) - 1) if fs else 0
            return Answer(str(len(self.pairs(f))))
        if re.match(r"^how many pairs of shapes overlap each other before anything moves\?$", ql):
            return Answer(str(len(self.pairs(0))))

        # ---- degree / isolated ---------------------------------------------
        m = re.match(rf"^how many other shapes does {_REF} overlap at the (end|start)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            f = self.n - 1 if m.group(3) == "end" else 0
            return Answer(str(self.degree(i, f)) if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")
        m = re.match(rf"^how many other shapes does {_REF} overlap before anything moves\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            return Answer(str(self.degree(i, 0)) if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")
        m = re.match(rf"^not counting the moving shapes, how many other shapes does {_REF} overlap\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.degree(i, self.n - 1, skip=set(self.movers))))

        m = re.match(rf"^which of these shapes overlaps {_REF} at the (end|start)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            f = self.n - 1 if m.group(3) == "end" else 0
            hits = [j for j in self.present(f) if j != i and self.overlap(i, j, f)]
            for o in opts:
                if self.resolve(o) in hits:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^which of these shapes overlaps {_REF} before anything moves\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            for o in opts:
                j = self.resolve(o)
                if j is not None and self.overlap(i, j, 0):
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")

        for fam, fdef in [
            (r"^which of these shapes overlaps no other shape at the end\?$", self.n - 1),
            (r"^which of these shapes overlaps no other shape at the start\?$", 0),
            (r"^which of these shapes overlaps no other shape before anything moves\?$", 0),
        ]:
            if re.match(fam, ql):
                iso = set(self.isolated(fdef))
                for o in opts:
                    if self.resolve(o) in iso:
                        return Answer(str(o))
                return Answer(None, confidence="unsupported")
        if re.match(r"^not counting the moving shapes, which of these shapes overlaps no other shape\?$", ql):
            iso = set(self.isolated(self.n - 1, skip=set(self.movers)))
            for o in opts:
                if self.resolve(o) in iso:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")

        if re.match(r"^not counting the moving shapes, how many pairs of shapes overlap each other\?$", ql):
            return Answer(str(len(self.pairs(self.n - 1, skip=set(self.movers)))))
        if re.match(r"^at the most crowded spot at the end, how many shapes are stacked on top of each other\?$", ql):
            return Answer(str(self.depth(self.n - 1)))
        if re.match(r"^at the most crowded spot at the start, how many shapes are stacked on top of each other\?$", ql):
            return Answer(str(self.depth(0)))
        if re.match(r"^not counting the moving shapes, at the most crowded spot, how many shapes are stacked on top of each other\?$", ql):
            return Answer(str(self.depth(self.n - 1, skip=set(self.movers))))

        # ---- draw order / on top -------------------------------------------
        m = re.match(rf"^which of these shapes lies on top of part of {_REF}\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            later = self.order[self.order.index(i) + 1:]
            for o in opts:
                j = self.resolve(o)
                if j in later and self.overlap(i, j, self.n - 1):
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^if the same shapes had been laid down in the reverse order, which of these shapes would lie on top of part of {_REF}\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            rev = list(reversed(self.order))
            later = rev[rev.index(i) + 1:]
            for o in opts:
                j = self.resolve(o)
                if j in later and self.overlap(i, j, self.n - 1):
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")

        # ---- appearance overlaps -------------------------------------------
        m = re.match(rf"^when {_REF} appears, it overlaps exactly one shape that is already on the board\. which one\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            f = self._appear_frame(i)
            hits = [j for j in self.present(f) if j != i and self.overlap(i, j, f)]
            for o in opts:
                if self.resolve(o) in hits:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^at the moment {_REF} (?:is laid down|appears), how many of the shapes already on the board does it overlap\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self._overlap_at_event(i)))
        if re.match(r"^not counting the first shape, how many shapes overlapped nothing already on the board at the moment they appeared\?$", ql):
            if self.scenario not in ("paper", "arrive"):
                return Answer(None, confidence="unsupported")
            return Answer(str(self._order_count_none_before(self.order)))
        if re.match(r"^if the first shape to appear had never appeared, how many of the other shapes \(not counting the new first one\) would overlap nothing already on the board at the moment they appeared\?$", ql):
            if self.scenario not in ("paper", "arrive"):
                return Answer(None, confidence="unsupported")
            return Answer(str(self._order_count_none_before(self.order[1:])))
        if re.match(r"^if the same shapes had appeared in the reverse order, how many of them \(not counting the first\) would overlap nothing already on the board at the moment they appeared\?$", ql):
            if self.scenario not in ("paper", "arrive"):
                return Answer(None, confidence="unsupported")
            return Answer(str(self._order_count_none_before(list(reversed(self.order)))))
        if re.match(r"^which shape, at the moment it appeared, overlapped the most shapes that were already on the board\?$", ql):
            if self.scenario not in ("paper", "arrive"):
                return Answer(None, confidence="unsupported")
            vals = [(self._overlap_at_event(i), i) for i in self.order[1:]]
            best = max(vals)
            return Answer(self.opt_for(opts, best[1]))

        # ---- disappearance overlaps ----------------------------------------
        m = re.match(rf"^just before {_REF} starts to disappear, how many of the shapes still on the board does it overlap\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            fr = self._disappear_frame(i) if i is not None else None
            if fr is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.degree(i, fr - 1)))

        # ---- moving-shape questions ----------------------------------------
        if re.match(r"^which shape slides across the board\?$", ql):
            if not self.movers:
                return Answer(None, confidence="unsupported")
            for o in opts:
                if self.resolve(o) in self.movers:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        if re.match(r"^which of these shapes slides across the board\?$", ql):
            for o in opts:
                if self.resolve(o) in self.movers:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^in which direction does {_REF} move\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None or i not in self.movers:
                return Answer(None, confidence="unsupported")
            return Answer(self._direction(i))

        m = re.match(rf"^which shape does the moving {_REF} (overlap first|overlap last|stay on top of for the longest time)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None or i not in self.movers:
                return Answer(None, confidence="unsupported")
            if m.group(3) == "overlap first":
                fo = self.first_overlaps(i)
                if not fo:
                    return Answer(None, confidence="unsupported")
                j = min(fo, key=lambda k: fo[k])
            elif m.group(3) == "overlap last":
                iv = self.intervals(i)
                if not iv:
                    return Answer(None, confidence="unsupported")
                j = max(iv, key=lambda k: iv[k][1])
            else:
                sf = self.stay_frames(i)
                if not sf:
                    return Answer(None, confidence="unsupported")
                j = max(sf, key=lambda k: sf[k])
            return Answer(self.opt_for(opts, j))
        m = re.match(rf"^which of these shapes does the moving {_REF} slide over\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None or i not in self.movers:
                return Answer(None, confidence="unsupported")
            hit = self.distinct_overlapped(i)
            for o in opts:
                if self.resolve(o) in hit:
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^how many different shapes does the moving {_REF} overlap on its way\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.distinct_overlapped(i))))
        m = re.match(rf"^how many shapes is the moving {_REF} overlapping when the clock reads ([\d.]+) s\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.degree(i, round(float(m.group(3)) * 24), eps=100.0)))
        m = re.match(rf"^which shape is the moving {_REF} overlapping when the clock reads ([\d.]+) s\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            f = round(float(m.group(3)) * 24)
            for o in opts:
                j = self.resolve(o)
                if j is not None and self.overlap(i, j, f, eps=100.0):
                    return Answer(str(o))
            return Answer(None, confidence="unsupported")
        m = re.match(rf"^what is the largest number of shapes the moving {_REF} overlaps at the same moment\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.max_degree(i)))
        m = re.match(rf"^between the moments the clock reads ([\d.]+) s and ([\d.]+) s, how many shapes does the moving {_REF} start to overlap\?$", ql)
        if m:
            i = self.resolve(f"{m.group(3)} {m.group(4)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            a, b = round(float(m.group(1)) * 24), round(float(m.group(2)) * 24)
            fo = self.first_overlaps(i)
            return Answer(str(sum(1 for f in fo.values() if a <= f <= b)))
        m = re.match(rf"^in what order does the moving {_REF} start to overlap the {_REF}, the {_REF}, the {_REF}\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            refs = [self.resolve(f"{m.group(3)} {m.group(4)}"),
                    self.resolve(f"{m.group(5)} {m.group(6)}"),
                    self.resolve(f"{m.group(7)} {m.group(8)}")]
            if i is None or any(r is None for r in refs):
                return Answer(None, confidence="unsupported")
            fo = self.first_overlaps(i)
            if any(r not in fo for r in refs):
                return Answer(None, confidence="unsupported")
            order = sorted(refs, key=lambda r: fo[r])
            return Answer(self.order_opt(opts, order))
        if re.match(r"^of the two moving shapes, which one overlaps more different shapes on its way\?$", ql):
            if len(self.movers) < 2:
                return Answer(None, confidence="unsupported")
            a, b = self.movers[0], self.movers[1]
            ca, cb = len(self.distinct_overlapped(a)), len(self.distinct_overlapped(b))
            if ca == cb:
                for o in opts:
                    if "same" in str(o).lower():
                        return Answer(str(o))
                return Answer(None, confidence="unsupported")
            return Answer(self.opt_for(opts, a if ca > cb else b))

        # ---- counterfactual paths ------------------------------------------
        m = re.match(rf"^if {_REF} had made the same trip backwards \(starting where it stops and ending where it starts\), which shape would it overlap first\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None or i not in self.movers:
                return Answer(None, confidence="unsupported")
            f0, f1 = self.moving_frames(i)
            first = None
            for f in range(f0, f1 + 1):
                prog = 0.0 if f <= f0 else (1.0 if f >= f1 else (f - f0) / max(1, f1 - f0))
                x, y, rot = self._pos(i, f)
                path = self.shapes[i]["path"]
                # reverse: start at the end point at f0, end at the start point at f1
                rx = float(path["x1"]) + (float(self.shapes[i]["x"]) - float(path["x1"])) * prog
                ry = float(path["y1"]) + (float(self.shapes[i]["y"]) - float(path["y1"])) * prog
                s = self.shapes[i]
                size = float(s["size"])
                c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                poly = Polygon([(rx + (u * c - v * sn) * size, ry + (u * sn + v * c) * size)
                                for u, v in self._unit[i]])
                hit = None
                for j in self.present(f):
                    if j == i:
                        continue
                    pj = self.poly(j, f)
                    if poly.intersects(pj) and poly.intersection(pj).area > 100.0:
                        hit = j
                        break
                if hit is not None:
                    first = hit
                    break
            return Answer(self.opt_for(opts, first))
        m = re.match(rf"^if the moving {_REF} had stopped for good when the clock read ([\d.]+) s, how many different shapes would it have overlapped\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.distinct_overlapped(i, round(float(m.group(3)) * 24)))))
        m = re.match(rf"^if {_REF} had moved at half its speed \(starting at the same moment, along the same line\), how many different shapes would it have overlapped by the end of the video\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            path = self.shapes[i]["path"]
            f0, f1 = float(path["f0"]), float(path["f1"])
            prog_end = max(0.0, min(1.0, 0.5 * (self.n - 1 - f0) / max(1.0, f1 - f0)))
            hit = set()
            steps = 80
            for k in range(steps + 1):
                prog = prog_end * k / steps
                poly = self.poly(i, progress=prog)
                for j in self.present(self.n - 1):
                    if j == i:
                        continue
                    pj = self.poly(j, self.n - 1)
                    if poly.intersects(pj) and poly.intersection(pj).area > 100.0:
                        hit.add(j)
            return Answer(str(len(hit)))

        # ---- counterfactuals on the set ------------------------------------
        if re.match(r"^between the start and the end of the video, how many pairs of overlapping shapes are lost\?$", ql):
            ps, pe = self.pairs(0), self.pairs(self.n - 1)
            return Answer(str(len(ps - pe)))
        if re.match(r"^how many of the shapes still on the board at the end overlapped another shape at the start but overlap nothing at the end\?$", ql):
            p0 = self.pairs(0)
            pe = self.pairs(self.n - 1)
            deg0 = {i for p in p0 for i in p}
            deg1 = {i for p in pe for i in p}
            return Answer(str(sum(1 for i in self.present(self.n - 1) if i in deg0 and i not in deg1)))

        m = re.match(rf"^if {_REF} had never disappeared, how many of the shapes on the board at the end would overlap no other shape\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.isolated(self.n - 1, extra={i}))))
        m = re.match(rf"^if {_REF} had never disappeared \(everything else the same\), how many pairs of shapes would overlap at the end\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.pairs(self.n - 1, extra={i}))))
        m = re.match(rf"^if {_REF} (?:were taken away|had never been on the board \(everything else the same\)), how many of the other (non-moving )?shapes would overlap no other shape (at the end|at the start|before anything moves)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            if m.group(4) == "before anything moves":
                skip = set(self.movers) | {i}
                idx = [j for j in self.present(0) if j not in skip]
                used = {x for p in self.pairs(0, skip={i} | set(self.movers)) for x in p}
                return Answer(str(sum(1 for j in idx if j not in used)))
            f = self.n - 1 if m.group(4) == "at the end" else 0
            idx = [j for j in self.present(f) if j != i]
            used = {x for p in self.pairs(f, remove={i}) for x in p}
            return Answer(str(sum(1 for j in idx if j not in used)))
        # separate regexes for the three "taken away" variants
        m = re.match(rf"^if {_REF} were taken away, how many of the other shapes would overlap no other shape at the (end|start)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            f = self.n - 1 if m.group(3) == "end" else 0
            idx = [j for j in self.present(f) if j != i]
            used = {x for p in self.pairs(f, remove={i}) for x in p}
            return Answer(str(sum(1 for j in idx if j not in used)))
        m = re.match(rf"^if {_REF} were taken away, how many of the other non-moving shapes would overlap no other shape before anything moves\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            skip = set(self.movers) | {i}
            idx = [j for j in self.present(0) if j not in skip]
            used = {x for p in self.pairs(0, skip={i} | set(self.movers)) for x in p}
            return Answer(str(sum(1 for j in idx if j not in used)))
        m = re.match(rf"^if {_REF} had never been on the board \(everything else the same\), how many pairs of shapes would overlap at the (end|start)\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            f = self.n - 1 if m.group(3) == "end" else 0
            return Answer(str(len(self.pairs(f, remove={i}))))
        m = re.match(rf"^if {_REF} had never been on the board \(everything else the same\), how many pairs of shapes would overlap before anything moves\?$", ql)
        if m:
            i = self.resolve(f"{m.group(1)} {m.group(2)}")
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.pairs(0, remove={i}))))

        return Answer(None, confidence="unsupported")

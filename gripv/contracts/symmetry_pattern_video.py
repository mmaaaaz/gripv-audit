"""Layer-1 contract for `symmetry_pattern_video`.

The generator builds a symmetric picture and animates it in one of four ways:

* ``turns``    — the picture sits on a disc that is flipped and turned.
* ``reveal``   — the shapes pop in one at a time.
* ``assemble`` — groups of shapes glide in from outside and settle.
* ``waves``    — groups are traced one wave at a time.

Everything the questions ask is recoverable from ``params``: the ideal picture
(``base``), the rendered picture (``shapes``), the one shape that may break the
symmetry (``break``) and the per-scenario timeline.

Conventions pinned against the answer keys (25,000 rows):

* on-screen clocks truncate to one decimal (``int(f/24*10)/10``);
* a glide/trace interval ``[f0, f0+dur)`` completes at ``f0+dur-1``;
* a mirror axis is a line through the centre; reflection maps
  ``rot -> 2a - rot`` and flips the local ``mirrored`` flag;
* a rotation by ``t`` adds ``t`` to ``rot``;
* the rendered ``cap`` participates in picture equality (an unfinished
  outline breaks the symmetry), the trace ``dir`` does not;
* groups are the (colour, type) pairs of ``base``; when the break renames a
  shape (kind ``color`` / ``shape``) that shape is dropped from its group.
"""
from __future__ import annotations

import collections
import math
import re

from ..engine import Answer

PLURAL = {
    "bolt": "lightning bolts",
    "chevron": "chevrons",
    "circle": "circles",
    "diamond": "diamonds",
    "hexagon": "hexagons",
    "kite": "kites",
    "pentagon": "pentagons",
    "square": "squares",
    "star": "stars",
    "triangle": "triangles",
}
SINGULAR = {v: k for k, v in PLURAL.items()}

DIRECTIONS = [(0, "right"), (45, "top-right"), (90, "top"), (135, "top-left"),
              (180, "left"), (225, "bottom-left"), (270, "bottom"), (315, "bottom-right")]

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}

MOVE_NAMES = ["after the first move", "after the second move", "after the third move",
              "after the fourth move", "after the fifth move"]

BREAK_OPTIONS = {
    "shape": "one shape has a different outline",
    "fill": "one shape has a different fill",
    "color": "one shape has a different colour",
    "size": "one shape is a different size",
    "rotation": "one shape is turned differently",
    "position": "one shape is out of place",
    "direction": "one shape is traced the other way round",
    "incomplete": "one shape's outline is never finished",
}

TRANSFORM_OPTIONS = {
    "a 45° turn": ("rot", 45),
    "a 60° turn": ("rot", 60),
    "a 72° turn": ("rot", 72),
    "a quarter turn (90°)": ("rot", 90),
    "a 120° turn": ("rot", 120),
    "a half turn (180°)": ("rot", 180),
    "a left-to-right flip": ("ref", 90),
    "an upside-down flip": ("ref", 0),
}

# ---------------------------------------------------------------- geometry
TOL_XY = 0.02
TOL_ROT = 0.6
TOL_SIZE = 0.02
TOL_R = 0.05


def _norm(a: float) -> float:
    return a % 360.0


def transform(shapes, kind: str, param: float):
    out = []
    if kind == "rot":
        c, s = math.cos(math.radians(param)), math.sin(math.radians(param))
        for sh in shapes:
            x, y = float(sh["x"]), float(sh["y"])
            out.append(dict(sh, x=c * x - s * y, y=s * x + c * y,
                            rot=_norm(float(sh["rot"]) + param)))
    else:
        a = math.radians(param)
        c2, s2 = math.cos(2 * a), math.sin(2 * a)
        for sh in shapes:
            x, y = float(sh["x"]), float(sh["y"])
            out.append(dict(sh, x=c2 * x + s2 * y, y=s2 * x - c2 * y,
                            rot=_norm(2 * param - float(sh["rot"])),
                            mirrored=not bool(sh["mirrored"])))
    return out


def same_shape(a, b) -> bool:
    for f in ("type", "color", "fill", "dash", "mirrored"):
        if a[f] != b[f]:
            return False
    if abs(float(a["size"]) - float(b["size"])) > TOL_SIZE:
        return False
    if abs(float(a.get("cap", 1.0)) - float(b.get("cap", 1.0))) > 1e-3:
        return False
    if abs(float(a["x"]) - float(b["x"])) > TOL_XY or abs(float(a["y"]) - float(b["y"])) > TOL_XY:
        return False
    d = abs(_norm(float(a["rot"])) - _norm(float(b["rot"])))
    return min(d, 360 - d) <= TOL_ROT


def same_picture(pic_a, pic_b, cmp=same_shape) -> bool:
    used = [False] * len(pic_b)
    for a in pic_a:
        hit = -1
        for i, b in enumerate(pic_b):
            if not used[i] and cmp(a, b):
                hit = i
                break
        if hit < 0:
            return False
        used[hit] = True
    return True


def is_symmetric(shapes, kind: str, param: float, cmp=same_shape) -> bool:
    return same_picture(transform(shapes, kind, param), shapes, cmp)


def _polar(s):
    x, y = float(s["x"]), float(s["y"])
    return math.degrees(math.atan2(y, x)), math.hypot(x, y)


def _compatible(a, b) -> bool:
    if a["type"] != b["type"] or a["color"] != b["color"] or a["fill"] != b["fill"]:
        return False
    if bool(a["dash"]) != bool(b["dash"]):
        return False
    return abs(float(a["size"]) - float(b["size"])) <= TOL_SIZE


def _rot_candidates(shapes):
    cands, info = set(), [_polar(s) for s in shapes]
    for i, a in enumerate(shapes):
        for j, b in enumerate(shapes):
            if i == j or not _compatible(a, b):
                continue
            if abs(info[i][1] - info[j][1]) > TOL_R:
                continue
            cands.add(_norm(info[j][0] - info[i][0]))
    return cands


def _ref_candidates(shapes):
    cands, info = set(), [_polar(s) for s in shapes]
    for i, a in enumerate(shapes):
        for j, b in enumerate(shapes):
            if not _compatible(a, b):
                continue
            if abs(info[i][1] - info[j][1]) > TOL_R:
                continue
            cands.add(((info[i][0] + info[j][0]) / 2) % 180)
    return cands


def mirror_lines(shapes, cmp=same_shape):
    out, seen = [], []
    for a in sorted(_ref_candidates(shapes)):
        if any(abs(a - b) < 0.5 for b in seen):
            continue
        if is_symmetric(shapes, "ref", a, cmp):
            out.append(a)
            seen.append(a)
    return out


def lenient_shape(a, b) -> bool:
    """Circle glyphs look identical under any rotation or mirroring."""
    if a["type"] == "circle" and b["type"] == "circle":
        if a["color"] != b["color"] or a["fill"] != b["fill"] or bool(a["dash"]) != bool(b["dash"]):
            return False
        if abs(float(a["size"]) - float(b["size"])) > TOL_SIZE:
            return False
        return (abs(float(a["x"]) - float(b["x"])) <= TOL_XY
                and abs(float(a["y"]) - float(b["y"])) <= TOL_XY)
    return same_shape(a, b)


def smallest_turn(shapes):
    for r in sorted(r for r in _rot_candidates(shapes) if r > 0.5):
        if is_symmetric(shapes, "rot", r):
            return r
    return None


def direction_of(x: float, y: float) -> str:
    a = math.degrees(math.atan2(y, x)) % 360
    return min(DIRECTIONS, key=lambda d: min(abs(a - d[0]), 360 - abs(a - d[0])))[1]


# ---------------------------------------------------------------- contract
class SymmetryPattern:
    def __init__(self, params: dict):
        self.p = params
        self.shapes = params["shapes"]
        self.base = params["base"]
        self.break_ = params.get("break") or None
        self.scenario = params.get("scenario")
        self.timeline = params.get("timeline") or {}
        self.n_frames = params.get("n_frames", 0)

    # -- groups ---------------------------------------------------------
    def groups(self):
        drop = set()
        if self.break_ and self.break_.get("kind") in ("color", "shape"):
            drop = {self.break_["index"]}
        out = collections.OrderedDict()
        for i, s in enumerate(self.base):
            if i in drop:
                continue
            out.setdefault((s["color"], s["type"]), []).append(i)
        return out

    def orbits(self):
        m = {}
        for s in self.base:
            m.setdefault(s["orbit"], (s["color"], s["type"]))
        return [m[k] for k in sorted(m)]

    @staticmethod
    def gname(key) -> str:
        return f"the {key[0]} {PLURAL[key[1]]}"

    @staticmethod
    def parse_group(text: str):
        m = re.search(r"the (\w+) (" + "|".join(sorted(SINGULAR, key=len, reverse=True)) + r")\b", text)
        if not m:
            return None
        return (m.group(1), SINGULAR[m.group(2)])

    @staticmethod
    def option_groups(options):
        out = set()
        for o in options:
            m = re.match(r"the (\w+) (.+)$", str(o).strip())
            if m and m.group(2) in SINGULAR:
                out.add((m.group(1), SINGULAR[m.group(2)]))
        return out

    @staticmethod
    def pick(scored, reducer, restrict=None):
        keys = [k for k in scored if restrict is None or k in restrict]
        if not keys:
            return None
        target = reducer([scored[k] for k in keys])
        for k in keys:
            if scored[k] == target:
                return SymmetryPattern.gname(k)
        return None

    # -- clocks ---------------------------------------------------------
    @staticmethod
    def clock(frames: float) -> str:
        return f"{int(frames / 24 * 10) / 10:.1f} s"

    # -- disc -----------------------------------------------------------
    def moves(self):
        return self.timeline.get("moves") or []

    @staticmethod
    def move_name(m):
        if m["kind"] == "flip":
            return "a left-right flip" if m["axis"] == "vertical" else "an upside-down flip"
        return "a counter-clockwise turn" if m["angle"] > 0 else "a clockwise turn"

    @staticmethod
    def apply_angle(angle, move):
        if move["kind"] == "turn":
            return (angle + move["angle"]) % 360
        return (180 - angle) % 360 if move["axis"] == "vertical" else (-angle) % 360

    @staticmethod
    def apply_shapes(shapes, move):
        if move["kind"] == "turn":
            return transform(shapes, "rot", move["angle"])
        return transform(shapes, "ref", 90 if move["axis"] == "vertical" else 0)

    # -- dispatch -------------------------------------------------------
    def answer(self, question: str, options) -> Answer:
        q = str(question).strip()
        text = q.lower()
        try:
            got = self._answer(q, text, options)
        except Exception:  # never let a handler crash the run
            got = None
        if got is None:
            return Answer(strict=None, confidence="unsupported",
                          note="no handler for this template")
        return Answer(strict=got)

    def _answer(self, q: str, text: str, options):
        # ---- symmetry statements -------------------------------------
        if text.startswith("compare every shape with its symmetric partners"):
            if not self.break_:
                return "every shape matches its partners"
            return BREAK_OPTIONS.get(self.break_["kind"])

        if text.startswith("imagine applying each of these"):
            shapes = self.shapes
            good = []
            for o in options:
                spec = TRANSFORM_OPTIONS.get(str(o).strip())
                if spec and is_symmetric(shapes, spec[0], spec[1]):
                    good.append(o)
            if not good:
                return "none of these"
            if len(good) == 1:
                return str(good[0])
            return "none of these"

        if text.startswith("how many mirror lines through the centre does the finished"):
            return str(len(mirror_lines(self.shapes)))
        if text.startswith("how many mirror lines through the centre does the picture as"):
            return str(len(mirror_lines(self.shapes)))
        if text.startswith("what is the smallest turn about the centre that makes the finished"):
            return self._turn_text(smallest_turn(self.shapes))
        if text.startswith("what is the smallest turn about the centre that makes the picture as"):
            return self._turn_text(smallest_turn(self.shapes))

        if text.startswith("if the last shape to appear had never appeared"):
            ap = self.timeline.get("appear") or []
            if not ap:
                return None
            idx = max(range(len(ap)), key=lambda i: ap[i])
            rest = [s for i, s in enumerate(self.shapes) if i != idx]
            if "how many mirror lines" in text:
                return str(len(mirror_lines(rest)))
            if "smallest turn" in text:
                return self._turn_text(smallest_turn(rest))
            return None

        if text.startswith("if the shape that breaks the symmetry were removed"):
            if not self.break_:
                return None
            idx = self.break_["index"]
            rest = [s for i, s in enumerate(self.base) if i != idx]
            return str(len(mirror_lines(rest)))

        if text.startswith("there is one shape that breaks the symmetry"):
            if not self.break_:
                return None
            s = self.shapes[self.break_["index"]]
            return direction_of(float(s["x"]), float(s["y"]))
        if text.startswith("one shape breaks the symmetry"):
            return self._break_shape_after_move(text)

        if text.startswith("one shape is a different size from its symmetric partners"):
            if not self.break_:
                return None
            i = self.break_["index"]
            base = float(self.base[i]["size"])
            if base <= 0:
                return None
            return f"about {float(self.shapes[i]['size']) / base:.1f} times"

        if text.startswith("as the shapes appear one by one"):
            return self._first_symmetric_prefix(text)

        # ---- shape counts --------------------------------------------
        if text.startswith("how many different colours do the shapes have at the end"):
            return str(len({s["color"] for s in self.shapes}))
        if text.startswith("how many shapes are in the picture at the end"):
            return str(len(self.shapes))
        m = re.match(r"how many (.+?) (?:are|is) in the picture at the end of the video", text)
        if m and m.group(1) in SINGULAR:
            typ = SINGULAR[m.group(1)]
            return str(sum(1 for s in self.shapes if s["type"] == typ))

        # ---- disc -----------------------------------------------------
        if text.startswith("how many times is the disc flipped over"):
            return str(sum(1 for m in self.moves() if m["kind"] == "flip"))
        if text.startswith("the disc moves several times"):
            return str(len(self.moves()))
        m = re.match(r"what is the (first|second|third|fourth|fifth) move the disc makes", text)
        if m:
            k = ORDINALS[m.group(1)]
            mv = self.moves()
            return self.move_name(mv[k - 1]) if k <= len(mv) else None
        if text.startswith("what does the clock show when the disc finishes its last move"):
            mv = self.moves()
            return self.clock(mv[-1]["f1"]) if mv else None
        if text.startswith("adding up all the turns the disc makes"):
            return f"{int(sum(abs(m['angle']) for m in self.moves() if m['kind'] == 'turn'))}°"
        m = re.match(r"by the end of the disc's (first|second|third|fourth|fifth) move", text)
        if m:
            k = ORDINALS[m.group(1)]
            return f"{int(sum(abs(mv['angle']) for mv in self.moves()[:k] if mv['kind'] == 'turn'))}°"
        if text.startswith("ignoring the disc and its notch, after how many"):
            cur, cnt = self.shapes, 0
            for mv in self.moves():
                cur = self.apply_shapes(cur, mv)
                if same_picture(cur, self.shapes):
                    cnt += 1
            return str(cnt)
        if text.startswith("ignoring the disc and its notch, after which move"):
            cur, ans = self.shapes, "never"
            for i, mv in enumerate(self.moves()):
                cur = self.apply_shapes(cur, mv)
                if same_picture(cur, self.shapes):
                    ans = MOVE_NAMES[i]
                    break
            return ans
        m = re.match(r"right after the disc's (first|second|third|fourth|fifth) move", text)
        if m and "notch" in text:
            k = ORDINALS[m.group(1)]
            a = float(self.timeline.get("notch", 0.0))
            for mv in self.moves()[:k]:
                a = self.apply_angle(a, mv)
            return direction_of(math.cos(math.radians(a)), math.sin(math.radians(a)))
        m = re.match(r"if the disc had skipped its (first|second|third|fourth|fifth) move", text)
        if m:
            k = ORDINALS[m.group(1)]
            a = float(self.timeline.get("notch", 0.0))
            for i, mv in enumerate(self.moves()):
                if i == k - 1:
                    continue
                a = self.apply_angle(a, mv)
            return direction_of(math.cos(math.radians(a)), math.sin(math.radians(a)))

        # ---- reveal ---------------------------------------------------
        if self.scenario == "reveal":
            got = self._reveal(q, text, options)
            if got is not None:
                return got

        # ---- assemble -------------------------------------------------
        if self.scenario == "assemble":
            got = self._assemble(q, text, options)
            if got is not None:
                return got

        # ---- waves ----------------------------------------------------
        if self.scenario == "waves":
            got = self._waves(q, text, options)
            if got is not None:
                return got

        return None

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _turn_text(turn):
        return "no turn short of a full turn" if turn is None else f"{round(turn)}°"

    def _break_shape_after_move(self, text):
        if not self.break_:
            return None
        m = re.search(r"disc's (first|second|third|fourth|fifth) move", text)
        k = ORDINALS[m.group(1)] if m else 0
        cur = self.shapes
        for mv in self.moves()[:k]:
            cur = self.apply_shapes(cur, mv)
        s = cur[self.break_["index"]]
        return direction_of(float(s["x"]), float(s["y"]))

    def _first_symmetric_prefix(self, text):
        ap = self.timeline.get("appear") or []
        if not ap:
            return None
        order = sorted(range(len(ap)), key=lambda i: ap[i])
        for cmp in (same_shape, lenient_shape):
            vis = []
            for i in order:
                vis.append(self.shapes[i])
                if len(vis) < 2:
                    continue
                if "half turn" in text:
                    if is_symmetric(list(vis), "rot", 180, cmp):
                        return str(len(vis))
                elif "mirror line" in text:
                    if mirror_lines(list(vis), cmp):
                        return str(len(vis))
        return None

    # -- reveal ---------------------------------------------------------
    def _reveal(self, q, text, options):
        ap = self.timeline.get("appear") or []
        groups = self.groups()
        if not ap:
            return None
        if text.startswith("what does the clock show when the last shape appears"):
            return self.clock(max(ap))
        if text.startswith("the first shape to appear belongs to which group"):
            i = min(range(len(ap)), key=lambda k: ap[k])
            s = self.base[i]
            return self.gname((s["color"], s["type"]))
        if text.startswith("which group is the first to have all of its shapes on screen"):
            return self.pick({k: max(ap[i] for i in v) for k, v in groups.items()}, min)
        if text.startswith("which group is the last to have all of its shapes on screen"):
            return self.pick({k: max(ap[i] for i in v) for k, v in groups.items()}, max)
        if text.startswith("of these groups, which one is the first to have a shape appear"):
            return self.pick({k: min(ap[i] for i in v) for k, v in groups.items()},
                             min, self.option_groups(options))
        if text.startswith("of these groups, which one is the first to have all of its shapes on screen"):
            return self.pick({k: max(ap[i] for i in v) for k, v in groups.items()},
                             min, self.option_groups(options))
        if text.startswith("how many shapes are on screen when the clock shows"):
            m = re.search(r"clock shows ([\d.]+) s", q)
            if not m:
                return None
            f = float(m.group(1)) * 24
            return str(sum(1 for a in ap if a <= f))
        if text.startswith("how many shapes are on screen at the moment the last of the"):
            key = self.parse_group(text)
            if key is None or key not in groups:
                return None
            f = max(ap[i] for i in groups[key])
            return str(sum(1 for a in ap if a <= f))
        m = re.match(r"counting the shapes in the order they appear, which group does the (\w+) shape", text)
        if m:
            k = ORDINALS.get(m.group(1))
            if k is None or k > len(ap):
                return None
            i = sorted(range(len(ap)), key=lambda j: ap[j])[k - 1]
            s = self.base[i]
            return self.gname((s["color"], s["type"]))
        return None

    # -- assemble -------------------------------------------------------
    def _assemble(self, q, text, options):
        names = self.orbits()
        starts = self.timeline.get("starts") or []
        glides = self.timeline.get("glides") or []
        if not starts:
            return None
        settles = [s + g - 1 for s, g in zip(starts, glides)]
        sc_start = {names[i]: starts[i] for i in range(len(names))}
        sc_settle = {names[i]: settles[i] for i in range(len(names))}
        if text.startswith("which group of shapes is the first to start moving"):
            return self.pick(sc_start, min)
        if text.startswith("of these groups, which one starts moving first"):
            return self.pick(sc_start, min, self.option_groups(options))
        if text.startswith("of these groups, which one is the first to have a shape appear"):
            return self.pick(sc_start, min, self.option_groups(options))
        if text.startswith("which group of shapes is the first to settle into place") or \
                text.startswith("which group is the first to settle into place"):
            return self.pick(sc_settle, min)
        if text.startswith("which group of shapes is the last to settle into place") or \
                text.startswith("which group is the last to settle into place"):
            return self.pick(sc_settle, max)
        m = re.match(r"which group is the (second|third|fourth|fifth) to settle into place", text)
        if m:
            k = ORDINALS[m.group(1)]
            order = sorted(range(len(settles)), key=lambda i: (settles[i], i))
            return self.gname(names[order[k - 1]]) if k <= len(order) else None
        if text.startswith("what does the clock show when the last group settles into place"):
            return self.clock(max(settles))
        if text.startswith("how many groups of shapes have settled into place when the clock shows"):
            m = re.search(r"clock shows ([\d.]+) s", q)
            if not m:
                return None
            f = float(m.group(1)) * 24
            return str(sum(1 for s in settles if s <= f))
        if text.startswith("how many groups of shapes are moving when the clock shows"):
            m = re.search(r"clock shows ([\d.]+) s", q)
            if not m:
                return None
            f = float(m.group(1)) * 24
            return str(sum(1 for st, se in zip(starts, settles) if st <= f < se))
        if text.startswith("if the ") and "had started moving" in text and "last to settle" in text:
            m = re.search(r"if the (\w+) (.+?) had started moving ([\d.]+) second", text)
            if not m or m.group(2) not in SINGULAR:
                return None
            key = (m.group(1), SINGULAR[m.group(2)])
            delay = float(m.group(3)) * 24
            new = {names[i]: settles[i] + (delay if names[i] == key else 0) for i in range(len(names))}
            return self.pick(new, max)
        return None

    # -- waves ----------------------------------------------------------
    def _waves(self, q, text, options):
        names = self.orbits()
        starts = self.timeline.get("starts") or []
        trace = self.timeline.get("trace") or 0
        if not starts:
            return None
        n = len(self.shapes)
        per = n // len(names)
        finish = [s + trace - 1 for s in starts]
        word = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

        def incomplete_done(done):
            if not self.break_ or self.break_.get("kind") != "incomplete":
                return 0
            orb = self.shapes[self.break_["index"]]["orbit"]
            return 1 if orb in done else 0

        if text.startswith("the picture is drawn in waves"):
            return str(len(starts))
        if text.startswith("which group of shapes is drawn first"):
            return self.pick({names[i]: starts[i] for i in range(len(names))}, min)
        if text.startswith("of these groups, which one starts being drawn first"):
            return self.pick({names[i]: starts[i] for i in range(len(names))}, min,
                             self.option_groups(options))
        if text.startswith("in which wave are the"):
            key = self.parse_group(text)
            if key is None or key not in names:
                return None
            return word[names.index(key) + 1]
        if text.startswith("what does the clock show when the") and "start to be drawn" in text:
            key = self.parse_group(text)
            if key is None or key not in names:
                return None
            return self.clock(starts[names.index(key)])
        if text.startswith("how many shapes have been traced all the way round when the clock shows"):
            m = re.search(r"clock shows ([\d.]+) s", q)
            if not m:
                return None
            f = float(m.group(1)) * 24
            done = [k for k in range(len(starts)) if finish[k] <= f]
            return str(sum(per for k in done) - incomplete_done(done))
        if text.startswith("what does the clock show when the last wave has been traced"):
            return self.clock(max(finish))
        if text.startswith("how many shapes have been traced all the way round by the moment the"):
            key = self.parse_group(text)
            if key is None or key not in names:
                return None
            f = starts[names.index(key)]
            done = [k for k in range(len(starts)) if finish[k] <= f]
            return str(sum(per for k in done) - incomplete_done(done))
        if text.startswith("suppose one more wave"):
            return str(n + per)
        if text.startswith("how many shapes are drawn after the"):
            key = self.parse_group(text)
            if key is None or key not in names:
                return None
            return str(n - per * (names.index(key) + 1))
        if text.startswith("suppose every wave took twice as long"):
            pause = starts[1] - (starts[0] + trace) if len(starts) > 1 else 0
            total = starts[0] + len(starts) * trace * 2 + (len(starts) - 1) * pause - 1
            return self.clock(total)
        if text.startswith("find the shape that breaks the symmetry"):
            if not self.break_:
                return None
            return word[self.shapes[self.break_["index"]]["orbit"] + 1]
        if text.startswith("follow the dot at the tip"):
            dirs = [s.get("dir") for s in self.shapes]
            return str(max(sum(1 for d in dirs if d == "clockwise"),
                           sum(1 for d in dirs if d == "counter_clockwise")))
        return None

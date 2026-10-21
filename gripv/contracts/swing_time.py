"""swing_time — deterministic replay of the rope-swing route puzzle.

The recipe (`params`) pins the board exactly:

  * `w` x `h` squares; square (i, j) has its centre at ``(i + 0.5, j + 0.5)``
    and the board's y axis points up (y = 0 is the bottom row).
  * `posts` stand on lattice *vertices* ``(x, y)``.
  * A swing hooks a post `P` at most `cap` units away and carries the particle
    to the centre of another square `B` with ``|P - B| == |P - A|`` (same
    radius), turning the shorter way round, the other way being allowed only on
    an exact half turn.  The swept sector must stay on the board and may not
    pass over another standing post; `pits` are squares that cannot be landed
    on.
  * Every swing costs its rope length, and the particle always takes the
    cheapest route from `start` to `goal` (``cost`` + ``margin`` in the params).

Calibration: this solver reproduces the generator's own ``swings`` list and
``cost`` on **2,500/2,500 clips** (2,500 exact paths, 0 cost errors).
Clock: the on-screen timer truncates to one decimal (``floor(t*10)/10``);
"about how long" durations are rounded to one decimal.  Both match the key text
on every clip (1,219 + 1,186 clock rows and 1,108 duration rows).
"""
from __future__ import annotations

import heapq
import math
import re

from ..engine import Answer

FPS = 24.0
TAU = 2 * math.pi
EPS = 1e-9

# colour -> the two words used in the option text
POST_RE = re.compile(r"the (\w+) (\w+) post")

_COLOURS = (
    "red|orange|yellow|green|teal|blue|purple|pink|brown|grey|gray|black|white",
)
_SHAPES = ("circle|square|triangle|diamond|star|hexagon|pentagon")

# module-level caches keyed by a stable board signature (never by id(): the
# per-row contract instance owns fresh lists, and a recycled id would poison
# the cache).
_EDGE_CACHE: dict[tuple, dict] = {}
_ROUTE_CACHE: dict[tuple, tuple] = {}


def _centre(cell):
    return (cell[0] + 0.5, cell[1] + 0.5)


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _angle(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0]) % TAU


def _arc_clear(p, r, a0, sweep, others, w, h):
    """The swept sector stays on the board and sweeps over no other post.

    The angular interval is closed at both ends: a post sitting exactly on the
    start or landing ray is passed over by the rope itself, which the generator
    forbids (this is what rules out a landing square that is collinear with a
    nearer post — e.g. ``swing_time_000003``).
    """
    lo, hi = (a0, a0 + sweep) if sweep >= 0 else (a0 + sweep, a0)
    for k in range(-6, 8):
        for base in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
            a = base + TAU * k
            if lo - 1e-12 <= a <= hi + 1e-12:
                x, y = p[0] + r * math.cos(a), p[1] + r * math.sin(a)
                if x < -1e-9 or x > w + 1e-9 or y < -1e-9 or y > h + 1e-9:
                    return False
    for q in others:
        if _d2(p, q) >= r * r - 1e-9:
            continue
        a = _angle(p, q)
        for k in range(-4, 5):
            aa = a + TAU * k
            if lo - 1e-9 <= aa <= hi + 1e-9:
                return False
    return True


def _build_edges(w, h, posts, cap, drop=None):
    """{(cell_a, cell_b): (rope, colour)} for every legal swing."""
    sig = (w, h, round(cap, 9), tuple(posts), drop)
    cached = _EDGE_CACHE.get(sig)
    if cached is not None:
        return cached
    live = [(c, xy) for c, xy in posts if c != drop]
    cells = [(i, j) for i in range(w) for j in range(h)]
    edges: dict[tuple, tuple] = {}
    for colour, p in live:
        others = [q for c2, q in live if q != p]
        groups: dict[float, list] = {}
        for c in cells:
            groups.setdefault(round(_d2(p, _centre(c)), 9), []).append(c)
        for r2, grp in groups.items():
            if r2 == 0:
                continue
            r = math.sqrt(r2)
            if r > cap + EPS:
                continue
            for a in grp:
                for b in grp:
                    if a == b:
                        continue
                    cur = edges.get((a, b))
                    if cur is not None and cur[0] <= r:
                        continue
                    aa, ab = _angle(p, _centre(a)), _angle(p, _centre(b))
                    s_ccw = (ab - aa) % TAU
                    if abs(s_ccw - math.pi) < 1e-9:
                        cands = ((aa, s_ccw), (aa, -s_ccw))
                    elif s_ccw < math.pi:
                        cands = ((aa, s_ccw),)
                    else:
                        cands = ((aa, s_ccw - TAU),)
                    for a0, sw in cands:
                        if _arc_clear(p, r, a0, sw, others, w, h):
                            edges[(a, b)] = (r, colour)
                            break
    _EDGE_CACHE[sig] = edges
    return edges


def _route(w, h, posts, cap, start, goal, pits, drop=None, cap_limit=None,
           filled=False):
    """Cheapest route: (cost, [(colour, landing_cell), ...]) or (None, None)."""
    ck = (w, h, tuple(posts), round(cap, 9), start, goal,
          tuple(sorted(pits)), drop, cap_limit, filled)
    hit = _ROUTE_CACHE.get(ck)
    if hit is not None:
        return hit
    limit = cap if cap_limit is None else min(cap, cap_limit)
    blocked = set() if filled else {tuple(x) for x in pits}
    edges = _build_edges(w, h, posts, limit, drop=drop)
    adj: dict[tuple, list] = {}
    for (a, b), (r, colour) in edges.items():
        if b in blocked:
            continue
        adj.setdefault(a, []).append((b, r, colour))
    s, g = tuple(start), tuple(goal)
    dist = {s: 0.0}
    prev: dict[tuple, tuple] = {}
    pq = [(0.0, s)]
    while pq:
        cst, u = heapq.heappop(pq)
        if cst > dist.get(u, 1e9) + EPS:
            continue
        for b, r, colour in adj.get(u, ()):
            nd = cst + r
            if nd < dist.get(b, 1e9) - EPS:
                dist[b] = nd
                prev[b] = (u, colour)
                heapq.heappush(pq, (nd, b))
    if g not in dist:
        out = (None, None)
    else:
        path, node = [], g
        while node != s:
            u, colour = prev[node]
            path.append((colour, node))
            node = u
        path.reverse()
        out = (dist[g], path)
    _ROUTE_CACHE[ck] = out
    return out


def _floor1(x: float) -> str:
    """On-screen clock: truncate to one decimal."""
    return f"{math.floor(x * 10 + 1e-9) / 10:.1f}"


def _round1(x: float) -> str:
    return f"{round(x + 1e-12, 1):.1f}"


class SwingTime:
    def __init__(self, params: dict):
        self.p = params
        self.w = int(params["w"])
        self.h = int(params["h"])
        self.cap = float(params["cap"])
        self.posts = [(q["colour"], tuple(q["xy"])) for q in params["posts"]]
        self.by_colour = {c: xy for c, xy in self.posts}
        self.shape = {q["colour"]: q["shape"] for q in params["posts"]}
        self.pits = [tuple(x) for x in params.get("pits") or []]
        self.start = tuple(params["start"])
        self.goal = tuple(params["goal"])
        self.swings = [(s["post"], tuple(s["land"]), int(s["dir"]))
                       for s in params["swings"]]
        self.timing = params["timing"]
        self._lengths = self._rope_lengths()

    # ---------- geometry ----------
    def _rope_lengths(self) -> list[float]:
        out, prev = [], self.start
        for colour, land, _dir in self.swings:
            p = self.by_colour[colour]
            out.append(math.hypot(p[0] - _centre(prev)[0], p[1] - _centre(prev)[1]))
            prev = land
        return out

    def _post_text(self, colour: str) -> str:
        return f"the {colour} {self.shape[colour]} post"

    def _swing_index(self, colour: str) -> int | None:
        idx = [i for i, s in enumerate(self.swings) if s[0] == colour]
        return idx[0] if idx else None

    def _frames(self) -> tuple[int, int, int, list[int]]:
        t = self.timing
        return int(t["intro"]), int(t["pause"]), int(t["outro"]), list(t["dur"])

    # ---------- question dispatch ----------
    def answer(self, question: str, options: list[str]) -> Answer:
        q = " ".join(str(question).split())
        m = POST_RE.search(q)
        colour = m.group(1) if m and m.group(1) in self.by_colour else None
        intro, pause, outro, durs = self._frames()

        # -- board census -------------------------------------------------
        if q.startswith("How many posts are standing"):
            return Answer(str(len(self.posts)))
        if q.startswith("How many hatched pit squares"):
            return Answer(str(len(self.pits)))
        if q.startswith("How many swings does the particle make"):
            return Answer(str(len(self.swings)))
        if q.startswith("How many different posts does the rope attach"):
            return Answer(str(len({s[0] for s in self.swings})))

        # -- route order --------------------------------------------------
        if q.startswith("Which post does the rope attach to on the particle's first"):
            return Answer(self._post_text(self.swings[0][0]))
        if q.startswith("Which post does the particle swing around on its last"):
            return Answer(self._post_text(self.swings[-1][0]))
        m = re.match(r"Which post does the particle swing around on swing (\d+)", q)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(self.swings):
                return Answer(self._post_text(self.swings[i][0]))
            return Answer(None, confidence="unsupported")
        if q.startswith("Of these posts, which one does the rope attach to first"):
            opts = {str(o).strip() for o in options}
            for colour_, _land, _d in self.swings:
                if self._post_text(colour_) in opts:
                    return Answer(self._post_text(colour_))
            return Answer(self._post_text(self.swings[0][0]))
        if q.startswith("Right after the swing around"):
            if colour is None:
                return Answer(None, confidence="unsupported")
            idx = [i for i, s in enumerate(self.swings) if s[0] == colour]
            nxt = [self.swings[i + 1][0] for i in idx if i + 1 < len(self.swings)]
            if len(set(nxt)) == 1:
                return Answer(self._post_text(nxt[0]))
            return Answer(None, confidence="unsupported")
        if q.startswith("In order, what colours are the posts"):
            return Answer(" \u2192 ".join(s[0] for s in self.swings))
        if q.startswith("Which post does the rope attach to more than once"):
            counts: dict[str, int] = {}
            for s in self.swings:
                counts[s[0]] = counts.get(s[0], 0) + 1
            reps = [c for c, n in counts.items() if n > 1]
            if not reps:
                return Answer("no post is used twice")
            if len(reps) == 1:
                return Answer(self._post_text(reps[0]))
            return Answer(None, confidence="unsupported")

        # -- swing shape --------------------------------------------------
        if q.startswith("When the particle swings around"):
            if colour is None:
                return Answer(None, confidence="unsupported")
            i = self._swing_index(colour)
            if i is None:
                return Answer(None, confidence="unsupported")
            d = self.swings[i][2]
            prev = self.start if i == 0 else self.swings[i - 1][1]
            p = self.by_colour[colour]
            a = _centre(prev)
            b = _centre(self.swings[i][1])
            v1 = (a[0] - p[0], a[1] - p[1])
            v2 = (b[0] - p[0], b[1] - p[1])
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            way = "counter-clockwise" if d > 0 else "clockwise"
            far = "more" if ang > 90.0 else "less"
            return Answer(f"{way}, {far} than a quarter turn")
        if q.startswith("Where does the particle land after its swing"):
            if colour is None:
                return Answer(None, confidence="unsupported")
            i = self._swing_index(colour)
            if i is None:
                return Answer(None, confidence="unsupported")
            p = self.by_colour[colour]
            b = _centre(self.swings[i][1])
            dx, dy = b[0] - p[0], b[1] - p[1]
            vert = "up" if dy > 0 else "down"
            horiz = "right" if dx > 0 else "left"
            return Answer(f"{vert} and to the {horiz} of it")
        if q.startswith("How many of the particle's swings go clockwise"):
            return Answer(str(sum(1 for s in self.swings if s[2] < 0)))
        if q.startswith("Around which post does the particle make its longest-lasting"):
            best = max(range(len(durs)), key=lambda i: (durs[i], -i))
            return Answer(self._post_text(self.swings[best][0]))

        # -- clocks -------------------------------------------------------
        if q.startswith("What does the clock show at the moment the rope first appears"):
            return Answer(f"{_floor1(intro / FPS)} s")
        if q.startswith("What does the clock show at the moment the particle lands"):
            land = intro + sum(durs) + pause * (len(durs) - 1)
            return Answer(f"{_floor1(land / FPS)} s")
        if q.startswith("Reading the clock, about how long does the swing around"):
            if colour is None:
                return Answer(None, confidence="unsupported")
            i = self._swing_index(colour)
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{_round1(durs[i] / FPS)} s")

        # -- rope lengths -------------------------------------------------
        if "How long is the rope when the particle swings around" in q:
            if colour is None:
                return Answer(None, confidence="unsupported")
            i = self._swing_index(colour)
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._lengths[i]:.2f}")
        if "Adding up the rope length of every swing" in q:
            return Answer(f"{sum(self._lengths):.2f}")
        if "How long is the longest rope" in q:
            return Answer(f"{max(self._lengths):.2f}")
        if "How many swings use a rope longer than" in q:
            lim = self._asked_length(q)
            return Answer(str(sum(1 for r in self._lengths if r > lim + EPS)))

        # -- counterfactuals ---------------------------------------------
        if q.startswith("If the hatched pits were filled in"):
            cost, _ = _route(self.w, self.h, self.posts, self.cap, self.start,
                             self.goal, self.pits, filled=True)
            return Answer("no route" if cost is None else f"{cost:.2f}")
        if q.startswith("Suppose no rope could be longer than"):
            lim = self._asked_length(q)
            cost, _ = _route(self.w, self.h, self.posts, self.cap, self.start,
                             self.goal, self.pits, cap_limit=lim)
            return Answer("no route" if cost is None else f"{cost:.2f}")
        if q.startswith("If the") and "post were pulled out" in q:
            if colour is None:
                return Answer(None, confidence="unsupported")
            cost, path = _route(self.w, self.h, self.posts, self.cap, self.start,
                                self.goal, self.pits, drop=colour)
            if "how many swings" in q:
                return Answer("no route" if cost is None else str(len(path)))
            if "which post would the rope attach to" in q:
                if cost is None:
                    return Answer("no route")
                return Answer(self._post_text(path[0][0]))
            return Answer("no route" if cost is None else f"{cost:.2f}")

        return Answer(None, confidence="unsupported")

    @staticmethod
    def _asked_length(q: str) -> float:
        """The unit length named in the question (grid-square width or cap)."""
        m = re.search(r"(?:longer than|no rope could be longer than)\s+(\d+(?:\.\d+)?)", q)
        if m:
            return float(m.group(1))
        m = re.search(r"Each grid square is\s+(\d+(?:\.\d+)?)\s+unit", q)
        return float(m.group(1)) if m else 1.0

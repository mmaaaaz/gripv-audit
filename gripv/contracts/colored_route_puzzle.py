"""Deterministic replay for colored_route_puzzle.

Every clip ships the generator's recipe: `stops` (shape + grid node) and
`routes` (colour, style, endpoints `a`/`b`, polyline `nodes`, growing end
`grow`, draw window `draw`, erase window `erase`), plus an optional `token`
(the ring) with its ride list.

Conventions pinned against the keys (25,000/25,000 exact):

  frame        clips run at 24 fps.  A window `[s, e]` means the route occupies
               frames s..e: drawing is visible from s, complete at e.
  clock        clock(t) = floor(frame / 24 * 10) / 10 with one decimal
               ("5.4 s").  Clock readings truncate, they do not round
               (110/24 = 4.583 -> "4.5 s", 130/24 = 5.417 -> "5.4 s").
  clock->frame a question that says "when the clock shows X s" means
               frame = round(X * 24); the generator only asks at multiples of
               0.5 s, so the frame is exact.
  growing      a route is growing at frame f iff draw_start <= f < draw_end;
               it is completely drawn iff draw_end <= f.  (No question lands on
               a growing boundary, so <= vs < there is unpinned; the partition
               keeps the two families disjoint.  The complete boundary is
               pinned: 433 rows flip if draw_end < f is used instead.)
  on screen    a route is on screen at f iff f < erase_start; erase_end always
               falls at least 15 frames before the last frame of the clip.
  end of clip  last drawn frame = n_frames - 1.
  end at       "how many routes end at the X stop" counts routes having X at
               either endpoint (`a` or `b`); a dead-end route (b is null) only
               contributes its `a` stop.
  "at the end" the end-of-video variants drop routes that have been erased by
               the last frame; the plain variants count every route ever drawn.
  corners      a corner is a direction change between consecutive unit
               segments of `nodes`.
  longest      "measured along the line" = number of unit segments
               (len(nodes) - 1); ties go to the lowest route id.
  two stops    pairs are written in the fixed shape order
               circle < square < triangle < diamond < star < cross.
  reach        "how many other stops can be reached from S" = size of S's
               connected component minus S itself, over routes as edges.
  ring         `token.start` is the stop the ring sits on at frame 0; each ride
               has route / from / to / t=[depart, arrive].  The ring arrives at
               `to` at t[1].
  first link   "which route, once completely drawn, first makes it possible to
               travel A -> B" = the first route, in completion order, at whose
               completion A and B are joined using only the routes that are then
               completely drawn and still on screen (an erased route cannot
               carry the path).  The reverse-order counterfactual instead adds
               routes last-drawn-first and keeps everything drawn so far.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24
SHAPES = ("circle", "square", "triangle", "diamond", "star", "cross")
SHAPE_RANK = {s: i for i, s in enumerate(SHAPES)}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}
CLOCK_RE = r"(\d+(?:\.\d+)?) s"
NEVER = "it would never become possible"
NO_PATH = "it could not get there"


def clock(frame: float) -> str:
    """Clock readings truncate to one decimal."""
    return f"{math.floor(round(frame / FPS, 9) * 10) / 10:.1f} s"


def frame_of(text: str) -> int:
    return int(round(float(text) * FPS))


def corners(nodes: list) -> int:
    dirs = [(v[0] - u[0], v[1] - u[1]) for u, v in zip(nodes, nodes[1:])]
    return sum(1 for p, q in zip(dirs, dirs[1:]) if p != q)


def length(nodes: list) -> int:
    return sum(abs(v[0] - u[0]) + abs(v[1] - u[1]) for u, v in zip(nodes, nodes[1:]))


class ColoredRoutePuzzle:
    def __init__(self, params: dict):
        self.p = params
        self.routes = list(params.get("routes") or [])
        self.stops = {int(s["id"]): dict(s) for s in params.get("stops") or []}
        self.n_frames = int(params.get("n_frames") or 0)
        self.token = params.get("token")
        self.by_color = {r["color"]: r for r in self.routes}
        self.by_id = {int(r["id"]): r for r in self.routes}

    # ---------------- helpers ----------------
    def shape(self, sid: int) -> str:
        return self.stops[int(sid)]["shape"]

    def stop_label(self, sid: int) -> str:
        return f"the {self.shape(sid)} stop"

    def route_label(self, r: dict) -> str:
        return f"the {r['color']} route"

    def route_by_label(self, label: str) -> dict | None:
        s = str(label).strip().lower()
        for r in self.routes:
            if self.route_label(r) == s:
                return r
        return None

    def stop_by_label(self, label: str) -> int | None:
        s = str(label).strip().lower()
        for sid in self.stops:
            if self.stop_label(sid) == s:
                return sid
        return None

    def route_stops(self, r: dict) -> list[int]:
        out = [int(r["a"])]
        if r.get("b") is not None:
            out.append(int(r["b"]))
        return out

    def present(self, r: dict, f: float) -> bool:
        """On screen at frame f (erasure has started => gone)."""
        e = r.get("erase")
        return e is None or f < e[0]

    def complete(self, r: dict, f: float) -> bool:
        return r["draw"][1] <= f and self.present(r, f)

    def draw_start(self, r: dict) -> int:
        return int(r["draw"][0])

    def draw_end(self, r: dict) -> int:
        return int(r["draw"][1])

    # ---------------- graph ----------------
    def adjacency(self, routes=None) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {sid: set() for sid in self.stops}
        for r in routes if routes is not None else self.routes:
            if r.get("b") is None:
                continue
            a, b = int(r["a"]), int(r["b"])
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj

    def reachable(self, sid: int, routes=None) -> set[int]:
        adj = self.adjacency(routes)
        seen, stack = {sid}, [sid]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def hop_distance(self, src: int, dst: int, routes=None) -> int | None:
        adj = self.adjacency(routes)
        dist = {src: 0}
        queue = [src]
        for cur in queue:
            if cur == dst:
                return dist[cur]
            for nxt in adj.get(cur, ()):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist.get(dst)

    @staticmethod
    def _find(parent: dict, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def first_connector(self, a: int, b: int, order: list[dict]) -> dict | None:
        """First route, in completion order, at whose completion a and b are joined.

        The graph only ever holds routes that are completely drawn *and still on
        screen* at that moment: a route erased before the candidate completes is
        gone and cannot carry the path.
        """
        for r in sorted(order, key=self.draw_end):
            f = self.draw_end(r)
            live = [x for x in order if self.draw_end(x) <= f and self.present(x, f)]
            if b in self.reachable(a, live):
                return r
        return None

    def first_connector_cumulative(self, a: int, b: int, order: list[dict]) -> dict | None:
        """Union-find along `order` (used by the reverse-order counterfactual,
        where the generator keeps every route that has been drawn so far)."""
        parent = {sid: sid for sid in self.stops}
        for r in order:
            if r.get("b") is None:
                continue
            ra, rb = self._find(parent, int(r["a"])), self._find(parent, int(r["b"]))
            if ra != rb:
                parent[ra] = rb
            if self._find(parent, a) == self._find(parent, b):
                return r
        return None

    def by_start(self, routes=None) -> list[dict]:
        return sorted(routes if routes is not None else self.routes, key=self.draw_start)

    # ---------------- option helpers ----------------
    def _growing_at(self, f: float) -> list[dict]:
        return [r for r in self.routes if self.draw_start(r) <= f < self.draw_end(r)]

    # ---------------- answer ----------------
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = question.strip()
        options = list(options or [])
        end = self.n_frames - 1

        m = re.match(r"^How many stops \(dark shape markers\) are on the map\?$", q)
        if m:
            return Answer(str(len(self.stops)))

        m = re.match(r"^How many routes stop at a dead end", q)
        if m:
            return Answer(str(sum(1 for r in self.routes if r.get("b") is None)))

        m = re.match(r"^How many coloured routes are drawn(?: during the video)?\?$", q)
        if m:
            return Answer(str(len(self.routes)))

        m = re.match(r"^How many coloured routes are still on screen at the end of the video\?$", q)
        if m:
            return Answer(str(sum(1 for r in self.routes if self.present(r, end))))

        m = re.match(r"^How many routes are drawn as (solid|dashed|dotted) lines(?: during the video)?\?$", q)
        if m:
            return Answer(str(sum(1 for r in self.routes if r["style"] == m.group(1))))

        m = re.match(r"^From which end does the (.+?) route start growing\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None:
                return Answer(None, confidence="unsupported")
            if r["grow"] == "a":
                return Answer(self.stop_label(int(r["a"])))
            if r.get("b") is None:
                return Answer("its dead end (the crossbar)")
            return Answer(self.stop_label(int(r["b"])))

        m = re.match(r"^Which two stops does the (.+?) route connect\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None or r.get("b") is None:
                return Answer(None, confidence="unsupported")
            pair = sorted([self.shape(int(r["a"])), self.shape(int(r["b"]))], key=SHAPE_RANK.get)
            return Answer(f"the {pair[0]} stop and the {pair[1]} stop")

        m = re.match(r"^How many routes end at the (.+?) stop\?$", q)
        if m:
            shape = m.group(1)
            n = sum(1 for r in self.routes
                    if any(self.shape(s) == shape for s in self.route_stops(r)))
            return Answer(str(n))

        m = re.match(r"^At the end of the video, how many routes end at the (.+?) stop\?$", q)
        if m:
            shape = m.group(1)
            n = sum(1 for r in self.routes if self.present(r, end)
                    and any(self.shape(s) == shape for s in self.route_stops(r)))
            return Answer(str(n))

        m = re.match(r"^How many routes directly connect the (.+?) stop and the (.+?) stop\?$", q)
        if m:
            a, b = m.group(1), m.group(2)
            n = sum(1 for r in self.routes if r.get("b") is not None
                    and {self.shape(int(r["a"])), self.shape(int(r["b"]))} == {a, b})
            return Answer(str(n))

        m = re.match(r"^At the end of the video, how many routes directly connect the (.+?) stop and the (.+?) stop\?$", q)
        if m:
            a, b = m.group(1), m.group(2)
            n = sum(1 for r in self.routes if r.get("b") is not None and self.present(r, end)
                    and {self.shape(int(r["a"])), self.shape(int(r["b"]))} == {a, b})
            return Answer(str(n))

        # ---- drawing order -------------------------------------------------
        m = re.match(r"^Which route is the first to start being drawn\?$", q)
        if m:
            return Answer(self.route_label(min(self.routes, key=self.draw_start)))

        m = re.match(r"^Which route is the last to start being drawn\?$", q)
        if m:
            return Answer(self.route_label(max(self.routes, key=self.draw_start)))

        m = re.match(r"^Which route is the (\w+) one to start being drawn\?$", q)
        if m and m.group(1) in ORDINALS:
            order = self.by_start()
            i = ORDINALS[m.group(1)] - 1
            if i >= len(order):
                return Answer(None, confidence="unsupported")
            return Answer(self.route_label(order[i]))

        m = re.match(r"^Which route is the next one to start being drawn after the (.+?) route starts\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None:
                return Answer(None, confidence="unsupported")
            later = [x for x in self.routes if self.draw_start(x) > self.draw_start(r)]
            if not later:
                return Answer(None, confidence="unsupported")
            return Answer(self.route_label(min(later, key=self.draw_start)))

        m = re.match(r"^Which of these routes starts being drawn first\?$", q)
        if m:
            cands = [(o, self.route_by_label(o)) for o in options]
            cands = [(o, r) for o, r in cands if r is not None]
            if not cands:
                return Answer(None, confidence="unsupported")
            return Answer(min(cands, key=lambda t: self.draw_start(t[1]))[0])

        m = re.match(r"^Which of these routes is the first to be completely drawn\?$", q)
        if m:
            cands = [(o, self.route_by_label(o)) for o in options]
            cands = [(o, r) for o, r in cands if r is not None]
            if not cands:
                return Answer(None, confidence="unsupported")
            return Answer(min(cands, key=lambda t: self.draw_end(t[1]))[0])

        m = re.match(r"^Which of these routes disappears during the video\?$", q)
        if m:
            hits = [o for o in options
                    if (r := self.route_by_label(o)) is not None and r.get("erase")]
            return Answer(hits[0] if hits else None,
                          confidence="exact" if hits else "unsupported")

        m = re.match(r"^Which route takes the longest time to draw, from first appearing to complete\?$", q)
        if m:
            r = max(self.routes, key=lambda x: self.draw_end(x) - self.draw_start(x))
            return Answer(self.route_label(r))

        # ---- drawing clocks ------------------------------------------------
        m = re.match(rf"^What does the clock show at the moment the (.+?) route becomes completely drawn\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None:
                return Answer(None, confidence="unsupported")
            return Answer(clock(self.draw_end(r)))

        m = re.match(r"^What does the clock show at the moment the (.+?) route has completely disappeared\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None or not r.get("erase"):
                return Answer(None, confidence="unsupported")
            return Answer(clock(r["erase"][1]))

        m = re.match(rf"^How many routes are completely drawn and on screen when the clock shows {CLOCK_RE}\?$", q)
        if m:
            f = frame_of(m.group(1))
            return Answer(str(sum(1 for r in self.routes if self.complete(r, f))))

        m = re.match(rf"^How many routes are growing \(being drawn\) at the same time when the clock shows {CLOCK_RE}\?$", q)
        if m:
            f = frame_of(m.group(1))
            return Answer(str(len(self._growing_at(f))))

        m = re.match(rf"^Which of these routes is growing \(being drawn\) when the clock shows {CLOCK_RE}\?$", q)
        if m:
            f = frame_of(m.group(1))
            grow = {r["color"] for r in self._growing_at(f)}
            hit = [o for o in options
                   if (r := self.route_by_label(o)) is not None and r["color"] in grow]
            return Answer(hit[0] if hit else None,
                          confidence="exact" if hit else "unsupported")

        m = re.match(r"^How many routes are already completely drawn at the moment the first (solid|dashed|dotted) route starts to appear\?$", q)
        if m:
            style = m.group(1)
            cand = [r for r in self.routes if r["style"] == style]
            if not cand:
                return Answer(None, confidence="unsupported")
            f0 = min(self.draw_start(r) for r in cand)
            return Answer(str(sum(1 for r in self.routes if self.complete(r, f0))))

        m = re.match(r"^What is the largest number of completely drawn routes on screen at the same moment\?$", q)
        if m:
            frames = sorted({self.draw_end(r) for r in self.routes})
            best = max((sum(1 for r in self.routes if self.complete(r, f)) for f in frames), default=0)
            return Answer(str(best))

        m = re.match(r"^How many corners \(turns\) does the (.+?) route make\?$", q)
        if m:
            r = self.by_color.get(m.group(1))
            if r is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(corners(r["nodes"])))

        m = re.match(r"^Measured along the line, which route(?: drawn during the video)? is the longest\?$", q)
        if m:
            r = max(self.routes, key=lambda x: (length(x["nodes"]), -int(x["id"])))
            return Answer(self.route_label(r))

        # ---- connectivity --------------------------------------------------
        m = re.match(r"^Which route, once completely drawn, first makes it possible to travel "
                     r"from the (.+?) stop to the (.+?) stop along the routes", q)
        if m:
            a, b = self.stop_by_label("the " + m.group(1) + " stop"), \
                   self.stop_by_label("the " + m.group(2) + " stop")
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            r = self.first_connector(a, b, self.by_start())
            return Answer(self.route_label(r) if r else NEVER)

        m = re.match(r"^Suppose the routes had been drawn one at a time in the reverse order "
                     r"\(the last one first\)\. Which route, once completely drawn, "
                     r"would first make it possible to travel from the (.+?) stop to the (.+?) stop", q)
        if m:
            a, b = self.stop_by_label("the " + m.group(1) + " stop"), \
                   self.stop_by_label("the " + m.group(2) + " stop")
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            order = sorted(self.routes, key=self.draw_start, reverse=True)
            r = self.first_connector_cumulative(a, b, order)
            return Answer(self.route_label(r) if r else NEVER)

        m = re.match(r"^Suppose the (.+?) route had never been drawn \(everything else the same\)\. "
                     r"Which route, once completely drawn, would first make it possible to travel "
                     r"from the (.+?) stop to the (.+?) stop", q)
        if m:
            col = m.group(1)
            a, b = self.stop_by_label("the " + m.group(2) + " stop"), \
                   self.stop_by_label("the " + m.group(3) + " stop")
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            rest = [r for r in self.by_start() if r["color"] != col]
            r = self.first_connector(a, b, rest)
            return Answer(self.route_label(r) if r else NEVER)

        m = re.match(r"^When the clock shows (\d+(?:\.\d+)?) s, how many other stops can be reached "
                     r"from the (.+?) stop along the completely drawn routes", q)
        if m:
            f = frame_of(m.group(1))
            sid = self.stop_by_label("the " + m.group(2) + " stop")
            if sid is None:
                return Answer(None, confidence="unsupported")
            live = [r for r in self.routes if self.complete(r, f)]
            return Answer(str(len(self.reachable(sid, live)) - 1))

        m = re.match(r"^Suppose the (.+?) route had never been drawn\. How many other stops could be "
                     r"reached from the (.+?) stop( at the end of the video)? along the routes", q)
        if m:
            col, shape, at_end = m.group(1), m.group(2), bool(m.group(3))
            sid = self.stop_by_label("the " + shape + " stop")
            if sid is None:
                return Answer(None, confidence="unsupported")
            rest = [r for r in self.routes if r["color"] != col
                    and (not at_end or self.present(r, end))]
            return Answer(str(len(self.reachable(sid, rest)) - 1))

        m = re.match(r"^Suppose the (.+?) route had never been erased\. How many other stops could be "
                     r"reached from the (.+?) stop at the end of the video", q)
        if m:
            col, shape = m.group(1), m.group(2)
            sid = self.stop_by_label("the " + shape + " stop")
            if sid is None:
                return Answer(None, confidence="unsupported")
            rest = [r for r in self.routes
                    if r["color"] == col or self.present(r, end)]
            return Answer(str(len(self.reachable(sid, rest)) - 1))

        m = re.match(r"^Of the routes that end at the (.+?) stop, which one starts being drawn first\?$", q)
        if m:
            shape = m.group(1)
            cand = [r for r in self.routes
                    if any(self.shape(s) == shape for s in self.route_stops(r))]
            if not cand:
                return Answer(None, confidence="unsupported")
            return Answer(self.route_label(min(cand, key=self.draw_start)))

        m = re.match(r"^Suppose the (.+?) route had been the first route drawn and the others had kept "
                     r"their order\. Which route would then be the (\w+) one to start being drawn\?$", q)
        if m and m.group(2) in ORDINALS:
            col = m.group(1)
            if col not in self.by_color:
                return Answer(None, confidence="unsupported")
            rest = [r for r in self.by_start() if r["color"] != col]
            order = [self.by_color[col]] + rest
            i = ORDINALS[m.group(2)] - 1
            if i >= len(order):
                return Answer(None, confidence="unsupported")
            return Answer(self.route_label(order[i]))

        m = re.match(r"^Suppose every (solid|dashed|dotted) route were removed\. "
                     r"How many stops would have no route left at all\?$", q)
        if m:
            style = m.group(1)
            rest = [r for r in self.routes if r["style"] != style]
            touched = {s for r in rest for s in self.route_stops(r)}
            return Answer(str(sum(1 for sid in self.stops if sid not in touched)))

        # ---- the ring ------------------------------------------------------
        m = re.match(r"^On which stop does the (.+?) ring sit at the start of the video\?$", q)
        if m:
            if not self.token:
                return Answer(None, confidence="unsupported")
            return Answer(self.stop_label(int(self.token["start"])))

        m = re.match(r"^On which stop is the (.+?) ring at the end of the video\?$", q)
        if m:
            if not self.token or not self.token.get("rides"):
                return Answer(None, confidence="unsupported")
            return Answer(self.stop_label(int(self.token["rides"][-1]["to"])))

        m = re.match(r"^How many different stops does the (.+?) ring visit, counting the one it starts on\?$", q)
        if m:
            if not self.token:
                return Answer(None, confidence="unsupported")
            seen = [int(self.token["start"])] + [int(x["to"]) for x in self.token["rides"]]
            return Answer(str(len(set(seen))))

        m = re.match(r"^Which route does the (.+?) ring ride along on its (\w+) trip\?$", q)
        if m and m.group(2) in ORDINALS:
            if not self.token:
                return Answer(None, confidence="unsupported")
            rides = self.token["rides"]
            i = ORDINALS[m.group(2)] - 1
            if i >= len(rides):
                return Answer(None, confidence="unsupported")
            return Answer(self.route_label(self.by_id[int(rides[i]["route"])]))

        m = re.match(r"^At which stop does the (.+?) ring arrive at the end of its (\w+) trip\?$", q)
        if m and m.group(2) in ORDINALS:
            if not self.token:
                return Answer(None, confidence="unsupported")
            rides = self.token["rides"]
            i = ORDINALS[m.group(2)] - 1
            if i >= len(rides):
                return Answer(None, confidence="unsupported")
            return Answer(self.stop_label(int(rides[i]["to"])))

        m = re.match(r"^What does the clock show when the (.+?) ring arrives at the (.+?) stop\?$", q)
        if m:
            if not self.token:
                return Answer(None, confidence="unsupported")
            sid = self.stop_by_label("the " + m.group(2) + " stop")
            hits = [x for x in self.token["rides"] if int(x["to"]) == sid]
            if not hits:
                return Answer(None, confidence="unsupported")
            return Answer(clock(hits[0]["t"][1]))

        m = re.match(r"^Adding up the corners \(turns\) of every route the (.+?) ring rides along, "
                     r"how many corners are there in total\?$", q)
        if m:
            if not self.token:
                return Answer(None, confidence="unsupported")
            total = sum(corners(self.by_id[int(x["route"])]["nodes"]) for x in self.token["rides"])
            return Answer(str(total))

        m = re.match(r"^Suppose the (.+?) route had never been drawn\. What is the smallest number of "
                     r"routes the (.+?) ring would have to ride to get from the stop where it starts "
                     r"to the stop where it ends\?$", q)
        if m:
            col = m.group(1)
            if not self.token or not self.token.get("rides"):
                return Answer(None, confidence="unsupported")
            src = int(self.token["start"])
            dst = int(self.token["rides"][-1]["to"])
            rest = [r for r in self.routes if r["color"] != col]
            d = self.hop_distance(src, dst, rest)
            return Answer(NO_PATH if d is None else str(d))

        m = re.match(r"^Suppose the (.+?) route had never been drawn\. How many other stops could be "
                     r"reached from the (.+?) stop", q)
        if m:
            col, shape = m.group(1), m.group(2)
            sid = self.stop_by_label("the " + shape + " stop")
            if sid is None:
                return Answer(None, confidence="unsupported")
            rest = [r for r in self.routes if r["color"] != col]
            return Answer(str(len(self.reachable(sid, rest)) - 1))

        return Answer(None, confidence="unsupported")

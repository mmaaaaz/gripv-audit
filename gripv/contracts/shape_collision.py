"""Deterministic replay for `shape_collision`.

Params
------
  scenario : swap | holes | posts | rails | infect
  rule     : swap | infect
  tag      : the infectious pattern (infect scenario only)
  box      : [x0, y0, x1, y1]
  bodies   : [{shape, color, pattern, r, x, y, vx, vy}]
  holes    : [{x, y, r}]      -- in the floor of the box
  posts    : [{x, y, r}]      -- static round pillars
  lanes    : [y, ...]         -- rails the shapes ride on (rails scenario)
  duration : seconds

Law (pinned by aud_18_collision_battery.py on clip 001265, 10/10)
----------------------------------------------------------------
* Free flight at constant velocity inside the box.  A wall contact is a
  radius-aware reflection: the body sits at ``x0 + r`` / ``x1 - r`` /
  ``y0 + r`` / ``y1 - r`` and the normal velocity flips.
* Body-body contact is a **mass-weighted elastic** impulse with
  ``m ∝ r²`` (the normal component only).  Equal-mass exchange and
  full-velocity exchange both fail the aud_18 keys.
* On every body-body contact the two bodies **trade fill patterns**.
  Walls, posts and holes never change a pattern.
* Posts are static, infinitely heavy: reflection about the contact
  normal, no pattern effect.
* A body falls into a hole when its centre crosses the hole rim
  (``|p - c| < r_hole``); it is then out of the box for good.
* ``infect`` clips: every body starts ``solid`` except one that carries
  ``tag``; a contact between a tagged and an untagged body passes the
  tag on (both keep it).  Nothing is traded back.

The simulation is event-driven (exact time of impact from the quadratic
for each wall / pair / post / hole), so a 10 s clip costs a few dozen
event steps instead of 20 000 fixed 0.0005 s frames.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

EPS = 1e-9
HOLE_MODE = "center"   # "center" = centre crosses the rim, "overlap" = discs touch

# ---------------------------------------------------------------- simulator


class Obs:
    """Recorded timeline of one simulation."""

    __slots__ = ("n", "names", "pair", "wall", "post", "fall", "segs",
                 "initial", "final", "infections", "rule", "tag")

    def __init__(self, n):
        self.n = n
        self.names = []
        self.pair = []      # (t, i, j)
        self.wall = []      # (t, i)
        self.post = []      # (t, i, k)
        self.fall = []      # (t, i)
        self.segs = []      # per body [(t, pattern), ...]
        self.initial = []
        self.final = []
        self.infections = []  # (t, source, target)
        self.rule = "swap"
        self.tag = None

    # -- convenience -------------------------------------------------
    def pair_count(self, i):
        return sum(1 for _, a, b in self.pair if a == i or b == i)

    def partners(self, i):
        return {b if a == i else a for _, a, b in self.pair if a == i or b == i}

    def wall_count(self, i):
        return sum(1 for _, a in self.wall if a == i)

    def post_count(self, i):
        return sum(1 for _, a, _k in self.post if a == i)

    def first_pair_with(self, i):
        for t, a, b in self.pair:
            if a == i or b == i:
                return t, (b if a == i else a)
        return None

    def pattern_at(self, i, t):
        cur = self.segs[i][0][1]
        for ts, p in self.segs[i]:
            if ts <= t + 1e-12:
                cur = p
            else:
                break
        return cur

    def patterns_seen(self, i):
        out = []
        for _, p in self.segs[i]:
            if p not in out:
                out.append(p)
        return out

    def alive_at(self, t):
        dead = {i for tf, i in self.fall if tf <= t + 1e-12}
        return [i for i in range(self.n) if i not in dead]

    def fall_time(self, i):
        for t, a in self.fall:
            if a == i:
                return t
        return None

    def post_first(self, i):
        ts = [t for t, a, _k in self.post if a == i]
        return min(ts) if ts else None

    def wall_first(self, i):
        ts = [t for t, a in self.wall if a == i]
        return min(ts) if ts else None


def _reflection_times(x, y, vx, vy, r, box):
    """Candidate wall-contact times (dt, axis) for one body."""
    x0, y0, x1, y1 = box
    out = []
    if vx > EPS:
        out.append(((x1 - r - x) / vx, "x", 1))
    elif vx < -EPS:
        out.append(((x0 + r - x) / vx, "x", -1))
    if vy > EPS:
        out.append(((y1 - r - y) / vy, "y", 1))
    elif vy < -EPS:
        out.append(((y0 + r - y) / vy, "y", -1))
    return out


def _circle_hit(dx, dy, ux, uy, R):
    """Earliest dt>=-eps at which |d + u dt| = R, or None."""
    c = dx * dx + dy * dy - R * R
    b = dx * ux + dy * uy
    a = ux * ux + uy * uy
    if c < 0.0:
        # already inside the ring: contact now unless already separating
        return 0.0 if b < 0.0 else None
    if a < 1e-15 or b >= 0.0:
        return None
    disc = b * b - a * c
    if disc < 0.0:
        return None
    dt = (-b - math.sqrt(disc)) / a
    return dt if dt >= 0.0 else 0.0


def simulate(bodies, box, posts=(), holes=(), lanes=(), duration=0.0, *,
             rule="swap", tag=None, hole_mode="center", swap_skip=(),
             skip_pairs=(), pattern_override=None, remove=(), t_end=None):
    """Run the billiard and return an :class:`Obs`."""
    keep = [i for i in range(len(bodies)) if i not in remove]
    n = len(keep)
    obs = Obs(n)
    obs.rule = rule
    obs.tag = tag
    obs.names = [bodies[i] for i in keep]

    x = [float(bodies[i]["x"]) for i in keep]
    y = [float(bodies[i]["y"]) for i in keep]
    vx = [float(bodies[i]["vx"]) for i in keep]
    vy = [float(bodies[i]["vy"]) for i in keep]
    r = [float(bodies[i]["r"]) for i in keep]
    pat = [(pattern_override or {}).get(i, bodies[i]["pattern"]) for i in keep]
    alive = [True] * n

    m = [ri * ri for ri in r]          # mass ∝ r²  (aud_18 law)
    obs.initial = list(pat)
    obs.segs = [[(0.0, p)] for p in pat]
    meet = {}
    skip = set(swap_skip)
    x0, y0, x1, y1 = [float(v) for v in box]
    limit = float(duration if t_end is None else t_end)

    t = 0.0
    guard = 0
    while guard < 20000:
        guard += 1
        best = None            # (dt, kind, payload)
        best_dt = float("inf")

        # walls
        for i in range(n):
            if not alive[i]:
                continue
            for dt, axis, side in _reflection_times(x[i], y[i], vx[i], vy[i], r[i],
                                                    (x0, y0, x1, y1)):
                if dt < 0.0:
                    continue
                if dt < best_dt - 1e-12:
                    best_dt = dt
                    best = ("wall", i, axis, side)

        # body-body
        for i in range(n):
            if not alive[i]:
                continue
            xi, yi, vxi, vyi, ri = x[i], y[i], vx[i], vy[i], r[i]
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                dt = _circle_hit(x[j] - xi, y[j] - yi,
                                 vx[j] - vxi, vy[j] - vyi, ri + r[j])
                if dt is not None and dt < best_dt - 1e-12:
                    best_dt = dt
                    best = ("pair", i, j)

        # posts
        for i in range(n):
            if not alive[i]:
                continue
            for k, po in enumerate(posts):
                dt = _circle_hit(x[i] - po["x"], y[i] - po["y"],
                                 vx[i], vy[i], r[i] + po["r"])
                if dt is not None and dt < best_dt - 1e-12:
                    best_dt = dt
                    best = ("post", i, k)

        # holes
        if holes:
            for i in range(n):
                if not alive[i]:
                    continue
                for k, ho in enumerate(holes):
                    R = ho["r"] + (r[i] if hole_mode == "overlap" else 0.0)
                    dt = _circle_hit(x[i] - ho["x"], y[i] - ho["y"],
                                     vx[i], vy[i], R)
                    if dt is not None and dt < best_dt - 1e-12:
                        best_dt = dt
                        best = ("hole", i, k)

        if best is None or best_dt == float("inf") or t + best_dt > limit + 1e-12:
            break

        dt = max(0.0, best_dt)
        t += dt
        for i in range(n):
            if alive[i]:
                x[i] += vx[i] * dt
                y[i] += vy[i] * dt

        kind = best[0]
        if kind == "wall":
            _, i, axis, side = best
            if axis == "x":
                x[i] = (x1 - r[i]) if side > 0 else (x0 + r[i])
                vx[i] = -abs(vx[i]) if side > 0 else abs(vx[i])
            else:
                y[i] = (y1 - r[i]) if side > 0 else (y0 + r[i])
                vy[i] = -abs(vy[i]) if side > 0 else abs(vy[i])
            obs.wall.append((t, i))
        elif kind == "post":
            _, i, k = best
            po = posts[k]
            dx, dy = x[i] - po["x"], y[i] - po["y"]
            d = math.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            nx, ny = dx / d, dy / d
            rv = vx[i] * nx + vy[i] * ny
            vx[i] -= 2.0 * rv * nx
            vy[i] -= 2.0 * rv * ny
            R = r[i] + po["r"]
            x[i] = po["x"] + nx * R
            y[i] = po["y"] + ny * R
            obs.post.append((t, i, k))
        elif kind == "hole":
            _, i, _k = best
            alive[i] = False
            obs.fall.append((t, i))
            obs.final = None
        elif kind == "pair":
            _, i, j = best
            dx, dy = x[j] - x[i], y[j] - y[i]
            d = math.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            nx, ny = dx / d, dy / d
            rv = (vx[i] - vx[j]) * nx + (vy[i] - vy[j]) * ny
            if rv > 0.0:
                f = 2.0 * rv / (m[i] + m[j])
                vx[i] -= f * m[j] * nx
                vy[i] -= f * m[j] * ny
                vx[j] += f * m[i] * nx
                vy[j] += f * m[i] * ny
                # pin the contact so no phantom overlap survives
                R = r[i] + r[j]
                x[i] = x[j] - nx * R
                y[i] = y[j] - ny * R
                obs.pair.append((t, i, j))
                key = (i, j)
                meet[key] = meet.get(key, 0) + 1
                k = meet[key]
                if rule == "infect":
                    if (i, j, k) in skip or key in skip_pairs:
                        continue
                    if pat[i] == tag and pat[j] != tag:
                        pat[j] = tag
                        obs.segs[j].append((t, tag))
                        obs.infections.append((t, i, j))
                    elif pat[j] == tag and pat[i] != tag:
                        pat[i] = tag
                        obs.segs[i].append((t, tag))
                        obs.infections.append((t, j, i))
                elif (i, j, k) not in skip and key not in skip_pairs:
                    pat[i], pat[j] = pat[j], pat[i]
                    obs.segs[i].append((t, pat[i]))
                    obs.segs[j].append((t, pat[j]))
        # end event switch

    obs.final = list(pat)
    return obs


# ---------------------------------------------------------------- contract

SHAPES = ("circle", "square", "triangle", "diamond", "heart", "star",
          "pentagon", "hexagon", "octagon", "cross", "trapezoid")
COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "pink",
          "cyan", "lime", "white", "grey", "gray")


def _name(b):
    return f"the {b['color']} {b['shape']}"


class _Clip:
    """One clip: params + lazily simulated variants."""

    def __init__(self, params):
        self.p = params
        self.box = [float(v) for v in params["box"]]
        self.bodies = params["bodies"]
        self.holes = params.get("holes") or []
        self.posts = params.get("posts") or []
        self.lanes = params.get("lanes") or []
        self.duration = float(params.get("duration") or 0.0)
        # the video shows frames 0..n_frames-1 at 24 fps, so events after the
        # last frame are not visible (and are not counted by the keys)
        nf = params.get("n_frames")
        self.t_end = ((nf - 1) / 24.0) if nf else self.duration
        self.rule = "infect" if (params.get("rule") == "infect") else "swap"
        self.tag = params.get("tag")
        self._base = None
        self._cache = {}

    # -- base run ----------------------------------------------------
    @property
    def base(self) -> Obs:
        if self._base is None:
            self._base = simulate(self.bodies, self.box, self.posts, self.holes,
                                  self.lanes, self.duration,
                                  rule=self.rule, tag=self.tag,
                                  hole_mode=HOLE_MODE, t_end=self.t_end)
        return self._base

    def variant(self, key, **kw):
        if key not in self._cache:
            self._cache[key] = simulate(self.bodies, self.box,
                                        kw.pop("posts", self.posts),
                                        kw.pop("holes", self.holes),
                                        kw.pop("lanes", self.lanes),
                                        self.duration,
                                        rule=kw.pop("rule", self.rule),
                                        tag=kw.pop("tag", self.tag),
                                        t_end=kw.pop("t_end", self.t_end), **kw)
        return self._cache[key]

    def without(self, i, holes=True, posts=True):
        return self.variant(("rm", i, holes, posts), remove={i})

    # -- naming ------------------------------------------------------
    def idx_of(self, desc):
        """Map a question phrase ('the lime diamond') to a body index."""
        d = " " + desc.lower().strip() + " "
        d = re.sub(r"\s+", " ", d)
        best = None
        for i, b in enumerate(self.bodies):
            for cand in (f" {b['color']} {b['shape']} ", f" {b['color']} shape ",
                         f" {b['color']} "):
                if cand in d:
                    if best is None or len(cand) > best[0]:
                        best = (len(cand), i)
        return best[1] if best else None


_DIRECT_CACHE: dict = {}
_CACHE_ORDER: list = []


def _fingerprint(params) -> tuple:
    """Content key, so the cache is safe even if caller dicts are transient."""
    return (
        tuple(params["box"]),
        tuple((b["shape"], b["color"], b["pattern"], b["r"], b["x"], b["y"],
               b["vx"], b["vy"]) for b in params["bodies"]),
        tuple(sorted((h["x"], h["y"], h["r"]) for h in (params.get("holes") or []))),
        tuple(sorted((p["x"], p["y"], p["r"]) for p in (params.get("posts") or []))),
        tuple(params.get("lanes") or ()),
        params.get("duration"), params.get("n_frames"), params.get("rule"),
        params.get("tag"),
    )


def _clip(params) -> _Clip:
    key = _fingerprint(params)
    c = _DIRECT_CACHE.get(key)
    if c is None:
        c = _Clip(params)
        _DIRECT_CACHE[key] = c
        _CACHE_ORDER.append(key)
        if len(_CACHE_ORDER) > 64:
            _DIRECT_CACHE.pop(_CACHE_ORDER.pop(0), None)
    return c


class ShapeCollision:
    def __init__(self, params: dict):
        self.p = params
        self.c = _clip(params)

    # ---------- helpers ----------
    def nm(self, i):
        return _name(self.c.bodies[i])

    def count(self, n):
        return str(int(n))

    def clock(self, t):
        """The clock reads the frame the event is first seen on, truncated to 0.1 s.

        24 fps frames: an event at t is drawn on frame ceil(24 t); the timer
        shows that frame's time with the tenths digit truncated (aud_18:
        1.848 s -> frame 45 = 1.875 s -> "1.8 s").  156/156 clock keys agree.
        """
        return f"{math.floor(self._ft(t) * 10.0 + 1e-9) / 10.0:.1f} s"

    @classmethod
    def _clock_val(cls, t):
        return math.floor(cls._ft(t) * 10.0 + 1e-9) / 10.0

    @staticmethod
    def _ft(t):
        """Time of the first frame that shows an event happening at t."""
        return math.ceil(t * 24.0 - 1e-9) / 24.0

    def _cf_pattern_answer(self, q, sub):
        """Answer the pattern tails shared by the pattern counterfactuals."""
        c = self.c
        m = re.search(r"what fill pattern would the (.+?) have at the end", q)
        if m:
            t = c.idx_of(m.group(1))
            if t is None:
                return Answer(None, confidence="unsupported")
            return Answer(sub.final[t])
        if "how many of the shapes still in the box at the end" in q:
            alive = sub.alive_at(c.t_end)
            return Answer(self.count(sum(1 for x in alive if sub.final[x] == sub.initial[x])))
        if "how many shapes would end with the same fill pattern" in q:
            return Answer(self.count(sum(1 for x in range(sub.n) if sub.final[x] == sub.initial[x])))
        m = re.search(r"how many shapes would wear the (.+?) pattern at the end", q)
        if m:
            pat = m.group(1)
            alive = sub.alive_at(c.t_end)
            return Answer(self.count(sum(1 for x in alive if sub.final[x] == pat)))
        return Answer(None, confidence="unsupported", note="counterfactual tail")

    def first_collision(self, obs):
        return obs.pair[0] if obs.pair else None

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        c = self.c
        obs = c.base
        opts = [str(o) for o in (options or [])]
        opt_l = [o.lower() for o in opts]

        # ---------------- static facts ----------------
        if re.search(r"how many shapes are in the box at the start", q):
            return Answer(self.count(len(c.bodies)))

        if re.search(r"how many horizontal rails", q):
            return Answer(self.count(len(c.lanes)))

        m = re.search(r"how many round (\w+) posts stand in the box", q)
        if m:
            return Answer(self.count(len(c.posts)))

        if re.search(r"how many holes are there in the floor", q):
            return Answer(self.count(len(c.holes)))

        m = re.search(r"what colour is the (\w+)\?", q)
        if m:
            sh = m.group(1)
            hits = [b for b in c.bodies if b["shape"] == sh]
            if len(hits) == 1:
                return Answer(hits[0]["color"])
            return Answer(None, confidence="uncertain", note=f"{len(hits)} {sh}s")

        m = re.search(r"what outline does the (?:(\w+) )?shape have\?", q)
        if m:
            col = m.group(1)
            hits = [b for b in c.bodies if b["color"] == col]
            if len(hits) == 1:
                return Answer(hits[0]["shape"])
            return Answer(None, confidence="uncertain", note=f"{len(hits)} {col} bodies")

        if re.search(r"which of these shapes is moving fastest at the very start", q):
            sp = {self.nm(i): math.hypot(b["vx"], b["vy"]) for i, b in enumerate(c.bodies)}
            cand = [o for o in opts if o.lower() in {k.lower() for k in sp}]
            pool = cand or list(sp)
            best = max(pool, key=lambda k: sp.get(k, sp.get(k.title(), 0)))
            fast = max(sp.values())
            slow = min(sp.values())
            if abs(fast - slow) < 1e-9:
                return Answer(None, confidence="uncertain", note="all speeds equal")
            return Answer(best)

        # ---------------- collision counts ----------------
        if re.search(r"how many times do two shapes collide with each other", q):
            return Answer(self.count(len(obs.pair)))

        m = re.search(r"how many collisions between two shapes have happened when the clock shows ([\d.]+) s", q)
        if m:
            T = float(m.group(1))
            return Answer(self.count(sum(1 for t, _a, _b in obs.pair if t <= T + 1e-9)))

        m = re.search(r"how many times does the (.+?) collide with another shape", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(obs.pair_count(i)))

        m = re.search(r"how many times does the (.+?) bounce off a wall", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(obs.wall_count(i)))

        m = re.search(r"how many times does the (.+?) bounce off a post", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(obs.post_count(i)))

        m = re.search(r"how many different shapes does the (.+?) collide with", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(len(obs.partners(i))))

        m = re.search(r"how much time passes between the first and the last collision", q)
        if m:
            if len(obs.pair) < 2:
                return Answer(None, confidence="unsupported", note="<2 collisions")
            # the key subtracts the two *displayed* clock readings (78/78)
            dt = round(self._clock_val(obs.pair[-1][0]) - self._clock_val(obs.pair[0][0]), 1)
            return Answer(f"{dt:.1f} s")

        # ---------------- collision identity ----------------
        if re.search(r"which two shapes are the first to collide", q):
            fc = self.first_collision(obs)
            if not fc:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.nm(fc[1])} and {self.nm(fc[2])}")

        m = re.search(r"which two shapes make the (second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh) collision", q)
        if m:
            k = ["second", "third", "fourth", "fifth", "sixth", "seventh",
                 "eighth", "ninth", "tenth", "eleventh"].index(m.group(1)) + 1
            if k >= len(obs.pair):
                return Answer(None, confidence="unsupported", note=f"only {len(obs.pair)} collisions")
            t, a, b = obs.pair[k]
            return Answer(f"{self.nm(a)} and {self.nm(b)}")

        m = re.search(r"which of these pairs of shapes is the first to collide", q)
        if m:
            times = {}
            for t, a, b in obs.pair:
                times.setdefault(frozenset((a, b)), t)
            best, best_t = None, None
            for o in opts:
                ids = {c.idx_of(x) for x in _split_pair(o)}
                ids.discard(None)
                if len(ids) != 2:
                    continue
                t = times.get(frozenset(ids))
                if t is not None and (best_t is None or t < best_t):
                    best, best_t = o, t
            if best is not None:
                return Answer(best)
            return Answer(None, confidence="unsupported")

        m = re.search(r"which shape is the first one that the (.+?) collides with", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            fp = obs.first_pair_with(i)
            if not fp:
                return Answer(None, confidence="unsupported", note="no collision")
            return Answer(self.nm(fp[1]))

        m = re.search(r"which of these shapes does the (.+?) collide with at some point", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            partners = obs.partners(i)
            for o in opts:
                if c.idx_of(o) in partners:
                    return Answer(o)
            return Answer(None, confidence="unsupported")

        # ---------------- walls / posts / holes identity ----------------
        if re.search(r"which of these shapes is the first to bounce off a wall", q):
            best, best_t = None, None
            for o in opts:
                i = c.idx_of(o)
                if i is None:
                    continue
                t = obs.wall_first(i)
                if t is not None and (best_t is None or t < best_t):
                    best, best_t = o, t
            return Answer(best) if best else Answer(None, confidence="unsupported")

        if re.search(r"which of these shapes is the first to bounce off one of the posts", q):
            best, best_t = None, None
            for o in opts:
                i = c.idx_of(o)
                if i is None:
                    continue
                t = obs.post_first(i)
                if t is not None and (best_t is None or t < best_t):
                    best, best_t = o, t
            return Answer(best) if best else Answer(None, confidence="unsupported")

        if re.search(r"which shape bounces off the posts the most times", q):
            counts = [obs.post_count(i) for i in range(obs.n)]
            if max(counts) == 0:
                return Answer(None, confidence="unsupported", note="no post contacts")
            i = max(range(obs.n), key=lambda k: counts[k])
            if counts.count(max(counts)) > 1:
                return Answer(None, confidence="uncertain", note="tie")
            return Answer(self.nm(i))

        m = re.search(r"which of these shapes is the first to fall into a hole", q)
        if m:
            best, best_t = None, None
            for o in opts:
                i = c.idx_of(o)
                if i is None:
                    continue
                t = obs.fall_time(i)
                if t is not None and (best_t is None or t < best_t):
                    best, best_t = o, t
            return Answer(best) if best else Answer(None, confidence="unsupported")

        m = re.search(r"of the shapes that fall into a hole, which one falls in last", q)
        if m:
            if not obs.fall:
                return Answer(None, confidence="unsupported", note="no falls")
            return Answer(self.nm(max(obs.fall)[1]))

        m = re.search(r"what does the clock show when the (.+?) falls into a hole", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            t = obs.fall_time(i)
            if t is None:
                return Answer(None, confidence="unsupported", note="body never falls")
            return Answer(self.clock(t))

        m = re.search(r"what does the clock show when the (.+?) and the (.+?) collide", q)
        if m:
            i, j = c.idx_of(m.group(1)), c.idx_of(m.group(2))
            if i is None or j is None:
                return Answer(None, confidence="unsupported")
            for t, a, b in obs.pair:
                if {a, b} == {i, j}:
                    return Answer(self.clock(t))
            return Answer(None, confidence="unsupported", note="pair never meets")

        # ---------------- box occupancy ----------------
        m = re.search(r"how many shapes are still in the box when the clock shows ([\d.]+) s", q)
        if m:
            T = float(m.group(1))
            return Answer(self.count(len(obs.alive_at(T))))

        if re.search(r"how many shapes are still in the box at the end", q):
            return Answer(self.count(obs.n - len(obs.fall)))

        # ---------------- patterns ----------------
        m = re.search(r"what fill pattern does the (.+?) have when the clock shows ([\d.]+) s", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(obs.pattern_at(i, float(m.group(2))))

        m = re.search(r"what fill pattern does the (.+?) have at the very start", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(c.bodies[i]["pattern"])

        m = re.search(r"what fill pattern does the (.+?) have at the end", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(obs.final[i])

        m = re.search(r"how many different fill patterns does the (.+?) wear", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(len(obs.patterns_seen(i))))

        m = re.search(r"how many different shapes wear the (.+?) pattern at some point", q)
        if m:
            pat = m.group(1)
            return Answer(self.count(sum(1 for i in range(obs.n) if pat in obs.patterns_seen(i))))

        m = re.search(r"how many shapes wear the (.+?) pattern when the clock shows ([\d.]+) s", q)
        if m:
            pat, T = m.group(1), float(m.group(2))
            return Answer(self.count(sum(1 for i in obs.alive_at(T) if obs.pattern_at(i, T) == pat)))

        m = re.search(r"how many shapes wear the (.+?) pattern at the end", q)
        if m:
            pat = m.group(1)
            return Answer(self.count(sum(1 for i in range(obs.n)
                                        if i in obs.alive_at(self.c.duration)
                                        and obs.final[i] == pat)))

        m = re.search(r"which shape wears the (.+?) pattern at the start", q)
        if m:
            pat = m.group(1)
            hits = [i for i in range(obs.n) if c.bodies[i]["pattern"] == pat]
            if len(hits) != 1:
                return Answer(None, confidence="uncertain", note=f"{len(hits)} wear {pat}")
            return Answer(self.nm(hits[0]))

        m = re.search(r"at the end of the video, which shape wears the pattern that the (.+?) started with", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            want = c.bodies[i]["pattern"]
            alive = obs.alive_at(c.duration)
            hits = [k for k in alive if obs.final[k] == want]
            if len(hits) > 1:
                cand = [k for k in hits if _name(c.bodies[k]) in opt_l]
                if len(cand) == 1:
                    return Answer(self.nm(cand[0]))
                return Answer(None, confidence="uncertain", note=f"{len(hits)} wear {want}")
            if not hits:
                return Answer(None, confidence="unsupported")
            return Answer(self.nm(hits[0]))

        # same-pattern-they-started-with counts
        m = re.search(r"how many of the shapes still in the box wear the same fill pattern they started with", q)
        if m:
            alive = obs.alive_at(c.duration)
            return Answer(self.count(sum(1 for i in alive if obs.final[i] == c.bodies[i]["pattern"])))

        m = re.search(r"how many shapes wear the same fill pattern they started with", q)
        if m:
            return Answer(self.count(sum(1 for i in range(obs.n)
                                        if obs.final[i] == c.bodies[i]["pattern"])))

        # just-before-the-last-collision snapshot
        m = re.search(r"if the video had stopped just before the last collision between two shapes, what fill pattern would the (.+?) have", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None or not obs.pair:
                return Answer(None, confidence="unsupported")
            t = obs.pair[-1][0]
            return Answer(obs.pattern_at(i, max(0.0, t - 1e-6)))

        # ---------------- most collisions ----------------
        if re.search(r"which of these shapes collides with other shapes the most times", q):
            counts = {self.nm(i): obs.pair_count(i) for i in range(obs.n)}
            cand = [o for o in opts if o.lower() in {k.lower() for k in counts}]
            pool = cand or list(counts)
            best = max(pool, key=lambda k: counts[k])
            if list(counts.values()).count(counts[best]) > 1 and len(pool) == len(counts):
                return Answer(None, confidence="uncertain", note="tie")
            return Answer(best)

        if re.search(r"which shape collides with other shapes the most times over the whole video", q):
            counts = [obs.pair_count(i) for i in range(obs.n)]
            if max(counts) == 0:
                return Answer(None, confidence="unsupported", note="no collisions")
            if counts.count(max(counts)) > 1:
                return Answer(None, confidence="uncertain", note="tie")
            return Answer(self.nm(counts.index(max(counts))))

        # ---------------- counterfactuals ----------------
        m = re.search(r"suppose the (.+?) had not been in the box at all", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            sub = c.without(i)
            if "how many times would two shapes collide" in q:
                return Answer(self.count(len(sub.pair)))
            if "how many shapes would fall into a hole" in q:
                return Answer(self.count(len(sub.fall)))
            if "which two shapes would be the first to collide" in q:
                if not sub.pair:
                    return Answer(None, confidence="unsupported")
                t, a, b = sub.pair[0]
                return Answer(f"{nm_of(sub, a)} and {nm_of(sub, b)}")
            return Answer(None, confidence="unsupported", note="removal variant")

        m = re.search(r"suppose the holes had been covered over", q)
        if m:
            sub = c.variant("noholes", holes=())
            if "how many times would two shapes collide" in q:
                return Answer(self.count(len(sub.pair)))
            return Answer(None, confidence="unsupported")

        m = re.search(r"suppose the posts had been taken out of the box", q)
        if m:
            sub = c.variant("noposts", posts=())
            if "how many times would two shapes collide" in q:
                return Answer(self.count(len(sub.pair)))
            return Answer(None, confidence="unsupported")

        # ---- "they had not traded patterns" counterfactuals ----
        m = re.search(r"when the (.+?) and the (.+?) collided (?:the (first|second|third|fourth) time )?they had not (?:traded patterns|passed on the pattern)", q)
        if m:
            i, j = c.idx_of(m.group(1)), c.idx_of(m.group(2))
            if i is None or j is None:
                return Answer(None, confidence="unsupported")
            pair = (min(i, j), max(i, j))
            if m.group(3):
                k = ["first", "second", "third", "fourth"].index(m.group(3)) + 1
                sub = c.variant(("skip", pair, k), swap_skip={pair + (k,)})
            else:
                sub = c.variant(("skippair", pair), skip_pairs={pair})
            return self._cf_pattern_answer(q, sub)

        # ---- "started with a different pattern" counterfactual ----
        m = re.search(r"but the (.+?) had started with the (.+?) pattern instead of the (.+?)(?: pattern)? \(which would start solid\)", q)
        if m:
            i = c.idx_of(m.group(1))
            new_pat = m.group(2)
            donor = c.idx_of(m.group(3))
            if i is None:
                return Answer(None, confidence="unsupported")
            override = {i: new_pat}
            if donor is not None and donor != i:
                override[donor] = "solid"
            sub = c.variant(("pat", i, new_pat, donor), pattern_override=override)
            return self._cf_pattern_answer(q, sub)

        # ---------------- infect-only ----------------
        m = re.search(r"how many shapes does the (.+?) pass the (.+?) pattern on to", q)
        if m:
            i = c.idx_of(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.count(len({b for _t, a, b in obs.infections if a == i})))

        m = re.search(r"which shape passes the (.+?) pattern on to the (.+?)\?", q)
        if m:
            j = c.idx_of(m.group(2))
            if j is None:
                return Answer(None, confidence="unsupported")
            for t, a, b in obs.infections:
                if b == j:
                    return Answer(self.nm(a))
            return Answer(None, confidence="unsupported", note="never infected")

        m = re.search(r"which of these shapes is the first to get the (.+?) pattern", q)
        if m:
            pat = m.group(1)
            order = sorted({b: t for t, a, b in obs.infections if pat == obs.tag}.items(),
                           key=lambda kv: kv[1])
            for o in opts:
                i = c.idx_of(o)
                if i is not None and i in dict(order):
                    times = {k: v for k, v in order}
                    cand = [(times[c.idx_of(x)], x) for x in opts
                            if c.idx_of(x) in times]
                    if cand:
                        return Answer(min(cand)[1])
            return Answer(None, confidence="unsupported")

        m = re.search(r"not counting the shape that starts with it, which shape is the (second|third|fourth|fifth) to get the (.+?) pattern", q)
        if m:
            k = ["second", "third", "fourth", "fifth"].index(m.group(1)) + 2
            seen, order = set(), []
            for t, a, b in obs.infections:
                if b not in seen:
                    seen.add(b)
                    order.append(b)
            if len(order) < k:
                return Answer(None, confidence="unsupported", note=f"{len(order)} infected")
            return Answer(self.nm(order[k - 1]))

        return Answer(None, confidence="unsupported", note="no handler for this template")


def nm_of(obs, i):
    b = obs.names[i]
    return f"the {b['color']} {b['shape']}"


def _split_pair(text):
    return re.split(r"\s+and\s+", text.strip())

"""Deterministic replay for `bouncing_ball_restitution`.

Params:
  g, settle_px, wall_l, wall_r, duration,
  shelves: [{floor_y, top_y}], objects: [{id, color, pattern, shape, lane,
  r, release, x, vx, e, h0}]

Model (verified on clip 000000):
  fall time to first impact  t0 = release + sqrt(2 h0 / g)
  impact k (0-based)         t_k = t0 + 2 fall * Σ_{i=1..k} e^i
  apex after impact k        a_k = h0 * e^{2(k+1)}
  a bounce is counted while  a_k >= settle_px
  stop time                  = t_k of the first impact with a_k < settle_px
  wall hits                  = reflections between wall_l+r and wall_r-r at |vx|
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0


class BouncingBall:
    def __init__(self, params: dict):
        self.p = params
        self.g = float(params.get("g", 600))
        self.settle = float(params.get("settle_px", 30))
        self.wall_l = float(params.get("wall_l", 0)) 
        self.wall_r = float(params.get("wall_r", 0))
        self.duration = float(params.get("duration", 0))
        self.objects = params.get("objects", [])
        self.e_defined = all(o.get("e") is not None for o in self.objects)
        for o in self.objects:
            if o.get("e") is None:
                o["e"] = 0.0
            if o.get("vx") is None:
                o["vx"] = 0.0

    # ---------- helpers ----------
    def find(self, desc: str) -> int | None:
        d = " " + desc.lower() + " "
        for o in self.objects:
            phrase = f"{o['pattern']} {o['color']} {o['shape']}"
            if phrase in d:
                return o["id"]
        return None

    def name(self, i: int) -> str:
        o = self.objects[i]
        return f"the {o['pattern']} {o['color']} {o['shape']}"

    def fall_time(self, o: dict) -> float:
        return math.sqrt(max(0.0, 2.0 * o["h0"] / self.g))

    def impact_times(self, o: dict, limit: int = 200) -> list[float]:
        t0 = o.get("release", 0.0) + self.fall_time(o)
        out = [t0]
        dt = 2.0 * o["e"] * self.fall_time(o)
        while len(out) < limit and out[-1] < self.duration + 10:
            out.append(out[-1] + dt)
            dt *= o["e"]
        return out

    def bounce_count(self, o: dict, e: float | None = None) -> int:
        e = o["e"] if e is None else e
        n = 0
        k = 0
        fall = self.fall_time(o)
        t = o.get("release", 0.0) + fall
        dt = 2.0 * e * fall
        while t <= self.duration + 1e-9:
            n += 1
            if o["h0"] * (e ** (2 * (k + 1))) < self.settle:
                break
            t += dt
            dt *= e
            k += 1
        return n

    def stop_time(self, o: dict) -> float:
        fall = self.fall_time(o)
        t = o.get("release", 0.0) + fall
        dt = 2.0 * o["e"] * fall
        k = 0
        while t <= self.duration + 1e-9 and o["h0"] * (o["e"] ** (2 * (k + 1))) >= self.settle:
            t += dt
            dt *= o["e"]
            k += 1
        return t

    def wall_hits(self, o: dict, vx: float | None = None) -> int:
        vx = o["vx"] if vx is None else vx
        if abs(vx) < 1e-9:
            return 0
        lo = self.wall_l + o["r"]
        hi = self.wall_r - o["r"]
        x = o["x"]
        t = o.get("release", 0.0)
        # time to first wall
        bound = hi if vx > 0 else lo
        dt = (bound - x) / vx
        if dt < 0:
            return 0
        t += dt
        n = 1 if t <= self.duration + 1e-9 else 0
        period = 2.0 * (hi - lo) / abs(vx)
        t += period
        while t <= self.duration + 1e-9:
            n += 1
            t += period
        return n

    def first_wall_time(self, o: dict) -> float | None:
        if abs(o["vx"]) < 1e-9:
            return None
        lo = self.wall_l + o["r"]
        hi = self.wall_r - o["r"]
        bound = hi if o["vx"] > 0 else lo
        dt = (bound - o["x"]) / o["vx"]
        if dt < 0:
            return None
        return o.get("release", 0.0) + dt

    def landed_by(self, o: dict, frame: float) -> int:
        limit = frame / FPS
        t0 = o.get("release", 0.0) + self.fall_time(o)
        n, t, dt, k = 0, t0, 2.0 * o["e"] * self.fall_time(o), 0
        while t <= limit + 1e-9:
            n += 1
            if o["h0"] * (o["e"] ** (2 * (k + 1))) < self.settle:
                break
            t += dt
            dt *= o["e"]
            k += 1
        return n

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        if not self.e_defined or str(self.p.get("scenario")) == "pads":
            return Answer(None, confidence="unsupported",
                          note="pad scenario: restitution is not stored in params")
        q = question.lower().strip()

        if re.search(r"how many bouncing objects", q):
            return Answer(str(sum(1 for o in self.objects if o["e"] > 0)))
        if re.search(r"which object is the smallest", q):
            return Answer(self.name(min(self.objects, key=lambda o: o["r"])["id"]))
        if re.search(r"which object starts highest", q):
            return Answer(self.name(max(self.objects, key=lambda o: o["h0"])["id"]))

        m = re.search(r"^how many times does the (.+?) land on its shelf", q)
        if m:
            i = self.find(m.group(1))
            return Answer(str(self.bounce_count(self.objects[i])) if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")

        m = re.search(r"which lands on its shelf more times: the (.+?) or the (.+?)\?", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            ca, cb = self.bounce_count(self.objects[a]), self.bounce_count(self.objects[b])
            if ca == cb:
                return Answer("they land equally often")
            return Answer(self.name(a if ca > cb else b))

        m = re.search(r"which keeps more of its height from one bounce to the next: the (.+?) or the (.+?)\?", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            ea, eb = self.objects[a]["e"], self.objects[b]["e"]
            if abs(ea - eb) < 1e-9:
                return Answer("they keep the same share")
            return Answer(self.name(a if ea > eb else b))

        if re.search(r"which object is the first to stop bouncing", q):
            o = min(self.objects, key=self.stop_time)
            return Answer(self.name(o["id"]))
        if re.search(r"which object is the last to come to rest", q):
            o = max(self.objects, key=self.stop_time)
            return Answer(self.name(o["id"]))

        m = re.search(r"between ([\d.]+) s and ([\d.]+) s on the clock, how many times does the (.+?) land", q)
        if m:
            lo, hi, i = float(m.group(1)), float(m.group(2)), self.find(m.group(3))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.landed_by(self.objects[i], hi * FPS) - self.landed_by(self.objects[i], lo * FPS)))

        m = re.search(r"how much time passes between the first and the second time the (.+?) lands", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            o = self.objects[i]
            return Answer(f"{2.0 * o['e'] * self.fall_time(o):.1f} s")

        if re.search(r"which object lands on its shelf the most times", q):
            o = max(self.objects, key=lambda x: self.bounce_count(x))
            return Answer(self.name(o["id"]))

        m = re.search(r"if the (.+?) were exactly as bouncy as the (.+?) .* how many times would it land", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.bounce_count(self.objects[a], self.objects[b]["e"])))

        m = re.search(r"if the (.+?) slid sideways half as fast .* how many times would it bounce off a side wall", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(self.wall_hits(self.objects[i], self.objects[i]["vx"] / 2.0)))

        if re.search(r"which object is the first to hit the floor", q):
            o = min(self.objects, key=lambda x: x.get("release", 0) + self.fall_time(x))
            return Answer(self.name(o["id"]))
        if re.search(r"which object is the first to bounce off a side wall", q):
            times = [(self.first_wall_time(o), o) for o in self.objects if self.first_wall_time(o) is not None]
            if not times:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(min(times, key=lambda t: t[0])[1]["id"]))

        if re.search(r"after its first bounce, which object rises highest", q):
            o = max(self.objects, key=lambda x: x["h0"] * x["e"] ** 2)
            return Answer(self.name(o["id"]))
        if re.search(r"which object is the most bouncy", q):
            return Answer(self.name(max(self.objects, key=lambda o: o["e"])["id"]))
        if re.search(r"which object is the least bouncy", q):
            return Answer(self.name(min(self.objects, key=lambda o: o["e"])["id"]))

        if re.search(r"which object hits the floor the most times", q):
            o = max(self.objects, key=lambda x: self.bounce_count(x))
            return Answer(self.name(o["id"]))
        if re.search(r"counting every object, how many times in total does an object hit the floor", q):
            return Answer(str(sum(self.bounce_count(o) for o in self.objects)))
        if re.search(r"counting every object, how many times in total does an object bounce off a side wall", q):
            return Answer(str(sum(self.wall_hits(o) for o in self.objects)))

        if re.search(r"how many objects are lying still on the floor at the end", q):
            return Answer(str(sum(1 for o in self.objects if self.stop_time(o) <= self.duration)))

        return Answer(None, confidence="unsupported", note="no handler for this template")

"""Deterministic replay for `inertia_race`.

Params:
  ramp: {angle_deg, length_m, g}
  t0: timer offset (the clips animate from clock 0; finish_f is the race clock)
  racers: [{colour, shape, k_shape, r_px, radius, start, release_f, finish_f, violates}]
    start is a fraction of the ramp length; finish_f and release_f are frames at 24 fps.

Rolling model:
  a = g * sin(angle) / (1 + k)
  k: slider 0, solid_sphere 2/5, hollow_sphere 2/3, solid_disc 1/2, hoop 1
  position(t) = start*L + 0.5*a*(t - release)^2, capped at L
  finish clock = finish_f / 24 (the stored animation time)
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0
K = {
    "slider": 0.0,
    "solid_sphere": 0.4,
    "hollow_sphere": 2.0 / 3.0,
    "solid_disc": 0.5,
    "hoop": 1.0,
    "thin_ring": 1.0,
}
KIND = {
    "slider": "sliding block",
    "solid_sphere": "solid ball",
    "hollow_sphere": "hollow shell",
    "solid_disc": "solid disc",
    "hoop": "thin ring",
}
SHAPE_WORD = {
    "sliding block": "slider",
    "solid ball": "solid_sphere",
    "hollow shell": "hollow_sphere",
    "solid disc": "solid_disc",
    "thin ring": "hoop",
}


class InertiaRace:
    def __init__(self, params: dict):
        self.p = params
        self.ramp = params.get("ramp", {})
        self.L = float(self.ramp.get("length_m", 1.0))
        self.angle = math.radians(float(self.ramp.get("angle_deg", 0.0)))
        self.g = float(self.ramp.get("g", 9.81))
        self.racers = params.get("racers", [])
        self.n_frames = int(params.get("n_frames", 0))
        self.run = float((params.get("view") or {}).get("run", 0.0))

    def colour(self, r: dict) -> str:
        return str(r["colour"]).lower()

    def name(self, r: dict) -> str:
        return f"the {self.colour(r)} body"

    def find(self, text: str) -> dict | None:
        low = text.lower()
        hits = [r for r in self.racers if self.colour(r) in low]
        return hits[0] if len(hits) == 1 else None

    def find_pair(self, q: str) -> tuple[dict, dict] | None:
        found = []
        for r in self.racers:
            if re.search(rf"\b{re.escape(self.colour(r))}\b", q):
                if r not in found:
                    found.append(r)
        return (found[0], found[1]) if len(found) >= 2 else None

    # ---------- motion ----------
    def a(self, r: dict) -> float:
        return self.g * math.sin(self.angle) / (1.0 + K.get(r.get("k_shape") or r.get("shape"), 0.0))

    def release(self, r: dict) -> float:
        return float(r.get("release_f", 0)) / FPS

    def finish(self, r: dict) -> float:
        v = r.get("finish_f")
        return float("inf") if v is None else float(v) / FPS

    def finish_display(self, r: dict) -> float:
        # the on-screen timer floors to 0.1 s
        f = self.finish(r)
        return float("inf") if f == float("inf") else math.floor(f * 10 + 1e-9) / 10

    def start_m(self, r: dict) -> float:
        return float(r.get("start", 0.0)) * self.L

    def pos(self, r: dict, t: float) -> float:
        rel = self.release(r)
        if t <= rel:
            return self.start_m(r)
        x = self.start_m(r) + 0.5 * self.a(r) * (t - rel) ** 2
        return min(x, self.L)

    def frac(self, r: dict, t: float) -> float:
        return self.pos(r, t) / self.L if self.L else 0.0

    def vel(self, r: dict, t: float) -> float:
        if t <= self.release(r) or self.pos(r, t) >= self.L - 1e-9:
            return 0.0
        return self.a(r) * (t - self.release(r))

    def race_time(self, r: dict, start_m: float | None = None) -> float:
        d = self.L - (self.start_m(r) if start_m is None else start_m)
        return math.sqrt(max(0.0, 2.0 * d / self.a(r)))

    def ordered(self) -> list[dict]:
        return sorted(self.racers, key=lambda r: (self.finish(r), -self.pos(r, self.n_frames / FPS)))

    def place(self, r: dict) -> int:
        return self.ordered().index(r) + 1

    # ---------- answer ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        # Families below need a renderer convention we have not pinned; keep them
        # out of the verified lane rather than guess.
        if ("gap along the ramp" in q or "overtake the other" in q
                or "complete turns" in q or "turned through more revolutions" in q
                or "times as hard" in q or "gravity were" in q
                or "replaced by a" in q or "crossed the finish line by" in q):
            return Answer(None, confidence="unsupported", note="timing/geometry family not calibrated")

        if "how many bodies race down the ramp" in q:
            return Answer(str(len(self.racers)), confidence="exact")

        m = re.search(r"what kind of body is the (\w+) body", q)
        if m and (r := self.find(m.group(1))):
            return Answer(KIND.get(r.get("shape"), str(r.get("shape", ""))), confidence="exact")

        if "largest radius" in q:
            r = max(self.racers, key=lambda x: x.get("r_px", 0))
            return Answer(self.name(r), confidence="exact")

        if "starts furthest down the ramp" in q or "closest to the finish line" in q:
            r = max(self.racers, key=lambda x: x.get("start", 0))
            return Answer(self.name(r), confidence="exact")

        # clock values
        m = re.search(r"what does the clock read when the (\w+) body (?:crosses|reaches) the finish", q)
        if m and (r := self.find(m.group(1))):
            return Answer(f"{self.finish_display(r):.1f}", confidence="exact")

        m = re.search(r"what does the clock read when the (\w+) body is let go", q)
        if m and (r := self.find(m.group(1))):
            return Answer(f"{self.release(r):.1f}", confidence="exact")

        # count crossed / still to finish
        m = re.search(r"how many bodies have crossed the finish line by ([\d.]+) s", q)
        if m:
            t = float(m.group(1))
            return Answer(str(sum(1 for r in self.racers if self.finish_display(r) <= t + 1e-9)), confidence="exact")

        if "still not reached the finish line when the clip ended" in q:
            end = self.n_frames / FPS
            r = [x for x in self.racers if self.finish(x) > end]
            return Answer(self.name(r[0]) if len(r) == 1 else "none", confidence="exact")

        # winner / last
        if "how many seconds after the winner does the last body" in q:
            times = [self.finish(r) for r in self.racers if self.finish(r) != float("inf")]
            if len(times) >= 2:
                return Answer(f"{round(round(max(times) - min(times), 2), 1):.1f}", confidence="exact")

        if "held at the very top" in q and "which would reach" in q:
            # every distance = L; rank by acceleration (smallest k wins)
            r = max(self.racers, key=lambda x: self.a(x))
            return Answer(self.name(r), confidence="exact")

        # two-body comparisons
        if any(s in q for s in ["further down the ramp", "which crosses the finish line first",
                                "do the ", "reach the finish line at the same moment",
                                "which has turned through more revolutions"]):
            pair = self.find_pair(q)
            if pair:
                a, b = pair
                if "further down" in q:
                    t = float(re.search(r"at ([\d.]+) s", q).group(1))
                    da, db = self.pos(a, t), self.pos(b, t)
                    if abs(da - db) < 1e-6:
                        return Answer("they are level", confidence="exact")
                    return Answer(self.name(a) if da > db else self.name(b), confidence="exact")
                if "same moment" in q:
                    return Answer("yes" if abs(self.finish(a) - self.finish(b)) < 1e-6 else "no", confidence="exact")
                if "turned through more revolutions" in q:
                    ra = self.frac(a, self.n_frames / FPS) * self.run / max(a.get("r_px", 1), 1)
                    rb = self.frac(b, self.n_frames / FPS) * self.run / max(b.get("r_px", 1), 1)
                    return Answer(self.name(a) if ra > rb else self.name(b), confidence="exact")
                fa, fb = self.finish(a), self.finish(b)
                if abs(fa - fb) < 1e-6:
                    return Answer("they are level", confidence="exact")
                return Answer(self.name(a) if fa < fb else self.name(b), confidence="exact")

        # distance / gap at a clock time
        m = re.search(r"how far down the ramp is the (\w+) body at ([\d.]+) s", q)
        if m and (r := self.find(m.group(1))):
            return Answer(f"{self.frac(r, float(m.group(2))):.3f} units", confidence="exact")

        m = re.search(r"how far apart along the ramp are the (\w+) body and the (\w+) body", q)
        if m and (a := self.find(m.group(1))) and (b := self.find(m.group(2))):
            mt = re.search(r"at ([\d.]+) s", q)
            if mt:
                t = float(mt.group(1))
                return Answer(f"{abs(self.frac(a, t) - self.frac(b, t)):.3f} units", confidence="exact")

        m = re.search(r"at ([\d.]+) s on the clock, which body is furthest down the ramp", q)
        if m:
            t = float(m.group(1))
            r = max(self.racers, key=lambda x: self.pos(x, t))
            return Answer(self.name(r), confidence="exact")

        # order
        if "give the order in which the bodies cross" in q:
            order = [r for r in self.ordered() if self.finish(r) != float("inf")]
            return Answer(" then ".join(self.colour(r) for r in order), confidence="exact")

        # overtaking
        m = re.search(r"does either of the (\w+) body and the (\w+) body overtake the other", q)
        if m and (a := self.find(m.group(1))) and (b := self.find(m.group(2))):
            mt = re.search(r"before the clock reaches ([\d.]+) s", q)
            end = float(mt.group(1)) if mt else self.n_frames / FPS
            vals = [self.pos(a, t) - self.pos(b, t) for t in [i / 10 for i in range(int(end * 10) + 1)]
                    if self.pos(a, t) < self.L - 1e-6 and self.pos(b, t) < self.L - 1e-6]
            signs = {1 if v > 1e-6 else (-1 if v < -1e-6 else 0) for v in vals}
            return Answer("yes" if 1 in signs and -1 in signs else "no", confidence="exact")

        # gap opening/closing
        m = re.search(r"gap along the ramp between the (\w+) body and the (\w+) body.*?(opening up|closing)", q)
        if m and (a := self.find(m.group(1))) and (b := self.find(m.group(2))):
            t = float(re.search(r"at ([\d.]+) s", q).group(1))
            va = self.vel(a, t)
            vb = self.vel(b, t)
            ahead = self.pos(a, t) >= self.pos(b, t)
            opening = (va - vb) > 0 if ahead else (vb - va) > 0
            return Answer("opening up" if opening else "closing", confidence="exact")

        # revolutions by time
        m = re.search(r"how many complete turns has the (\w+) body made by ([\d.]+) s", q)
        if m and (r := self.find(m.group(1))):
            dist_px = self.frac(r, float(m.group(2))) * self.run
            turns = dist_px / (2 * math.pi * max(r.get("r_px", 1), 1))
            return Answer(str(int(round(turns))), confidence="exact")

        # place
        m = re.search(r"in what place does the (\w+) body finish", q)
        if m and (r := self.find(m.group(1))):
            return Answer(str(self.place(r)), confidence="exact")

        # counterfactuals
        m = re.search(r"ramp were tilted so that it pulled the bodies along ([\d.]+) times as hard.*?(\w+) body", q)
        if m and (r := self.find(m.group(2))):
            f = float(m.group(1))
            return Answer(f"{self.race_time(r) / math.sqrt(f):.1f}", confidence="exact")

        m = re.search(r"gravity were ([\d.]+) times stronger.*?(\w+) body", q)
        if m and (r := self.find(m.group(2))):
            f = float(m.group(1))
            return Answer(f"{self.race_time(r) / math.sqrt(f):.1f}", confidence="exact")

        m = re.search(r"suppose the (\w+) body had started at the very top of the ramp", q)
        if m and (r := self.find(m.group(1))):
            return Answer(f"{self.race_time(r, start_m=0.0):.1f}", confidence="exact")

        m = re.search(r"suppose the (\w+) body were replaced by a (.+?) of the same size", q)
        if m and (r := self.find(m.group(1))):
            kind = SHAPE_WORD.get(m.group(2).strip())
            if kind:
                d = self.L - self.start_m(r)
                a = self.g * math.sin(self.angle) / (1.0 + K[kind])
                return Answer(f"{math.sqrt(2 * d / a) + self.release(r):.1f}", confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

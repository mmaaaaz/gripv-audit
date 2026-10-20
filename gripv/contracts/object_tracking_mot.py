"""Deterministic replay for object_tracking_mot.

Each object follows an ellipse:
  x(t) = cx + ax*sin(wx*t + px)
  y(t) = cy + ay*sin(wy*t + py)
with t in seconds from the clip start. Quadrants split the arena at its centre.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0


class ObjectTracking:
    def __init__(self, params: dict):
        self.p = params
        self.arena = params.get("arena", {})
        self.all_objects = params.get("objects", [])
        self.objects = [o for o in self.all_objects if all(k in o for k in ("cx", "cy", "ax", "ay", "wx", "wy", "px", "py"))]
        self.start_t = float(params.get("start_t") or 0.0)
        self.frames = int(params.get("frames") or 0)
        self.ax = (self.arena.get("x0", 0) + self.arena.get("x1", 0)) / 2.0
        self.ay = (self.arena.get("y0", 0) + self.arena.get("y1", 0)) / 2.0

    def pos(self, o: dict, t: float, speed: float = 1.0) -> tuple[float, float]:
        tt = max(0.0, (t - self.start_t) * speed)
        x = o["cx"] + o["ax"] * math.sin(o["wx"] * tt + o["px"])
        y = o["cy"] + o["ay"] * math.sin(o["wy"] * tt + o["py"])
        return x, y

    def quad(self, o: dict, t: float, speed: float = 1.0) -> str:
        x, y = self.pos(o, t, speed)
        return ("top" if y < self.ay else "bottom") + "-" + ("right" if x > self.ax else "left")

    def find(self, q: str) -> dict | None:
        hits = [o for o in self.objects if o["colour"].lower() in q and o["shape"].lower() in q]
        return hits[0] if len(hits) == 1 else None

    def end_t(self) -> float:
        return self.frames / FPS if self.frames else 0.0

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if "which pattern is on the" in q:
            o = self.find(q)
            if o is None:
                hits = [x for x in self.objects if x["colour"].lower() in q]
                o = hits[0] if len(hits) == 1 else None
            return Answer(o["pattern"], confidence="exact") if o else Answer(None, confidence="unsupported")

        if "how many moving objects are in the arena" in q:
            return Answer(str(len(self.all_objects)), confidence="exact")

        if "at the start some objects wear a ring" in q or "how many objects wear a ring at the start" in q:
            if not any(o.get("pattern") == "ringed" for o in self.all_objects):
                return Answer(None, confidence="unsupported", note="ring targets are stored outside object.pattern in this scenario")
            return Answer(str(sum(1 for o in self.all_objects if o.get("pattern") == "ringed")), confidence="exact")

        if not self.objects:
            return Answer(None, confidence="unsupported", note="no elliptical trajectories in params")

        # quadrant at a named time
        m = re.search(r"at ([\d.]+) s on the clock, in which quadrant.*? is the (.*?)(?:\?|$)", q)
        if m and (o := self.find(q)):
            return Answer(self.quad(o, float(m.group(1))), confidence="exact")

        m = re.search(r"in which quadrant is the (.+?) at the end of the clip", q)
        if m and (o := self.find(q)):
            return Answer(self.quad(o, self.end_t()), confidence="exact")

        # ringed / not ringed counts in a quadrant at a time or at the end
        m = re.search(r"at ([\d.]+) s on the clock, how many (of the objects that were ringed at the start|objects that were not ringed at the start) are in the (top-right|top-left|bottom-right|bottom-left) quadrant", q)
        end_mode = False
        if not m:
            m = re.search(r"at the end of the clip, how many (of the objects that were ringed at the start|objects that were not ringed at the start) are in the (top-right|top-left|bottom-right|bottom-left) quadrant", q)
            end_mode = True
        if m:
            t = self.end_t() if end_mode else float(m.group(1))
            ringed = "not ringed" not in m.group(1 if end_mode else 2)
            quad = m.group(2 if end_mode else 3)
            n = sum(1 for o in self.objects if (o.get("pattern") == "ringed") == ringed and self.quad(o, t) == quad)
            return Answer(str(n), confidence="exact")

        # farthest left at a time
        m = re.search(r"at ([\d.]+) s on the clock, which object is farthest to the left", q)
        if m:
            t = float(m.group(1))
            o = min(self.objects, key=lambda z: self.pos(z, t)[0])
            return Answer(f"the {o['colour']} {o['shape']}", confidence="exact")

        # quadrant changes over the clip
        m = re.search(r"how many times does the (.+?) move from one quadrant into another during the clip", q)
        if m and (o := self.find(m.group(1))):
            prev, n = None, 0
            for f in range(self.frames + 1):
                cur = self.quad(o, f / FPS)
                if prev is not None and cur != prev:
                    n += 1
                prev = cur
            return Answer(str(n), confidence="exact")

        # speed counterfactuals at the end
        m = re.search(r"suppose every object went round its path (1\.5|twice|half) times? as fast.*? in which quadrant would the (.+?) be at the end", q)
        if m and (o := self.find(m.group(2))):
            k = 1.5 if m.group(1) == "1.5" else (2.0 if m.group(1) == "twice" else 0.5)
            return Answer(self.quad(o, self.end_t(), k), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

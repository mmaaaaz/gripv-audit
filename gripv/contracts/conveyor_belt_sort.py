"""Deterministic replay for `conveyor_belt_sort`.

Params:
  objects: [{id, pattern, colour, shape, size, e, bin, slot, pending,
             keys: [[frame, x, y], ...]}]
  bins: {name: {cap, start, slots, rect}}
  gate: [x, y]; belt_y; mouth; n_frames; frames {belt, chute, in, slot}

Timeline (verified against clip 000000):
  e            = frame the object leaves the tunnel (first key at the mouth)
  gate_frame   = frame at the gate (x == gate[0])
  rest_frame   = last key frame (its slot in the bin)
  times are seconds = frame / 24
  pending=True means rest_frame > n_frames (not sorted by the end of the video)
"""
from __future__ import annotations

import re

from ..engine import Answer

FPS = 24.0


class ConveyorBelt:
    def __init__(self, params: dict):
        self.p = params
        self.objects = params.get("objects", [])
        self.n_frames = params.get("n_frames", 0)
        self.gate_x = (params.get("gate") or [0, 0])[0]
        self.belt_y = params.get("belt_y")
        self.bins = params.get("bins", {})

    # ---------- helpers ----------
    def find(self, desc: str) -> int | None:
        d = " " + desc.lower() + " "
        for o in self.objects:
            phrase = f"{o['pattern']} {o['colour']} {o['shape']}"
            if re.search(rf"\b{re.escape(phrase)}\b", d):
                return o["id"]
        return None

    def name(self, i: int) -> str:
        o = self.objects[i]
        return f"the {o['pattern']} {o['colour']} {o['shape']}"

    def gate_frame(self, o: dict) -> int:
        keys = o["keys"]
        for f, x, y in keys:
            if abs(x - self.gate_x) < 1.0:
                return f
        # fall back to the key before the bin approach
        return keys[-2][0] if len(keys) > 1 else keys[-1][0]

    def rest_frame(self, o: dict) -> int:
        return o["keys"][-1][0]

    def settled(self, o: dict) -> bool:
        return self.rest_frame(o) <= self.n_frames

    def state_at(self, o: dict, frame: float) -> str:
        if frame < o["e"]:
            return "still inside the tunnel"
        if frame < self.gate_frame(o):
            return "riding on the belt"
        if frame < self.rest_frame(o):
            return "sliding toward a bin"
        return "already resting in a bin"

    def sorted_rest(self) -> list[dict]:
        return sorted([o for o in self.objects if self.settled(o)], key=self.rest_frame)

    def in_bin_at(self, bin_name: str, frame: float) -> int:
        return sum(1 for o in self.objects if o["bin"] == bin_name and self.rest_frame(o) <= frame)

    def count_attr(self, attr: str, value: str) -> int:
        return sum(1 for o in self.objects if str(o.get(attr, "")).lower() == value.lower())

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if re.search(r"which object comes out of the tunnel first", q):
            o = min(self.objects, key=lambda x: x["e"])
            return Answer(self.name(o["id"]))
        if re.search(r"which object comes out of the tunnel last", q):
            o = max(self.objects, key=lambda x: x["e"])
            return Answer(self.name(o["id"]))

        m = re.search(r"which object comes out of the tunnel right after the (.+?)\?", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            e = self.objects[i]["e"]
            later = [o for o in self.objects if o["e"] > e]
            if not later:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(min(later, key=lambda x: x["e"])["id"]))

        m = re.search(r"which bin does the object that comes out of the tunnel right after the (.+?) end up in", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            e = self.objects[i]["e"]
            later = [o for o in self.objects if o["e"] > e]
            if not later:
                return Answer(None, confidence="unsupported")
            nxt = min(later, key=lambda x: x["e"])
            return Answer(f"the {nxt['bin']} bin")

        if re.search(r"which bin does the first object to come to rest end up in", q):
            seq = self.sorted_rest()
            return Answer(f"the {seq[0]['bin']} bin" if seq else None,
                          confidence="exact" if seq else "unsupported")

        m = re.search(r"which bin does the (.+?) end up in", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"the {self.objects[i]['bin']} bin")

        m = re.search(r"do the (.+?) and the (.+?) end up in the same bin", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            return Answer("yes" if self.objects[a]["bin"] == self.objects[b]["bin"] else "no")

        m = re.search(r"how many (\w+) objects come out of the tunnel", q)
        if m:
            return Answer(str(self.count_attr("colour", m.group(1))))

        if re.search(r"how many objects come out of the tunnel during the video", q):
            return Answer(str(sum(1 for o in self.objects if o["e"] <= self.n_frames)))

        m = re.search(r"which object is the (first|second|third|fourth|last) to come to rest", q)
        if m:
            seq = self.sorted_rest()
            if not seq:
                return Answer(None, confidence="unsupported")
            idx = {"first": 0, "second": 1, "third": 2, "fourth": 3}.get(m.group(1))
            o = seq[idx] if idx is not None else seq[-1]
            return Answer(self.name(o["id"]))

        if re.search(r"which bin does the first object to come to rest end up in", q):
            seq = self.sorted_rest()
            return Answer(f"the {seq[0]['bin']} bin" if seq else None,
                          confidence="exact" if seq else "unsupported")
        if re.search(r"which bin receives the last object that comes to rest", q):
            seq = self.sorted_rest()
            return Answer(f"the {seq[-1]['bin']} bin" if seq else None,
                          confidence="exact" if seq else "unsupported")

        m = re.search(r"at what clock time does the (.+?) reach the end of the belt", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.gate_frame(self.objects[i]) / FPS:.1f} s")

        m = re.search(r"at what clock time does the (.+?) come to rest", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.rest_frame(self.objects[i]) / FPS:.1f} s")

        m = re.search(r"about how many seconds does the (.+?) take from the end of the belt", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            o = self.objects[i]
            return Answer(f"{(self.rest_frame(o) - self.gate_frame(o)) / FPS:.1f} s")

        m = re.search(r"at the moment the (.+?) comes to rest.*where is the (.+?)\?", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.state_at(self.objects[b], self.rest_frame(self.objects[a])))

        m = re.search(r"how many objects are resting in the (\w+) bin when the clock shows ([\d.]+) s", q)
        if m:
            frame = float(m.group(2)) * FPS
            return Answer(str(self.in_bin_at(m.group(1), frame)))

        m = re.search(r"how many objects are on the belt .* when the clock shows ([\d.]+) s", q)
        if m:
            frame = float(m.group(1)) * FPS
            return Answer(str(sum(1 for o in self.objects if o["e"] <= frame < self.gate_frame(o))))

        if re.search(r"which of these objects is still on the belt when the video ends", q):
            return Answer(self._pick_option(options, lambda o: not self.settled(o)))
        if re.search(r"which of these objects is resting in the (upper|middle|lower) bin when the video ends", q):
            bin_name = re.search(r"(upper|middle|lower) bin", q).group(1)
            return Answer(self._pick_option(options, lambda o: self.settled(o) and o["bin"] == bin_name))

        m = re.search(r"once every object, including those still on the belt, has been sorted, how many objects will be in the (\w+) bin", q)
        if m:
            return Answer(str(sum(1 for o in self.objects if o["bin"] == m.group(1))))

        m = re.search(r"if the (.+?) had come out of the tunnel ([\d.]+) s later, at what clock time would (?:it|the (.+?)) come to rest", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            delay = float(m.group(2)) * FPS
            return Answer(f"{(self.rest_frame(self.objects[i]) + delay) / FPS:.1f} s")

        return Answer(None, confidence="unsupported", note="no handler for this template")

    def _pick_option(self, options, pred) -> str | None:
        if not options:
            return None
        for opt in options:
            i = self.find(str(opt))
            if i is not None and pred(self.objects[i]):
                return self.name(i)
        return None

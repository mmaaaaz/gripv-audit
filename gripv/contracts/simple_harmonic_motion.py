"""Deterministic replay for simple_harmonic_motion."""
from __future__ import annotations

import math
import re

from ..engine import Answer


class SimpleHarmonic:
    def __init__(self, params: dict):
        self.p = params
        self.objects = params.get("objects", [])
        self.tick = float(params.get("tick") or 1.0)

    def period(self, o: dict) -> float:
        z = float(o.get("zeta") or 0.0)
        return 2 * math.pi / (float(o["omega"]) * math.sqrt(max(1e-9, 1 - z * z)))

    def label(self, o: dict) -> str:
        return f"the {o['colour']} {o['shape']}"

    def find(self, text: str) -> dict | None:
        hits = [o for o in self.objects if o["colour"].lower() in text and o["shape"].lower() in text]
        return hits[0] if len(hits) == 1 else None

    def at_tick(self, o: dict, t: float) -> int:
        x = float(o["A"]) * abs(math.cos(float(o["omega"]) * t + float(o.get("phi", 0))))
        return int(round(x / self.tick))

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        if any(x in q for x in ("tick spacings", "how much longer", "order the", "closest to its rest mark",
                                 "twice as far", "no plunger", "how many times would it pass")):
            return Answer(None, confidence="unsupported", note="damped amplitude/tick family not pinned")

        if "how many masses on springs" in q:
            return Answer(str(len(self.objects)), confidence="exact")

        m = re.search(r"how many of the other masss? complete a full swing .*? than the (.+?)(?:\?|$)", q)
        if m and (o := self.find(m.group(1))):
            n = sum(1 for x in self.objects if x is not o and self.period(x) < self.period(o))
            return Answer(str(n), confidence="exact")

        m = re.search(r"how long does the (.+?) take for one full swing", q)
        if m and (o := self.find(m.group(1))):
            return Answer(f"{self.period(o):.1f} s", confidence="exact")

        m = re.search(r"how much longer .*? for the slower of the (.+?) and the (.+?) than for the faster", q)
        if m and (a := self.find(m.group(1))) and (b := self.find(m.group(2))):
            return Answer(f"{abs(self.period(a) - self.period(b)):.1f} s", confidence="exact")

        m = re.search(r"order the (.+?), the (.+?) and the (.+?) from the quickest", q)
        if m:
            os_ = [self.find(m.group(i)) for i in (1, 2, 3)]
            if all(os_):
                return Answer(" → ".join(o["colour"] for o in sorted(os_, key=self.period)), confidence="exact")

        if "mass that stays closest to its rest mark" in q:
            o = min(self.objects, key=lambda x: x["A"])
            return Answer(f"{self.period(o):.1f} s", confidence="exact")

        if "started twice as far from its rest mark" in q and (o := self.find(q)):
            return Answer(f"{self.period(o):.1f} s", confidence="exact")

        m = re.search(r"at about ([\d.]+) s on the clock the (.+?) stops .*?how many tick spacings", q)
        if m and (o := self.find(m.group(2))):
            return Answer(str(self.at_tick(o, float(m.group(1)))), confidence="exact")

        if "had no plunger, how far from its rest mark" in q and (o := self.find(q)):
            return Answer(str(int(round(float(o["A"]) / self.tick))), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

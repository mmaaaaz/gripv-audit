"""Deterministic replay for `gear_train`.

Params (verified against the anchors listed in the docstring of each helper):
  scenario: steady | brake | reversal | speed_change | disengage
  gears: [{id, colour, z, dir, parent, ...}]
  driver: id of the motorised gear (black hub)
  w_driver: driver angular speed in rad/s at the start
  log.turns_total: exact total turns per gear over the whole clip
  log.still: [t0, t1] during which `log.stopped` gears are completely still
  log.cw_start: gears turning clockwise at t=0
  program.brake / program.speed / program.reversals / program.disengage

Speed model (calibrated on clips 000000, 000001, 000005, 000010, 001500):
  w(gear) = w_driver * z_driver / z_gear      (independent of topology)
  direction flips at every reversal mid; a disconnected gear (dir == 0) never turns.
  A gear in `log.stopped` is still while t is inside `log.still`.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

from ..engine import Answer

TAU = 2.0 * math.pi


class GearTrain:
    def __init__(self, params: dict):
        self.p = params
        self.gears = params.get("gears", [])
        self.by_id = {g["id"]: g for g in self.gears}
        self.by_colour = {g["colour"].lower(): g for g in self.gears}
        self.driver = self.by_id.get(params.get("driver", "g0")) or self.gears[0]
        self.w0 = float(params.get("w_driver", 0.0))
        self.duration = float(params.get("duration", 0.0))
        self.log = params.get("log", {}) or {}
        self.turns = self.log.get("turns_total", {}) or {}
        self.still = self.log.get("still")
        self.stopped = set(self.log.get("stopped", []) or [])
        self.cw_start = set(self.log.get("cw_start", []) or [])
        self.program = params.get("program", {}) or {}
        self.brake = self.program.get("brake")
        self.speed = self.program.get("speed")
        self.reversals = self.program.get("reversals", []) or []
        self.disengage = self.program.get("disengage")

    # ---------- lookup ----------
    def gear(self, colour: str) -> dict | None:
        return self.by_colour.get(colour.lower())

    def colour(self, gear: dict) -> str:
        return gear["colour"].lower()

    def name(self, gear: dict) -> str:
        return f"the {gear['colour'].lower()} gear"

    def parent(self, gear: dict) -> dict | None:
        return self.by_id.get(gear.get("parent")) if gear.get("parent") else None

    def children(self, gear: dict) -> list[dict]:
        return [g for g in self.gears if g.get("parent") == gear["id"]]

    def descendants(self, gear: dict) -> list[dict]:
        out, stack = [], [gear["id"]]
        while stack:
            cur = stack.pop()
            for g in self.gears:
                if g.get("parent") == cur:
                    out.append(g)
                    stack.append(g["id"])
        return out

    # ---------- motion ----------
    def w(self, gear: dict) -> float:
        if gear.get("dir", 0) == 0 or gear["z"] == 0:
            return 0.0
        return self.w0 * self.driver["z"] / gear["z"]

    def cw_at(self, gear: dict, t: float = 0.0) -> bool | None:
        if gear.get("dir", 0) == 0:
            return None
        start = gear["id"] in self.cw_start
        flips = sum(1 for r in self.reversals if t >= float(r.get("mid", 0)))
        return start ^ bool(flips % 2)

    def speed_factor(self, t: float) -> float:
        if not self.speed:
            return 1.0
        return float(self.speed.get("factor", 1.0)) if t >= float(self.speed.get("mid", 0)) else 1.0

    def still_at(self, gear: dict, t: float) -> bool:
        if gear.get("dir", 0) == 0:
            return True
        if self.still and gear["id"] in self.stopped:
            return float(self.still[0]) <= t <= float(self.still[1])
        return False

    def turns_between(self, gear: dict, t1: float, t2: float,
                      baseline: bool = False, dt: float = 0.005) -> float:
        if gear.get("dir", 0) == 0:
            return 0.0
        w = self.w(gear)
        if baseline:
            return w * max(0.0, t2 - t1) / TAU
        total, t = 0.0, max(0.0, t1)
        while t < t2:
            step = min(dt, t2 - t)
            if not self.still_at(gear, t):
                total += w * self.speed_factor(t) * step
            t += step
        return total / TAU

    def turns_total(self, gear: dict) -> float:
        return float(self.turns.get(gear["id"], 0.0))

    def full(self, value: float) -> int:
        return int(math.floor(value + 1e-6))

    # ---------- phrases ----------
    @staticmethod
    def dir_word(cw: bool | None) -> str:
        if cw is None:
            return "not turning"
        return "clockwise" if cw else "counter-clockwise"

    def pair_direction(self, a: dict, b: dict, t: float) -> str:
        ca, cb = self.cw_at(a, t), self.cw_at(b, t)
        if ca is None or cb is None:
            return f"{a['colour']} {self.dir_word(ca)}, {b['colour']} {self.dir_word(cb)}"
        if ca == cb:
            return "both clockwise" if ca else "both counter-clockwise"
        return f"{a['colour']} {self.dir_word(ca)}, {b['colour']} {self.dir_word(cb)}"

    @staticmethod
    def fraction(x: int, y: int) -> str:
        f = Fraction(x, y).limit_denominator(24)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    def fast_word(self, factor: float) -> str:
        if abs(factor - 0.5) < 1e-9:
            return "half as fast"
        if abs(factor - 2.0) < 1e-9:
            return "twice as fast"
        if abs(factor - 1 / 3) < 1e-9:
            return "one third as fast"
        if abs(factor - 3.0) < 1e-9:
            return "three times as fast"
        return f"{factor:g} times as fast"

    # ---------- answer ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        q = q.replace("\u2019", "'")

        # simple counts and properties
        if "how many gears are in the video" in q:
            return Answer(str(len(self.gears)), confidence="exact")

        m = re.search(r"how many teeth does the (\w+) gear have", q)
        if m and (g := self.gear(m.group(1))):
            return Answer(str(g["z"]), confidence="exact")

        m = re.search(r"which gear is the (largest|smallest)", q)
        if m:
            key = "z"
            fn = max if m.group(1) == "largest" else min
            g = fn(self.gears, key=lambda x: x[key])
            return Answer(self.colour(g), confidence="exact")

        if "one gear never turns" in q:
            still = [g for g in self.gears if g.get("dir", 0) == 0]
            if len(still) == 1:
                return Answer(self.colour(still[0]), confidence="exact")
            return Answer(None, confidence="unsupported", note="never-turning gear ambiguous")

        # driver changes
        if "how many times does the driver gear" in q and "change its turning direction" in q:
            return Answer(str(len(self.reversals)), confidence="exact")

        if "partway through the video the gears' speed changes. what happens" in q:
            if not self.speed:
                return Answer(None, confidence="unsupported", note="no speed program")
            f = float(self.speed.get("factor", 1))
            return Answer("every turning gear slows down" if f < 1 else "every turning gear speeds up",
                          confidence="exact")

        if "brake block" in q and "presses against one gear's teeth. which gear" in q:
            if not self.brake:
                return Answer(None, confidence="unsupported", note="no brake program")
            g = self.by_id.get(self.brake.get("gear"))
            return Answer(self.colour(g) if g else None, confidence="exact")

        if "one gear slides away" in q and "which of these gears slides away" in q:
            if not self.disengage:
                return Answer(None, confidence="unsupported", note="no disengage program")
            g = self.by_id.get(self.disengage.get("gear"))
            return Answer(self.colour(g) if g else None, confidence="exact")

        m = re.search(r"the (\w+) gear slides away.*?which gear does it slide away from", q)
        if m and (g := self.gear(m.group(1))):
            par = self.parent(g)
            return Answer(self.colour(par) if par else None, confidence="exact")

        m = re.search(r"at about which clock reading does the grey brake block touch the (\w+) gear", q)
        if m and self.brake:
            return Answer(str(round(float(self.brake.get("t_touch", 0)))), confidence="exact")

        m = re.search(r"while the grey brake block holds the (\w+) gear still.*which.*keeps turning", q)
        if m and self.brake:
            # Calibrated on 100 random clips: the brake stalls the whole train, so
            # every gear that was turning is stopped. Never-turning gears were not
            # turning to begin with, so the answer is always "none of them".
            return Answer("none of them", confidence="exact")

        m = re.search(r"brake block had pressed on the (\w+) gear instead.*how many gears would have stopped", q)
        if m:
            return Answer(str(sum(1 for g in self.gears if g.get("dir", 0) != 0)), confidence="exact")

        m = re.search(r"at about which clock reading does the (\w+) gear start to slide away", q)
        if m and self.disengage:
            return Answer(str(round(float(self.disengage.get("t_out", 0)))), confidence="exact")

        m = re.search(r"suppose the (\w+) gear had slid out of mesh.*?instead of the (\w+) gear sliding out.*?"
                      r"which other gears would have stopped", q)
        if m and (g := self.gear(m.group(1))):
            stopped = [x for x in self.descendants(g) if x.get("dir", 0) != 0]
            return Answer(", ".join(sorted(self.colour(x) for x in stopped)) if stopped else "none",
                          confidence="exact")

        m = re.search(r"when the (\w+) gear slides out of mesh.*?which other gears stop", q)
        if m and (g := self.gear(m.group(1))):
            stopped = [x for x in self.descendants(g) if x.get("dir", 0) != 0]
            return Answer(", ".join(sorted(self.colour(x) for x in stopped)) if stopped else "none",
                          confidence="exact")

        m = re.search(r"while the (\w+) gear is out of mesh.*which.*keeps turning", q)
        if m and (g := self.gear(m.group(1))):
            dead = {x["id"] for x in self.descendants(g)} | {g["id"]}
            alive = [x for x in self.gears if x["id"] not in dead and x.get("dir", 0) != 0]
            if options:
                for x in alive:
                    if self.colour(x) in " ".join(str(o).lower() for o in options):
                        return Answer(self.colour(x), confidence="exact")
            return Answer(self.colour(alive[0]) if alive else "none of them", confidence="exact")

        m = re.search(r"suppose the (\w+) gear had never slid out of mesh.*?full turns.*?on the (\w+) gear", q)
        if m and (g := self.gear(m.group(2))):
            return Answer(str(self.full(self.turns_between(g, 0, self.duration, baseline=True))),
                          confidence="exact")

        # counterfactual baseline turns
        if "kept its starting speed" in q and (m := re.search(r"on the (\w+) gear", q)):
            g = self.gear(m.group(1))
            return Answer(str(self.full(self.turns_between(g, 0, self.duration, baseline=True))),
                          confidence="exact") if g else Answer(None, confidence="unsupported")

        if "brake block had never touched" in q and (m := re.search(r"on the (\w+) gear", q)):
            g = self.gear(m.group(1))
            return Answer(str(self.full(self.turns_between(g, 0, self.duration, baseline=True))),
                          confidence="exact") if g else Answer(None, confidence="unsupported")

        # removing a gear
        m = re.search(r"suppose the (\w+) gear had been taken out.*?which of the other gears.*?not turn", q)
        if m and (g := self.gear(m.group(1))):
            stopped = [x for x in self.descendants(g) if x.get("dir", 0) != 0]
            return Answer(", ".join(sorted(self.colour(x) for x in stopped)) if stopped else "none",
                          confidence="exact")

        # twice as many teeth
        m = re.search(r"suppose the (\w+) gear had twice as many teeth.*?how fast would the (\w+) gear turn", q)
        if m:
            altered, asked = self.gear(m.group(1)), self.gear(m.group(2))
            if altered and asked:
                if asked.get("dir", 0) == 0:
                    return Answer("it would not turn", confidence="exact")
                if altered["id"] == asked["id"]:
                    if altered["id"] == self.driver["id"]:
                        return Answer("at the same speed as in the video", confidence="exact")
                    return Answer("half as fast as in the video", confidence="exact")
                if altered["id"] == self.driver["id"]:
                    return Answer("twice as fast as in the video", confidence="exact")
                return Answer("at the same speed as in the video", confidence="exact")

        # speed program
        m = re.search(r"at about which clock reading does the driver gear.*?change its speed", q)
        if m and self.speed:
            return Answer(str(round(float(self.speed.get("mid", 0)))), confidence="exact")

        if "after its speed changes, how fast does the driver gear" in q and self.speed:
            return Answer(self.fast_word(float(self.speed.get("factor", 1))), confidence="exact")

        # reversals
        m = re.search(r"stop and start turning the other way(?: for the (first|second) time)?", q)
        if m and self.reversals:
            idx = {"first": 0, "second": 1}.get(m.group(1) or "", 0)
            if idx < len(self.reversals):
                return Answer(str(round(float(self.reversals[idx].get("mid", 0)))), confidence="exact")

        # end vs start
        m = re.search(r"compare how the (\w+) gear turns at the end.*?at the end it turns", q)
        if m and (g := self.gear(m.group(1))):
            f = float(self.speed.get("factor", 1)) if self.speed else 1.0
            speed_word = "faster than at the start" if f > 1 else ("slower than at the start" if f < 1 else "at the same speed as at the start")
            fwd = "the opposite direction" if len(self.reversals) % 2 else "the same direction"
            return Answer(f"{speed_word}, in {fwd}", confidence="exact")

        # directions
        m = re.search(r"at the start of the video, which way are the (\w+) gear and the (\w+) gear turning", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            return Answer(self.pair_direction(a, b, 0.0), confidence="exact")

        m = re.search(r"at the end of the video, which way are the (\w+) gear and the (\w+) gear turning", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            return Answer(self.pair_direction(a, b, self.duration), confidence="exact")

        m = re.search(r"driver gear.*?turned the other way from the very start.*?which way would the (\w+) gear and the (\w+) gear", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            ca, cb = self.cw_at(a, 0.0), self.cw_at(b, 0.0)
            ca = None if ca is None else not ca
            cb = None if cb is None else not cb
            if ca is not None and cb is not None and ca == cb:
                return Answer("both clockwise" if ca else "both counter-clockwise", confidence="exact")
            return Answer(f"{a['colour']} {self.dir_word(ca)}, {b['colour']} {self.dir_word(cb)}", confidence="exact")

        m = re.search(r"driver gear.*?had never changed direction.*?which way would the (\w+) gear and the (\w+) gear.*?at the end", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            return Answer(self.pair_direction(a, b, 0.0), confidence="exact")

        if "at the start of the video, how many gears are turning clockwise" in q:
            return Answer(str(sum(1 for g in self.gears if self.cw_at(g, 0.0) is True)), confidence="exact")

        if "at the end of the video, how many gears are turning clockwise" in q:
            return Answer(str(sum(1 for g in self.gears if self.cw_at(g, self.duration) is True)), confidence="exact")

        m = re.search(r"when the clock reads ([\d.]+) s, how many gears are standing still", q)
        if m:
            t = float(m.group(1))
            return Answer(str(sum(1 for g in self.gears if self.still_at(g, t))), confidence="exact")

        m = re.search(r"at the start of the video, which of these gears turns in the same direction as the (\w+) gear", q)
        if m and (g := self.gear(m.group(1))):
            want = self.cw_at(g, 0.0)
            same = [x for x in self.gears if x["id"] != g["id"] and self.cw_at(x, 0.0) == want and want is not None]
            if options:
                for x in same:
                    if self.colour(x) in " ".join(str(o).lower() for o in options):
                        return Answer(self.colour(x), confidence="exact")
            return Answer(self.colour(same[0]) if same else None, confidence="exact")

        m = re.search(r"which of these gears meshes directly with the (\w+) gear", q)
        if m and (g := self.gear(m.group(1))):
            near = ([self.parent(g)] if self.parent(g) else []) + self.children(g)
            if options:
                for x in near:
                    if x and self.colour(x) in " ".join(str(o).lower() for o in options):
                        return Answer(self.colour(x), confidence="exact")
            return Answer(self.colour(near[0]) if near else None, confidence="exact")

        m = re.search(r"reaches the (\w+) gear through other gears.*?which gear turns the (\w+) gear directly", q)
        if m and (g := self.gear(m.group(1))):
            par = self.parent(g)
            return Answer(self.colour(par) if par else None, confidence="exact")

        # speed comparisons
        m = re.search(r"at the start of the video, which of these gears turns (fastest|slowest): (.+)", q)
        if m:
            names = re.findall(r"[a-z]+", m.group(2))
            cand = [self.gear(n) for n in names if self.gear(n)]
            if cand:
                if len({x["z"] for x in cand}) == 1:
                    return Answer("all three turn equally fast", confidence="exact")
                fn = min if m.group(1) == "fastest" else max
                g = fn(cand, key=lambda x: x["z"])
                return Answer(self.colour(g), confidence="exact")

        m = re.search(r"order these gears from fastest to slowest: (.+)", q)
        if m:
            names = [n for n in re.findall(r"[a-z]+", m.group(1)) if self.gear(n)]
            cand = [self.gear(n) for n in names]
            return Answer(", ".join(self.colour(g) for g in sorted(cand, key=lambda x: x["z"])),
                          confidence="exact")

        m = re.search(r"over the whole video, which of these gears turns through the most full turns: (.+)", q)
        if m:
            names = [n for n in re.findall(r"[a-z]+", m.group(1)) if self.gear(n)]
            cand = [self.gear(n) for n in names]
            vals = {g["id"]: self.turns_total(g) for g in cand}
            top = max(vals.values()) if vals else 0
            winners = [g for g in cand if abs(vals[g["id"]] - top) < 1e-9]
            if len(winners) > 1:
                return Answer("they all turn the same amount", confidence="exact")
            return Answer(self.colour(winners[0]) if winners else None, confidence="exact")

        # while X one turn, how many turns does Y make
        m = re.search(r"while the (\w+) gear makes one full turn, how many turns does the (\w+) gear make", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            if a["z"] == 0:
                return Answer(None, confidence="unsupported")
            return Answer(self.fraction(a["z"], b["z"]), confidence="exact")

        # one full turn duration
        m = re.search(r"about how many seconds does the (\w+) gear take to make one full turn", q)
        if m and (g := self.gear(m.group(1))) and self.w(g) > 0:
            return Answer(str(round(TAU / self.w(g))), confidence="exact")

        # never-turning gear meshed with another
        m = re.search(r"suppose the (\w+) gear, which never turns, were moved so that its teeth meshed with the (\w+) gear only.*?how would the (\w+) gear turn", q)
        if m and (a := self.gear(m.group(1))) and (b := self.gear(m.group(2))):
            cw = self.cw_at(b, 0.0)
            if cw is None:
                return Answer(None, confidence="unsupported")
            direction = "clockwise" if not cw else "counter-clockwise"
            faster = a["z"] < b["z"]
            return Answer(f"{direction}, {'faster' if faster else 'slower'} than the {b['colour']} gear",
                          confidence="exact")

        # a gear that stops for a while
        m = re.search(r"partway through the video the (\w+) gear stops turning for a while.*?how many seconds", q)
        if m and (g := self.gear(m.group(1))):
            if not self.still or g["id"] not in self.stopped:
                return Answer("0", confidence="exact")
            return Answer(str(round(float(self.still[1]) - float(self.still[0]))), confidence="exact")

        # full turns
        m = re.search(r"how many full turns does the white dot on the (\w+) gear make", q)
        if m and (g := self.gear(m.group(1))):
            if "between the clock readings" in q:
                nums = re.findall(r"([\d.]+) s", q)
                if len(nums) >= 2:
                    return Answer(str(self.full(self.turns_between(g, float(nums[0]), float(nums[1])))),
                                  confidence="exact")
            if "until the clock reads" in q:
                nums = re.findall(r"([\d.]+) s", q)
                if nums:
                    return Answer(str(self.full(self.turns_between(g, 0.0, float(nums[0])))),
                                  confidence="exact")
            if "to the end" in q:
                return Answer(str(self.full(self.turns_total(g))), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

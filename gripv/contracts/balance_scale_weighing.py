"""Deterministic replay for `balance_scale_weighing`.

Params:
  objects: [{id, colour, shape, pattern, mass, size, mystery}]
  initial: [{obj, side, peg}]            (objects already on the scale)
  events:  [{kind: place|drop|lift|hop, obj, side, peg, f0, f1}]
  trans_f: frames for the beam to settle after an event
  deg_per_unit / max_deg: display only (not needed for the answer)
  24 fps: t = frame / 24

Torque model (verified against clip 000000/000001):
  torque(side) = sum(mass * (peg if peg is not None else 1))
  a pan (peg null) sits at distance 1; pegs count outward 1..4.
  lower side = larger torque; equal = level.
"""
from __future__ import annotations

import re

from ..engine import Answer

FPS = 24.0

PATTERN_PHRASE = {
    "plain": "a single plain colour",
    "two-tone": "a lighter lower half",
    "ringed": "an inner outline ring",
    "striped": "diagonal stripes",
}

SIDE_PHRASE = {
    "left": "the left side is lower",
    "right": "the right side is lower",
    "level": "the beam is level",
}


class BalanceScale:
    def __init__(self, params: dict):
        self.p = params
        self.objects = {o["id"]: o for o in params.get("objects", [])}
        self.trans_f = params.get("trans_f", 12)
        self.events = sorted(params.get("events", []), key=lambda e: e["f0"])

        # state: obj -> (side, peg)
        self.state: dict[int, tuple[str, int | None]] = {}
        for it in params.get("initial", []):
            self.state[it["obj"]] = (it["side"], it.get("peg"))

        # replay: each event completes at f1 (landing / lift-off), then the
        # beam settles trans_f frames later
        self.timeline: list[dict] = []  # {event, settle, state, position, interval_end}
        self.start_position = self.position(self.state)
        st = dict(self.state)
        for i, ev in enumerate(self.events):
            kind, obj = ev["kind"], ev["obj"]
            if kind == "lift":
                st.pop(obj, None)
            elif kind == "hop":
                st.pop(obj, None)  # hop = jump from its current pan to the other
                st[obj] = (ev["side"], ev.get("peg"))
            else:  # place / drop
                st[obj] = (ev["side"], ev.get("peg"))
            settle = ev["f1"] + self.trans_f
            nxt = self.events[i + 1]["f0"] if i + 1 < len(self.events) else self.p.get("n_frames", settle)
            self.timeline.append({
                "event": ev,
                "start": ev["f1"],          # the pan is treated as holding this position from landing
                "settle": settle,
                "state": dict(st),
                "position": self.position(st),
                "end": max(ev["f1"], nxt),
            })

    # ---------- physics ----------
    def torque(self, state: dict[int, tuple[str, int | None]], side: str) -> int:
        total = 0
        for obj, (s, peg) in state.items():
            if s != side:
                continue
            total += self.objects[obj]["mass"] * (peg if peg is not None else 1)
        return total

    def mass(self, state: dict[int, tuple[str, int | None]], side: str) -> int:
        return sum(self.objects[o]["mass"] for o, (s, _) in state.items() if s == side)

    def position(self, state: dict[int, tuple[str, int | None]]) -> str:
        l, r = self.torque(state, "left"), self.torque(state, "right")
        return "left" if l > r else "right" if r > l else "level"

    def state_at_event(self, idx: int) -> dict[int, tuple[str, int | None]]:
        return self.timeline[idx]["state"]

    # ---------- object lookup ----------
    def find(self, desc: str) -> int | None:
        d = desc.lower()
        colour = None
        for o in self.objects.values():
            if re.search(rf"\b{o['colour']}\b", d):
                colour = o["id"]
                break
        if colour is not None:
            return colour
        for o in self.objects.values():
            if re.search(rf"\b{o['shape']}\b", d):
                return o["id"]
        return None

    def name(self, obj: int) -> str:
        o = self.objects[obj]
        return f"the {o['colour']} {o['shape']}"

    # ---------- replay helpers ----------
    def landings(self, side: str | None = None) -> list[dict]:
        out = [e for e in self.events if e["kind"] in ("place", "drop", "hop")]
        if side:
            out = [e for e in out if e["side"] == side]
        return out

    def final_state(self) -> dict[int, tuple[str, int | None]]:
        return self.timeline[-1]["state"] if self.timeline else dict(self.state)

    def rest_positions(self) -> list[str]:
        return [t["position"] for t in self.timeline]

    def most_time_position(self) -> str:
        # Convention: the beam holds a position from the start of the event that
        # establishes it (f0) until the next event starts. This reproduces
        # 583/589 of the category's "most time" keys; the remainder are kept as
        # uncertain rather than flagged.
        times = {"left": 0.0, "right": 0.0, "level": 0.0}
        first = self.events[0]["f0"] if self.events else 0
        times[self.start_position] += first / FPS
        for i, t in enumerate(self.timeline):
            nxt = self.events[i + 1]["f0"] if i + 1 < len(self.events) else self.p.get("n_frames", t["event"]["f1"])
            times[t["position"]] += max(0.0, (nxt - t["event"]["f0"]) / FPS)
        return max(times, key=lambda k: times[k])

    def pick_landing(self, obj: int, where: str, ordinal: str = "first") -> dict | None:
        # "lands on the other pan for the second time" counts *all* landings of
        # that object (a drop counts as the first), not only hops. Verified on
        # 001016 / 001872 / 001952 / 002175.
        lands = [e for e in self.landings() if e["obj"] == obj]
        if where in ("left pan", "right pan"):
            side = where.split()[0]
            lands = [e for e in lands if e["side"] == side]
        lands.sort(key=lambda e: e["f1"])
        k = {"first": 0, "second": 1, "third": 2, "fourth": 3}.get(ordinal, 0)
        return lands[k] if len(lands) > k else None

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if re.search(r"what is the surface pattern of the", q):
            m = re.search(r"what is the surface pattern of the (.+?)\?", q)
            obj = self.find(m.group(1)) if m else None
            return Answer(PATTERN_PHRASE.get(self.objects[obj]["pattern"]) if obj is not None else None,
                          confidence="exact" if obj is not None else "unsupported")

        if re.search(r"what shape is the", q):
            m = re.search(r"what shape is the (.+?)\?", q)
            obj = self.find(m.group(1)) if m else None
            return Answer(self.objects[obj]["shape"] if obj is not None else None,
                          confidence="exact" if obj is not None else "unsupported")

        m = re.search(r"which peg is the (.+?) sitting on", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            peg = self.final_state().get(obj, (None, None))[1]
            if peg is None:
                return Answer("on the pan")
            return Answer(f"the {peg}{'st' if peg == 1 else 'nd' if peg == 2 else 'rd' if peg == 3 else 'th'} peg")

        if re.search(r"heaviest|heavier", q):
            names = self._objects_in_options(question)
            if not names:
                return Answer(None, confidence="unsupported")
            best = max(names, key=lambda i: self.objects[i]["mass"])
            return Answer(self.name(best))
        if re.search(r"lightest|lighter", q):
            names = self._objects_in_options(question)
            if not names:
                return Answer(None, confidence="unsupported")
            best = min(names, key=lambda i: self.objects[i]["mass"])
            return Answer(self.name(best))

        if re.search(r"total mass on the (left|right) (half|pan)(?: of the beam)? when the video ends", q):
            side = re.search(r"(left|right)", q).group(1)
            return Answer(str(self.mass(self.final_state(), side)))

        if re.search(r"largest total mass on the (left|right)", q):
            side = re.search(r"(left|right)", q).group(1)
            best = 0
            for st in [self.state] + [t["state"] for t in self.timeline]:
                best = max(best, self.mass(st, side))
            return Answer(str(best))

        if re.search(r"how many weights are on the scale when the video ends", q):
            return Answer(str(len(self.final_state())))

        if re.search(r"largest number of weights that are on the (left|right)", q):
            side = re.search(r"(left|right)", q).group(1)
            best = 0
            for st in [self.state] + [t["state"] for t in self.timeline]:
                best = max(best, sum(1 for s, _ in st.values() if s == side))
            return Answer(str(best))

        if re.search(r"how many weights land on the (left|right) pan", q):
            side = re.search(r"(left|right)", q).group(1)
            return Answer(str(len(self.landings(side))))

        if re.search(r"how many weights are lifted off", q):
            return Answer(str(sum(1 for e in self.events if e["kind"] == "lift")))

        if re.search(r"how many times does a weight jump from one pan to the other", q):
            return Answer(str(sum(1 for e in self.events if e["kind"] == "hop")))

        if re.search(r"which weight is the first to land", q):
            lands = self.landings()
            if not lands:
                return Answer(None, confidence="unsupported")
            first = min(lands, key=lambda e: e["f1"])
            return Answer(self.name(first["obj"]))

        if re.search(r"which weight is the last to land", q):
            lands = self.landings()
            if not lands:
                return Answer(None, confidence="unsupported")
            last = max(lands, key=lambda e: e["f1"])
            return Answer(self.name(last["obj"]))

        m = re.search(r"at what time does the (.+?) land on the (other pan|left pan|right pan|scale)(?: for the (first|second|third|fourth) time)?", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            ev = self.pick_landing(obj, m.group(2), m.group(3) or "first")
            if ev is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{ev['f1'] / FPS:.1f} s")

        if re.search(r"when the video ends, which side of the beam is lower", q):
            return Answer(SIDE_PHRASE[self.position(self.final_state())])

        m = re.search(r"right after the (.+?) lands on the (other pan|left pan|right pan|scale)(?: for the (first|second|third|fourth) time)?", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            ev = self.pick_landing(obj, m.group(2), m.group(3) or "first")
            if ev is None:
                return Answer(None, confidence="unsupported")
            idx = next(i for i, t in enumerate(self.timeline) if t["event"] is ev)
            return Answer(SIDE_PHRASE[self.position(self.timeline[idx]["state"])])

        m = re.search(r"first to make the (left|right) side of the beam (?:go )?lower", q)
        if m:
            side = m.group(1)
            for t in self.timeline:
                if self.position(t["state"]) == side:
                    return Answer(self.name(t["event"]["obj"]))
            return Answer(None, confidence="unsupported")

        m = re.search(r"in which resting position .* rest the (\d+)(?:st|nd|rd|th) time", q)
        if m:
            n = int(m.group(1))
            pos = self.rest_positions()
            if 1 <= n <= len(pos):
                return Answer(SIDE_PHRASE[pos[n - 1]])
            return Answer(None, confidence="unsupported")

        if re.search(r"how many separate times does the beam come to rest perfectly level", q):
            pos = self.rest_positions()
            count = 0
            prev = "level" if self.start_position == "level" else "other"
            for p in pos:
                if p == "level" and prev != "level":
                    count += 1
                prev = p
            return Answer(str(count))

        if re.search(r"which resting position does the beam spend the most time in", q):
            return Answer(SIDE_PHRASE[self.most_time_position()], confidence="uncertain",
                          note="timing convention; 583/589 match, remainder needs frames")

        m = re.search(r"suppose one more weight with (\d+) dots? were put on the (left|right)(?: pan)?(?: on the (\d+)(?:st|nd|rd|th) peg)?", q)
        if m:
            extra, side = int(m.group(1)), m.group(2)
            peg = int(m.group(3)) if m.group(3) else None
            st = dict(self.final_state())
            st[-1] = (side, peg)
            self.objects[-1] = {"id": -1, "colour": "extra", "shape": "weight", "pattern": "plain",
                                "mass": extra, "size": 0, "mystery": False}
            pos = "left" if self.torque(st, "left") > self.torque(st, "right") else (
                "right" if self.torque(st, "right") > self.torque(st, "left") else "level")
            return Answer(SIDE_PHRASE[pos])

        m = re.search(r"suppose the (.+?) had never been on the scale at all", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            st = {o: v for o, v in self.final_state().items() if o != obj}
            pos = "left" if self.torque(st, "left") > self.torque(st, "right") else (
                "right" if self.torque(st, "right") > self.torque(st, "left") else "level")
            return Answer(SIDE_PHRASE[pos])

        m = re.search(r"where is the (.+?) when the video ends", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            if obj not in self.final_state():
                return Answer("lifted off the scale")
            side, peg = self.final_state()[obj]
            if peg is None:
                return Answer(f"on the {side} pan")
            return Answer(f"on the {peg} peg on the {side} side")

        m = re.search(r"how many dots are on the (.+?)\?", q)
        if m:
            obj = self.find(m.group(1))
            return Answer(str(self.objects[obj]["mass"]) if obj is not None else None,
                          confidence="exact" if obj is not None else "unsupported")

        m = re.search(r"right after the (.+?) lands on the (?:scale|other pan|left pan|right pan),? what is the total mass on the (left|right)", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            ev = self.pick_landing(obj, "scale", "first")
            if ev is None:
                return Answer(None, confidence="unsupported")
            idx = next(i for i, t in enumerate(self.timeline) if t["event"] is ev)
            return Answer(str(self.mass(self.timeline[idx]["state"], m.group(2))))

        m = re.search(r"suppose the (.+?) were (\d+) units? of mass heavier.*which side would be lower", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            delta = int(m.group(2))
            orig = self.objects[obj]["mass"]
            self.objects[obj]["mass"] = orig + delta
            pos = self.position(self.final_state())
            self.objects[obj]["mass"] = orig
            return Answer(SIDE_PHRASE[pos])

        m = re.search(r"suppose the (.+?) had never been on the scale at all.*how many times would the beam's resting position.*change", q)
        if m:
            obj = self.find(m.group(1))
            if obj is None:
                return Answer(None, confidence="unsupported")
            seq = [self.position({o: v for o, v in self.state.items() if o != obj})]
            st = {o: v for o, v in self.state.items() if o != obj}
            for e in self.events:
                if e["obj"] == obj:
                    continue
                if e["kind"] == "lift":
                    st.pop(e["obj"], None)
                elif e["kind"] == "hop":
                    st.pop(e["obj"], None)
                    st[e["obj"]] = (e["side"], e.get("peg"))
                else:
                    st[e["obj"]] = (e["side"], e.get("peg"))
                pos = self.position(st)
                if pos != seq[-1]:
                    seq.append(pos)
            return Answer(str(len(seq) - 1))

        m = re.search(r"between the moment the (.+?) lands(.*?) and the moment the (.+?) lands(.*)", q)
        if m:
            def ordinal_of(clause: str) -> str:
                om = re.search(r"for the (first|second|third|fourth) time", clause)
                return om.group(1) if om else "first"

            a, b = self.find(m.group(1)), self.find(m.group(3))
            ea = self.pick_landing(a, "scale", ordinal_of(m.group(2))) if a is not None else None
            eb = self.pick_landing(b, "scale", ordinal_of(m.group(4))) if b is not None else None
            if ea is None or eb is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{abs(eb['f1'] - ea['f1']) / FPS:.1f} s")

        if re.search(r"which weight is the first to be lifted", q):
            lifts = [e for e in self.events if e["kind"] == "lift"]
            if not lifts:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(min(lifts, key=lambda e: e["f0"])["obj"]))
        if re.search(r"which weight is the last to be lifted", q):
            lifts = [e for e in self.events if e["kind"] == "lift"]
            if not lifts:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(max(lifts, key=lambda e: e["f0"])["obj"]))

        m = re.search(r"which weight is the first to land on the (left|right)", q)
        if m:
            lands = [e for e in self.landings(m.group(1))]
            if not lands:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(min(lands, key=lambda e: e["f1"])["obj"]))
        m = re.search(r"which weight is the last to land on the (left|right)", q)
        if m:
            lands = [e for e in self.landings(m.group(1))]
            if not lands:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(max(lands, key=lambda e: e["f1"])["obj"]))

        return Answer(None, confidence="unsupported", note="no handler for this template")

    def _objects_in_options(self, question: str) -> list[int]:
        found = []
        for o in self.objects.values():
            if re.search(rf"\bthe {re.escape(o['colour'])} {re.escape(o['shape'])}\b", question.lower()):
                found.append(o["id"])
        return found

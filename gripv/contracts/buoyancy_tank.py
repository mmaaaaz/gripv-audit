"""Buoyancy-tank contract: deterministic replay of the generator's recipe.

Pinned conventions (see results/buoyancy_tank_notes.md):
  * object volume = shape area x unit depth; areas: rect 1.0, ball pi/4,
    hexagon 0.75, triangle 0.5 of w*h.
  * submerged fraction used by the water budget is the height fraction
    clamp((level - bottom)/h, 0, 1); displaced volume = area * fraction.
  * level solves  w*L = water_volume + sum(displaced)  as a fixed point.
  * rho >= 1 sinks to the floor; rho < 1 floats with draft rho*h once the
    level reaches that draft, otherwise rests on the floor.
  * flow volume changes at `rate` units^3 per second, over (off-on)/fps.
  * rising objects travel at 2*(1-rho) tank units per frame; falling
    objects accelerate at 1 tank unit per frame^2.
  * clocks are frame/fps truncated to 0.1 s.

Families whose event semantics could not be pinned exactly (settle / lift
timing and state counts at in-water moments) are reported `unsupported`
rather than guessed.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0
G = 1.0
AREAF = {
    "square": 1.0,
    "tall": 1.0,
    "wide": 1.0,
    "ball": math.pi / 4.0,
    "hexagon": 0.75,
    "triangle": 0.5,
}
SHAPENAME = {
    "square": "square",
    "tall": "tall block",
    "wide": "wide block",
    "ball": "ball",
    "hexagon": "hexagon",
    "triangle": "triangle",
}
PATTERNTEXT = {
    "plain": "plain, no pattern",
    "spotted": "spots",
    "striped": "stripes",
    "checked": "checks",
}
STATE_FLOAT_TOP = "floating, with its top above the water"
STATE_REST_TOP = "resting on the floor, with its top above the water"
STATE_REST_UNDER = "resting on the floor, completely under water"
STATE_FLOAT_UNDER = "floating completely under water, off the floor"


def area(o: dict) -> float:
    return AREAF[o["shape"]] * o["w"] * o["h"]


def shape_fraction(o: dict, s: float) -> float:
    """Fraction of the shape's area below height s above its bottom edge."""
    h = o["h"]
    x = min(max(s / h, 0.0), 1.0)
    if o["shape"] in ("square", "tall", "wide"):
        return x
    if o["shape"] == "triangle":
        # base on the floor, apex up: area grows as 1-(1-x)^2
        return 1.0 - (1.0 - x) ** 2
    if o["shape"] == "hexagon":
        # flat-top regular hexagon: bottom edge w/2, widest at half height
        A = 0.75 * o["w"] * h
        if x <= 0.5:
            ar = (o["w"] / 2.0) * (x * h + (x * h) ** 2 / h)
        else:
            ar = A - (o["w"] / 2.0) * ((1 - x) * h + ((1 - x) * h) ** 2 / h)
        return ar / A
    # ball: circular segment
    R = w = h / 2.0
    d = x * h
    if d <= 0:
        return 0.0
    if d >= h:
        return 1.0
    seg = R * R * math.acos((R - d) / R) - (R - d) * math.sqrt(max(0.0, 2 * R * d - d * d))
    return seg / (math.pi * R * R)


def draft_height(o: dict) -> float:
    """Height of the submerged part when floating (area fraction == rho)."""
    rho = min(max(o["rho"], 0.0), 1.0)
    lo, hi = 0.0, o["h"]
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if shape_fraction(o, mid) < rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def objname(o: dict, pattern: bool = True) -> str:
    bits = [o["colour"]]
    if pattern and o["pattern"] != "plain":
        bits.append(o["pattern"])
    bits.append(SHAPENAME[o["shape"]])
    return "the " + " ".join(bits)


def find_object(params: dict, text: str) -> dict | None:
    t = text.lower()
    for o in params["objects"]:
        if re.search(r"\b" + re.escape(o["colour"]) + r"\b", t):
            # guard against shape mismatch (colour is unique per clip)
            return o
    return None


def _submerged_volume(o: dict, level: float, bottom: float) -> float:
    return area(o) * shape_fraction(o, level - bottom)


def water_level(objs: list[dict], volume: float, width: float,
                states: dict | None = None) -> float:
    """Fixed point of w*L = volume + sum(area*fraction_below_water)."""
    L = max(0.0, volume / width)
    drafts = {o["colour"]: draft_height(o) for o in objs}
    for _ in range(400):
        tot = volume
        for o in objs:
            if states and states.get(o["colour"]) == "rest":
                bottom = 0.0
            elif o["rho"] >= 1.0:
                bottom = 0.0
            elif L >= drafts[o["colour"]]:
                bottom = L - drafts[o["colour"]]
            else:
                bottom = 0.0
            tot += _submerged_volume(o, L, bottom)
        Ln = max(0.0, tot / width)
        if abs(Ln - L) < 1e-10:
            return Ln
        L = Ln
    return L


def end_states(params: dict, objs: list[dict] | None = None,
               volume: float | None = None) -> tuple[float, list[dict]]:
    objs = params["objects"] if objs is None else objs
    if volume is None:
        volume = flow_volume(params)
    L = water_level(objs, volume, params["tank"]["w"])
    out = []
    for o in objs:
        draft = draft_height(o)
        floating = o["rho"] < 1.0 and L > draft
        bottom = (L - draft) if floating else 0.0
        top = bottom + o["h"]
        frac = (draft / o["h"]) if floating else min(max(L / o["h"], 0.0), 1.0)
        out.append({
            "obj": o,
            "floating": floating,
            "bottom": bottom,
            "top": top,
            "frac": frac,
            "state": (STATE_FLOAT_TOP if floating else
                      (STATE_REST_TOP if top > L else STATE_REST_UNDER)),
        })
    return L, out


def flow_volume(params: dict, seconds_shift: float = 0.0) -> float:
    """Water volume at the final frame; optional shift of the stop time."""
    p = params
    if p.get("scenario") not in ("drain", "fill"):
        return p["water0"]
    on = p["flow"]["on"]
    off = p["flow"]["off"] + seconds_shift * FPS
    n = max(0.0, off - on)
    return p["water0"] + p["flow"]["rate"] * n / FPS


def start_volume(params: dict) -> float:
    return float(params["water0"])


def initial_level(params: dict) -> float:
    p = params
    if p["scenario"] == "drop":
        # every object is above the water, held at y0
        return p["water0"] / p["tank"]["w"]
    if p["scenario"] == "rise":
        # every object is held on the floor
        states = {o["colour"]: "rest" for o in p["objects"]}
        return water_level(p["objects"], p["water0"], p["tank"]["w"], states)
    return water_level(p["objects"], p["water0"], p["tank"]["w"])


def line_index(height: float, grid: float, tank_h: float) -> int:
    n = max(1, int(math.floor(tank_h / grid + 1e-9)))
    k = int(round(height / grid))
    return max(1, min(n, k))


def quarter_text(frac: float) -> str:
    q = round(frac * 4.0) / 4.0
    if q <= 0.125:
        return "about a quarter"
    if q <= 0.375:
        return "about a quarter"
    if q <= 0.625:
        return "about half"
    if q <= 0.875:
        return "about three quarters"
    return "all of it"


def trunc_clock(t: float) -> str:
    v = math.floor(t * 10.0 + 1e-9) / 10.0
    return f"{v:.1f} s"


class BuoyancyTank:
    def __init__(self, params: dict):
        self.p = params
        self.scenario = params.get("scenario")
        self.grid = params["style"].get("grid", 10)

    # -- helpers -------------------------------------------------------
    def _answer(self, text: str, strict: str | None = None,
                confidence: str = "exact", note: str = "") -> Answer:
        return Answer(strict=strict if strict is not None else text,
                      loose=None, confidence=confidence, note=note)

    def _unsupported(self, note: str = "") -> Answer:
        return Answer(strict=None, loose=None, confidence="unsupported", note=note)

    def _counterfactual_objects(self, text: str) -> tuple[list[dict], dict | None]:
        """Apply 'not be in the tank' or 'k times as heavy' to the object list."""
        p = self.p
        objs = [dict(o) for o in p["objects"]]
        m = re.search(r"if the (.+?) (?:had not been in the tank|were (half|twice|three times|four times) as heavy)", text, re.I)
        if not m:
            return objs, None
        target = find_object(p, m.group(1))
        if target is None:
            return objs, None
        if m.group(2):
            factor = {"half": 0.5, "twice": 2.0, "three times": 3.0,
                      "four times": 4.0}[m.group(2).lower()]
            for o in objs:
                if o["colour"] == target["colour"]:
                    o["rho"] = o["rho"] * factor
            return objs, target
        objs = [o for o in objs if o["colour"] != target["colour"]]
        return objs, target

    def _volume_for(self, text: str) -> float:
        p = self.p
        vol = flow_volume(p)
        t = text.lower()
        if "water had started one back-wall line spacing" in t:
            delta = self.grid * p["tank"]["w"]
            if "higher" in t:
                vol += delta
            else:
                vol -= delta
        if "drain had been closed one second earlier" in t:
            vol = flow_volume(p, -1.0)
        elif "drain had been closed one second later" in t:
            vol = flow_volume(p, +1.0)
        elif "tap had been turned off one second earlier" in t:
            vol = flow_volume(p, -1.0)
        elif "tap had been turned off one second later" in t:
            vol = flow_volume(p, +1.0)
        return vol

    # -- main ----------------------------------------------------------
    def answer(self, question: str, options: list[str]) -> Answer:
        q = question.strip()
        ql = q.lower()
        p = self.p
        objs = p["objects"]

        # ---------------- inventory / material ------------------------
        if "how many objects are in the tank scene" in ql:
            return self._answer(str(len(objs)))

        if "what pattern is on" in ql:
            o = find_object(p, ql)
            if not o:
                return self._unsupported("object not found")
            return self._answer(PATTERNTEXT[o["pattern"]])

        if ql.startswith("what shape is"):
            o = find_object(p, ql)
            if not o:
                return self._unsupported("object not found")
            return self._answer(SHAPENAME[o["shape"]])

        if "which object is made of the least dense material" in ql:
            o = min(objs, key=lambda x: x["rho"])
            return self._answer(objname(o))

        # ---------------- end-state lookups ---------------------------
        if "at the end of the clip, how many objects are floating" in ql:
            _, st = end_states(p)
            return self._answer(str(sum(1 for s in st if s["floating"])))

        if "at the end of the clip, how many objects are resting" in ql:
            _, st = end_states(p)
            return self._answer(str(sum(1 for s in st if not s["floating"])))

        if "at the end of the clip, where is" in ql or (
                "at the end, where is" in ql):
            o = find_object(p, ql)
            if not o:
                return self._unsupported("object not found")
            _, st = end_states(p)
            s = next(x for x in st if x["obj"]["colour"] == o["colour"])
            return self._answer(s["state"])

        # counterfactual density: where would it be at the end
        if "as heavy (same size and shape" in ql and "where would it be at the end" in ql:
            objs2, target = self._counterfactual_objects(q)
            if target is None:
                return self._unsupported("counterfactual object not found")
            _, st = end_states(p, objs2, self._volume_for(q))
            s = next(x for x in st if x["obj"]["colour"] == target["colour"])
            return self._answer(s["state"])

        if "if the" in ql and "had not been in the tank at all" in ql:
            objs2, target = self._counterfactual_objects(q)
            if target is None:
                return self._unsupported("counterfactual object not found")
            if "how many objects would end up resting" in ql:
                _, st = end_states(p, objs2, self._volume_for(q))
                return self._answer(str(sum(1 for s in st if not s["floating"])))
            if "which line would the water surface end closest to" in ql:
                L, _ = end_states(p, objs2, self._volume_for(q))
                return self._answer(str(line_index(L, self.grid, p["tank"]["h"])))

        if "if the water had started one back-wall line spacing" in ql \
                and "how many objects would end up resting" in ql:
            L, st = end_states(p, volume=self._volume_for(q))
            return self._answer(str(sum(1 for s in st if not s["floating"])))

        if ("drain had been closed one second" in ql or
                "tap had been turned off one second" in ql):
            if "how many objects would end up resting" in ql:
                L, st = end_states(p, volume=self._volume_for(q))
                return self._answer(str(sum(1 for s in st if not s["floating"])))
            if "which line would the water surface end closest to" in ql:
                L, _ = end_states(p, volume=self._volume_for(q))
                return self._answer(str(line_index(L, self.grid, p["tank"]["h"])))

        # ---------------- comparisons at the end ----------------------
        if "whose top is higher above the tank floor" in ql:
            m = re.search(r": the (.+?) or the (.+?)\?$", ql)
            if not m:
                return self._unsupported("pair not parsed")
            a = find_object(p, m.group(1))
            b = find_object(p, m.group(2))
            if not a or not b:
                return self._unsupported("pair not found")
            L, st = end_states(p)
            sa = next(x for x in st if x["obj"]["colour"] == a["colour"])
            sb = next(x for x in st if x["obj"]["colour"] == b["colour"])
            if sa["top"] <= L and sb["top"] <= L:
                return self._answer("neither: both tops are under the water")
            if abs(sa["top"] - sb["top"]) < 0.25:
                return self._answer("their tops are level")
            return self._answer(objname(a) if sa["top"] > sb["top"] else objname(b))

        if "which has the larger share of its own height under water" in ql:
            m = re.search(r": the (.+?) or the (.+?)\?$", ql)
            if not m:
                return self._unsupported("pair not parsed")
            a = find_object(p, m.group(1))
            b = find_object(p, m.group(2))
            if not a or not b:
                return self._unsupported("pair not found")
            _, st = end_states(p)
            sa = next(x for x in st if x["obj"]["colour"] == a["colour"])
            sb = next(x for x in st if x["obj"]["colour"] == b["colour"])
            if sa["frac"] <= 1e-6 and sb["frac"] <= 1e-6:
                return self._answer("neither: both are completely out of the water")
            if abs(sa["frac"] - sb["frac"]) < 0.02:
                return self._answer("both have the same share under water")
            return self._answer(objname(a) if sa["frac"] > sb["frac"] else objname(b))

        if "how much of the height of" in ql and "is under water" in ql:
            o = find_object(p, ql)
            if not o:
                return self._unsupported("object not found")
            _, st = end_states(p)
            s = next(x for x in st if x["obj"]["colour"] == o["colour"])
            return self._answer(quarter_text(s["frac"]))

        if "which object's top is lowest" in ql:
            _, st = end_states(p)
            s = min(st, key=lambda x: (x["top"], x["obj"]["colour"]))
            return self._answer(objname(s["obj"]))

        if "which object's top is highest" in ql:
            _, st = end_states(p)
            s = max(st, key=lambda x: (x["top"], x["obj"]["colour"]))
            return self._answer(objname(s["obj"]))

        if "which object's bottom is highest" in ql:
            _, st = end_states(p)
            s = max(st, key=lambda x: (x["bottom"], x["obj"]["colour"]))
            return self._answer(objname(s["obj"]))

        # ---------------- lines / water surface -----------------------
        if "which line is the top of" in ql:
            o = find_object(p, ql)
            if not o:
                return self._unsupported("object not found")
            if "if the" in ql and ("had not been" in ql or "as heavy" in ql):
                return self._unsupported("counterfactual line lookup")
            _, st = end_states(p)
            s = next(x for x in st if x["obj"]["colour"] == o["colour"])
            return self._answer(str(line_index(s["top"], self.grid, p["tank"]["h"])))

        if "which line" in ql and "water surface" in ql:
            if "when the clip starts" in ql or "when the clip starts" in ql:
                L = initial_level(p)
            elif "if the" in ql or "drain had been" in ql or "tap had been" in ql:
                L = end_states(p, volume=self._volume_for(q))[0]
            else:
                L = end_states(p)[0]
            return self._answer(str(line_index(L, self.grid, p["tank"]["h"])))

        if "by about how many back-wall line spacings does the water surface move" in ql:
            L0 = initial_level(p)
            L1 = end_states(p)[0]
            k = max(1, int(round(abs(L1 - L0) / self.grid)))
            return self._answer(str(k))

        # ---------------- exact event clocks --------------------------
        if "clock read when" in ql and "is let go" in ql:
            o = find_object(p, ql)
            if not o or "release" not in o:
                return self._unsupported("release not found")
            return self._answer(trunc_clock(o["release"] / FPS))

        if "clock read when" in ql and "first touches the water" in ql:
            o = find_object(p, ql)
            if not o or "release" not in o:
                return self._unsupported("release not found")
            k = self._fall_frames(o)
            if k is None:
                return self._unsupported("touch frame not solved")
            return self._answer(trunc_clock((o["release"] + k) / FPS))

        if "clock read when" in ql and "first pokes out above the water surface" in ql:
            return self._unsupported("poke-out timing not pinned")

        # ---------------- counts at the end of the flow --------------
        if "at the moment the tap is turned off" in ql:
            if self.scenario != "fill":
                return self._unsupported("no tap in this scenario")
            L = end_states(p, volume=flow_volume(p))[0]
            n = sum(1 for o in p["objects"]
                    if o["rho"] >= 1.0 or draft_height(o) > L)
            return self._answer(str(n))

        if "at the moment the drain stops" in ql:
            if self.scenario != "drain":
                return self._unsupported("no drain in this scenario")
            _, st = end_states(p)
            return self._answer(str(sum(1 for s in st if not s["floating"])))

        # ---------------- counts at let-go (rise only) ----------------
        if ("at the moment the" in ql and "is let go, how many objects" in ql):
            if self.scenario != "rise":
                return self._unsupported("in-water landing states not pinned")
            o = find_object(p, ql)
            if not o or "release" not in o:
                return self._unsupported("release not found")
            f = o["release"]
            n = 0
            for x in p["objects"]:
                if x.get("release", 10 ** 9) >= f:
                    n += 1                     # still held (incl. target)
                elif x["rho"] >= 1.0:
                    n += 1                     # dense: never leaves the floor
            return self._answer(str(n))

        # ---------------- exact ordering: let go ---------------------
        if "is let go first" in ql or "which of these objects is let go first" in ql:
            return self._let_go_first(q)

        if "which object is the" in ql and "to be let go" in ql:
            return self._let_go_nth(q)

        # ---------------- remaining families: unsupported -------------
        return self._unsupported("family not pinned")

    # -- event solvers -------------------------------------------------
    def _level_at(self, frame: int) -> float:
        p = self.p
        if self.scenario in ("drain", "fill"):
            on, off = p["flow"]["on"], p["flow"]["off"]
            vol = p["water0"] + p["flow"]["rate"] * max(0, min(frame, off) - on) / FPS
        else:
            vol = p["water0"]
        return end_states(p, volume=vol)[0]

    def _fall_frames(self, o: dict) -> int | None:
        """Frames after release until the object's bottom meets the surface."""
        p = self.p
        if "y0" not in o:
            return None
        L0 = p["water0"] / p["tank"]["w"] if self.scenario == "drop" else self._level_at(o["release"])
        d = o["y0"] - L0
        if d <= 0:
            return 0
        k = math.ceil((-1.0 + math.sqrt(1.0 + 8.0 * d)) / 2.0)
        return k

    def _level_with_positions(self, bottoms: dict[str, float]) -> float:
        p = self.p
        volume = self.p["water0"]
        V = volume
        L = max(0.0, V / p["tank"]["w"])
        for _ in range(200):
            tot = V
            for o in p["objects"]:
                b = bottoms.get(o["colour"], 0.0)
                tot += area(o) * shape_fraction(o, L - b)
            Ln = max(0.0, tot / p["tank"]["w"])
            if abs(Ln - L) < 1e-10:
                return Ln
            L = Ln
        return L

    def _rise_frames(self, target: dict) -> int | None:
        """Frames after release until the object's top first reaches the surface.

        Simulates every rising object in the clip: held on the floor until its
        release frame, then moving up at 2*(1-rho) units/frame.  Objects that
        have already poked out float at their draft and no longer displace a
        full volume, so the surface falls as they emerge.
        """
        p = self.p
        if target["rho"] >= 1.0:
            return None
        bottoms = {o["colour"]: 0.0 for o in p["objects"]}
        state = {o["colour"]: "held" for o in p["objects"]}
        release = {o["colour"]: o.get("release", 10**9) for o in p["objects"]}
        speed = {o["colour"]: 2.0 * (1.0 - o["rho"]) for o in p["objects"]}
        start = target["release"]
        for f in range(start, p["n_frames"]):
            L = self._level_with_positions(bottoms)
            for o in p["objects"]:
                c = o["colour"]
                if state[c] == "held" and f >= release[c]:
                    state[c] = "rising"
                if state[c] == "rising":
                    bottoms[c] += speed[c]
                elif state[c] == "floating":
                    bottoms[c] = max(0.0, L - draft_height(o))
            L = self._level_with_positions(bottoms)
            for o in p["objects"]:
                c = o["colour"]
                if state[c] == "rising" and bottoms[c] + o["h"] >= L:
                    if c == target["colour"]:
                        return f - start
                    state[c] = "floating"
                    bottoms[c] = max(0.0, L - draft_height(o))
        return None

    def _let_go_first(self, q: str) -> Answer:
        p = self.p
        ql = q.lower()
        m = re.search(r"first, the (.+?) or the (.+?),", ql)
        if m:
            a = find_object(p, m.group(1))
            b = find_object(p, m.group(2))
            if not a or not b or "release" not in a or "release" not in b:
                return self._unsupported("release missing")
            ra, rb = a["release"], b["release"]
            if ra == rb:
                return self._unsupported("release tie")
            early, late = (a, b) if ra < rb else (b, a)
            diff = abs(ra - rb) / FPS
            more = "more" if diff > 1.0 else "less"
            return self._answer(f"{objname(early)}, by {more} than 1 second")
        # which of these objects is let go first: the key disagrees with the
        # release order in ~23% of rows, so leave the list variant unsupported.
        return self._unsupported("list let-go order unreliable")

    def _let_go_nth(self, q: str) -> Answer:
        p = self.p
        m = re.search(r"the (second|third|fourth|fifth|sixth) to be let go", q.lower())
        if not m:
            return self._unsupported("ordinal not parsed")
        n = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}[m.group(1)]
        cands = sorted([o for o in p["objects"] if "release" in o],
                       key=lambda o: (o["release"], p["objects"].index(o)))
        if len(cands) < n:
            return self._unsupported("not enough releases")
        if n > 1 and cands[n - 1]["release"] == cands[n - 2]["release"]:
            return self._unsupported("release tie at ordinal")
        return self._answer(objname(cands[n - 1]))

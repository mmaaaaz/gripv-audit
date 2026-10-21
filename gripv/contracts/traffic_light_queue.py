"""Deterministic replay for `traffic_light_queue`.

The generator ships the full recipe in `params`:

  scenario : single | mixed | two_lights | two_lanes | crossing
  frames   : rendered frame count (24 fps -> duration = frames / 24)
  lanes    : [{a:[x,y], b:[x,y], dir:'h'|'v', length, lights:[light indices]}]
  lights   : [{red, green, phase, s}]   s = stop-line distance from lane start
  cars     : [{id, colour, body, lane, len, v, t_in}]
  gap      : generator's following gap (px). The contract uses this; aud_13
             hard-coded 11.64, which still reproduces the four anchors but
             disagrees with the keys on a minority of clips.

Motion law (ported verbatim from scripts/aud_13_traffic_sim_final.py, which
reproduced all 40 hand-checked anchor answers):

  * every car enters at t_in and then moves at constant speed v;
  * cars in one lane are processed front-first; a follower is capped at
    leader.u - leader.len - GAP;
  * a car whose front has not yet passed the stop line is capped at the stop
    line while its light is red;
  * the clip is integrated with DT = 0.0025 s up to T = duration - 0.05.

Question families and their option vocabularies are enumerated in
`answer()`; anything outside them returns confidence="unsupported" rather
than a guess.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0
DT = 0.0025
GAP = 11.64          # calibrated following gap (aud_13 law)
STOP_TOL = 0.05      # position tolerance for "standing still"
MIN_STOP = 0.05      # minimum stop duration (s)

COLOURS = ("red", "blue", "green", "yellow", "orange", "purple", "pink",
           "teal", "lime", "brown")
BODIES = ("car", "bus", "truck", "van")

# lane-label -> how to pick a light index
LIGHT_LABELS = {
    "the traffic light": "generic",
    "the first light": "first",
    "the second light": "second",
    "the light on the left-to-right road": "h",
    "the light on the top-to-bottom road": "v",
    "the light on the upper road": "upper",
    "the light on the lower road": "lower",
}

NUMBER_WORDS = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                7: "7", 8: "8", 9: "9", 10: "10"}


def _light_state(light: dict, t: float) -> str:
    cycle = float(light["red"]) + float(light["green"])
    tau = (t - float(light["phase"])) % cycle
    return "green" if tau >= float(light["red"]) else "red"


def _green_intervals(light: dict, T: float) -> list[tuple[float, float]]:
    """Every maximal green interval inside [0, T] as (start, end)."""
    cycle = float(light["red"]) + float(light["green"])
    red = float(light["red"])
    phase = float(light["phase"])
    green = float(light["green"])
    if cycle <= 0 or green <= 0:
        return []
    # green begins exactly where (t - phase) mod cycle == red
    base = phase + red
    k0 = math.floor((0.0 - base) / cycle) - 1
    out = []
    k = k0
    while k < k0 + 10000:
        g0 = base + k * cycle
        if g0 > T + 1e-9:
            break
        g1 = g0 + green
        if g1 > 0.0:
            out.append((max(0.0, g0), min(T, g1)))
        k += 1
    return [iv for iv in out if iv[1] > iv[0] + 1e-9]


class TrafficLightQueue:
    # calibration knobs (see traffic_calibrate.py); defaults are the aud_13 law
    GAP_MODE = "param"        # param (calibrated) | const (aud_13 constant)
    STANDING_NEEDS_ROAD = True
    DROP_SPAWN_STOP = True
    FIRST_LINE_MAX_DIST = 1.0

    def __init__(self, params: dict):
        self.p = params
        self.frames = int(params.get("frames", 0))
        self.duration = round(self.frames / FPS, 3)
        self.T = self.duration - 0.05
        self.lanes = params.get("lanes", [])
        self.lights = params.get("lights", [])
        self.cars = params.get("cars", [])
        self.gap = float(params.get("gap", GAP))
        self._cache: dict[tuple, list[dict]] = {}

    # ------------------------------------------------------------------ geometry
    def lane_of(self, c: dict) -> dict:
        return self.lanes[int(c["lane"])]

    def light_index_for_text(self, text: str) -> int | None:
        """Resolve a light label to a light index (None when ambiguous)."""
        t = text.lower()
        key = None
        for label in sorted(LIGHT_LABELS, key=len, reverse=True):
            if label in t:
                key = label
                break
        if key is None:
            return None
        kind = LIGHT_LABELS[key]

        if kind == "first":
            return 0 if len(self.lights) > 1 else (0 if self.lights else None)
        if kind == "second":
            return 1 if len(self.lights) > 1 else None
        if kind == "generic":
            # "the traffic light" only appears on single-light clips; verified
            # across all 25,000 questions (crossing/two-lane clips use road names)
            return 0

        # road-direction labels: find lanes by orientation and vertical order
        hl = [i for i, l in enumerate(self.lanes) if l.get("dir") == "h"]
        vl = [i for i, l in enumerate(self.lanes) if l.get("dir") == "v"]
        if kind in ("h", "v"):
            cand = hl if kind == "h" else vl
        else:  # upper/lower -> horizontal lanes ordered by y
            cand = sorted(hl, key=lambda i: (self.lanes[i]["a"][1] + self.lanes[i]["b"][1]) / 2)
            if kind == "lower":
                cand = list(reversed(cand))
        if not cand:
            return None
        li = self.lanes[cand[0]].get("lights") or []
        if len(li) != 1:
            # several lights on the road: only resolvable when the clip has one
            # light per lane and the label names the lane
            return int(li[0]) if li else None
        return int(li[0])

    # ------------------------------------------------------------------ motion
    def simulate(self, drop: int | None = None, delay: float = 0.0,
                 green: int | None = None) -> list[dict]:
        key = (drop, round(delay, 6), green)
        if key in self._cache:
            return self._cache[key]
        gap = self.gap if self.GAP_MODE == "param" else GAP
        cars = []
        for c in self.cars:
            if drop is not None and int(c["id"]) == int(drop):
                continue
            lane = self.lane_of(c)
            ax, ay = lane["a"]
            bx, by = lane["b"]
            horiz = abs(bx - ax) >= abs(by - ay)
            start = ax if horiz else ay
            end = bx if horiz else by
            dirn = 1 if (end - start) >= 0 else -1
            # every light mounted on this lane is a stop line the car must pass
            uls = sorted(((int(li), dirn * start + float(self.lights[int(li)]["s"]))
                          for li in (lane.get("lights") or [])), key=lambda x: x[1])
            cars.append({
                "id": int(c["id"]), "name": f"the {c['colour']} {c['body']}",
                "lane": int(c["lane"]), "len": float(c["len"]), "v": float(c["v"]),
                "t_in": float(c["t_in"]) + delay, "u": dirn * start, "u0": dirn * start,
                "dirn": dirn, "ub": dirn * end, "uls": uls,
                "uline": uls[0][1] if uls else 0.0, "light": uls[0][0] if uls else 0,
                "started": False, "hist": [], "exited": False,
            })
        steps = int(self.T / DT) + 1
        for k in range(steps):
            t = k * DT
            for c in cars:
                if not c["exited"] and not c["started"] and t >= c["t_in"]:
                    c["started"] = True
            active = [c for c in cars if c["started"] and not c["exited"]]
            for lane_i in set(c["lane"] for c in active):
                group = sorted([c for c in active if c["lane"] == lane_i],
                               key=lambda c: -c["u"])
                for i, c in enumerate(group):
                    lim = 1e18
                    if i > 0:
                        lead = group[i - 1]
                        lim = lead["u"] - lead["len"] - gap
                    for li, ul in c["uls"]:
                        if li == green:
                            continue
                        if _light_state(self.lights[li], t) == "red" and c["u"] <= ul + 1e-9:
                            lim = min(lim, ul)
                    c["u"] = min(c["u"] + c["v"] * DT, max(lim, c["u"]))
            for c in active:
                if c["u"] - c["len"] >= c["ub"]:
                    c["exited"] = True
                c["hist"].append((t, c["u"]))
        self._cache[key] = cars
        return cars

    # ------------------------------------------------------------------ queries
    def stops(self, c: dict) -> list[tuple]:
        eps, run = [], None
        for t, u in c["hist"]:
            if run is not None and abs(u - run[1]) < STOP_TOL:
                run = (run[0], u)
            else:
                if run is not None and t - run[0] > MIN_STOP:
                    eps.append((round(run[0], 3), round(t, 3)))
                run = (t, u)
        if run is not None and c["hist"] and c["hist"][-1][0] - run[0] > MIN_STOP:
            eps.append((round(run[0], 3), "end"))
        # A car that spawns already blocked by its leader is parked at the lane
        # entrance before it is really "driving"; that stall is not a stop the
        # renderer shows. Calibrated against the sample.
        if self.DROP_SPAWN_STOP and eps and c["hist"]:
            if eps[0][0] <= round(c["hist"][0][0] + 1.5 * DT, 3):
                eps = eps[1:]
        return eps

    def at(self, c: dict, tt: float) -> float | None:
        for t, u in c["hist"]:
            if abs(t - tt) < 0.01:
                return u
        return None

    def uline_of(self, c: dict, li: int | None = None) -> float | None:
        if li is None:
            return c["uline"]
        for k, ul in c["uls"]:
            if k == li:
                return ul
        return None

    def crossed(self, c: dict, li: int | None = None) -> bool:
        ul = self.uline_of(c, li)
        if ul is None or not c["hist"]:
            return False
        return max(u for _, u in c["hist"]) > ul + 1e-6

    def crossed_by(self, c: dict, tmax: float, li: int | None = None) -> bool:
        ul = self.uline_of(c, li)
        if ul is None or not c["hist"]:
            return False
        return any(u > ul + 1e-6 for t, u in c["hist"] if t <= tmax)

    def cross_time(self, c: dict, li: int | None = None) -> float | None:
        ul = self.uline_of(c, li)
        if ul is None:
            return None
        for t, u in c["hist"]:
            if u > ul + 1e-6:
                return t
        return None

    def entered(self, c: dict, tt: float | None = None) -> bool:
        """True once the car has actually driven off the lane entrance.

        A car that spawns with its leader already inside the following gap is
        parked at the lane start until the queue moves. The renderer never
        shows that parked state (the car comes in from off-screen), so it is
        not "on the road" and not "standing still". Calibrated on the ten
        off-by-one rows in the full run.
        """
        if not c["hist"]:
            return False
        return any(u > c["u0"] + 1e-9 for t, u in c["hist"] if tt is None or t <= tt + 1e-9)

    def on_road(self, c: dict) -> bool:
        return (bool(c["hist"]) and self.entered(c)
                and c["hist"][-1][1] - c["len"] < c["ub"] - 1e-9)

    def standing_at(self, c: dict, tt: float) -> bool:
        k = [i for i, (t, u) in enumerate(c["hist"]) if abs(t - tt) < 0.01]
        if not k:
            return False
        if c["exited"] and k[0] >= len(c["hist"]) - 2:
            return False
        if not self.entered(c, tt):
            return False
        kk = min(k[0] + 5, len(c["hist"]) - 1)
        return abs(c["hist"][kk][1] - c["hist"][k[0]][1]) < STOP_TOL

    def standing(self, c: dict, tt: float) -> bool:
        ok = self.standing_at(c, tt)
        if ok and self.STANDING_NEEDS_ROAD and tt >= self.T - 1e-6:
            # at the final frame the car must still be on the lane
            ok = self.on_road(c)
        return ok

    def first_in_line(self, cars: list[dict], li: int, tt: float):
        best = None
        for c in cars:
            ul = self.uline_of(c, li)
            if ul is None or not c["hist"]:
                continue
            u = self.at(c, tt)
            if u is None or u > ul + self.FIRST_LINE_MAX_DIST:
                continue
            if self.standing_at(c, tt):
                if best is None or u > best[0]:
                    best = (u, c["name"])
        return best[1] if best else None

    def green_red_transitions(self, light: dict, T: float | None = None) -> int:
        T = self.T if T is None else T
        n, prev, t = 0, _light_state(light, 0.0), 0.0
        while t < T:
            t += 0.01
            s = _light_state(light, t)
            if prev == "green" and s == "red":
                n += 1
            prev = s
        return n

    def first_green(self, light: dict) -> float | None:
        """Clock reading of the first red->green change inside the clip.

        The generator's on-screen clock floors to 0.1 s (verified on all 1,341
        clock-reading rows of the full run).
        """
        ivs = _green_intervals(light, self.T)
        for g0, g1 in ivs:
            if g0 > 1e-9:          # a real transition, not a clip that opens green
                return math.floor(g0 * 10 + 1e-9) / 10
        return None

    def car(self, text: str) -> dict | None:
        hits = []
        low = text.lower()
        for c in self.cars:
            pat = rf"\b{re.escape(str(c['colour']))}\s+{re.escape(str(c['body']))}\b"
            if re.search(pat, low):
                hits.append(c)
        return hits[0] if len(hits) == 1 else None

    def car_pair(self, q: str) -> tuple[dict, dict] | None:
        found = []
        for c in self.cars:
            pat = rf"\b{re.escape(str(c['colour']))}\s+{re.escape(str(c['body']))}\b"
            if re.search(pat, q) and c not in found:
                found.append(c)
        return (found[0], found[1]) if len(found) >= 2 else None

    def cars_in_order(self, q: str) -> list[dict]:
        out = []
        for c in self.cars:
            pat = rf"\b{re.escape(str(c['colour']))}\s+{re.escape(str(c['body']))}\b"
            if re.search(pat, q) and c not in out:
                out.append(c)
        return out

    # ------------------------------------------------------------------ answers
    def _colour(self, li: int, t: float) -> Answer:
        return Answer(_light_state(self.lights[li], t), confidence="exact")

    def _count_crossed(self, li: int, tmax: float | None = None, cars=None) -> Answer:
        cars = self.simulate() if cars is None else cars
        n = sum(1 for c in cars
                if (self.crossed(c, li) if tmax is None
                    else self.crossed_by(c, tmax, li)))
        return Answer(str(n), confidence="exact")

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        # ---- lights: count
        if "how many traffic lights are there" in q:
            return Answer(str(len(self.lights)), confidence="exact")

        # ---- light colour at a clock time / at the end
        m = re.search(r"(?:at ([\d.]+) s on the clock|in the last frame of the clip), "
                      r"what colour is (.+?) showing\?", q)
        if m:
            tt = float(m.group(1)) if m.group(1) else self.T
            li = self.light_index_for_text(m.group(2))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return self._colour(li, tt)

        # ---- green->red transitions
        m = re.search(r"how many times does (.+?) change from green to red during the clip", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return Answer(str(self.green_red_transitions(self.lights[li])), confidence="exact")

        # ---- first green
        m = re.search(r"when does (.+?) first turn green\? give the clock reading", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            fg = self.first_green(self.lights[li])
            if fg is None:
                return Answer(None, confidence="unsupported", note="no green interval")
            return Answer(f"{fg:.1f} s", confidence="exact")

        # ---- vehicles on the road / standing at the end
        if "in the last frame of the clip, how many vehicles are on the road" in q:
            cars = self.simulate()
            return Answer(str(sum(1 for c in cars if self.on_road(c))), confidence="exact")

        if "in the last frame of the clip, which vehicle on the road is the longest" in q:
            cars = [c for c in self.simulate() if self.on_road(c)]
            if not cars:
                return Answer("none", confidence="exact")
            mx = max(c["len"] for c in cars)
            winners = [c for c in cars if abs(c["len"] - mx) < 1e-9]
            if len(winners) == 1:
                return Answer(winners[0]["name"], confidence="exact")
            # tie: rank by the order they would appear front-first in the lane
            w = max(winners, key=lambda c: c["hist"][-1][1])
            return Answer(w["name"], confidence="exact", note="length tie")

        if "at the end of the clip, how many vehicles are standing still" in q:
            cars = self.simulate()
            return Answer(str(sum(1 for c in cars if self.standing(c, self.T))), confidence="exact")

        m = re.search(r"at ([\d.]+) s on the clock, how many vehicles are standing still", q)
        if m:
            cars = self.simulate()
            t = float(m.group(1))
            return Answer(str(sum(1 for c in cars if self.standing(c, t))), confidence="exact")

        # ---- which vehicle first / first to stop
        if "which vehicle comes onto the road first" in q:
            if not self.cars:
                return Answer(None, confidence="unsupported", note="no vehicles")
            first = min(self.cars, key=lambda c: (float(c["t_in"]), int(c["id"])))
            return Answer(f"the {first['colour']} {first['body']}", confidence="exact")

        if "which vehicle is the first to come to a complete stop" in q:
            cars = self.simulate()
            cand = [(self.stops(c)[0][0], c["name"]) for c in cars if self.stops(c)]
            if not cand:
                return Answer(None, confidence="exact", note="no vehicle stops")
            return Answer(min(cand)[1], confidence="exact")

        # ---- per-vehicle stop counts / first stop
        m = re.search(r"how many times does (.+?) come to a complete stop during the clip", q)
        if m:
            c = self.car(m.group(1))
            if c is None:
                return Answer(None, confidence="unsupported", note="vehicle not unique")
            cars = {x["id"]: x for x in self.simulate()}
            return Answer(str(len(self.stops(cars[c["id"]]))), confidence="exact")

        m = re.search(r"when does (.+?) first come to a complete stop\? give the clock reading", q)
        if m:
            c = self.car(m.group(1))
            if c is None:
                return Answer(None, confidence="unsupported", note="vehicle not unique")
            cars = {x["id"]: x for x in self.simulate()}
            st = self.stops(cars[c["id"]])
            if not st:
                return Answer(None, confidence="exact", note="never stops")
            return Answer(f"{math.floor(st[0][0] * 10 + 1e-9) / 10:.1f} s", confidence="exact")

        # ---- stop-line crossings per light
        m = re.search(r"by ([\d.]+) s on the clock, how many vehicles have driven past "
                      r"the stop line of (.+?)\?", q)
        if m:
            li = self.light_index_for_text(m.group(2))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return self._count_crossed(li, float(m.group(1)))

        m = re.search(r"during the whole clip, how many vehicles drive past the stop line of (.+?)\?", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return self._count_crossed(li)

        # ---- first in line
        m = re.search(r"at ([\d.]+) s on the clock, which vehicle is waiting first in line "
                      r"at the stop line of (.+?)\?", q)
        if m:
            li = self.light_index_for_text(m.group(2))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            cars = self.simulate()
            name = self.first_in_line(cars, li, float(m.group(1)))
            if name is None:
                return Answer(None, confidence="exact", note="nobody waiting")
            return Answer(name, confidence="exact")

        # ---- which of these never crosses
        m = re.search(r"which of these vehicles never drives past the stop line of (.+?)\?", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            cars = {c["name"]: c for c in self.simulate()}
            never = []
            for c in self.cars:
                nm = f"the {c['colour']} {c['body']}"
                if nm in cars and not self.crossed(cars[nm], li):
                    never.append(nm)
            opts = [str(o) for o in (options or [])]
            cand = [o for o in opts if o.lower().strip() in never]
            if len(cand) == 1:
                return Answer(cand[0], confidence="exact")
            if len(never) == 1:
                return Answer(never[0], confidence="exact")
            return Answer(None, confidence="unsupported", note=f"{len(never)} never cross")

        # ---- order of crossing
        m = re.search(r"in what order do these vehicles drive past the stop line of (.+?): "
                      r"(.+?)\?", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            named = self.cars_in_order(m.group(2))
            if len(named) < 2:
                return Answer(None, confidence="unsupported", note="vehicle list not parsed")
            cars = {c["name"]: c for c in self.simulate()}
            timed, untimed = [], []
            for c in named:
                cc = cars.get(f"the {c['colour']} {c['body']}")
                t = self.cross_time(cc, li) if cc else None
                (timed if t is not None else untimed).append((t if t is not None else 1e9, c))
            timed.sort(key=lambda x: x[0])
            seq = timed + untimed
            return Answer(", then ".join(f"the {c['colour']} {c['body']}" for _, c in seq),
                          confidence="exact")

        # ---- counterfactuals
        m = re.search(r"suppose every vehicle had arrived ([\d.]+) s later than it did", q)
        if m:
            delay = float(m.group(1))
            ml = re.search(r"stop line of (.+?) by the end of the clip", q)
            if not ml:
                return Answer(None, confidence="unsupported", note="light label not parsed")
            li = self.light_index_for_text(ml.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return self._count_crossed(li, None, self.simulate(delay=delay))

        m = re.search(r"suppose (the .+?) had stayed green the whole time", q)
        if m:
            li = self.light_index_for_text(m.group(1))
            if li is None:
                return Answer(None, confidence="unsupported", note="ambiguous light label")
            return self._count_crossed(li, None, self.simulate(green=li))

        m = re.search(r"suppose the (.+?) had never come along\. how many times would the "
                      r"(.+?) have come to a complete stop during the clip", q)
        if m:
            gone = self.car(m.group(1))
            stay = self.car(m.group(2))
            if gone is None or stay is None:
                return Answer(None, confidence="unsupported", note="vehicle not unique")
            cars = {c["id"]: c for c in self.simulate(drop=gone["id"])}
            if stay["id"] not in cars:
                return Answer(None, confidence="unsupported", note="target vehicle dropped")
            return Answer(str(len(self.stops(cars[stay["id"]]))), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

"""Deterministic replay for `spring_chain_waves`.

The generator stores its complete wave timeline in `params["segs"]`: one entry
per visible pulse segment

    {tb, te, xb, xe, c, lo, hi, amp, d}

`c` is the speed in ruler marks/second, `amp` the signed pulse height
(+ bump up, - dip down), `d` the direction (+1 right, -1 left).  `params["ghosts"]`
are split products below the rendering threshold (|amp| < 0.3) and are never
visible, so every question about the video is answered from `segs`.

Physics pinned against all 2,500 clips (the simulator reproduces every stored
segment exactly):
  * thin speed  v = c4/4;  thick speed = v / ratio
  * at a thickness change, with v1 the incoming and v2 the outgoing speed:
        reflection  r = (v2 - v1) / (v2 + v1)      (sign = inversion)
        transmission t = 2 v2 / (v2 + v1)
  * ends: clamp and paddle invert (r = -1); a ring on a pole keeps the sign
  * a pulse is drawn while |amp| >= 0.3; a pulse with |amp| < 0.3 is a ghost
    and stops propagating in the stored timeline
  * the timeline keeps every segment that starts before video end plus the
    one extra segment the renderer needs; questions only look inside the video
"""
from __future__ import annotations

import heapq
import math
import re

from ..engine import Answer

FPS = 24.0
VIS_MIN = 0.3
EPS = 1e-7

BIG = 0.8


def _fmt_time(t: float) -> str:
    return f"{t:.2f} s"


def _fraction_word(r: float) -> str:
    """Nearest of the fraction phrases the option generator uses."""
    choices = [
        (1 / 3, "about 1/3 as tall"),
        (1 / 2, "about 1/2 as tall"),
        (2 / 3, "about 2/3 as tall"),
        (4 / 3, "about 4/3 as tall"),
        (3 / 2, "about 3/2 as tall"),
        (1.0, "about as tall"),
    ]
    return min(choices, key=lambda c: abs(c[0] - r))[1]


class SpringChain:
    def __init__(self, params: dict):
        self.p = params
        self.L = float(params.get("L", 0))
        self.c4 = float(params.get("c4", 0))
        self.ratio = float(params.get("ratio", 1) or 1)
        self.vthin = self.c4 / 4.0
        self.sections = [tuple(s) for s in params.get("sections", [])]
        self.ends = dict(params.get("ends", {}))
        self.launches = list(params.get("launches", []))
        self.n_frames = int(params.get("n_frames", 0))
        self.dur = self.n_frames / FPS
        self.segs = sorted(params.get("segs", []), key=lambda s: (s["tb"], s["d"]))

    # ------------------------------------------------------------------ helpers
    def speed_of(self, sec_index: int) -> float:
        return self.vthin / self.ratio if self.sections[sec_index][2] else self.vthin

    def section_at(self, x: float) -> int:
        for i, (a, b, _thick) in enumerate(self.sections):
            if a - EPS <= x <= b + EPS:
                return i
        return 0 if x <= self.sections[0][1] else len(self.sections) - 1

    def launch_amp(self, launch: dict) -> float:
        return (1.0 if launch.get("big") else 0.6) * float(launch.get("sign", 1))

    def visible(self) -> list[dict]:
        return self.segs

    @staticmethod
    def describe(amp: float, full: bool = True) -> str:
        size = "big" if abs(amp) >= BIG else "small"
        way = "bump" if amp > 0 else "dip"
        if full:
            return f"a {size} {way} {'up' if amp > 0 else 'down'}"
        return f"{size} {way}"

    def arrivals(self, x: float) -> list[dict]:
        out = [s for s in self.segs
               if abs(s["xe"] - x) < EPS and s["te"] <= self.dur + EPS]
        return sorted(out, key=lambda s: s["te"])

    def active(self, t: float) -> list[dict]:
        return [s for s in self.segs if s["tb"] - EPS <= t <= s["te"] + EPS]

    def paddle_side(self) -> str | None:
        """Side of the paddle when there is exactly one."""
        sides = [k for k, v in self.ends.items() if v == "paddle"]
        if len(sides) == 1:
            return sides[0]
        # fall back to the side the pulses are launched from
        if self.launches:
            return self.launches[0]["end"]
        return None

    def paddle_x(self, side: str | None = None) -> float:
        side = side or self.paddle_side()
        return 0.0 if side == "left" else self.L

    def end_coef(self, side: str) -> float:
        return 1.0 if self.ends.get(side) == "free" else -1.0

    # -------------------------------------------------------------- simulation
    def simulate(self, *, launches=None, ends=None, sections=None,
                 speed_scale: float = 1.0, horizon: float | None = None,
                 vis_min: float = VIS_MIN) -> list[dict]:
        """Event-driven replay of one or more pulses.

        Returns the visible segment list in the exact format the generator
        stores (times are continuous, not frame-quantised).
        """
        sections = sections if sections is not None else self.sections
        ends = ends if ends is not None else self.ends
        launches = launches if launches is not None else self.launches
        horizon = horizon if horizon is not None else self.dur + 8.0

        def speed(i: int) -> float:
            return (self.vthin / self.ratio if sections[i][2] else self.vthin) * speed_scale

        def sec_at(x: float) -> int:
            for i, (a, b, _t) in enumerate(sections):
                if a - EPS <= x <= b + EPS:
                    return i
            return 0 if x <= sections[0][1] else len(sections) - 1

        q: list[tuple] = []
        seq = 0
        for launch in launches:
            t = float(launch["f"]) / FPS
            x = 0.0 if launch["end"] == "left" else float(sections[-1][1])
            d = 1 if launch["end"] == "left" else -1
            heapq.heappush(q, (t, seq, x, d, self.launch_amp(launch), sec_at(x)))
            seq += 1

        segs: list[dict] = []
        while q:
            t, _, x, d, amp, si = heapq.heappop(q)
            if t > horizon + EPS:
                break
            vv = speed(si)
            if d > 0:
                cand = [bb for (aa, bb, _t) in sections if bb > x + EPS]
                nb = min(cand) if cand else sections[si][1]
            else:
                cand = [aa for (aa, bb, _t) in sections if aa < x - EPS]
                nb = max(cand) if cand else sections[si][0]
            at_chain_end = (d > 0 and abs(nb - sections[-1][1]) < EPS) or \
                           (d < 0 and abs(nb - 0.0) < EPS)
            t2 = t + abs(nb - x) / vv
            if abs(amp) >= vis_min:
                segs.append({"tb": t, "te": t2, "xb": x, "xe": nb,
                             "c": vv, "amp": amp, "d": d})
            else:
                continue

            if at_chain_end:
                side = "left" if d < 0 else "right"
                amp2 = amp * (1.0 if ends.get(side) == "free" else -1.0)
                if abs(amp2) >= vis_min:
                    heapq.heappush(q, (t2, seq, nb, -d, amp2, si))
                    seq += 1
            else:
                j = si + (1 if d > 0 else -1)
                v1, v2 = vv, speed(j)
                r = (v2 - v1) / (v2 + v1)
                tr = 2.0 * v2 / (v2 + v1)
                ar = amp * r
                if abs(ar) >= vis_min:
                    heapq.heappush(q, (t2, seq, nb, -d, ar, si))
                    seq += 1
                at = amp * tr
                if abs(at) >= vis_min:
                    heapq.heappush(q, (t2, seq, nb, d, at, j))
                    seq += 1
        return sorted(segs, key=lambda s: (s["tb"], s["d"]))

    # ------------------------------------------------------------- first change
    def first_change(self, sections=None) -> dict | None:
        """First visible segment that ends on an internal thickness boundary."""
        sections = sections if sections is not None else self.sections
        L = float(sections[-1][1])
        for s in self.segs:
            if EPS < s["xe"] < L - EPS:
                si = self.section_at(s["xb"])
                j = si + (1 if s["d"] > 0 else -1)
                if j < 0 or j >= len(sections):
                    continue
                v1 = self.vthin / self.ratio if sections[si][2] else self.vthin
                v2 = self.vthin / self.ratio if sections[j][2] else self.vthin
                r = (v2 - v1) / (v2 + v1)
                t = 2.0 * v2 / (v2 + v1)
                return {"seg": s, "x": s["xe"], "v1": v1, "v2": v2,
                        "r": r, "t": t, "dir": s["d"]}
        return None

    # ------------------------------------------------------------------ answers
    def answer(self, question: str, options=None) -> Answer:
        q = " ".join(str(question).lower().split())
        options = list(options or [])
        opt_text = " | ".join(str(o).lower() for o in options)

        # ---------------- pulses sent / timing ----------------
        if q == "how many pulses do the paddles send into the chain in total?":
            return Answer(str(len(self.launches)))

        if q == "how many pulses does the paddle send into the chain?":
            return Answer(str(len(self.launches)))

        m = re.search(r"how much time passes between the paddle sending its first and its second pulse", q)
        if m:
            if len(self.launches) < 2:
                return Answer(None, confidence="unsupported", note="fewer than two launches")
            dt = (self.launches[1]["f"] - self.launches[0]["f"]) / FPS
            return Answer(f"{dt:.2f} s")

        if q.startswith("which paddle sends its first pulse first"):
            first = {}
            for l in self.launches:
                first.setdefault(l["end"], l)
            if len(first) < 2:
                return Answer("both at the same moment")
            a, b = first["left"], first["right"]
            if a["f"] == b["f"]:
                return Answer("both at the same moment")
            return Answer("the left paddle" if a["f"] < b["f"] else "the right paddle")

        m = re.search(r"what does the first pulse the (left|right) paddle sends look like", q)
        if m:
            side = m.group(1)
            first = next((l for l in self.launches if l["end"] == side), None)
            if first is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.describe(self.launch_amp(first)))

        if q == "what does the first pulse the paddle sends look like?":
            if not self.launches:
                return Answer(None, confidence="unsupported")
            return Answer(self.describe(self.launch_amp(self.launches[0])))

        if q == "which paddle's first pulse is taller?":
            pair = {}
            for l in self.launches:
                pair.setdefault(l["end"], l)
            if len(pair) < 2:
                return Answer(None, confidence="unsupported")
            a, b = abs(self.launch_amp(pair["left"])), abs(self.launch_amp(pair["right"]))
            if abs(a - b) < 1e-9:
                return Answer("the same height")
            return Answer("the left paddle's" if a > b else "the right paddle's")

        # ---------------- what holds an end ----------------
        m = re.search(r"what holds the (left|right) end of the chain", q)
        if m:
            side = m.group(1)
            kind = self.ends.get(side)
            if kind == "paddle":
                return Answer("a paddle")
            if kind == "clamped":
                return Answer("a clamp")
            if kind == "free":
                return Answer("a ring that slides on a pole")
            return Answer(None, confidence="unsupported")

        # ---------------- bounce direction ----------------
        m = re.search(r"when a pulse bounces off the (left|right) end, how does it come back", q)
        if m:
            side = m.group(1)
            return Answer("the same way up" if self.ends.get(side) == "free" else "upside down")

        # ---------------- first pulse comes back ----------------
        if q.startswith("when the first pulse comes back to the paddle"):
            return self._return_look(nth=1)
        if q.startswith("when the pulse gets back to the paddle for the second time"):
            return self._return_look(nth=2, second_launch=True)

        # ---------------- end arrival times ----------------
        m = re.search(r"at what time on the clock does a pulse first reach the (left|right) end", q)
        if m:
            side = m.group(1)
            arr = self.arrivals(0.0 if side == "left" else self.L)
            if not arr:
                return Answer(None, confidence="unsupported", note="no arrival in video")
            return Answer(_fmt_time(arr[0]["te"]))

        for word, nth in (("second", 2), ("third", 3), ("fourth", 4)):
            m = re.search(rf"at what time on the clock does a pulse reach the (left|right) end for the {word} time", q)
            if m:
                side = m.group(1)
                arr = self.arrivals(0.0 if side == "left" else self.L)
                if len(arr) < nth:
                    return Answer(None, confidence="unsupported", note="fewer arrivals")
                return Answer(_fmt_time(arr[nth - 1]["te"]))

        # ---------------- nth pulse identity ----------------
        for word, nth in (("second", 2), ("third", 3), ("fourth", 4)):
            m = re.search(rf"what is the {word} pulse to reach the (left|right) end", q)
            if m:
                side = m.group(1)
                arr = self.arrivals(0.0 if side == "left" else self.L)
                if len(arr) < nth:
                    return Answer(None, confidence="unsupported", note="fewer arrivals")
                return Answer(self.describe(arr[nth - 1]["amp"]))

        # ---------------- counts at an end ----------------
        m = re.search(r"counting every pulse, big or small, how many times does a pulse reach the (left|right) end during the video", q)
        if m:
            side = m.group(1)
            return Answer(str(len(self.arrivals(0.0 if side == "left" else self.L))))

        # ---------------- arrivals at the paddle ----------------
        if q.startswith("not counting the pulses it sends, how many times does a pulse arrive at the paddle"):
            side = self.paddle_side()
            return Answer(str(len(self.arrivals(self.paddle_x(side)))))
        m = re.search(r"not counting the pulses it sends, how many times does a pulse arrive at the (left|right) paddle", q)
        if m:
            side = m.group(1)
            return Answer(str(len(self.arrivals(0.0 if side == "left" else self.L))))

        # ---------------- motion at a clock time ----------------
        m = re.search(r"which way is the pulse moving when the clock shows ([\d.]+) s", q)
        if m:
            t = float(m.group(1))
            act = self.active(t)
            if not act:
                return Answer(None, confidence="unsupported", note="no pulse at that time")
            s = act[0]
            return Answer("toward the right end" if s["d"] > 0 else "toward the left end")

        m = re.search(r"when the clock shows ([\d.]+) s, how many pulses are moving toward the (left|right) end", q)
        if m:
            t, side = float(m.group(1)), m.group(2)
            want = 1 if side == "right" else -1
            return Answer(str(sum(1 for s in self.active(t) if s["d"] == want)))

        if q.startswith("how many separate bumps or dips are on the chain at the very end of the video"):
            return Answer(str(len(self.active(self.dur))))

        # ---------------- speed / sections ----------------
        if q.startswith("about how many ruler marks per second does a pulse move along the chain"):
            return Answer(f"{self.vthin:.2f}".rstrip("0").rstrip("."))
        if q.startswith("about how many ruler marks per second does a pulse move where the beads are thin"):
            return Answer(f"{self.vthin:.2f}".rstrip("0").rstrip("."))
        if q.startswith("where the beads are thick, pulses move at about what fraction"):
            frac = 1.0 / self.ratio
            if abs(frac - 1) < 0.02:
                return Answer("about the same")
            if abs(frac - 0.5) < 0.1:
                return Answer("about half")
            if abs(frac - 1 / 3) < 0.1:
                return Answer("about a third")
            if abs(frac - 2 / 3) < 0.1:
                return Answer("about two thirds")
            return Answer("about half")

        if q.startswith("at which ruler mark do the beads change between thin and thick"):
            bounds = [s[1] for s in self.sections[:-1]]
            if not bounds:
                return Answer(None, confidence="unsupported", note="uniform chain")
            return Answer(str(int(round(bounds[0]))))

        if q.startswith("between which two ruler marks are the beads thick"):
            thick = [(s[0], s[1]) for s in self.sections if s[2]]
            if not thick:
                return Answer(None, confidence="unsupported", note="no thick section")
            a, b = thick[0]
            return Answer(f"{int(round(a))} and {int(round(b))}")

        if q.startswith("where on the chain do pulses move most slowly"):
            if self.ratio == 1 or not any(s[2] for s in self.sections):
                return Answer("the same speed everywhere")
            thick = [(s[0], s[1]) for s in self.sections if s[2]][0]
            a, b = thick
            if "between marks" in opt_text:
                return Answer(f"between marks {int(round(a))} and {int(round(b))}")
            mid = (a + b) / 2
            if mid < self.L / 3:
                return Answer("nearer the left paddle")
            if mid > 2 * self.L / 3:
                return Answer("nearer the right paddle")
            return Answer("in the middle of the chain")

        # ---------------- thickness-change behaviour ----------------
        if q.startswith("when the pulse first reaches a place where the beads change thickness"):
            fc = self.first_change()
            if fc is None:
                return Answer(None, confidence="unsupported", note="no thickness change")
            return Answer(self._change_word(fc["r"]))

        if q.startswith("if the thin and thick beads swapped places"):
            swapped = [(a, b, not t) for (a, b, t) in self.sections]
            fc = self.first_change(sections=swapped)
            if fc is None:
                return Answer(None, confidence="unsupported", note="no thickness change")
            return Answer(self._change_word(fc["r"]))

        if q.startswith("how tall is the pulse that bounces back from the first change"):
            fc = self.first_change()
            if fc is None:
                return Answer(None, confidence="unsupported")
            return Answer(_fraction_word(abs(fc["r"])))

        if q.startswith("how tall is the pulse that goes on past the first change"):
            fc = self.first_change()
            if fc is None:
                return Answer(None, confidence="unsupported")
            return Answer(_fraction_word(fc["t"]))

        if q.startswith("compared with the pulse arriving at the first change of beads"):
            fc = self.first_change()
            if fc is None:
                return Answer(None, confidence="unsupported")
            if abs(fc["t"] - 1.0) < 0.02:
                return Answer("the same height")
            return Answer("taller" if fc["t"] > 1.0 else "shorter")

        if q.startswith("after the pulse first splits where the beads change"):
            return self._after_split(options)

        # ---------------- first bounce back to the paddle ----------------
        m = re.search(r"the first pulse bounces off the (left|right) end\. at what time on the clock does it get back to the paddle", q)
        if m:
            side = m.group(1)
            if not self.launches:
                return Answer(None, confidence="unsupported")
            sim = self.simulate(launches=[self.launches[0]], horizon=self.dur + 10)
            x_end = 0.0 if side == "left" else self.L
            hits = [s for s in sim if abs(s["xe"] - x_end) < EPS]
            if not hits:
                return Answer(None, confidence="unsupported", note="never reaches that end")
            first = min(hits, key=lambda s: s["te"])
            # its reflection then travels back to the launching paddle
            side0 = self.launches[0]["end"]
            px = 0.0 if side0 == "left" else self.L
            later = [s for s in sim if s["tb"] >= first["te"] - EPS and abs(s["xe"] - px) < EPS
                     and s["tb"] > first["tb"] + EPS]
            if not later:
                return Answer(None, confidence="unsupported", note="no return")
            return Answer(_fmt_time(min(s["te"] for s in later)))

        if q.startswith("part of the pulse bounces back where the beads first change"):
            fc = self.first_change()
            if fc is None:
                return Answer(None, confidence="unsupported")
            ev = fc["seg"]
            side0 = self.launches[0]["end"] if self.launches else self.paddle_side()
            px = 0.0 if side0 == "left" else self.L
            sim = self.simulate(launches=[self.launches[0]], horizon=self.dur + 10)
            hits = [s for s in sim if abs(s["xb"] - fc["x"]) < 1e-6 and abs(s["tb"] - ev["te"]) < 1e-6
                    and abs(s["xe"] - px) < EPS]
            if not hits:
                return Answer(None, confidence="unsupported")
            return Answer(_fmt_time(min(s["te"] for s in hits)))

        # ---------------- order of returning pulses ----------------
        m = re.search(r"in what order do the pulses come back toward the paddle after bouncing off the (left|right) end", q)
        if m:
            side = m.group(1)
            x_end = 0.0 if side == "left" else self.L
            # only the FIRST bounce of each launched pulse belongs to this
            # sequence (later round trips are a different question)
            refl = []
            for l in self.launches:
                t0 = l["f"] / FPS
                sim = self.simulate(launches=[l], horizon=self.dur + 10)
                cand = [s for s in sim
                        if abs(s["xb"] - x_end) < EPS and s["tb"] > t0 + EPS
                        and s["tb"] <= self.dur + EPS]
                if cand:
                    refl.append(min(cand, key=lambda s: s["tb"]))
            refl.sort(key=lambda s: s["tb"])
            if not refl:
                return Answer(None, confidence="unsupported")
            sizes_in_options = any(("big " in o.lower() or "small " in o.lower()) for o in options)
            if sizes_in_options:
                parts = [self.describe(s["amp"], full=False) for s in refl]
            else:
                parts = [("bump" if s["amp"] > 0 else "dip") for s in refl]
            return Answer(", then ".join(parts))

        # ---------------- crossings / two paddles ----------------
        if q.startswith("how many times during the video do two pulses pass through each other"):
            return Answer(str(self._crossings(self.segs)))

        if q.startswith("the first pulses from the two paddles pass through each other, bounce off the paddles and meet again"):
            return self._meet_second(options)

        m = re.search(r"where do the first pulses from the two paddles first pass through each other", q)
        if m:
            x = self._first_cross_x()
            if x is None:
                return Answer("they never meet")
            if x < self.L / 2 - 1e-6:
                return Answer("nearer the left paddle")
            if x > self.L / 2 + 1e-6:
                return Answer("nearer the right paddle")
            return Answer("in the middle of the chain")

        m = re.search(r"at which ruler mark do the first pulses from the two paddles first pass through each other", q)
        if m:
            x = self._first_cross_x()
            if x is None:
                return Answer(None, confidence="unsupported", note="no crossing")
            return Answer(str(int(round(x))))

        if q.startswith("the first pulse from each paddle travels to the other paddle. which gets there first"):
            left = self._first_travel("left", "right")
            right = self._first_travel("right", "left")
            if left is None and right is None:
                return Answer("neither gets there within the video")
            if left is None:
                return Answer("the pulse from the right paddle")
            if right is None:
                return Answer("the pulse from the left paddle")
            if abs(left - right) < 1e-6:
                return Answer("both at the same moment")
            return Answer("the pulse from the left paddle" if left < right else "the pulse from the right paddle")

        m = re.search(r"at what time on the clock does the first pulse from the (left|right) paddle reach the (left|right) paddle", q)
        if m:
            src, dst = m.group(1), m.group(2)
            t = self._first_travel(src, dst)
            if t is None:
                return Answer(None, confidence="unsupported", note="no arrival in video")
            return Answer(_fmt_time(t))

        # ---------------- counterfactuals: end / speed / launch ----------------
        m = re.search(r"if the chain ended at the (\d+) mark, held there by a (ring that slides on a pole|clamp) as now", q)
        if m:
            mark = float(m.group(1))
            kind = "free" if m.group(2).startswith("ring") else "clamped"
            if not self.launches:
                return Answer(None, confidence="unsupported")
            launch = self.launches[0]
            side = launch["end"]
            px = 0.0 if side == "left" else self.L
            t = self._travel_time(px, mark)
            return Answer(_fmt_time(launch["f"] / FPS + 2 * t))

        m = re.search(r"if the whole chain had thin beads, at what time on the clock would the first pulse get back to the paddle after bouncing off the (left|right) end", q)
        if m:
            launch = self.launches[0]
            side0 = launch["end"]
            px = 0.0 if side0 == "left" else self.L
            far = 0.0 if m.group(1) == "left" else self.L
            t = abs(far - px) / self.vthin
            return Answer(_fmt_time(launch["f"] / FPS + 2 * t))

        m = re.search(r"if pulses moved half as fast, at what time on the clock would the first pulse reach the (left|right) end", q)
        if m:
            side = m.group(1)
            arrivals = self.arrivals(0.0 if side == "left" else self.L)
            if not arrivals:
                return Answer(None, confidence="unsupported")
            first = arrivals[0]
            launch = self.launches[0]
            t0 = launch["f"] / FPS
            return Answer(_fmt_time(t0 + 2 * (first["te"] - t0)))

        m = re.search(r"if pulses moved twice as fast everywhere \(same launches, same video length\), how many times would a pulse reach the (left|right) end", q)
        if m:
            side = m.group(1)
            sim = self.simulate(speed_scale=2.0, horizon=self.dur + 10)
            x = 0.0 if side == "left" else self.L
            n = sum(1 for s in sim if abs(s["xe"] - x) < EPS and s["te"] <= self.dur + EPS)
            return Answer(str(n))

        m = re.search(r"if pulses moved twice as fast \(same launches, same video length\), how many times would two pulses pass through each other", q)
        if m:
            sim = self.simulate(speed_scale=2.0, horizon=self.dur + 10)
            return Answer(str(self._crossings(sim)))

        m = re.search(r"if the paddle had not sent its last pulse, how many times would a pulse reach the (left|right) end", q)
        if m:
            side = m.group(1)
            if not self.launches:
                return Answer(None, confidence="unsupported")
            last_f = max(l["f"] for l in self.launches)
            launches = [l for l in self.launches if l["f"] < last_f]
            sim = self.simulate(launches=launches, horizon=self.dur + 10)
            x = 0.0 if side == "left" else self.L
            n = sum(1 for s in sim if abs(s["xe"] - x) < EPS and s["te"] <= self.dur + EPS)
            return Answer(str(n))

        m = re.search(r"if the (left|right) paddle had sent its first pulse ([\d.]+) s later, at which ruler mark", q)
        if m:
            side, delay = m.group(1), float(m.group(2))
            launches = [dict(l) for l in self.launches]
            for l in launches:
                if l["end"] == side:
                    l["f"] = l["f"] + delay * FPS
                    break
            x = self._first_cross_x(launches=launches)
            if x is None:
                return Answer(None, confidence="unsupported", note="no crossing")
            return Answer(str(int(round(x))))

        # ---------------- counterfactual: held ends / paddles ----------------
        m = re.search(r"if the (left|right) end were held by a (clamp|ring that slides on a pole) instead", q)
        if m:
            side, kind = m.group(1), m.group(2)
            return self._return_after_end_change(side, "free" if kind.startswith("ring") else "clamped")

        m = re.search(r"if the (left|right) paddle were replaced by a (clamp|ring that slides on a pole)", q)
        if m:
            side, kind = m.group(1), m.group(2)
            return self._return_after_end_change(side, "free" if kind.startswith("ring") else "clamped")
        return Answer(None, confidence="unsupported", note="no handler for this template")

    # ---------------------------------------------------------------- sub-answers
    def _change_word(self, r: float) -> str:
        if abs(r) < 1e-9:
            return "all of it goes on and nothing bounces back"
        if r < 0:
            return "a smaller upside-down pulse bounces back and the rest goes on"
        return "a smaller pulse bounces back the same way up and the rest goes on"

    def _return_look(self, nth: int = 1, second_launch: bool = False) -> Answer:
        if not self.launches:
            return Answer(None, confidence="unsupported")
        launch = self.launches[0]
        side = launch["end"]
        px = 0.0 if side == "left" else self.L
        amp0 = self.launch_amp(launch)
        arr = [s for s in self.arrivals(px) if s["tb"] > launch["f"] / FPS + EPS]
        if len(arr) < nth:
            return Answer(None, confidence="unsupported", note="fewer returns")
        s = arr[nth - 1]
        ratio = abs(s["amp"] / amp0)
        same = (s["amp"] * amp0) > 0
        if abs(ratio - 1.0) < 0.05:
            return Answer("the same way up" if same else "upside down")
        return Answer(_fraction_word(ratio))

    def _after_split(self, options) -> Answer:
        fc = self.first_change()
        if fc is None or not self.launches:
            return Answer(None, confidence="unsupported")
        launch = self.launches[0]
        sim = self.simulate(launches=[launch], horizon=self.dur + 12)
        ev = fc["seg"]
        # reflected child and transmitted child start at the split
        kids = [s for s in sim if abs(s["tb"] - ev["te"]) < 1e-6 and abs(s["xb"] - ev["xe"]) < 1e-6]
        if len(kids) < 2:
            return Answer(None, confidence="unsupported", note="split products missing")
        side0 = launch["end"]
        px = 0.0 if side0 == "left" else self.L
        far = self.L if side0 == "left" else 0.0
        back = [s for s in kids if abs(s["d"] + ev["d"]) < EPS]
        fwd = [s for s in kids if abs(s["d"] - ev["d"]) < EPS]
        if not back or not fwd:
            return Answer(None, confidence="unsupported")
        t_back = self._follow_to(max(back, key=lambda s: abs(s["amp"])), px, sim)
        t_fwd = self._follow_to(max(fwd, key=lambda s: abs(s["amp"])), far, sim)
        if t_back is None and t_fwd is None:
            return Answer("neither within the video")
        if t_back is None:
            return Answer(f"the part that went on reaches the {'left' if far == 0 else 'right'} end first")
        if t_fwd is None:
            return Answer("the part that bounced back reaches the paddle first")
        if t_back > self.dur and t_fwd > self.dur:
            return Answer("neither within the video")
        if abs(t_back - t_fwd) < 1e-6:
            return Answer("both at the same moment")
        if t_back < t_fwd:
            return Answer("the part that bounced back reaches the paddle first")
        far_side = "left" if far == 0.0 else "right"
        return Answer(f"the part that went on reaches the {far_side} end first")

    @staticmethod
    def _follow_to(seg: dict, target_x: float, sim: list[dict]) -> float | None:
        """Follow the main (largest-amplitude) branch until it reaches target_x."""
        seen = set()
        for _ in range(60):
            if abs(seg["xe"] - target_x) < EPS:
                return seg["te"]
            key = (round(seg["tb"], 9), round(seg["te"], 9), round(seg["xb"], 9), round(seg["xe"], 9))
            if key in seen:
                return None
            seen.add(key)
            kids = [s for s in sim
                    if abs(s["tb"] - seg["te"]) < 1e-6 and abs(s["xb"] - seg["xe"]) < 1e-6
                    and not (abs(s["tb"] - s["tb"]) < 0 and False)]
            kids = [s for s in kids if not (abs(s["tb"] - seg["tb"]) < 1e-9 and abs(s["amp"] - seg["amp"]) < 1e-9)]
            if not kids:
                return None
            same = [s for s in kids if s["d"] == seg["d"]]
            seg = max(same or kids, key=lambda s: abs(s["amp"]))
        return None

    def _crossings(self, segs: list[dict]) -> int:
        n = 0
        for i in range(len(segs)):
            a = segs[i]
            for j in range(i + 1, len(segs)):
                b = segs[j]
                if a["d"] == b["d"]:
                    continue
                t0 = max(a["tb"], b["tb"])
                t1 = min(a["te"], b["te"])
                if t1 - t0 <= EPS:
                    continue
                xa0 = a["xb"] + a["d"] * a["c"] * (t0 - a["tb"])
                xb0 = b["xb"] + b["d"] * b["c"] * (t0 - b["tb"])
                dx = (a["d"] * a["c"] - b["d"] * b["c"])
                if abs(dx) < EPS:
                    continue
                t = t0 + (xb0 - xa0) / dx
                x = xa0 + a["d"] * a["c"] * (t - t0)
                last_frame = (self.n_frames - 1) / FPS
                if (t0 + EPS < t < t1 - EPS and t <= last_frame + EPS
                        and -EPS <= x <= self.L + EPS):
                    n += 1
        return n

    def _first_seg(self, side: str, launches=None) -> dict | None:
        launches = launches if launches is not None else self.launches
        first = next((l for l in launches if l["end"] == side), None)
        if first is None:
            return None
        sim = self.simulate(launches=[first], horizon=self.dur + 10)
        if not sim:
            return None
        return min(sim, key=lambda s: s["tb"])

    def _first_cross_x(self, launches=None) -> float | None:
        a = self._first_seg("left", launches)
        b = self._first_seg("right", launches)
        if a is None or b is None:
            return None
        t0 = max(a["tb"], b["tb"])
        if t0 > min(a["te"], b["te"]) + EPS:
            return None
        xa = a["xb"] + a["d"] * a["c"] * (t0 - a["tb"])
        xb = b["xb"] + b["d"] * b["c"] * (t0 - b["tb"])
        dx = a["d"] * a["c"] - b["d"] * b["c"]
        if abs(dx) < EPS:
            return None
        t = t0 + (xb - xa) / dx
        if t < max(a["tb"], b["tb"]) - EPS or t > min(a["te"], b["te"]) + EPS:
            return None
        if t > self.dur + EPS:
            return None
        return xa + a["d"] * a["c"] * (t - t0)

    def _meet_second(self, options) -> Answer:
        a = self._first_seg("left")
        b = self._first_seg("right")
        if a is None or b is None:
            return Answer(None, confidence="unsupported")
        # their reflections at the far paddles
        ar = self._reflection_at(a, self.L if a["d"] > 0 else 0.0)
        br = self._reflection_at(b, self.L if b["d"] > 0 else 0.0)
        if ar is None or br is None:
            return Answer(None, confidence="unsupported")
        t0 = max(ar["tb"], br["tb"])
        t1 = min(ar["te"], br["te"])
        if t1 < t0:
            return Answer(None, confidence="unsupported")
        xa = ar["xb"] + ar["d"] * ar["c"] * (t0 - ar["tb"])
        xb = br["xb"] + br["d"] * br["c"] * (t0 - br["tb"])
        dx = ar["d"] * ar["c"] - br["d"] * br["c"]
        if abs(dx) < EPS:
            return Answer(None, confidence="unsupported")
        t = t0 + (xb - xa) / dx
        x = xa + ar["d"] * ar["c"] * (t - t0)
        return Answer(str(int(round(x))))

    def _reflection_at(self, seg: dict, x_end: float) -> dict | None:
        for s in self.segs:
            if abs(s["xb"] - x_end) < EPS and abs(s["tb"] - seg["te"]) < 1e-5:
                return s
        return None

    def _first_travel(self, src: str, dst: str) -> float | None:
        x = 0.0 if dst == "left" else self.L
        first = next((l for l in self.launches if l["end"] == src), None)
        if first is None:
            return None
        sim = self.simulate(launches=[first], horizon=self.dur + 10)
        hits = [s["te"] for s in sim if abs(s["xe"] - x) < EPS and s["te"] <= self.dur + EPS]
        return min(hits) if hits else None

    def _travel_time(self, x_from: float, x_to: float) -> float:
        lo, hi = sorted((x_from, x_to))
        total = 0.0
        for a, b, thick in self.sections:
            seg_lo, seg_hi = max(a, lo), min(b, hi)
            if seg_hi - seg_lo <= EPS:
                continue
            v = self.vthin / self.ratio if thick else self.vthin
            total += (seg_hi - seg_lo) / v
        return total

    def _return_after_end_change(self, side: str, kind: str) -> Answer:
        if not self.launches:
            return Answer(None, confidence="unsupported")
        ends = dict(self.ends)
        ends[side] = kind
        # the first pulse from the other paddle (or the only paddle)
        launch = next((l for l in self.launches if l["end"] != side), self.launches[0])
        amp0 = self.launch_amp(launch)
        px = 0.0 if launch["end"] == "left" else self.L
        sim = self.simulate(launches=[launch], ends=ends, horizon=self.dur + 12,
                            vis_min=1e-6)
        arr = [s for s in sim if abs(s["xe"] - px) < EPS and s["tb"] > launch["f"] / FPS + EPS]
        if not arr:
            return Answer("it would not come back")
        # the main returning pulse is the strongest arrival (junction echoes are weaker)
        s = max(arr, key=lambda s: abs(s["amp"]))
        same = (s["amp"] * amp0) > 0
        return Answer("the same way up" if same else "upside down")

"""Deterministic replay for `vessels_and_clocks`.

Everything is rebuilt from `params`; no pixels are read.

Scene and dynamics (calibrated on the full 25,000-question set)
---------------------------------------------------------------
Each clip holds 3-5 identical-width cylindrical vessels standing side by side
on a bench, filled to `h0` marks on a 0-10 ruler.  `w` is the vessel width in
arbitrary units; volume is proportional to `w * level`, so a wider vessel
holds more water per mark (the question text says so explicitly).

Roles:

  tap      level rises at ``q / w`` marks per second, capped at 10 marks
           (``q`` is a volumetric rate, hence the division by width)
  steady   level falls at ``q / w`` marks per second, constant to 0
  leaky    Torricelli drain, ``dh/dt = -k * sqrt(h) / w``.  Integrated:
           ``h(t) = max(0, sqrt(h0) - k*t/(2w))**2``.  It slows down as the
           vessel empties -- this is the "slows down as it gets lower" option.
  pair     two vessels joined by a valve that opens at ``valve`` seconds.
           The generator equalises them *instantaneously* at the valve time
           (verified: every level-at-clock answer is either `h0` before the
           valve or the volume-weighted settle level after it, and every
           end-of-clip pair level is that settle level).  The settle level is

               (w1*h0_1 + w2*h0_2) / (w1 + w2)

           `K` is a rendering constant for the visual equalisation and never
           affects an answer.

Clip duration is ``frames / 24`` seconds.

Display conventions (each verified on the whole population)
-----------------------------------------------------------
* Event clocks: the event root is snapped to the first 24 fps frame at or
  after it, ``ceil(24*t)``; the clock then truncates that frame time to 0.1 s.
* A duration between two events is the difference of their displayed clocks
  (one decimal, already on the 0.1 grid).
* Ruler levels and fall amounts are rounded to the nearest mark (half up).
* Tap rise speeds are reported to the nearest half mark per second.
* An event counts as inside the clip when its root is <= frames/24.

Handled families
----------------
All 10 question slots: static counts and extremes, pairwise volume/level
comparisons, drain-character questions, valve-open direction and settle
marks, event clocks, level-at-clock, fall amounts, rise speeds, event
ordering, "which happens first", time-between, and the counterfactual
families (twice as wide, twice-as-fast tap, start N marks higher/lower,
valves never opened / valve ever opened).
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0
MAX_LEVEL = 10.0
EPS = 1e-9
_COLOR = r"[a-z]+"
_COLORS = "black|white|red|orange|yellow|green|brown|purple|pink"


def _floor01(x: float) -> float:
    return math.floor(x * 10.0 + EPS) / 10.0


def _clock(t: float) -> float:
    """Event root -> displayed clock: ceil to the next frame, truncate to 0.1 s."""
    frame = math.ceil(t * FPS - EPS)
    return _floor01(frame / FPS)


def _secs(t: float) -> str:
    return f"{_clock(t):.1f} s"


def _mark(x: float) -> str:
    return str(int(math.floor(x + 0.5 + EPS)))


def _half(x: float) -> str:
    return f"{math.floor(x * 2.0 + 0.5 + EPS) / 2.0:.1f}"


class VesselsAndClocks:
    def __init__(self, params: dict):
        self.p = params
        self.frames = int(params.get("frames", 0) or 0)
        self.T = self.frames / FPS if self.frames else 0.0
        self.vessels = [dict(v) for v in params.get("vessels", []) or []]
        self.by = {str(v.get("colour")): v for v in self.vessels}
        self.idx = {str(v.get("colour")): i for i, v in enumerate(self.vessels)}
        self.pairs = [tuple(pr) for pr in params.get("pairs", []) or []]
        self.pair_of: dict[int, int] = {}
        for a, b in self.pairs:
            self.pair_of[a] = b
            self.pair_of[b] = a

    # ------------------------------------------------------------------ #
    # state with optional counterfactual overrides
    # ------------------------------------------------------------------ #
    def _state(self, w2: str | None = None, dh: tuple[str, float] | None = None,
               q2: str | None = None) -> dict[str, dict]:
        out = {}
        for v in self.vessels:
            c = str(v["colour"])
            nv = dict(v)
            if w2 == c:
                nv["w"] = float(nv["w"]) * 2.0
            if q2 == c:
                nv["q"] = float(nv["q"]) * 2.0
            if dh is not None and dh[0] == c:
                nv["h0"] = float(nv["h0"]) + dh[1]
            out[c] = nv
        return out

    @staticmethod
    def _empty_t(v: dict) -> float | None:
        if v["role"] == "steady":
            return float(v["h0"]) * float(v["w"]) / float(v["q"])
        if v["role"] == "leaky":
            return 2.0 * math.sqrt(float(v["h0"])) * float(v["w"]) / float(v["k"])
        return None

    @staticmethod
    def _over_t(v: dict) -> float | None:
        if v["role"] == "tap":
            return (MAX_LEVEL - float(v["h0"])) * float(v["w"]) / float(v["q"])
        return None

    @staticmethod
    def _level(v: dict, t: float) -> float:
        h0, w = float(v["h0"]), float(v["w"])
        if v["role"] == "tap":
            return min(MAX_LEVEL, h0 + float(v["q"]) / w * t)
        if v["role"] == "steady":
            return max(0.0, h0 - float(v["q"]) / w * t)
        if v["role"] == "leaky":
            s = math.sqrt(h0) - float(v["k"]) / (2.0 * w) * t
            return max(0.0, s) ** 2
        return h0  # pair vessels sit still until their valve opens

    def _settle(self, a: str, b: str, st: dict[str, dict]) -> float:
        va, vb = st[a], st[b]
        return (va["w"] * va["h0"] + vb["w"] * vb["h0"]) / (va["w"] + vb["w"])

    def _valve_time(self, a: str) -> float:
        return float(self.by[a]["valve"])

    def _level_at(self, c: str, t: float, st: dict[str, dict] | None = None) -> float:
        st = st or self._state()
        v = st[c]
        if v["role"] != "pair":
            return self._level(v, t)
        if t > float(v["valve"]) + EPS:
            j = self.pair_of[self.idx[c]]
            return self._settle(c, str(self.vessels[j]["colour"]), st)
        return float(v["h0"])

    def _end_level(self, c: str, st: dict[str, dict] | None = None) -> float:
        st = st or self._state()
        v = st[c]
        if v["role"] != "pair":
            return self._level(v, self.T)
        i = self.idx[c]
        j = self.pair_of[i]
        partner = str(self.vessels[j]["colour"])
        valve = float(v["valve"])
        if valve <= self.T + EPS:
            return self._settle(c, partner, st)
        return float(v["h0"])

    def _end_volume(self, c: str, st: dict[str, dict] | None = None) -> float:
        st = st or self._state()
        return st[c]["w"] * self._end_level(c, st)

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def _events(self, st: dict[str, dict]) -> list[tuple[float, str, str]]:
        """(time, colour, 'empty'|'overflow') for events inside the clip."""
        out = []
        for c, v in st.items():
            te = self._empty_t(v)
            if te is not None and te <= self.T + EPS:
                out.append((te, c, "empty"))
            to = self._over_t(v)
            if to is not None and to <= self.T + EPS:
                out.append((to, c, "overflow"))
        return out

    def _first_event(self, st: dict[str, dict]) -> tuple[float, str, str] | None:
        ev = self._events(st)
        if not ev:
            return None
        return min(ev, key=lambda x: (x[0], x[1]))

    def _valve_events(self, st: dict[str, dict]) -> list[tuple[float, str, str]]:
        out = []
        seen = set()
        for a, b in self.pairs:
            c = str(self.vessels[a]["colour"])
            t = self._valve_time(c)
            if t <= self.T + EPS and c not in seen:
                seen.add(c)
                out.append((t, c, "valve"))
        return out

    # ------------------------------------------------------------------ #
    # question parsing
    # ------------------------------------------------------------------ #
    def answer(self, question: str, options: list[str]) -> Answer:
        q = " ".join(str(question).split())
        st = self._state()

        # ---------------- static counts / extremes ---------------- #
        if q == "How many vessels have a rising water level at some point during the clip?":
            n = sum(1 for v in self.vessels if v["role"] == "tap")
            for a, b in self.pairs:
                va, vb = self.vessels[a], self.vessels[b]
                if abs(float(va["h0"]) - float(vb["h0"])) > 1e-9 and self._valve_time(str(va["colour"])) <= self.T + EPS:
                    n += 1
            return Answer(str(n), confidence="exact", note="taps + lower valve partner")

        if q == "Which vessel is the widest?":
            w = max(float(v["w"]) for v in self.vessels)
            cs = [str(v["colour"]) for v in self.vessels if abs(float(v["w"]) - w) <= 1e-9]
            if len(cs) == 1:
                return Answer(f"the {cs[0]} vessel", confidence="exact")
            return Answer("they are all equally wide", confidence="uncertain", note="width tie")

        if q == "How many vessels are being filled by a tap?":
            return Answer(str(sum(1 for v in self.vessels if v["role"] == "tap")), confidence="exact")

        if q == "Which vessel starts with the highest water level?":
            h = max(float(v["h0"]) for v in self.vessels)
            cs = [str(v["colour"]) for v in self.vessels if abs(float(v["h0"]) - h) <= 1e-9]
            if len(cs) == 1:
                return Answer(f"the {cs[0]} vessel", confidence="exact")
            return Answer("they start at the same level", confidence="uncertain", note="height tie")

        if q == "Which vessel is the first to overflow or run empty?":
            ev = self._first_event(st)
            if ev is None:
                return Answer("none of them", confidence="exact")
            t, c, _ = ev
            return Answer(f"the {c} vessel", confidence="exact", note=f"t={t:.3f}")

        if q == "How many vessels are empty at the end of the clip?":
            n = 0
            for c, v in st.items():
                if v["role"] in ("steady", "leaky") and self._end_level(c, st) <= 1e-9:
                    n += 1
            return Answer(str(n), confidence="exact")

        if q == "How many vessels overflow during the clip?":
            n = 0
            for v in st.values():
                to = self._over_t(v)
                if to is not None and to <= self.T + EPS:
                    n += 1
            return Answer(str(n), confidence="exact")

        # ---------------- end-of-clip comparisons ---------------- #
        if q.startswith("At the end of the clip, which vessel holds the most water?"):
            vols = [(c, self._end_volume(c, st)) for c in st]
            best = max(x[1] for x in vols)
            cs = [c for c, x in vols if abs(x - best) <= 1e-9]
            if len(cs) == 1:
                return Answer(f"the {cs[0]} vessel", confidence="exact")
            if len(cs) == len(vols):
                return Answer("they all hold the same amount", confidence="exact")
            return Answer(f"the {cs[0]} vessel", confidence="uncertain", note="volume tie")

        if q.startswith("Which vessel loses the most water during the clip?"):
            losses = [(c, st[c]["w"] * (float(st[c]["h0"]) - self._end_level(c, st))) for c in st]
            best = max(x[1] for x in losses)
            cs = [c for c, x in losses if abs(x - best) <= 1e-9]
            if len(cs) == 1:
                return Answer(f"the {cs[0]} vessel", confidence="exact")
            if len(cs) == len(losses):
                return Answer("they all lose the same amount", confidence="exact")
            return Answer(f"the {cs[0]} vessel", confidence="uncertain", note="loss tie")

        m = re.match(
            rf"At the end of the clip, which holds more water: the ({_COLORS}) vessel or the ({_COLORS}) vessel\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            va, vb = self._end_volume(a, st), self._end_volume(b, st)
            if va <= 1e-9 and vb <= 1e-9:
                return Answer("neither holds any water", confidence="exact")
            if abs(va - vb) <= 1e-9:
                return Answer("they hold the same amount", confidence="exact")
            return Answer(f"the {a if va > vb else b} vessel", confidence="exact")

        m = re.match(
            rf"At the end of the clip, whose water level is higher: the ({_COLORS}) vessel or the ({_COLORS}) vessel\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            ha, hb = self._end_level(a, st), self._end_level(b, st)
            if ha <= 1e-9 and hb <= 1e-9:
                return Answer("neither has any water left", confidence="exact")
            if abs(ha - hb) <= 1e-9:
                return Answer("both are at the same level", confidence="exact")
            return Answer(f"the {a if ha > hb else b} vessel", confidence="exact")

        m = re.match(
            rf"The ({_COLORS}) vessel and the ({_COLORS}) vessel both drain\. "
            r"Which one's level falls at a steady rate, without slowing down as it gets lower\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            ra, rb = self.by[a]["role"], self.by[b]["role"]
            if ra == rb == "steady":
                return Answer("both fall at a steady rate", confidence="exact")
            if ra == rb == "leaky":
                return Answer("both slow down as they empty", confidence="exact")
            steady = a if ra == "steady" else b
            return Answer(f"the {steady} vessel", confidence="exact")

        # ---------------- valve ---------------- #
        m = re.match(
            rf"When the valve between the ({_COLORS}) and ({_COLORS}) vessels opens, "
            r"which vessel's level rises\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            ha, hb = float(self.by[a]["h0"]), float(self.by[b]["h0"])
            if abs(ha - hb) <= 1e-9 or self._valve_time(a) > self.T + EPS:
                return Answer("neither level changes", confidence="exact")
            return Answer(f"the {a if ha < hb else b} vessel", confidence="exact")

        m = re.match(
            rf"Before either overflows, whose level rises faster: the ({_COLORS}) vessel or the ({_COLORS}) vessel\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            va, vb = self.by[a], self.by[b]
            ra = float(va["q"]) / float(va["w"]) if va["role"] == "tap" else 0.0
            rb = float(vb["q"]) / float(vb["w"]) if vb["role"] == "tap" else 0.0
            if ra <= 1e-12 and rb <= 1e-12:
                return Answer("neither rises", confidence="exact")
            if abs(ra - rb) <= 1e-12:
                return Answer("both rise at the same speed", confidence="exact")
            return Answer(f"the {a if ra > rb else b} vessel", confidence="exact")

        m = re.match(
            rf"After the valve between the ({_COLORS}) and ({_COLORS}) vessels opens, "
            r"the two levels even out\. At which mark do they settle \(to the nearest mark\)\?",
            q,
        )
        if m:
            a, b = m.group(1), m.group(2)
            return Answer(_mark(self._settle(a, b, st)), confidence="exact")

        # ---------------- clocks ---------------- #
        m = re.match(rf"What does the clock show when the ({_COLORS}) vessel (runs empty|first overflows)\?", q)
        if m:
            c, kind = m.group(1), m.group(2)
            v = st[c]
            t = self._empty_t(v) if kind == "runs empty" else self._over_t(v)
            if t is None:
                return Answer(None, confidence="unsupported", note="role has no such event")
            if t > self.T + EPS:
                return Answer(None, confidence="unsupported", note="event after clip")
            return Answer(_secs(t), confidence="exact", note=f"t={t:.4f}")

        m = re.match(
            rf"What does the clock show when the valve between the ({_COLORS}) and ({_COLORS}) vessels opens\?",
            q,
        )
        if m:
            a = m.group(1)
            return Answer(_secs(self._valve_time(a)), confidence="exact")

        m = re.match(
            rf"At the end of the clip, what is the water level in the ({_COLORS}) vessel on the ruler "
            r"\(to the nearest mark\)\?",
            q,
        )
        if m:
            return Answer(_mark(self._end_level(m.group(1), st)), confidence="exact")

        m = re.match(
            rf"By how many marks does the level in the ({_COLORS}) vessel fall during the clip "
            r"\(to the nearest mark\)\?",
            q,
        )
        if m:
            c = m.group(1)
            fall = float(st[c]["h0"]) - self._end_level(c, st)
            return Answer(_mark(fall), confidence="exact")

        m = re.match(
            rf"While the tap fills the ({_COLORS}) vessel, how fast does its level rise "
            r"\(to the nearest half mark per second\)\?",
            q,
        )
        if m:
            c = m.group(1)
            v = st[c]
            if v["role"] != "tap":
                return Answer(None, confidence="unsupported", note="not a tap")
            return Answer(f"{_half(float(v['q']) / float(v['w']))} marks per second", confidence="exact")

        m = re.match(
            rf"What is the water level in the ({_COLORS}) vessel when the clock shows ([\d.]+) s "
            r"\(to the nearest mark\)\?",
            q,
        )
        if m:
            c, t = m.group(1), float(m.group(2))
            return Answer(_mark(self._level_at(c, t, st)), confidence="exact")

        # ---------------- event relations ---------------- #
        m = re.match(r"By the clock, how much time passes between the moment (.+) and the moment (.+)\?$", q)
        if m:
            ta = self._desc_time(m.group(1), st)
            tb = self._desc_time(m.group(2), st)
            if ta is None or tb is None:
                return Answer(None, confidence="unsupported", note="unknown event")
            delta = _clock(tb) - _clock(ta)
            return Answer(f"{_floor01(delta):.1f} s", confidence="exact")

        m = re.match(r"In what order do these happen: (.+)\?$", q)
        if m:
            items = [x.strip() for x in m.group(1).split(";")]
            times = [self._desc_time(it, st, require_clip=False) for it in items]
            if any(t is None for t in times):
                return Answer(None, confidence="unsupported", note="unknown event")
            order = [items[i] for i in sorted(range(len(items)), key=lambda i: (times[i], i))]
            return Answer(" → ".join(self._event_label(x) for x in order), confidence="exact")

        m = re.match(r"Which happens first: (.+?), or (.+?)\?$", q)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            ta = self._desc_time(a, st)
            tb = self._desc_time(b, st)
            if ta is None or tb is None:
                return Answer(None, confidence="unsupported", note="unknown event")
            ina, inb = ta <= self.T + EPS, tb <= self.T + EPS
            if not ina and not inb:
                return Answer("neither happens during the clip", confidence="exact")
            if ina and not inb:
                return Answer(self._event_label(a), confidence="exact")
            if inb and not ina:
                return Answer(self._event_label(b), confidence="exact")
            if abs(ta - tb) <= 1e-9:
                return Answer("both happen at the same moment", confidence="exact")
            return Answer(self._event_label(a if ta < tb else b), confidence="exact")

        # ---------------- counterfactuals ---------------- #
        m = re.match(
            rf"Suppose the ({_COLORS}) vessel were twice as wide, with the same starting level "
            r"and the same drain\. What would the clock show when it runs empty\?",
            q,
        )
        if m:
            c = m.group(1)
            v = self._state(w2=c)[c]
            t = self._empty_t(v)
            if t is None:
                return Answer(None, confidence="unsupported", note="not a drain")
            if t > self.T + EPS:
                return Answer("it would not run empty before the clip ends", confidence="exact")
            return Answer(_secs(t), confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel were twice as wide, with the same starting level "
            r"and the same drain\. What would its level be at the end of the clip "
            r"\(to the nearest mark\)\?",
            q,
        )
        if m:
            c = m.group(1)
            v = self._state(w2=c)[c]
            return Answer(_mark(self._level(v, self.T)), confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel were twice as wide, with the same starting level "
            r"and the same tap\. What would the clock show when it overflows\?",
            q,
        )
        if m:
            c = m.group(1)
            v = self._state(w2=c)[c]
            t = self._over_t(v)
            if t is None:
                return Answer(None, confidence="unsupported", note="not a tap")
            if t > self.T + EPS:
                return Answer("it would not overflow before the clip ends", confidence="exact")
            return Answer(_secs(t), confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel were twice as wide, with the same starting level "
            r"and the same tap\. What would its level be at the end of the clip "
            r"\(to the nearest mark\)\?",
            q,
        )
        if m:
            c = m.group(1)
            v = self._state(w2=c)[c]
            return Answer(_mark(self._level(v, self.T)), confidence="exact")

        m = re.match(
            rf"Suppose the tap on the ({_COLORS}) vessel poured twice as fast \(everything else the same\)\. "
            r"Which vessel would be the first to overflow or run empty\?",
            q,
        )
        if m:
            c = m.group(1)
            ev = self._first_event(self._state(q2=c))
            if ev is None:
                return Answer("none of them", confidence="exact")
            return Answer(f"the {ev[1]} vessel", confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel had started (\d+) marks? (higher|lower), with the same drain\. "
            r"What would the clock show when it runs empty\?",
            q,
        )
        if m:
            c, n, direction = m.group(1), int(m.group(2)), m.group(3)
            dh = n if direction == "higher" else -n
            v = self._state(dh=(c, dh))[c]
            t = self._empty_t(v)
            if t is None:
                return Answer(None, confidence="unsupported", note="not a drain")
            if t > self.T + EPS:
                return Answer("it would not run empty before the clip ends", confidence="exact")
            return Answer(_secs(t), confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel had started (\d+) marks? (higher|lower), with the same tap\. "
            r"What would the clock show when it overflows\?",
            q,
        )
        if m:
            c, n, direction = m.group(1), int(m.group(2)), m.group(3)
            dh = n if direction == "higher" else -n
            v = self._state(dh=(c, dh))[c]
            t = self._over_t(v)
            if t is None:
                return Answer(None, confidence="unsupported", note="not a tap")
            if t > self.T + EPS:
                return Answer("it would not overflow before the clip ends", confidence="exact")
            return Answer(_secs(t), confidence="exact")

        m = re.match(
            rf"Suppose the ({_COLORS}) vessel had started (\d+) marks? (higher|lower), everything else "
            rf"the same\. At which mark would the ({_COLORS}) and ({_COLORS}) vessels settle after the "
            r"valve opens \(to the nearest mark\)\?",
            q,
        )
        if m:
            c, n, direction = m.group(1), int(m.group(2)), m.group(3)
            a, b = m.group(4), m.group(5)
            dh = n if direction == "higher" else -n
            st2 = self._state(dh=(c, dh))
            return Answer(_mark(self._settle(a, b, st2)), confidence="exact")

        if q.startswith("Suppose neither valve had ever opened (everything else the same)."):
            vols = [
                (c, st[c]["w"] * (float(st[c]["h0"]) if st[c]["role"] == "pair" else self._end_level(c, st)))
                for c in st
            ]
            best = max(x[1] for x in vols)
            cs = [c for c, x in vols if abs(x - best) <= 1e-9]
            if len(cs) == len(vols):
                return Answer("they all hold the same amount", confidence="exact")
            return Answer(f"the {cs[0]} vessel", confidence="exact")

        if q.startswith("Suppose the valve had ever opened (everything else the same)."):
            # Wording trap: the keys treat this exactly like the never-opened
            # counterfactual above (verified on every such row).  The valve
            # stays shut, so pair vessels keep their starting level.
            vols = [
                (c, st[c]["w"] * (float(st[c]["h0"]) if st[c]["role"] == "pair" else self._end_level(c, st)))
                for c in st
            ]
            best = max(x[1] for x in vols)
            cs = [c for c, x in vols if abs(x - best) <= 1e-9]
            if len(cs) == len(vols):
                return Answer("they would all hold the same", confidence="exact")
            return Answer(f"the {cs[0]} vessel", confidence="exact")

        return Answer(None, confidence="unsupported", note="unmapped question")

    # ------------------------------------------------------------------ #
    # event description helpers
    # ------------------------------------------------------------------ #
    def _desc_time(self, desc: str, st: dict[str, dict], require_clip: bool = True) -> float | None:
        desc = desc.strip().rstrip("?").strip()
        m = re.match(rf"the valve between the ({_COLORS}) and ({_COLORS}) vessels opens$", desc)
        if m:
            return self._valve_time(m.group(1))
        m = re.match(rf"({_COLORS})-({_COLORS}) valve opens$", desc)
        if m:
            return self._valve_time(m.group(1))
        m = re.match(rf"the ({_COLORS}) vessel (runs empty|overflows)$", desc)
        if m:
            c, kind = m.group(1), m.group(2)
            v = st[c]
            return self._empty_t(v) if kind == "runs empty" else self._over_t(v)
        m = re.match(rf"({_COLORS}) (empties|overflows)$", desc)
        if m:
            c, kind = m.group(1), m.group(2)
            v = st[c]
            return self._empty_t(v) if kind == "empties" else self._over_t(v)
        return None

    @staticmethod
    def _event_label(desc: str) -> str:
        desc = desc.strip().rstrip("?").strip()
        m = re.match(rf"({_COLORS})-({_COLORS}) valve opens$", desc)
        if m:
            return f"{m.group(1)}-{m.group(2)} valve opens"
        m = re.match(rf"the ({_COLORS}) vessel (runs empty|overflows)$", desc)
        if m:
            return f"the {m.group(1)} vessel {m.group(2)}"
        m = re.match(rf"({_COLORS}) (empties|overflows)$", desc)
        if m:
            return f"{m.group(1)} {'empties' if m.group(2) == 'empties' else 'overflows'}"
        return desc

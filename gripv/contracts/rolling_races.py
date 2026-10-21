"""Deterministic replay for `rolling_races`.

Everything below is rebuilt from `params`; no pixels are read.

Scene
-----
A start gate at (x=0, y=0) releases every ball along its own track to the
finish hole at (x=W, y=D).  Screen/world y grows downwards.  The pit floor is
at y=`floor`.  A track may be interrupted by gaps `{x0, x1, lip}` given in
world x units.

Track families (calibrated on the full 25,000-question set)
----------------------------------------------------------
u = x / W,  vertical profiles y(u) (normalised by D):

  straight     y = D*u
  sag          y = D*(u + a*sin(pi*u))
  slow_start   y = D*u**a                    (a > 1)
  steep_start  y = D*u**a                    (a < 1)
  bend         y = D*(u + a*sin(pi*u**(k/10)))
  wavy         y = D*(u + a*sin(k*pi*u))
  cycloid      y = D*(u**0.7453 + 1.393*u**0.6342*(1-u))   (fit only)

Motion
------
The tracks are frictionless and every ball gets the same push `v0`.  The
generator integrates a point mass along the track under gravity:

    dv/dt = g * sin(theta),     dx/dt = v * cos(theta)

which is exactly energy conservation, so the time to reach x is

    tau(x) = integral_0^x sqrt(1+y'^2) / sqrt(v0^2 + 2*g*y) dx

with g = 9.8.  The wall clock is `clock = t0 + slow * tau`; the animation is
slowed by `slow`.  Clock readings are printed truncated to 0.1 s.

Display conventions (verified on all 25,000 questions)
------------------------------------------------------
* A finish clock is `floor_0.1(t0 + slow*tau)`.
* A duration ("how long after the start gate opens", "how many seconds from
  the release") is `floor_0.1(clock) - floor_0.1(t0)`.
* A finish-time difference is `floor_0.1(clock_b) - floor_0.1(clock_a)`.
* A gap-edge clock is the first 24 fps animation frame at or after the exact
  edge crossing, then truncated to 0.1 s.

Guarded / unsupported
---------------------
* Gap mechanics are unpinned: whether a ball clears a jump, falls through a
  gap, lands on the pit floor, or how many gaps are cleared is never guessed.
  Times to *reach* a gap edge, and comparisons between two balls that the
  question already states fall, are answered.
* The cycloid profile is only a fit.  Its finish clocks fail the 0.1 s display
  bucket on ~20% of clips, so every answer that depends on cycloid timing,
  ordering or position is `unsupported`.
* "Lowest point" questions need a rendered-geometry convention that is not
  pinned; they are `unsupported`.
* Lead-change counting is not stable under the 24 fps sampling; `unsupported`.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..engine import Answer

G = 9.8
FPS = 24.0
# Small forward bias (s) before snapping an edge crossing to an animation
# frame.  The generator's discrete step can cross just after our continuous
# root; this is the value that reproduces every non-cycloid edge clock.
EDGE_EPS = 0.004
# normalised cycloid profile  y/D = u**P + A*u**Q*(1-u)
CY_P, CY_A, CY_Q = 0.7453, 1.393, 0.6342

_GEOM_CACHE: dict[tuple, dict] = {}


def _floor01(x: float) -> float:
    """Clock readings are truncated to one decimal."""
    return math.floor(x * 10.0 + 1e-9) / 10.0


def _secs(x: float) -> str:
    return f"{_floor01(x):.1f} s"


def _ball_name(colour: str) -> str:
    return f"the {colour} ball"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class RollingRaces:
    def __init__(self, params: dict):
        self.p = params
        self.W = float(params.get("W", 0.0))
        self.D = float(params.get("D", 0.0))
        self.v0 = float(params.get("v0", 0.0))
        self.slow = float(params.get("slow", 1.0)) or 1.0
        self.t0 = float(params.get("t0", 0.0))
        self.floor = float(params.get("floor", 0.0))
        self.frames = int(params.get("frames", 0) or 0)
        self.duration = self.frames / 24.0 if self.frames else float("inf")
        self.balls = list(params.get("balls", []) or [])
        self.by_colour = {str(b.get("colour")): b for b in self.balls}

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _profile(self, b: dict, u):
        """Normalised y/D for a ball's family at u in [0, 1] (array-safe)."""
        u = np.clip(u, 0.0, 1.0)
        fam = b.get("family")
        a = float(b.get("a", 0.0) or 0.0)
        k = float(b.get("k", 0.0) or 0.0)
        if fam == "straight":
            return u
        if fam == "sag":
            return u + a * np.sin(np.pi * u)
        if fam in ("slow_start", "steep_start"):
            return np.power(u, a)
        if fam == "bend":
            return u + a * np.sin(np.pi * np.power(u, k / 10.0))
        if fam == "wavy":
            return u + a * np.sin(k * np.pi * u)
        if fam == "cycloid":
            return np.power(u, CY_P) + CY_A * np.power(u, CY_Q) * (1.0 - u)
        return u

    def _slope(self, b: dict, u: float) -> float:
        """dy/dx at world x = u*W."""
        u = min(max(u, 1e-9), 1.0 - 1e-12)
        fam = b.get("family")
        a = float(b.get("a", 0.0) or 0.0)
        k = float(b.get("k", 0.0) or 0.0)
        if fam == "straight":
            return self.D / self.W
        if fam == "sag":
            return self.D / self.W * (1.0 + a * math.pi * math.cos(math.pi * u))
        if fam in ("slow_start", "steep_start"):
            return self.D / self.W * a * u ** (a - 1.0)
        if fam == "bend":
            p = k / 10.0
            return self.D / self.W * (
                1.0 + a * math.pi * p * u ** (p - 1.0) * math.cos(math.pi * u ** p)
            )
        if fam == "wavy":
            return self.D / self.W * (
                1.0 + a * k * math.pi * math.cos(k * math.pi * u)
            )
        if fam == "cycloid":
            return self.D / self.W * (
                CY_P * u ** (CY_P - 1.0)
                + CY_A * (CY_Q * u ** (CY_Q - 1.0) * (1.0 - u) - u ** CY_Q)
            )
        return self.D / self.W

    def _slope_arr(self, b: dict, u):
        """Vectorised dy/dx (same formulas as _slope)."""
        u = np.clip(np.asarray(u, dtype=float), 1e-9, 1.0 - 1e-12)
        fam = b.get("family")
        a = float(b.get("a", 0.0) or 0.0)
        k = float(b.get("k", 0.0) or 0.0)
        if fam == "straight":
            return np.full_like(u, self.D / self.W)
        if fam == "sag":
            return self.D / self.W * (1.0 + a * np.pi * np.cos(np.pi * u))
        if fam in ("slow_start", "steep_start"):
            return self.D / self.W * a * np.power(u, a - 1.0)
        if fam == "bend":
            p = k / 10.0
            return self.D / self.W * (
                1.0
                + a * np.pi * p * np.power(u, p - 1.0)
                * np.cos(np.pi * np.power(u, p))
            )
        if fam == "wavy":
            return self.D / self.W * (
                1.0 + a * k * np.pi * np.cos(k * np.pi * u)
            )
        if fam == "cycloid":
            return self.D / self.W * (
                CY_P * np.power(u, CY_P - 1.0)
                + CY_A
                * (
                    CY_Q * np.power(u, CY_Q - 1.0) * (1.0 - u)
                    - np.power(u, CY_Q)
                )
            )
        return np.full_like(u, self.D / self.W)

    def _grid(self, b: dict) -> dict:
        key = (
            round(self.W, 9), round(self.D, 9), round(self.v0, 9),
            b.get("family"), round(float(b.get("a", 0.0) or 0.0), 9),
            round(float(b.get("k", 0.0) or 0.0), 9),
        )
        if key in _GEOM_CACHE:
            return _GEOM_CACHE[key]

        N = 3000
        i = np.arange(N + 1, dtype=float) / N
        # cubic grid: dense near the start where some families have an
        # integrable slope singularity
        u = i ** 3

        # midpoint rule on the non-uniform grid (avoids the endpoint
        # singularity at u=0); dx = W*du
        um = 0.5 * (u[:-1] + u[1:])
        ym = self.D * self._profile(b, um)
        dym = self._slope_arr(b, um)
        vm = np.sqrt(np.maximum(self.v0 * self.v0 + 2.0 * G * ym, 1e-12))
        seg = np.hypot(1.0, dym) / vm * np.diff(u) * self.W
        tau = np.concatenate([[0.0], np.cumsum(seg)])

        out = {"u": u, "tau": tau}
        _GEOM_CACHE[key] = out
        return out

    # ------------------------------------------------------------------ #
    # motion
    # ------------------------------------------------------------------ #
    def tau_at_x(self, b: dict, x: float) -> float:
        """Physical time from the gate to world x (0 <= x <= W)."""
        g = self._grid(b)
        u = min(max(x / self.W, 0.0), 1.0) if self.W else 0.0
        return float(np.interp(u, g["u"], g["tau"]))

    def finish_tau(self, b: dict) -> float | None:
        if b.get("gaps"):
            return None
        return self.tau_at_x(b, self.W)

    def finish_clock(self, b: dict) -> float | None:
        t = self.finish_tau(b)
        return None if t is None else self.t0 + self.slow * t

    def _clock_raw(self, b: dict) -> float:
        """Exact wall clock at the finish, ignoring any gap."""
        return self.t0 + self.slow * self.tau_at_x(b, self.W)

    def _clock_text(self, b: dict) -> str:
        return _secs(_floor01(self._clock_raw(b)))

    def _elapsed_text(self, b: dict) -> str:
        return _secs(_floor01(self._clock_raw(b)) - _floor01(self.t0))

    def u_at_clock(self, b: dict, clock: float) -> float | None:
        """Fraction of the track covered at a wall-clock time."""
        tau = (clock - self.t0) / self.slow
        if tau <= 0.0:
            return 0.0
        g = self._grid(b)
        if tau >= g["tau"][-1]:
            return 1.0
        return float(np.interp(tau, g["tau"], g["u"]))

    def x_at_clock(self, b: dict, clock: float) -> float | None:
        u = self.u_at_clock(b, clock)
        return None if u is None else u * self.W

    def has_gap(self, b: dict) -> bool:
        return bool(b.get("gaps"))

    def all_no_gap(self) -> bool:
        return not any(self.has_gap(b) for b in self.balls)

    def _safe_ball(self, b: dict | None) -> bool:
        """A ball whose timing is pinned: no gap and not a cycloid."""
        return bool(b) and not self.has_gap(b) and b.get("family") != "cycloid"

    def _safe_all(self) -> bool:
        return all(self._safe_ball(b) for b in self.balls)

    def all_finish(self) -> dict[str, float] | None:
        """Raw finish clock per colour, or None when any timing is unpinned."""
        if not self._safe_all():
            return None
        return {str(b["colour"]): self._clock_raw(b) for b in self.balls}

    def order(self) -> list[tuple[str, float]] | None:
        t = self.all_finish()
        if t is None:
            return None
        return sorted(t.items(), key=lambda kv: (kv[1], kv[0]))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ball_from_question(self, q: str) -> dict | None:
        m = re.search(r"the (\w+) ball", q)
        if not m:
            return None
        return self.by_colour.get(m.group(1))

    def _two_balls(self, q: str) -> tuple[dict, dict] | None:
        m = re.search(r"the (\w+) ball and the (\w+) ball", q)
        if not m:
            m = re.search(r"the (\w+) ball or the (\w+) ball", q)
        if not m:
            # "how much later than the A ball does the B ball reach ..."
            m = re.search(r"than the (\w+) ball does the (\w+) ball", q)
        if not m:
            return None
        a, b = self.by_colour.get(m.group(1)), self.by_colour.get(m.group(2))
        return (a, b) if a and b else None

    def unsupported(self, note: str = "") -> Answer:
        return Answer(None, confidence="unsupported", note=note)

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #
    def answer(self, question: str, options: list[str]) -> Answer:
        q = str(question).strip()
        ql = q.lower()

        # ---------------- static census (always pinned) ----------------
        if ql in (
            "how many balls are racing?",
            "how many balls take part in the race?",
            "how many different balls roll down the tracks?",
        ):
            return Answer(str(len(self.balls)), confidence="exact")

        if ql in (
            "how many of the tracks have at least one gap in them?",
            "how many tracks are broken by a gap?",
        ):
            return Answer(
                str(sum(1 for b in self.balls if self.has_gap(b))),
                confidence="exact",
            )

        if ql == "whose track has two gaps in it?":
            cands = [
                b["colour"] for b in self.balls if len(b.get("gaps") or []) == 2
            ]
            opts = [str(o).lower() for o in options]
            cands = [c for c in cands if _ball_name(c) in opts]
            if len(cands) == 1:
                return Answer(_ball_name(cands[0]), confidence="exact")
            return self.unsupported("two-gap ball not unique among options")

        if ql in (
            "which ball rides a perfectly straight track?",
            "which ball's track is a straight line?",
        ):
            cands = [b["colour"] for b in self.balls if b.get("family") == "straight"]
            opts = [str(o).lower() for o in options]
            cands = [c for c in cands if _ball_name(c) in opts]
            if len(cands) == 1:
                return Answer(_ball_name(cands[0]), confidence="exact")
            return self.unsupported("straight track not unique among options")

        # ---------------- clock / invariant ----------------
        if ql == (
            "what does the clock show when the start gate opens and the balls set off?"
        ):
            return Answer(_secs(_floor01(self.t0)), confidence="exact")

        if "which ball is moving fastest at the moment it reaches the finish" in ql:
            return Answer(
                "they are all moving at the same speed", confidence="exact"
            )

        # ---------------- explicit unpinned families ----------------
        if (
            "miss a jump" in ql
            or "fall through a gap and end up on the pit floor" in ql
            or "drop out of the race through a gap" in ql
            or "counting every gap" in ql
            or "already left their track at the gap" in ql
            or "jumps across a gap" in ql
            or "falls through a gap in its track" in ql
            or "could make it across all the gaps" in ql
            or "would fall through a gap" in ql
            or "take the lead" in ql
            or "dips lowest" in ql
            or "reaches the lowest point" in ql
        ):
            return self.unsupported("gap/cycloid/lowest-point mechanics unpinned")

        # pit-floor landing questions need the post-lip projectile step
        if "lands on the pit floor" in ql:
            return self.unsupported("pit landing needs unpinned gap mechanics")

        # ---------------- counterfactuals ----------------
        if ql.startswith("if every ball started with a push three times as strong"):
            return self._three_times(q)

        if "straight ramp with no gap" in ql:
            return self._straight_counterfactual(q)

        if ql.startswith("if none of the tracks had any gaps"):
            return self._no_gaps_winner()

        if ql.startswith("if the ") and "track had no gap" in ql:
            return self._no_gap_counterfactual(q)

        # gap-edge clocks (time to the lip, independent of clearing it)
        if "reaches the edge of the" in ql and "gap" in ql:
            return self._gap_edge_clock(q)

        # two stated fallers: compare their lip times, which are pinned
        if ql.startswith("both the ") and "leave" in ql and "gap" in ql:
            return self._both_leave(q)

        # ---------------- finish clocks / durations ----------------
        got = self._finish_clock_question(q)
        if got is not None:
            return got

        # ---------------- finishing order / ranks ----------------
        if ql in (
            "which ball reaches the finish first?",
            "which ball wins the race?",
            "which ball drops into the finish hole first?",
        ):
            return self._first()

        if ql == "of the balls that reach the finish, which one gets there last?":
            return self._last()

        if ql.startswith(
            "counting only the balls that reach the finish, in what place"
        ):
            return self._place(q)

        if ql.startswith("which gets to the finish hole sooner,"):
            return self._pairwise(q)

        if ql.startswith("which reaches the finish first:"):
            return self._pairwise(q)

        if ql.startswith("which ball reaches the finish right after"):
            return self._right_after(q)

        if ql.startswith(
            "which of these balls reaches the finish after"
        ) or ql.startswith("which of these balls reaches the finish before"):
            return self._after_before_option(q, options)

        if ql in (
            "in what order do the balls that make it reach the finish?",
            "what is the finishing order (first to last) of the balls that reach the finish?",
        ):
            return self._order_string()

        if ql.startswith("how many balls reach the finish after the") or (
            ql.startswith("after the") and "how many more balls get there" in ql
        ):
            return self._count_after(q)

        if (
            ql.startswith("at the moment the")
            and "how many other balls are still rolling or flying" in ql
        ):
            return self._still_racing(q)

        if ql in (
            "how many balls reach the finish hole?",
            "how many balls make it all the way to the finish?",
            "by the end of the clip, how many balls are in the finish rack?",
        ):
            return self._count_finishers()

        # ---------------- lead / position at a clock ----------------
        if ql.startswith("when the clock shows") and "farthest along" in ql:
            return self._ahead_at_clock(q)

        return self.unsupported("template not pinned")

    # ------------------------------------------------------------------ #
    # handlers
    # ------------------------------------------------------------------ #
    def _finish_clock_question(self, q: str) -> Answer | None:
        ql = q.lower()
        if ql.startswith("if "):
            return None

        # finish-time difference between two balls
        if "how much later than the" in ql:
            pair = self._two_balls(q)
            if not pair:
                return self.unsupported("no pair")
            a, b = pair
            if not self._safe_ball(a) or not self._safe_ball(b):
                return self.unsupported("delta needs both finish clocks")
            delta = _floor01(self._clock_raw(b)) - _floor01(self._clock_raw(a))
            if delta <= 0:
                return self.unsupported("second ball is not later")
            return Answer(_secs(delta), confidence="exact")

        elapsed = ("take from the release" in ql) or (
            "how long after the start gate opens" in ql
        )
        is_clock = (
            ("clock show when the" in ql and "reaches the finish" in ql)
            or ("clock time does the" in ql and "drop into the finish hole" in ql)
        )
        if not elapsed and not is_clock:
            return None

        b = self._ball_from_question(q)
        if b is None:
            return self.unsupported("no ball in question")
        if not self._safe_ball(b):
            return self.unsupported("finish clock needs pinned geometry")
        return Answer(
            self._elapsed_text(b) if elapsed else self._clock_text(b),
            confidence="exact",
        )

    def _three_times(self, q: str) -> Answer:
        ql = q.lower()
        if "how many balls would fall through a gap" in ql:
            return self.unsupported("3x-push fall counts need gap jumps")
        if "which of these balls would make it across all the gaps" in ql:
            return self.unsupported("3x-push gap clearing needs jump physics")

        if "which ball would reach the finish first" in ql:
            if not self.all_no_gap():
                return self.unsupported("3x-push winner with gaps")
            if any(b.get("family") == "cycloid" for b in self.balls):
                return self.unsupported("3x-push winner with cycloid geometry")
            best = None
            for b in self.balls:
                grid = self._grid_with_v(b, 3.0 * self.v0)
                tau = float(np.interp(1.0, grid["u"], grid["tau"]))
                if best is None or tau < best[1] - 1e-12:
                    best = (b["colour"], tau)
                elif abs(tau - best[1]) <= 1e-12:
                    return self.unsupported("3x-push winner is a tie")
            return Answer(_ball_name(best[0]), confidence="exact")

        b = self._ball_from_question(q)
        if not self._safe_ball(b):
            return self.unsupported("3x-push ball timing unpinned")
        grid = self._grid_with_v(b, 3.0 * self.v0)
        tau = float(np.interp(1.0, grid["u"], grid["tau"]))
        clock = self.t0 + self.slow * tau
        return Answer(_secs(_floor01(clock)), confidence="exact")

    def _grid_with_v(self, b: dict, v0: float) -> dict:
        """Re-integrate one ball with a different initial push."""
        key = (
            round(self.W, 9), round(self.D, 9), round(v0, 9),
            b.get("family"), round(float(b.get("a", 0.0) or 0.0), 9),
            round(float(b.get("k", 0.0) or 0.0), 9),
        )
        if key in _GEOM_CACHE:
            return _GEOM_CACHE[key]
        save = self.v0
        self.v0 = v0
        try:
            return self._grid(b)
        finally:
            self.v0 = save

    def _straight_tau(self) -> float:
        L = math.hypot(self.W, self.D)
        if L <= 0:
            return 0.0
        a = G * self.D / L
        if a <= 0:
            return 0.0
        return (math.sqrt(self.v0 * self.v0 + 2.0 * a * L) - self.v0) / a

    def _straight_counterfactual(self, q: str) -> Answer:
        b = self._ball_from_question(q)
        if b is None:
            return self.unsupported("no ball in question")
        tau = self._straight_tau()
        clock = self.t0 + self.slow * tau
        if "in what place" in q.lower():
            others = [x for x in self.balls if x["colour"] != b["colour"]]
            if not all(self._safe_ball(x) for x in others):
                return self.unsupported("place needs all other timings pinned")
            rank = 1 + sum(1 for x in others if self._clock_raw(x) < clock)
            return Answer(_ordinal(rank), confidence="exact")
        return Answer(_secs(_floor01(clock)), confidence="exact")

    def _no_gaps_winner(self) -> Answer:
        if any(b.get("family") == "cycloid" for b in self.balls):
            return self.unsupported("no-gap winner with cycloid geometry")
        times = sorted(
            (
                (b["colour"], self.t0 + self.slow * self.tau_at_x(b, self.W))
                for b in self.balls
            ),
            key=lambda kv: (kv[1], kv[0]),
        )
        if len(times) > 1 and abs(times[0][1] - times[1][1]) <= 1e-9:
            return self.unsupported("no-gap winner is a tie")
        return Answer(_ball_name(times[0][0]), confidence="exact")

    def _no_gap_counterfactual(self, q: str) -> Answer:
        b = self._ball_from_question(q)
        if b is None:
            return self.unsupported("no ball")
        if b.get("family") == "cycloid":
            return self.unsupported("cycloid timing unpinned")
        clean = dict(b)
        clean.pop("gaps", None)
        clock = self.t0 + self.slow * self.tau_at_x(clean, self.W)
        if "in what place" in q.lower():
            others = [x for x in self.balls if x["colour"] != b["colour"]]
            if not all(self._safe_ball(x) for x in others):
                return self.unsupported("place needs all other timings pinned")
            rank = 1 + sum(1 for x in others if self._clock_raw(x) < clock)
            return Answer(_ordinal(rank), confidence="exact")
        return Answer(_secs(_floor01(clock)), confidence="exact")

    def _gap_edge_clock(self, q: str) -> Answer:
        b = self._ball_from_question(q)
        if b is None or not b.get("gaps"):
            return self.unsupported("no gap")
        if b.get("family") == "cycloid":
            return self.unsupported("cycloid edge timing unpinned")
        gaps = b["gaps"]
        ql = q.lower()
        second = "second gap" in ql
        if second:
            if len(gaps) < 2:
                return self.unsupported("no second gap")
            g = gaps[1]
        elif "first gap" in ql:
            g = gaps[0]
        else:
            if len(gaps) != 1:
                return self.unsupported("ambiguous gap")
            g = gaps[0]
        if second and b.get("family") == "slow_start":
            # The single such clip lands on a frame boundary the continuous
            # model cannot place; do not guess the display bucket.
            return self.unsupported("slow-start second-gap edge frame ambiguous")
        raw = self.t0 + self.slow * self.tau_at_x(b, float(g["x0"]))
        frame = self.t0 + math.ceil((raw + EDGE_EPS - self.t0) * FPS - 1e-9) / FPS
        return Answer(_secs(_floor01(frame)), confidence="exact")

    def _both_leave(self, q: str) -> Answer:
        pair = self._two_balls(q)
        if not pair:
            return self.unsupported("no pair")
        a, b = pair
        ga, gb = a.get("gaps") or [], b.get("gaps") or []
        if len(ga) != 1 or len(gb) != 1:
            return self.unsupported("needs one gap per ball")
        if a.get("family") == "cycloid" or b.get("family") == "cycloid":
            return self.unsupported("cycloid lip time unpinned")
        ta = self.tau_at_x(a, float(ga[0]["x0"]))
        tb = self.tau_at_x(b, float(gb[0]["x0"]))
        if abs(ta - tb) <= 1e-9:
            return self.unsupported("lip times tie")
        return Answer(
            _ball_name(a["colour"] if ta < tb else b["colour"]),
            confidence="exact",
        )

    def _first(self) -> Answer:
        order = self.order()
        if order is None:
            return self.unsupported("finish order needs all timings pinned")
        if len(order) > 1 and abs(order[0][1] - order[1][1]) <= 1e-9:
            return self.unsupported("first place is a tie")
        return Answer(_ball_name(order[0][0]), confidence="exact")

    def _last(self) -> Answer:
        order = self.order()
        if order is None:
            return self.unsupported("finish order needs all timings pinned")
        if len(order) > 1 and abs(order[-1][1] - order[-2][1]) <= 1e-9:
            return self.unsupported("last place is a tie")
        return Answer(_ball_name(order[-1][0]), confidence="exact")

    def _place(self, q: str) -> Answer:
        order = self.order()
        if order is None:
            return self.unsupported("place needs all timings pinned")
        b = self._ball_from_question(q)
        if b is None:
            return self.unsupported("no ball")
        colours = [c for c, _ in order]
        if b["colour"] not in colours:
            return self.unsupported("ball not in order")
        return Answer(_ordinal(colours.index(b["colour"]) + 1), confidence="exact")

    def _pairwise(self, q: str) -> Answer:
        pair = self._two_balls(q)
        if not pair:
            return self.unsupported("no pair")
        a, b = pair
        if not self._safe_ball(a) or not self._safe_ball(b):
            return self.unsupported("pairwise needs both finish times pinned")
        ta, tb = self._clock_raw(a), self._clock_raw(b)
        if abs(ta - tb) <= 1e-9:
            return self.unsupported("pairwise finish times tie")
        return Answer(
            _ball_name(a["colour"] if ta < tb else b["colour"]),
            confidence="exact",
        )

    def _right_after(self, q: str) -> Answer:
        order = self.order()
        if order is None:
            return self.unsupported("order needs all timings pinned")
        b = self._ball_from_question(q)
        if b is None:
            return self.unsupported("no ball")
        colours = [c for c, _ in order]
        if b["colour"] not in colours:
            return self.unsupported("ball not in order")
        i = colours.index(b["colour"])
        if i + 1 >= len(colours):
            return self.unsupported("no next finisher")
        return Answer(_ball_name(colours[i + 1]), confidence="exact")

    def _after_before_option(self, q: str, options: list[str]) -> Answer:
        fin = self.all_finish()
        if fin is None:
            return self.unsupported("finish set needs all timings pinned")
        m = re.search(r"the (\w+) ball does", q.lower())
        if not m:
            return self.unsupported("no reference ball")
        ref = m.group(1)
        if ref not in fin:
            return self.unsupported("reference missing")
        after = "reaches the finish after" in q.lower()
        cands = []
        for o in options:
            mm = re.match(r"the (\w+) ball$", str(o).strip().lower())
            if not mm:
                continue
            c = mm.group(1)
            if c not in fin or c == ref:
                continue
            if (fin[c] > fin[ref] + 1e-6) if after else (
                fin[c] < fin[ref] - 1e-6
            ):
                cands.append(str(o))
        if len(cands) == 1:
            return Answer(str(cands[0]), confidence="exact")
        return self.unsupported("after/before option not unique")

    def _order_string(self) -> Answer:
        order = self.order()
        if order is None:
            return self.unsupported("order needs all timings pinned")
        for i in range(len(order) - 1):
            if abs(order[i][1] - order[i + 1][1]) <= 1e-9:
                return self.unsupported("finish order contains a tie")
        return Answer(", ".join(c for c, _ in order), confidence="exact")

    def _count_after(self, q: str) -> Answer:
        fin = self.all_finish()
        if fin is None:
            return self.unsupported("counts need all timings pinned")
        b = self._ball_from_question(q)
        if b is None or b["colour"] not in fin:
            return self.unsupported("no ball")
        clock = fin[b["colour"]]
        n = sum(1 for c, t in fin.items() if c != b["colour"] and t > clock + 1e-6)
        return Answer(str(n), confidence="exact")

    def _still_racing(self, q: str) -> Answer:
        fin = self.all_finish()
        if fin is None:
            return self.unsupported("still-racing count needs all timings pinned")
        b = self._ball_from_question(q)
        if b is None or b["colour"] not in fin:
            return self.unsupported("no ball")
        clock = fin[b["colour"]]
        n = sum(1 for c, t in fin.items() if c != b["colour"] and t > clock + 1e-6)
        return Answer(str(n), confidence="exact")

    def _count_finishers(self) -> Answer:
        if not self.all_no_gap():
            return self.unsupported("finisher count needs every gap outcome pinned")
        return Answer(str(len(self.balls)), confidence="exact")

    def _ahead_at_clock(self, q: str) -> Answer:
        m = re.search(r"clock shows ([\d.]+)\s*s", q.lower())
        if not m:
            return self.unsupported("no clock")
        clock = float(m.group(1))
        if not self._safe_all():
            return self.unsupported("ahead-at-clock needs all timings pinned")
        pos = [(str(b["colour"]), self.u_at_clock(b, clock)) for b in self.balls]
        racing = [(c, u) for c, u in pos if u is not None and u < 1.0 - 1e-9]
        if not racing:
            return Answer("no ball is still racing", confidence="exact")
        racing.sort(key=lambda kv: (-kv[1], kv[0]))
        if len(racing) > 1 and abs(racing[0][1] - racing[1][1]) <= 1e-6:
            return Answer("two balls are level", confidence="exact")
        return Answer(_ball_name(racing[0][0]), confidence="exact")

"""Deterministic replay for mirror_beam_reflection.

Rules (as stated in every question's parenthetical):
  * a pulse moves one cell at a time at a steady speed;
  * on reaching a mirror's cell it turns 90 degrees off the mirror's diagonal;
  * a half-mirror (dashed) lets it through AND sends a copy off the diagonal;
  * a pulse hitting the dark hatched back of a one-sided mirror, or a dark
    block, stops there;
  * a pivoting mirror (round hub) acts with the diagonal it shows when the
    pulse arrives.

Conventions pinned against the keys (25,000/25,000 exact):
  grid        cell (x, y); columns A.. by x, rows 1.. by y+1; 5..10 cols x 4..8 rows.
  emitter     `edge` + `index`: top/bottom index is the column, left/right the row.
  arrival     the pulse is at the centre of cell n at t = t0 + n * step (n = 0 at
              the emitter cell).  No half-cell offset.
  head        "head at T" = the last cell whose arrival time is <= T.
  turn        a direction change at a mirror is stamped with that cell's arrival
              time; a half-mirror copy counts as a turn of its colour.
  exit        a beam leaves the board at (arrival at its last cell) + 1.0 * step.
  stop        hitting a dark block (absorber) or the hatched back of a one-sided
              mirror stops the pulse at (arrival) + 0.5 * step.
  reach       "when does light first reach mirror C" = arrival + 0.5 * step.
  one-sided   `face` is the outward normal of the reflective side; a pulse whose
              travel direction d has d . face > 0 hits the BACK and stops.
  pivoting    `kind` is shown until `flip_t`, the opposite diagonal after it.
              (No arrival lands exactly on flip_t, so >= vs > is unpinned.)
  clock fmt   clock readings truncate to 1 decimal; durations round to 1 decimal.
  clip end    "at the end of the clip" = (frames - 1) / 24 s (last drawn frame);
              "how long after the clip ends" = frames / 24 s.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
REF = {"/": {"R": "U", "L": "D", "D": "L", "U": "R"},
       "\\": {"R": "D", "L": "U", "D": "R", "U": "L"}}
OTHER = {"/": "\\", "\\": "/"}
NAME = {"U": "up", "D": "down", "L": "left", "R": "right"}
COLOURS = ("magenta", "blue", "green", "red", "orange")
CELL_RE = re.compile(r"\b[A-J][1-8]\b")


def cell_name(x: int, y: int) -> str:
    return f"{chr(65 + x)}{y + 1}"


def clock(t: float) -> str:
    """Clock readings truncate to one decimal."""
    return f"{math.floor(round(t, 6) * 10) / 10:.1f} s"


def duration(t: float) -> str:
    """Durations round to one decimal."""
    return f"{t:.1f} s"


class Beam:
    __slots__ = ("colour", "stations", "turns", "outcome", "t_end", "stop_cell", "exit", "gen")

    def __init__(self, colour: str, gen: int = 0):
        self.colour = colour
        self.stations: list[tuple[tuple[int, int], float]] = []
        self.turns: list[tuple[tuple[int, int], float, str]] = []
        self.outcome = "running"
        self.t_end: float | None = None
        self.stop_cell = None
        self.exit = None
        self.gen = gen

    def head(self, t: float):
        seen = [p for p, tt in self.stations if tt <= t + 1e-9]
        return seen[-1] if seen else None

    def turns_by(self, t: float) -> int:
        return sum(1 for _p, tt, _d in self.turns if tt <= t + 1e-9)


class MirrorBeam:
    def __init__(self, params: dict):
        self.p = params
        self.cols = int(params["cols"])
        self.rows = int(params["rows"])
        self.step = float(params["step"])
        self.mirrors = {tuple(m["cell"]): dict(m) for m in params.get("mirrors", [])}
        self.absorbers = {tuple(a) for a in params.get("absorbers", [])}
        self.emitters = list(params.get("emitters") or [])
        self.frames = int(params.get("frames") or 0)

    # ---------------- simulation ----------------
    def simulate(self, *, pivot: bool = True, step_mul: float = 1.0,
                 remove=None, add=None, flip_cell=None, silver: bool = False,
                 no_absorbers: bool = False, delay: float = 0.0) -> dict[str, list[Beam]]:
        step = self.step * step_mul
        M = {c: dict(m) for c, m in self.mirrors.items()}
        for c in (remove or ()):
            if c is not None:
                M.pop(tuple(c), None)
        for c, kind in (add or {}).items():
            if c is not None:
                M[tuple(c)] = {"cell": list(c), "kind": kind, "type": "two"}
        if flip_cell is not None and tuple(flip_cell) in M:
            M[tuple(flip_cell)] = dict(M[tuple(flip_cell)],
                                       kind=OTHER[M[tuple(flip_cell)]["kind"]])
        if silver:
            for m in M.values():
                if m.get("type") in ("one", "split", "flip"):
                    m["type"] = "two"
                    m.pop("face", None)
        A = set() if no_absorbers else self.absorbers
        out: dict[str, list[Beam]] = {}

        def trace(pos, d, t, beam: Beam):
            budget = [12]          # split copies per emitter
            queue = [(pos, d, t, beam, False)]
            while queue:
                pos, d, t, b, skip = queue.pop(0)
                for _ in range(600):
                    if not skip:
                        m = M.get(pos)
                        if m:
                            kind, typ = m["kind"], m.get("type", "two")
                            if typ == "one":
                                face = m.get("face") or (0, 0)
                                dv = DIRS[d]
                                if dv[0] * face[0] + dv[1] * face[1] > 0:
                                    b.outcome, b.stop_cell = "stop", pos
                                    b.t_end = t + step / 2
                                    break
                            if typ == "flip" and pivot and t >= float(m.get("flip_t", 0.0)) - 1e-9:
                                kind = OTHER[kind]
                            nd = REF[kind][d]
                            if typ == "split":
                                b.turns.append((pos, t, nd))
                                if budget[0] > 0:
                                    budget[0] -= 1
                                    cb = Beam(b.colour, b.gen + 1)
                                    cb.stations.append((pos, t))
                                    out.setdefault(b.colour, []).append(cb)
                                    queue.append((pos, nd, t, cb, True))
                            else:
                                if nd != d:
                                    b.turns.append((pos, t, nd))
                                d = nd
                    skip = False
                    nx = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
                    if not (0 <= nx[0] < self.cols and 0 <= nx[1] < self.rows):
                        b.outcome, b.t_end = "exit", t + step
                        if nx[0] < 0:
                            b.exit = ("left", pos[1], f"leaves through the left edge at row {pos[1] + 1}")
                        elif nx[0] >= self.cols:
                            b.exit = ("right", pos[1], f"leaves through the right edge at row {pos[1] + 1}")
                        elif nx[1] < 0:
                            b.exit = ("top", pos[0], f"leaves through the top edge at column {chr(65 + pos[0])}")
                        else:
                            b.exit = ("bottom", pos[0], f"leaves through the bottom edge at column {chr(65 + pos[0])}")
                        break
                    pos = nx
                    t += step
                    b.stations.append((pos, t))
                    if pos in A:
                        b.outcome, b.stop_cell = "stop", pos
                        b.t_end = t + step / 2
                        break
                else:
                    b.outcome, b.t_end = "loop", t

        for em in self.emitters:
            edge, idx = em["edge"], int(em["index"])
            if edge == "top":
                pos, d = (idx, 0), "D"
            elif edge == "bottom":
                pos, d = (idx, self.rows - 1), "U"
            elif edge == "left":
                pos, d = (0, idx), "R"
            else:
                pos, d = (self.cols - 1, idx), "L"
            t = float(em.get("t0") or 0.0) + delay
            b = Beam(em.get("colour", "magenta"))
            b.stations.append((pos, t))
            out.setdefault(b.colour, []).append(b)
            trace(pos, d, t, b)
        return out

    # ---------------- helpers ----------------
    @staticmethod
    def beams(res) -> list[Beam]:
        return [b for bs in res.values() for b in bs]

    @staticmethod
    def colour(q: str) -> str | None:
        words = set(re.findall("[a-z]+", q.lower()))
        for c in COLOURS:
            if c in words:
                return c
        return None

    def cell(self, text: str):
        m = CELL_RE.search(text.upper())
        return (ord(m.group(0)[0]) - 65, int(m.group(0)[1:]) - 1) if m else None

    @staticmethod
    def outcome(b: Beam) -> str:
        if b.outcome == "exit":
            return b.exit[2]
        return f"is stopped in cell {cell_name(*b.stop_cell)}"

    def colour_outcome(self, res, colour: str) -> str | None:
        ends = [b for b in res.get(colour, []) if b.outcome in ("exit", "stop")]
        if not ends:
            return None
        return self.outcome(min(ends, key=lambda b: b.t_end))

    @staticmethod
    def exits(res) -> int:
        return len({b.exit[:2] for b in MirrorBeam.beams(res) if b.outcome == "exit"})

    def final(self, res, colour: str):
        """Event time of the colour's first-finishing beam, or None."""
        ends = [b for b in res.get(colour, []) if b.t_end is not None]
        if not ends:
            return None
        return min(ends, key=lambda b: b.t_end)

    @staticmethod
    def time_of(q: str, word: str = "at") -> float | None:
        m = re.search(rf"{word} ([\d.]+) s\b", q)
        return float(m.group(1)) if m else None

    # ---------------- answering ----------------
    def answer(self, question: str, options=None) -> Answer:
        q = re.sub(r"\(Rules:.*$", "", question, flags=re.S).strip()
        ql = q.lower()

        # ---- static counts ----
        if "how many dark blocks" in ql:
            return Answer(str(len(self.absorbers)), confidence="exact")
        if "how many mirrors are on the board" in ql:
            return Answer(str(len(self.mirrors)), confidence="exact")
        if "lean like" in ql:
            return Answer(str(sum(1 for m in self.mirrors.values() if m["kind"] == "/")), confidence="exact")
        if "drawn dashed" in ql:
            return Answer(str(sum(1 for m in self.mirrors.values() if m.get("type") == "split")), confidence="exact")
        if "round pivot hub" in ql:
            return Answer(str(sum(1 for m in self.mirrors.values() if m.get("type") == "flip")), confidence="exact")
        if "hatched backing" in ql:
            return Answer(str(sum(1 for m in self.mirrors.values() if m.get("type") == "one")), confidence="exact")
        if "in cells per second" in ql:
            return Answer(f"{1.0 / self.step:.1f} cells per second", confidence="exact")
        if "from which edge" in ql:
            c = self.colour(ql)
            for e in self.emitters:
                if e.get("colour") == c:
                    return Answer(e["edge"], confidence="exact")
            return Answer(None, confidence="unsupported", note="emitter colour not found")

        # ---- counterfactual board edits ----
        add, remove, flip_cell, silver, pivot, step_mul, delay, no_abs = {}, None, None, False, True, 1.0, 0.0, False
        m = re.search(r"extra two-sided mirror leaning like '(/|\\)' stood in the empty cell ([A-J][1-8])", q)
        if m:
            add = {self.cell(m.group(2)): m.group(1)}
        if "were taken away" in ql:
            remove = (self.cell(q),)
        if "leaned the other way" in ql:
            flip_cell = self.cell(q)
        if "silvered on both sides" in ql:
            silver = True
        if "pivoting mirror(s) never turned" in ql:
            pivot = False
        if "twice as fast" in ql:
            step_mul = 0.5
        if "dark blocks were removed" in ql:
            no_abs = True
        m = re.search(r"launched ([\d.]+) s later", ql)
        if m:
            delay = float(m.group(1))

        if "how many places" in ql:
            res = self.simulate(remove=remove, add=add, flip_cell=flip_cell, silver=silver,
                                no_absorbers=no_abs)
            return Answer(str(self.exits(res)), confidence="exact")

        res = self.simulate(remove=remove, add=add, flip_cell=flip_cell, silver=silver,
                            pivot=pivot, step_mul=step_mul, delay=delay, no_absorbers=no_abs)
        if any(b.outcome == "loop" for b in self.beams(res)):
            return Answer(None, confidence="unsupported", note="beam cycles forever")
        clip_end = self.frames / 24.0
        end_frame = (self.frames - 1) / 24.0

        c = self.colour(ql)

        # ---- outcome families ----
        if "end up" in ql:
            return Answer(self.colour_outcome(res, c), confidence="exact")
        if "how long after the clip ends" in ql:
            b = self.final(res, c)
            if b is None:
                return Answer(None, confidence="unsupported", note="beam never ends")
            return Answer(duration(max(0.0, b.t_end - clip_end)), confidence="exact")
        if "at what clock reading would" in ql:
            b = self.final(res, c)
            if b is None:
                return Answer(None, confidence="unsupported", note="beam never ends")
            return Answer(clock(b.t_end), confidence="exact")

        # ---- head / turns ----
        t = self.time_of(ql, "at")
        if "moving light heads" in ql and t is not None:
            n = 0
            for b in self.beams(res):
                if b.stations[0][1] <= t + 1e-9 and (b.t_end is None or b.t_end > t + 1e-9):
                    n += 1
            return Answer(str(n), confidence="exact")
        if "in which cell is the head" in ql and t is not None:
            heads = [b.head(t) for b in res.get(c, [])]
            heads = [h for h in heads if h is not None]
            if not heads:
                return Answer(None, confidence="unsupported", note="no head at that time")
            return Answer(cell_name(*heads[0]), confidence="exact")
        t = self.time_of(ql, "by")
        if "how many times has" in ql and t is not None:
            return Answer(str(sum(b.turns_by(t) for b in res.get(c, []))), confidence="exact")
        if "turn for the first time" in ql:
            turns = [(tt, p_) for b in res.get(c, []) for p_, tt, _d in b.turns]
            if not turns:
                return Answer(None, confidence="unsupported", note="no turn")
            return Answer(cell_name(*min(turns)[1]), confidence="exact")
        if "after its first turn" in ql:
            turns = sorted((tt, d) for b in res.get(c, []) for _p, tt, d in b.turns)
            if not turns:
                return Answer(None, confidence="unsupported", note="no turn")
            return Answer(NAME[turns[0][1]], confidence="exact")
        if "how many times in total" in ql:
            return Answer(str(sum(len(b.turns) for b in res.get(c, []))), confidence="exact")
        if "first split into two beams" in ql:
            splits = {c2 for c2, mm in self.mirrors.items() if mm.get("type") == "split"}
            hit = [(tt, p_) for b in self.beams(res) for p_, tt, _d in b.turns if p_ in splits]
            if not hit:
                return Answer(None, confidence="unsupported", note="no split")
            return Answer(cell_name(*min(hit)[1]), confidence="exact")

        # ---- timing / geometry ----
        if "when does light first reach" in ql:
            cell = self.cell(q)
            ts = [tt for b in self.beams(res) for p_, tt in b.stations if p_ == cell]
            if not ts:
                return Answer(None, confidence="unsupported", note="cell never reached")
            return Answer(clock(min(ts) + self.step * step_mul / 2), confidence="exact")
        if "how long does the light take" in ql:
            cells = CELL_RE.findall(q.upper())
            a, b2 = self.cell(cells[0]), self.cell(cells[1])
            for bm in self.beams(res):
                ta = next((tt for p_, tt in bm.stations if p_ == a), None)
                tb = next((tt for p_, tt in bm.stations if p_ == b2 and ta is not None and tt > ta), None)
                if ta is not None and tb is not None:
                    return Answer(duration(tb - ta), confidence="exact")
            return Answer(None, confidence="unsupported", note="leg not on one beam")
        if "how much time passes between" in ql:
            cells = CELL_RE.findall(q.upper())
            a, b2 = self.cell(cells[0]), self.cell(cells[1])
            for bm in res.get(c, []):
                turns = [(p_, tt) for p_, tt, _d in bm.turns]
                for i, (p_, tt) in enumerate(turns):
                    if p_ == a:
                        for p2, t2 in turns[i + 1:]:
                            if p2 == b2:
                                return Answer(duration(t2 - tt), confidence="exact")
            return Answer(None, confidence="unsupported", note="turn pair not found")
        if "how many cells does" in ql and "move from" in ql:
            cells = CELL_RE.findall(q.upper())
            a, b2 = self.cell(cells[0]), self.cell(cells[1])
            for bm in res.get(c, []):
                turns = [(p_, tt) for p_, tt, _d in bm.turns]
                for i, (p_, tt) in enumerate(turns):
                    if p_ == a:
                        for p2, t2 in turns[i + 1:]:
                            if p2 == b2:
                                return Answer(str(round((t2 - tt) / (self.step * step_mul))), confidence="exact")
            return Answer(None, confidence="unsupported", note="turn pair not found")
        if "lit trail" in ql:
            cells = set()
            for b in self.beams(res):
                for p_, tt in b.stations:
                    if tt <= end_frame + 1e-9:
                        cells.add(p_)
            return Answer(str(len(cells)), confidence="exact")
        if "beam start" in ql:
            bs = res.get(c, [])
            if not bs:
                return Answer(None, confidence="unsupported", note="colour not found")
            return Answer(clock(bs[0].stations[0][1]), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

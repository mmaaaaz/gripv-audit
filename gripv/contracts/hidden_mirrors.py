"""Deterministic replay for hidden_mirrors.

Board model (all of it comes from `params`):

  * a rectangular cell grid, `cols` x `rows`, cell (x, y) named
    ``chr(65+x) + str(y+1)`` (the video draws exactly those column letters and
    row numbers);
  * `mirrors` carries **every** mirror, shown or hidden, each with its diagonal
    `kind` and a `shown` flag — the pulses obey all of them, the viewer only
    sees the `shown` ones;
  * `blocks` are dark blocks: a pulse entering one stops there;
  * `pulses` are (colour, edge, index, t0): a pulse enters on `edge` at port
    `index` (the row for left/right, the column for top/bottom) and travels one
    cell per `step` seconds.

Conventions pinned against the keys (full 25,000/25,000 exact):

  grid        cell (x, y); columns A.. left-to-right, rows 1.. top-to-bottom.
  port        right index i -> (cols-1, i) moving left; left -> (0, i) moving
              right; top -> (i, 0) moving down; bottom -> (i, rows-1) moving up.
  arrival     the pulse is at the centre of cell n at t = t0 + n * step, with
              n = 0 on the entry cell (no half-cell offset).
  mirror      a cell holding a mirror turns the pulse 90 degrees on the
              diagonal it shows; "/" maps right->up and "\\" maps right->down.
              A mirror on the entry cell turns the pulse at t = t0 (that is a
              change of direction like any other).
  exit        the pulse leaves at (arrival on its last cell) + 1.0 * step, and
              exits through the edge it was heading for:
              left/right -> "leaves at the left|right end of row <y+1>",
              up/down    -> "leaves at the top|bottom of column <letter>".
  stop        a dark block stops the pulse at (arrival) + 0.5 * step.
  clock fmt   clock readings truncate to 1 decimal ("what does the clock
              show"); durations round to 1 decimal ("how long", "how much time
              passes").
  speed       "to the nearest half cell per second" = round(1/step * 2) / 2.
  counters    "cells passed through, counting the cells where it turns" counts
              every cell the pulse occupies, entry cell and turn cells
              included.
  no-pulse    "how many cells does no pulse ever enter" = all cells minus the
              cells at least one pulse occupied; the "(not counting the dark
              blocks)" variant removes the blocks from the board total *and*
              from the visited set.
  inference   "must hold a hidden mirror" = a cell some pulse turned in whose
              mirror is not drawn; "certainly has no mirror" = a cell some
              pulse occupied without turning there.
  counterfactual
              "a new pulse fired exactly where X leaves" enters on X's exit
              cell travelling back into the board (mirrors are two-sided, so it
              retraces X's path); "one more pulse at the right end of row N"
              enters that port at t = 0.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
REF = {
    "/": {"R": "U", "L": "D", "D": "L", "U": "R"},
    "\\": {"R": "D", "L": "U", "D": "R", "U": "L"},
}
NAME = {"U": "up", "D": "down", "L": "left", "R": "right"}
COLOURS = ("orange", "red", "gold", "blue", "pink", "green", "purple", "white", "black")
CELL_RE = re.compile(r"\b([A-J])([1-8])\b")
CELLS_RE = re.compile(r"\b[A-J][1-8]\b", re.I)


def cell_name(x: int, y: int) -> str:
    return f"{chr(65 + x)}{y + 1}"


def clock(t: float) -> str:
    """Clock readings truncate to one decimal."""
    return f"{math.floor(round(t, 6) * 10) / 10:.1f} s"


def duration(t: float) -> str:
    """Durations round to one decimal."""
    return f"{t:.1f} s"


def half(x: float) -> str:
    return f"{round(x * 2) / 2:.1f} cells per second"


class Pulse:
    __slots__ = ("colour", "stations", "turns", "outcome", "t_end", "exit_dir", "stop_cell")

    def __init__(self, colour: str):
        self.colour = colour
        self.stations: list[tuple[tuple[int, int], float]] = []
        self.turns: list[tuple[tuple[int, int], float, str]] = []
        self.outcome = "running"
        self.t_end: float | None = None
        self.exit_dir: str | None = None
        self.stop_cell: tuple[int, int] | None = None


class HiddenMirrors:
    def __init__(self, params: dict):
        self.p = params
        self.cols = int(params["cols"])
        self.rows = int(params["rows"])
        self.step = float(params["step"])
        self.mirrors = {tuple(m["cell"]): {"kind": m["kind"], "shown": bool(m.get("shown"))}
                        for m in params.get("mirrors") or []}
        self.blocks = {tuple(b) for b in params.get("blocks") or []}
        self.pulses = list(params.get("pulses") or [])
        self.frames = int(params.get("frames") or 0)

    # ---------------- geometry ----------------
    def port(self, edge: str, index: int) -> tuple[tuple[int, int], str]:
        if edge == "right":
            return (self.cols - 1, index), "L"
        if edge == "left":
            return (0, index), "R"
        if edge == "top":
            return (index, 0), "D"
        return (index, self.rows - 1), "U"

    def port_text(self, edge: str, index: int) -> str:
        if edge == "right":
            return f"right end of row {index + 1}"
        if edge == "left":
            return f"left end of row {index + 1}"
        if edge == "top":
            return f"top of column {chr(65 + index)}"
        return f"bottom of column {chr(65 + index)}"

    def edge_text(self, pos: tuple[int, int], d: str) -> str:
        if d == "L":
            return f"leaves at the left end of row {pos[1] + 1}"
        if d == "R":
            return f"leaves at the right end of row {pos[1] + 1}"
        if d == "U":
            return f"leaves at the top of column {chr(65 + pos[0])}"
        return f"leaves at the bottom of column {chr(65 + pos[0])}"

    @staticmethod
    def outcome_text(pulse: Pulse) -> str | None:
        if pulse.outcome == "exit":
            return None  # filled by caller (needs the cell)
        if pulse.outcome == "block":
            return f"is stopped by the block in {cell_name(*pulse.stop_cell)}"
        return None

    # ---------------- simulation ----------------
    def _trace(self, pos, d, t, step, M, B, pulse: Pulse) -> None:
        for _ in range(400):
            m = M.get(pos)
            if m is not None:
                nd = REF[m["kind"]][d]
                if nd != d:
                    pulse.turns.append((pos, t, nd))
                    d = nd
            nx = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
            if not (0 <= nx[0] < self.cols and 0 <= nx[1] < self.rows):
                pulse.outcome = "exit"
                pulse.exit_dir = d
                pulse.t_end = t + step
                return
            pos = nx
            t += step
            pulse.stations.append((pos, t))
            if pos in B:
                pulse.outcome = "block"
                pulse.stop_cell = pos
                pulse.t_end = t + step / 2
                return
        pulse.outcome = "loop"
        pulse.t_end = t

    def _board(self, remove_mirror=None, flip_cell=None, remove_block=None):
        M = {c: dict(m) for c, m in self.mirrors.items()}
        if remove_mirror is not None:
            M.pop(tuple(remove_mirror), None)
        if flip_cell is not None and tuple(flip_cell) in M:
            k = M[tuple(flip_cell)]["kind"]
            M[tuple(flip_cell)]["kind"] = "\\" if k == "/" else "/"
        B = set(self.blocks)
        if remove_block is not None:
            B.discard(tuple(remove_block))
        return M, B

    def simulate(self, *, remove_mirror=None, flip_cell=None, remove_block=None,
                 speed: float = 1.0, extra=()) -> tuple[dict[str, Pulse], list[Pulse]]:
        """Replay every pulse; returns ({colour: Pulse}, [extra pulses])."""
        M, B = self._board(remove_mirror, flip_cell, remove_block)
        step = self.step / speed
        out: dict[str, Pulse] = {}
        for pu in self.pulses:
            pos, d = self.port(pu["edge"], int(pu["index"]))
            t = float(pu.get("t0") or 0.0)
            p = Pulse(pu.get("colour", "white"))
            p.stations.append((pos, t))
            out[p.colour] = p
            if pos in B:
                p.outcome, p.stop_cell, p.t_end = "block", pos, t + step / 2
                continue
            self._trace(pos, d, t, step, M, B, p)
        extras: list[Pulse] = []
        for i, (edge, index, start_dir) in enumerate(extra):
            t = 0.0
            pos = (index[0], index[1]) if isinstance(index, tuple) else self.port(edge, index)[0]
            d = start_dir or self.port(edge, index)[1]
            p = Pulse(f"extra{i}")
            p.stations.append((pos, t))
            if pos in B:
                p.outcome, p.stop_cell, p.t_end = "block", pos, t + step / 2
            else:
                self._trace(pos, d, t, step, M, B, p)
            extras.append(p)
        return out, extras

    # ---------------- helpers ----------------
    @staticmethod
    def colours_in(q: str) -> list[str]:
        words = re.findall(r"[a-z]+", q.lower())
        out: list[str] = []
        for w in words:
            if w in COLOURS and w not in out:
                out.append(w)
        return out

    def colour(self, q: str) -> str | None:
        c = self.colours_in(q)
        return c[0] if c else None

    @staticmethod
    def cells_in(q: str) -> list[tuple[int, int]]:
        return [(ord(m.group(1).upper()) - 65, int(m.group(2)) - 1) for m in CELL_RE.finditer(q)]

    def describe(self, p: Pulse) -> str | None:
        if p.outcome == "exit":
            return self.edge_text(p.stations[-1][0], p.exit_dir)
        if p.outcome == "block":
            return f"is stopped by the block in {cell_name(*p.stop_cell)}"
        return None

    def visited(self, res: dict[str, Pulse], extras: list[Pulse] = ()) -> set:
        cells = set()
        for p in list(res.values()) + list(extras):
            cells.update(pos for pos, _t in p.stations)
        return cells

    def turn_cells(self, res) -> set:
        return {pos for p in res.values() for pos, _t, _d in p.turns}

    # ---------------- answering ----------------
    def answer(self, question: str, options=None) -> Answer:
        q = str(question)
        ql = q.lower()

        # ---------- static board facts ----------
        if "how many light pulses are fired" in ql:
            return Answer(str(len(self.pulses)), confidence="exact")
        if "how many mirrors are drawn on the board" in ql:
            return Answer(str(sum(1 for m in self.mirrors.values() if m["shown"])), confidence="exact")
        if "how many dark blocks are on the board" in ql:
            return Answer(str(len(self.blocks)), confidence="exact")
        if "about how fast do the pulses travel" in ql:
            return Answer(half(1.0 / self.step), confidence="exact")
        if "which pulse is fired first" in ql:
            if not self.pulses:
                return Answer(None, confidence="unsupported", note="no pulses")
            first = min(self.pulses, key=lambda p: float(p.get("t0") or 0.0))
            return Answer(str(first["colour"]), confidence="exact")

        base, extras = self.simulate()

        if "how many of the pulses cross the board without ever changing direction" in ql:
            return Answer(str(sum(1 for p in base.values() if not p.turns)), confidence="exact")
        if "how many changes of direction do all the pulses make altogether" in ql:
            return Answer(str(sum(len(p.turns) for p in base.values())), confidence="exact")
        if "in how many different cells does some pulse change direction" in ql:
            return Answer(str(len(self.turn_cells(base))), confidence="exact")
        if "no pulse ever enter" in ql:
            visited = self.visited(base)
            if "not counting the dark blocks" in ql:
                total = self.cols * self.rows - len(self.blocks)
                return Answer(str(total - len(visited - self.blocks)), confidence="exact")
            return Answer(str(self.cols * self.rows - len(visited)), confidence="exact")
        if "must hold a hidden mirror" in ql:
            cells = self.cells_in(q)
            hidden_turns = {pos for p in base.values() for pos, _t, _d in p.turns
                            if not self.mirrors.get(pos, {}).get("shown", True)}
            hits = [c for c in cells if c in hidden_turns]
            if len(hits) == 1:
                return Answer(cell_name(*hits[0]), confidence="exact")
            return Answer(cell_name(*hits[0]) if hits else None, confidence="exact" if hits else "unsupported",
                          note=f"hidden-turn candidates={[cell_name(*c) for c in hits]}")
        if "certainly has no mirror" in ql:
            cells = self.cells_in(q)
            visited = self.visited(base)
            turns = self.turn_cells(base)
            blank = visited - turns - self.blocks
            hits = [c for c in cells if c in blank]
            if len(hits) == 1:
                return Answer(cell_name(*hits[0]), confidence="exact")
            return Answer(cell_name(*hits[0]) if hits else None, confidence="exact" if hits else "unsupported",
                          note=f"no-mirror candidates={[cell_name(*c) for c in hits]}")

        # ---------- ordering / timing of the base run ----------
        if "in what order are" in ql and "fired" in ql:
            cols = self.colours_in(q)
            got = [p["colour"] for p in sorted(self.pulses, key=lambda p: float(p.get("t0") or 0.0))
                   if p["colour"] in cols]
            return Answer(", ".join(got), confidence="exact")
        if "in what order do" in ql and "leave the board" in ql:
            cols = self.colours_in(q)
            ends = [p for p in base.values() if p.colour in cols and p.t_end is not None]
            if len(ends) < len(cols):
                return Answer(None, confidence="unsupported", note="a pulse never ends")
            ends.sort(key=lambda p: p.t_end)
            return Answer(", ".join(p.colour for p in ends), confidence="exact")
        if "how much time passes between" in ql:
            cols = self.colours_in(q)
            if len(cols) < 2:
                return Answer(None, confidence="unsupported", note="two colours expected")
            t0s = {p["colour"]: float(p.get("t0") or 0.0) for p in self.pulses}
            if cols[0] not in t0s or cols[1] not in t0s:
                return Answer(None, confidence="unsupported", note="colour not fired")
            return Answer(duration(abs(t0s[cols[1]] - t0s[cols[0]])), confidence="exact")

        c = self.colour(q)

        # ---------- counterfactuals ----------
        if "if every pulse moved twice as fast" in ql:
            fast, _ = self.simulate(speed=2.0)
            p = fast.get(c)
            if p is None or p.t_end is None:
                return Answer(None, confidence="unsupported", note="pulse never ends")
            return Answer(clock(p.t_end), confidence="exact")

        if "if a new pulse were fired into the board exactly where" in ql:
            src = base.get(c)
            if src is None or src.outcome != "exit":
                return Answer(None, confidence="unsupported", note="source pulse does not leave")
            cell = src.stations[-1][0]
            back = {"L": "R", "R": "L", "U": "D", "D": "U"}[src.exit_dir]
            _, ex = self.simulate(extra=(("", cell, back),))
            return Answer(self.describe(ex[0]), confidence="exact")

        m = re.search(r"if one more pulse were fired in at the (right|left) end of row (\d+)", ql)
        if m:
            idx = int(m.group(2)) - 1
            _, ex = self.simulate(extra=((m.group(1), idx, None),))
            return Answer(self.describe(ex[0]), confidence="exact")
        m = re.search(r"if one more pulse were fired in at the (top|bottom) of column ([a-j])", ql)
        if m:
            idx = ord(m.group(2).upper()) - 65
            _, ex = self.simulate(extra=((m.group(1), idx, None),))
            return Answer(self.describe(ex[0]), confidence="exact")

        cell = self.cells_in(q)[0] if self.cells_in(q) else None
        if "if there were no mirror in" in ql and cell is not None:
            res, _ = self.simulate(remove_mirror=cell)
            p = res.get(c)
            if p is None:
                return Answer(None, confidence="unsupported", note="colour not fired")
            if "how many times" in ql:
                return Answer(str(len(p.turns)), confidence="exact")
            return Answer(self.describe(p), confidence="exact")
        if "if the mirror in" in ql and "leaned the other way" in ql and cell is not None:
            res, _ = self.simulate(flip_cell=cell)
            p = res.get(c)
            if p is None:
                return Answer(None, confidence="unsupported", note="colour not fired")
            if "how many times" in ql:
                return Answer(str(len(p.turns)), confidence="exact")
            return Answer(self.describe(p), confidence="exact")
        if "if the dark block in" in ql and "were taken away" in ql and cell is not None:
            res, _ = self.simulate(remove_block=cell)
            p = res.get(c)
            if p is None:
                return Answer(None, confidence="unsupported", note="colour not fired")
            if "how many times" in ql:
                return Answer(str(len(p.turns)), confidence="exact")
            return Answer(self.describe(p), confidence="exact")

        # ---------- per-pulse facts ----------
        p = base.get(c)
        if p is None:
            return Answer(None, confidence="unsupported", note=f"colour {c!r} not fired")

        if "changes direction for the" in ql and "hidden mirror" in ql:
            word = re.search(r"for the (first|second|third|fourth) time", ql)
            k = {"first": 1, "second": 2, "third": 3, "fourth": 4}.get(word.group(1) if word else "first", 1)
            if len(p.turns) < k:
                return Answer(None, confidence="unsupported", note=f"fewer than {k} turns")
            pos = p.turns[k - 1][0]
            return Answer(f"{cell_name(*pos)}, {self.mirrors[pos]['kind']}", confidence="exact")
        if "after its first change of direction" in ql:
            if not p.turns:
                return Answer(None, confidence="unsupported", note="pulse never turns")
            return Answer(NAME[p.turns[0][2]], confidence="exact")
        if "in which cell does" in ql and "change direction for the first time" in ql:
            if not p.turns:
                return Answer(None, confidence="unsupported", note="pulse never turns")
            return Answer(cell_name(*p.turns[0][0]), confidence="exact")
        if "how many times does" in ql and "change direction" in ql:
            return Answer(str(len(p.turns)), confidence="exact")
        if "how many cells does" in ql and "pass through" in ql:
            return Answer(str(len(p.stations)), confidence="exact")
        if "where is the" in ql and "pulse fired into the board" in ql:
            for pu in self.pulses:
                if pu["colour"] == c:
                    return Answer(self.port_text(pu["edge"], int(pu["index"])), confidence="exact")
            return Answer(None, confidence="unsupported", note="colour not fired")
        if "for how long is the" in ql and "on the board" in ql:
            if p.t_end is None:
                return Answer(None, confidence="unsupported", note="pulse never ends")
            return Answer(duration(p.t_end - p.stations[0][1]), confidence="exact")
        if "what does the clock show when" in ql:
            if "changes direction for the first time" in ql:
                if not p.turns:
                    return Answer(None, confidence="unsupported", note="pulse never turns")
                # a turn is stamped half a step after the pulse reaches the
                # mirror's cell (the interaction instant), same as "when does
                # the light reach X" in mirror_beam_reflection.
                return Answer(clock(p.turns[0][1] + self.step / 2), confidence="exact")
            if p.t_end is None:
                return Answer(None, confidence="unsupported", note="pulse never ends")
            return Answer(clock(p.t_end), confidence="exact")
        if "where does the" in ql and "pulse leave the board" in ql:
            return Answer(self.describe(p), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

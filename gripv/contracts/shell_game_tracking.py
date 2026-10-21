"""Shell-game tracking: cups swap on a table; where does each token end up?

The recipe (`params`) pins the whole game:

  scenario  classic | grid | peek | multi | rotate
  n         number of cups (3-6)
  slots     [x, y] centre of every cup position, in index order
            (row-major when ``scenario == "grid"``, left-to-right otherwise)
  cols      columns in the grid layout (grid only: 2 or 3)
  tokens    [{colour, shape, slot}] tokens placed under the cup in ``slot``
  moves     [{start, dur, perm}] one "move"; ``perm`` is a list of
            ``[from_slot, to_slot]`` pairs applied simultaneously (a 2-cycle
            for a plain swap, a 3-cycle for "three cups turning", two 2-cycles
            for "two swaps at once")
  peeks     [{start, up, hold, down, slot}] a single cup lifted to show the
            table beneath it
  n_frames  video length in frames

Conventions calibrated on the 25,000 questions (all literal matches):

* The clock is 24 fps and the first frame is time zero, so a frame ``f``
  displays ``(f - 1) / 24`` seconds, rounded to one decimal with ties going
  down (the generator's own rounding: 1.75 s -> "1.7 s").
* A move *starts* on the clock at ``(start - 1) / 24``.  A peek is announced
  at mid-lift, ``(start + 2) / 24`` — ``up`` is 4 frames.
* "When the clock shows T" reads the board after every move that has finished
  by the frame showing T; the generator never samples a moment inside a move.
* A cup's position is the slot index; grid positions are named
  top/bottom x left(/middle)/right by column count, linear positions are
  leftmost / "the Nth position from the left" / rightmost, with "middle"
  reserved for the three-cup line.
* "Before the last swap" excludes the last move itself.
* Counterfactuals ("had never happened", "opposite order", "trade places once
  more") replay the same ``perm`` list with the requested edit.
"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_DOWN

from ..engine import Answer

FPS = 24
ORDW = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
ORD = {w: i + 1 for i, w in enumerate(ORDW)}
ORD["last"] = -1
COLOURS = "blue cyan green orange pink purple red yellow".split()
SHAPES = "ball cube diamond star triangle".split()
TOKRE = (
    r"the (blue|cyan|green|orange|pink|purple|red|yellow) "
    r"(ball|cube|diamond|star|triangle)"
)
POSRE = (
    r"the (?:top|bottom)-(?:left|middle|right) position"
    r"|the (?:leftmost|rightmost|middle) position"
    r"|the (?:first|second|third|fourth|fifth|sixth) position from the (?:left|right)"
)


def clock(t: float) -> str:
    """Frame-time -> displayed clock string, generator rounding (ties down)."""
    d = Decimal(str(round(t, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_DOWN)
    return f"{d:.1f} s"


class ShellGameTracking:
    def __init__(self, params: dict):
        self.p = params
        self.n = int(params["n"])
        self.moves = list(params.get("moves") or [])
        self.peeks = list(params.get("peeks") or [])
        self.tokens = list(params.get("tokens") or [])
        self.slots = params.get("slots") or []
        self._names = self._position_names()

    # ------------------------------------------------------------------ names
    def _position_names(self) -> list[str]:
        n = self.n
        out: list[str] = []
        if self.p.get("scenario") == "grid":
            cols = int(self.p.get("cols") or 3)
            labels = ["left", "right"] if cols == 2 else ["left", "middle", "right"]
            for i in range(n):
                row, col = divmod(i, cols)
                out.append(f"the {'top' if row == 0 else 'bottom'}-{labels[col]} position")
        else:
            for i in range(n):
                if i == 0:
                    out.append("the leftmost position")
                elif i == n - 1:
                    out.append("the rightmost position")
                elif n == 3 and i == 1:
                    out.append("the middle position")
                else:
                    out.append(f"the {ORDW[i]} position from the left")
        return out

    def _slot(self, phrase: str) -> int | None:
        phrase = phrase.strip().rstrip("?.,")
        try:
            return self._names.index(phrase)
        except ValueError:
            return None

    def _token(self, colour: str, shape: str) -> dict | None:
        for t in self.tokens:
            if t["colour"] == colour and t["shape"] == shape:
                return t
        return None

    @staticmethod
    def _tk(t: dict) -> str:
        return f"the {t['colour']} {t['shape']}"

    def _x(self, slot: int) -> float:
        try:
            return float(self.slots[slot][0])
        except Exception:
            return float(slot)

    # ------------------------------------------------------------- simulation
    def _arr(self, moves=None, upto: int | None = None) -> list[int]:
        """arr[slot] = index of the cup standing there originally."""
        arr = list(range(self.n))
        for m in self.moves if moves is None else moves:
            if upto is not None and not (m["start"] + m["dur"] <= upto):
                continue
            new = arr.copy()
            for a, b in m["perm"]:
                new[b] = arr[a]
            arr = new
        return arr

    def _end_slot(self, start_slot: int, moves=None) -> int:
        return self._arr(moves).index(start_slot)

    def _slot_of_cup_at(self, start_slot: int, frame: int) -> int:
        return self._arr(upto=frame).index(start_slot)

    def _move_index(self, word: str) -> int | None:
        o = ORD.get(word)
        if o is None:
            return None
        idx = len(self.moves) - 1 if o == -1 else o - 1
        return idx if 0 <= idx < len(self.moves) else None

    def _peek_at(self, t: float) -> dict | None:
        if not self.peeks:
            return None
        frame = t * FPS - 2
        pk = min(self.peeks, key=lambda x: abs(x["start"] - frame))
        # The question's one-decimal clock lands within a couple of frames of
        # the peek's lift; anything further is not a peek we can vouch for.
        return pk if abs(pk["start"] - frame) <= 8 else None

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, options: list[str]) -> Answer:
        q = " ".join(str(question).split())
        p = self.p
        m: re.Match | None

        # ---- static census ------------------------------------------------
        if q == "How many cups are on the table?":
            return Answer(str(self.n))
        if q == "How many tokens are shown under the cups at the start?":
            return Answer(str(len(self.tokens)))
        if q == "How many swaps happen in the video?":
            return Answer(str(sum(len(mv["perm"]) // 2 for mv in self.moves)))
        if q.startswith("How many moves (a swap, three cups turning"):
            return Answer(str(len(self.moves)))
        if q == "After the start, how many times is a single cup lifted to show what is under it?":
            return Answer(str(len(self.peeks)))

        # ---- token position lookups ---------------------------------------
        if (m := re.match(rf"At the start, under the cup in which position is {TOKRE}\?", q)):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            return Answer(self._names[t["slot"]])

        if (m := re.match(
            rf"At the end of the video, under the cup in which position is {TOKRE}\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            return Answer(self._names[self._end_slot(t["slot"])])

        if (m := re.match(
            rf"At the end of the video, what is under the cup in ({POSRE})\?", q
        )):
            slot = self._slot(m.group(1))
            if slot is None:
                return Answer(None, confidence="unsupported")
            orig = self._arr()[slot]
            for t in self.tokens:
                if t["slot"] == orig:
                    return Answer(self._tk(t))
            return Answer("no token")

        if (m := re.match(rf"Follow the cup that starts in ({POSRE})\. Where is it at the end of the video\?", q)):
            slot = self._slot(m.group(1))
            if slot is None:
                return Answer(None, confidence="unsupported")
            return Answer(self._names[self._end_slot(slot)])

        if (m := re.match(
            rf"When the clock shows ([\d.]+) s, under the cup in which position is {TOKRE}\?", q
        )):
            t = self._token(m.group(2), m.group(3))
            if t is None:
                return Answer(None, confidence="unsupported")
            frame = round(float(m.group(1)) * FPS) + 1
            return Answer(self._names[self._slot_of_cup_at(t["slot"], frame)])

        # ---- left / right --------------------------------------------------
        if (m := re.match(
            rf"At the end, is {TOKRE} further left, further right, or in the same "
            r"position compared with where it started\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            end = self._end_slot(t["slot"])
            if end == t["slot"]:
                return Answer("in the same position")
            if self._x(end) == self._x(t["slot"]):
                return Answer(None, confidence="uncertain", note="moved, same x")
            return Answer("further left" if self._x(end) < self._x(t["slot"]) else "further right")

        if (m := re.match(
            rf"At the end, which is further left: {TOKRE} or {TOKRE}\?", q
        )):
            a = self._token(m.group(1), m.group(2))
            b = self._token(m.group(3), m.group(4))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            ea, eb = self._end_slot(a["slot"]), self._end_slot(b["slot"])
            if self._x(ea) == self._x(eb):
                return Answer(None, confidence="uncertain", note="same x at the end")
            return Answer(self._tk(a) if self._x(ea) < self._x(eb) else self._tk(b))

        # ---- move / swap census -------------------------------------------
        if (m := re.match(r"In the (\w+) (?:swap|move), the cups in which two positions trade places\?", q)):
            idx = self._move_index(m.group(1))
            if idx is None:
                return Answer(None, confidence="unsupported")
            pairs = self.moves[idx]["perm"]
            if len(pairs) != 2:
                return Answer(None, confidence="unsupported",
                              note=f"move has {len(pairs)} pairs, not a single swap")
            a, b = sorted([pairs[0][0], pairs[0][1]])
            return Answer(f"{self._names[a]} and {self._names[b]}")

        if (m := re.match(
            rf"Does the cup hiding {TOKRE} take part in the (\w+) (?:swap|move)\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            idx = self._move_index(m.group(3))
            if t is None or idx is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            for mv in self.moves[:idx]:
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
            here = arr.index(t["slot"])
            return Answer("yes" if here in {a for a, _ in self.moves[idx]["perm"]} else "no")

        if (m := re.match(rf"How many times does the cup hiding {TOKRE} change position\?", q)):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            count = 0
            for mv in self.moves:
                if arr.index(t["slot"]) in {a for a, _ in mv["perm"]}:
                    count += 1
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
            return Answer(str(count))

        if (m := re.match(
            rf"At what time on the clock does the cup hiding {TOKRE} start to move "
            r"for the first time\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            for mv in self.moves:
                if arr.index(t["slot"]) in {a for a, _ in mv["perm"]}:
                    return Answer(clock((mv["start"] - 1) / FPS))
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
            return Answer("never")

        if (m := re.match(r"At what time on the clock does the (\w+) (?:swap|move) start\?", q)):
            idx = self._move_index(m.group(1))
            if idx is None:
                return Answer(None, confidence="unsupported")
            return Answer(clock((self.moves[idx]["start"] - 1) / FPS))

        if (m := re.match(r"How many times does a cup leave (the .+?)\?", q)):
            slot = self._slot(m.group(1))
            if slot is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(sum(1 for mv in self.moves if slot in {a for a, _ in mv["perm"]})))

        if (m := re.match(
            rf"Counting where it starts, how many different positions does {TOKRE} "
            r"occupy over the video\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            seen = {arr.index(t["slot"])}
            for mv in self.moves:
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
                seen.add(arr.index(t["slot"]))
            return Answer(str(len(seen)))

        if (m := re.match(
            rf"The first time the cup hiding {TOKRE} moves, to which position does it go\?", q
        )):
            t = self._token(m.group(1), m.group(2))
            if t is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            for mv in self.moves:
                here = arr.index(t["slot"])
                dest = dict(mv["perm"]).get(here)
                if dest is not None:
                    return Answer(self._names[dest])
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
            return Answer("never")

        # ---- first / most-often comparisons -------------------------------
        if (m := re.match(
            r"Which position does a cup leave first: (the .+?) or (the .+?)\?", q
        )):
            a, b = self._slot(m.group(1)), self._slot(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            for mv in self.moves:
                src = {x for x, _ in mv["perm"]}
                hit = [s for s in (a, b) if s in src]
                if len(hit) == 1:
                    return Answer(self._names[hit[0]])
                if len(hit) == 2:
                    return Answer(None, confidence="uncertain", note="both leave in the same move")
            return Answer(None, confidence="unsupported", note="neither position is ever left")

        if (m := re.match(
            r"Which cup changes position more often over the video: the cup that starts in "
            r"(the .+?) or the cup that starts in (the .+?)\?", q
        )):
            a, b = self._slot(m.group(1)), self._slot(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            counts = {a: 0, b: 0}
            arr = list(range(self.n))
            for mv in self.moves:
                src = {x for x, _ in mv["perm"]}
                for s in (a, b):
                    if arr.index(s) in src:
                        counts[s] += 1
                new = arr.copy()
                for x, y in mv["perm"]:
                    new[y] = arr[x]
                arr = new
            if counts[a] == counts[b]:
                return Answer(None, confidence="uncertain",
                              note=f"tie at {counts[a]} changes")
            best = a if counts[a] > counts[b] else b
            return Answer("the cup that starts in " + self._names[best])

        if (m := re.match(
            rf"Which cup moves first: the one hiding {TOKRE} or the one hiding {TOKRE}\?", q
        )):
            ta = self._token(m.group(1), m.group(2))
            tb = self._token(m.group(3), m.group(4))
            if ta is None or tb is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            for mv in self.moves:
                src = {x for x, _ in mv["perm"]}
                hit = [t for t in (ta, tb) if arr.index(t["slot"]) in src]
                if len(hit) == 1:
                    return Answer(self._tk(hit[0]) + "'s cup")
                if len(hit) == 2:
                    return Answer(None, confidence="uncertain", note="both move in the same move")
                new = arr.copy()
                for x, y in mv["perm"]:
                    new[y] = arr[x]
                arr = new
            return Answer(None, confidence="unsupported", note="neither cup ever moves")

        if (m := re.match(r"How many (?:swaps|moves) start between ([\d.]+) s and ([\d.]+) s on the clock\?", q)):
            lo, hi = float(m.group(1)), float(m.group(2))
            count = sum(1 for mv in self.moves if lo <= (mv["start"] - 1) / FPS <= hi)
            return Answer(str(count))

        # ---- return-to-start question -------------------------------------
        if (m := re.match(
            rf"Before the (\w+) (?:swap|move), does the cup hiding {TOKRE} ever come "
            r"back to the position it started in after having left it\?", q
        )):
            t = self._token(m.group(2), m.group(3))
            idx = self._move_index(m.group(1))
            if t is None or idx is None:
                return Answer(None, confidence="unsupported")
            arr = list(range(self.n))
            here = arr.index(t["slot"])
            start = here
            left = False
            for mv in self.moves[:idx]:  # exclude the move named in the question
                new = arr.copy()
                for a, b in mv["perm"]:
                    new[b] = arr[a]
                arr = new
                here = arr.index(t["slot"])
                if here != start:
                    left = True
                elif left:
                    return Answer("yes")
            return Answer("no")

        # ---- peeks ---------------------------------------------------------
        if (m := re.match(r"In which position is the cup that is lifted at ([\d.]+) s\?", q)):
            pk = self._peek_at(float(m.group(1)))
            if pk is None:
                return Answer(None, confidence="unsupported")
            return Answer(self._names[pk["slot"]])

        if (m := re.match(r"Where did the cup that is lifted at ([\d.]+) s stand at the very start\?", q)):
            pk = self._peek_at(float(m.group(1)))
            if pk is None:
                return Answer(None, confidence="unsupported")
            arr = self._arr(upto=pk["start"])
            return Answer(self._names[arr[pk["slot"]]])

        if (m := re.match(r"When a single cup is lifted at ([\d.]+) s, is the token under it\?", q)):
            pk = self._peek_at(float(m.group(1)))
            if pk is None:
                return Answer(None, confidence="unsupported")
            orig = self._arr(upto=pk["start"])[pk["slot"]]
            return Answer("yes" if any(t["slot"] == orig for t in self.tokens) else "no")

        # ---- counterfactuals ----------------------------------------------
        if (m := re.match(
            rf"Suppose {TOKRE} had started under the cup in ({POSRE}) instead, and the "
            r"cups had moved exactly the same way\. Where would it be at the end\?", q
        )):
            slot = self._slot(m.group(3))
            if slot is None:
                return Answer(None, confidence="unsupported")
            return Answer(self._names[self._end_slot(slot)])

        if (m := re.match(
            rf"Suppose that after the video ends, the cups in ({POSRE}) and ({POSRE}) "
            rf"trade places once more\. Under the cup in which position would {TOKRE} then be\?", q
        )):
            a, b = self._slot(m.group(1)), self._slot(m.group(2))
            t = self._token(m.group(3), m.group(4))
            if a is None or b is None or t is None:
                return Answer(None, confidence="unsupported")
            here = self._arr().index(t["slot"])
            if here == a:
                here = b
            elif here == b:
                here = a
            return Answer(self._names[here])

        if (m := re.match(
            rf"Suppose the (\w+) (?:swap|move) had never happened and every other "
            rf"(?:swap|move) had happened just the same\. Under the cup in which position "
            rf"would {TOKRE} be at the end\?", q
        )):
            idx = self._move_index(m.group(1))
            t = self._token(m.group(2), m.group(3))
            if idx is None or t is None:
                return Answer(None, confidence="unsupported")
            moves = self.moves[:idx] + self.moves[idx + 1:]
            return Answer(self._names[self._end_slot(t["slot"], moves)])

        if (m := re.match(
            rf"Suppose the (\w+) (?:swap|move) and the (\w+) (?:swap|move) had happened "
            rf"in the opposite order, with everything else the same\. Under the cup in "
            rf"which position would {TOKRE} be at the end\?", q
        )):
            i1, i2 = self._move_index(m.group(1)), self._move_index(m.group(2))
            t = self._token(m.group(3), m.group(4))
            if i1 is None or i2 is None or t is None:
                return Answer(None, confidence="unsupported")
            moves = list(self.moves)
            moves[i1], moves[i2] = moves[i2], moves[i1]
            return Answer(self._names[self._end_slot(t["slot"], moves)])

        return Answer(None, confidence="unsupported", note="no handler for this template")

"""Contract for chess_dance_trace (Layer-1 params replay).

The generator stores the whole choreography: pieces (type/colour/start), the
missing squares, every move of every round (from -> to, plus an optional
``capture`` = index of the captured piece and ``illegal`` = pattern|blocked),
and the frame timing (intro_f / move_f / pause_f / hold_f).

Round r animates over frames
    start_r = intro_f + r * (move_f + pause_f)
    end_r   = start_r + move_f
and the on-screen clock shows ``floor(frame / 24 * 10) / 10`` seconds.

Conventions pinned against the keys (see results/chess_dance_trace_notes.md):
  * "travels N squares" = Chebyshev distance (the larger of file/rank steps),
    for knights too (a knight move is 2).
  * a piece is "moving" while its move animates; the clock value asked in a
    question is turned into frame ``int(t * 24)``.
  * "when the clock shows X" uses the piece's destination square once its
    move has started (the generator only asks at rest times for positions).
  * a capture removes the victim at the end of the round; victims are always
    static at the target square at the start of their round.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

TYPE_NAME = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
NAME_TYPE = {v: k for k, v in TYPE_NAME.items()}
ORD = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
       "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8}
ORD_RE = "|".join(ORD)

COLOURS = (
    "sky-blue", "magenta", "purple", "yellow", "orange", "green",
    "teal", "blue", "pink", "lime", "red",
)
_COLOUR_RE = "|".join(sorted(COLOURS, key=len, reverse=True))
PIECE_RE = re.compile(rf"the ({_COLOUR_RE}) (king|queen|rook|bishop|knight)")
SQUARE_RE = re.compile(r"\b([A-H][1-8])\b")


def csq(s: str) -> tuple[int, int]:
    return (ord(s[0].upper()) - 65, int(s[1:]) - 1)


def nsq(c: tuple[int, int]) -> str:
    return chr(65 + c[0]) + str(c[1] + 1)


def cheb(a: str, b: str) -> int:
    (x1, y1), (x2, y2) = csq(a), csq(b)
    return max(abs(x2 - x1), abs(y2 - y1))


class ChessDance:
    def __init__(self, params: dict):
        self.p = params
        self.N = int(params["N"])
        self.pieces = list(params["pieces"])
        self.holes = set(params.get("holes") or [])
        self.rounds = list(params["moves"])
        self.t = params["timing"]

        self.moves_of: list[list[tuple[int, dict]]] = [[] for _ in self.pieces]
        for ri, rd in enumerate(self.rounds):
            for m in rd:
                self.moves_of[int(m["p"])].append((ri, m))

        self.start = {i: x["start"] for i, x in enumerate(self.pieces)}
        self.captures: list[tuple[int, int, int]] = []  # (round, mover, victim)
        self.captured_by = {i: 0 for i in range(len(self.pieces))}
        self.captured_at: dict[int, int] = {}
        self.illegal_move = None
        for ri, rd in enumerate(self.rounds):
            for m in rd:
                if m.get("capture") is not None:
                    victim = int(m["capture"])
                    self.captures.append((ri, int(m["p"]), victim))
                    self.captured_by[int(m["p"])] += 1
                    self.captured_at[victim] = ri
                if m.get("illegal"):
                    self.illegal_move = (ri, int(m["p"]), m)

        # final positions by replaying the stored moves
        self.pos_after: list[dict[int, str]] = []
        pos = dict(self.start)
        for rd in self.rounds:
            for m in rd:
                pos[int(m["p"])] = m["to"]
            self.pos_after.append(dict(pos))
        self.final = pos

        self.round_start = [
            self.t["intro_f"] + r * (self.t["move_f"] + self.t["pause_f"])
            for r in range(len(self.rounds))
        ]
        self.round_end = [s + self.t["move_f"] for s in self.round_start]

    # ------------------------------------------------------------------ geometry
    def piece_index(self, colour: str, name: str) -> int | None:
        typ = NAME_TYPE[name]
        for i, x in enumerate(self.pieces):
            if x["color"] == colour and x["type"] == typ:
                return i
        return None

    def refs(self, question: str) -> list[int]:
        out = []
        for m in PIECE_RE.finditer(question):
            i = self.piece_index(m.group(1), m.group(2))
            if i is not None:
                out.append(i)
        return out

    def label(self, i: int) -> str:
        return f"the {self.pieces[i]['color']} {TYPE_NAME[self.pieces[i]['type']]}"

    @staticmethod
    def pattern_ok(typ: str, a: str, b: str) -> bool:
        (x1, y1), (x2, y2) = csq(a), csq(b)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 and dy == 0:
            return False
        if typ == "K":
            return dx <= 1 and dy <= 1
        if typ == "N":
            return sorted((dx, dy)) == [1, 2]
        if typ == "R":
            return dx == 0 or dy == 0
        if typ == "B":
            return dx == dy
        if typ == "Q":
            return dx == 0 or dy == 0 or dx == dy
        return False

    @staticmethod
    def between(a: str, b: str) -> list[str] | None:
        (x1, y1), (x2, y2) = csq(a), csq(b)
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return []
        if dx == 0 or dy == 0:
            n = max(abs(dx), abs(dy))
            sx = 0 if dx == 0 else dx // abs(dx)
            sy = 0 if dy == 0 else dy // abs(dy)
        elif abs(dx) == abs(dy):
            n = abs(dx)
            sx = dx // n
            sy = dy // n
        else:
            return None
        return [nsq((x1 + sx * k, y1 + sy * k)) for k in range(1, n)]

    def can_move(self, typ: str, a: str, b: str, occupied: dict[int, str],
                 extra_holes: set[str] | None = None) -> bool:
        if not self.pattern_ok(typ, a, b):
            return False
        if typ == "N":
            return True
        occ = set(occupied.values()) | self.holes | (extra_holes or set())
        return not any(c in occ for c in (self.between(a, b) or []))

    def attack_count(self, i: int, pos: dict[int, str], alive: set[int],
                     extra_holes: set[str] | None = None) -> int:
        typ = self.pieces[i]["type"]
        target_pos = {j: pos[j] for j in alive if j != i}
        return sum(1 for j, sq in target_pos.items()
                   if self.can_move(typ, pos[i], sq, target_pos, extra_holes))

    # ------------------------------------------------------------------ timeline
    def frame_for_clock(self, seconds: float) -> int:
        return int(seconds * 24 + 1e-9)

    def clock_text(self, frame: int) -> str:
        return f"{math.floor(frame / 24 * 10 + 1e-9) / 10:.1f} s"

    def move_interval(self, i: int, k: int) -> tuple[int, int]:
        ri, _m = self.moves_of[i][k]
        return self.round_start[ri], self.round_end[ri]

    def moving_at(self, frame: int) -> int:
        n = 0
        for i in range(len(self.pieces)):
            for ri, _m in self.moves_of[i]:
                if self.round_start[ri] <= frame < self.round_end[ri]:
                    n += 1
                    break
        return n

    def position_at(self, i: int, frame: int) -> str | None:
        pos = self.start[i]
        for ri, m in self.moves_of[i]:
            if frame < self.round_start[ri]:
                return pos
            pos = m["to"]
            if frame < self.round_end[ri]:
                return pos
        return pos

    def rest_frame(self, i: int, k: int) -> int:
        return self.move_interval(i, k)[1]

    # ------------------------------------------------------------------ answers
    def answer(self, question: str, options: list[str]) -> Answer:
        q = " ".join(str(question).split()).lower()
        ql = q

        # ---- static board facts
        if ql.startswith("how many chess pieces are on the board at the start"):
            return Answer(str(len(self.pieces)))
        if ql.startswith("how many chess pieces are on the board at the end"):
            return Answer(str(len(self.pieces) - len(self.captures)))
        if "how many squares are cut out of the board" in ql:
            return Answer(str(len(self.holes)))
        m = re.match(r"how many (kings|queens|rooks|bishops|knights) are on the board at the start", ql)
        if m:
            typ = NAME_TYPE[m.group(1)[:-1]]
            return Answer(str(sum(1 for x in self.pieces if x["type"] == typ)))
        m = re.match(r"what colour is the (king|queen|rook|bishop|knight)", ql)
        if m:
            typ = NAME_TYPE[m.group(1)]
            same = [x for x in self.pieces if x["type"] == typ]
            if len(same) == 1:
                return Answer(same[0]["color"])
            return Answer(None, confidence="unsupported", note="type not unique")

        # ---- move counts / captures
        if "how many moves are made in total" in ql:
            return Answer(str(sum(len(rd) for rd in self.rounds)))
        if ql.startswith("how many captures happen during the video"):
            return Answer(str(len(self.captures)))
        if ql.startswith("which piece is captured first"):
            if not self.captures:
                return Answer("no piece is captured")
            return Answer(self.label(self.captures[0][2]))
        if ql.startswith("which piece captures the most pieces"):
            best = max(range(len(self.pieces)), key=lambda i: self.captured_by[i])
            return Answer(self.label(best))
        if ql.startswith("which piece moves the most times"):
            best = max(range(len(self.pieces)), key=lambda i: len(self.moves_of[i]))
            return Answer(self.label(best))
        m = re.match(r"which piece captures the (?:[\w-]+) (king|queen|rook|bishop|knight)", ql)
        if m:
            refs = self.refs(q)
            if refs:
                victim = refs[0]
                for _ri, mover, v in self.captures:
                    if v == victim:
                        return Answer(self.label(mover))
            return Answer("no piece captures it")

        # ---- first / timing comparisons
        if "is the first to move" in ql:
            cand = [self.piece_index(*self._opt_piece(o)) for o in options]
            cand = [i for i in cand if i is not None]
            if not cand:
                return Answer(None, confidence="unsupported")
            first = min(cand, key=lambda i: self.moves_of[i][0][0] if self.moves_of[i] else 10**6)
            return Answer(self.label(first))
        if ql.startswith("which starts moving earlier"):
            refs = self.refs(q)
            if len(refs) < 2:
                return Answer(None, confidence="unsupported")
            a, b = refs[0], refs[1]
            fa = self.moves_of[a][0][0] if self.moves_of[a] else None
            fb = self.moves_of[b][0][0] if self.moves_of[b] else None
            if fa is None and fb is None:
                return Answer("neither of them moves")
            if fa is None:
                return Answer(self.label(b))
            if fb is None:
                return Answer(self.label(a))
            if fa == fb:
                return Answer("they first move at the same moment")
            return Answer(self.label(a) if fa < fb else self.label(b))
        if ql.startswith("which moves more times"):
            refs = self.refs(q)
            if len(refs) < 2:
                return Answer(None, confidence="unsupported")
            a, b = refs[0], refs[1]
            na, nb = len(self.moves_of[a]), len(self.moves_of[b])
            if na == 0 and nb == 0:
                return Answer("neither of them moves")
            if na == nb:
                return Answer("they move equally often")
            return Answer(self.label(a) if na > nb else self.label(b))
        if "moves at the same moment as" in ql:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            x = refs[0]
            xr = {ri for ri, _m in self.moves_of[x]}
            for o in options:
                got = self._opt_piece(o)
                if got is None:
                    continue
                j = self.piece_index(*got)
                if j is None or j == x:
                    continue
                if xr & {ri for ri, _m in self.moves_of[j]}:
                    return Answer(o)
            return Answer("no piece moves at the same moment")

        # ---- square facts
        m = re.match(r"on which square does the (.+?) start\b", q)
        if m:
            refs = self.refs(q)
            return Answer(self.start[refs[0]]) if refs else Answer(None, confidence="unsupported")
        m = re.match(r"on which square is the (.+?) at the end of the video", q)
        if m:
            refs = self.refs(q)
            return Answer(self.final[refs[0]]) if refs else Answer(None, confidence="unsupported")
        m = re.match(r"on which square is the (.+?) when the clock shows ([\d.]+)\s*s", q)
        if m:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            frame = self.frame_for_clock(float(m.group(2)))
            return Answer(self.position_at(refs[0], frame))
        if "never stand on" in ql:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            stand = {self.start[refs[0]]} | {mv["to"] for _ri, mv in self.moves_of[refs[0]]}
            for o in options:
                if o not in stand:
                    return Answer(o)
            return Answer(None, confidence="uncertain", note="every option stood on")

        # ---- distances
        m = re.match(r"at the end, how many king steps away from its starting square is the (.+?) \(", q)
        if m:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            i = refs[0]
            return Answer(str(cheb(self.final[i], self.start[i])))
        m = re.match(rf"how many squares does the (.+?) travel in its ({ORD_RE}) move", q)
        if m:
            refs = self.refs(q)
            k = ORD[m.group(2)]
            if not refs or k >= len(self.moves_of[refs[0]]):
                return Answer(None, confidence="unsupported")
            mv = self.moves_of[refs[0]][k][1]
            return Answer(str(cheb(mv["from"], mv["to"])))
        m = re.match(r"adding up the length of each of its moves, how many squares does the (.+?) travel in total", q)
        if m:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            total = sum(cheb(mv["from"], mv["to"]) for _ri, mv in self.moves_of[refs[0]])
            return Answer(str(total))
        m = re.match(r"how many times does the (.+?) move during the video", q)
        if m:
            refs = self.refs(q)
            if not refs:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.moves_of[refs[0]])))

        # ---- clocks
        m = re.match(rf"what does the clock show when the (.+?) comes to rest after its ({ORD_RE}) move", q)
        if m:
            refs = self.refs(q)
            k = ORD[m.group(2)]
            if not refs or k >= len(self.moves_of[refs[0]]):
                return Answer(None, confidence="unsupported")
            return Answer(self.clock_text(self.rest_frame(refs[0], k)))
        if ql.startswith("how many pieces are moving when the clock shows"):
            m = re.search(r"clock shows ([\d.]+)\s*s", ql)
            frame = self.frame_for_clock(float(m.group(1)))
            return Answer(str(self.moving_at(frame)))

        # ---- illegal move family
        if "does any piece make a move its type is not allowed to make" in ql:
            if self.illegal_move is None:
                return Answer("no piece does")
            return Answer(self.label(self.illegal_move[1]))
        if ql.startswith("one piece makes a move its type is not allowed to make"):
            if self.illegal_move is None:
                return Answer(None, confidence="unsupported")
            ri, _p, _mv = self.illegal_move
            return Answer(self.clock_text(self.round_end[ri]))

        # ---- counterfactuals
        if "had never left its starting square" in ql:
            return self._blocked_by_stationary(q, options)
        if "had skipped its" in ql and "attack right after that round" in ql:
            return self._skipped_attack(q)
        if "had never moved" in ql and "how many pieces would be on the board at the end" in ql:
            return self._never_moved(q)
        if "had stayed put instead of making its illegal move" in ql:
            return self._stayed_put(q)
        if "had also been cut out of the board" in ql:
            return self._extra_hole(q)

        return Answer(None, confidence="unsupported", note="no handler")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _opt_piece(opt: str) -> tuple[str, str] | None:
        m = PIECE_RE.search(opt)
        return (m.group(1), m.group(2)) if m else None

    def _blocked_by_stationary(self, q: str, options: list[str]) -> Answer:
        refs = self.refs(q)
        if not refs:
            return Answer(None, confidence="unsupported")
        x = refs[0]
        s = self.start[x]
        for ri, rd in enumerate(self.rounds):
            # if the piece is standing on its start square in the real video at
            # this moment, the move was already blocked there -> not a new block
            pos, alive = self._board_before_round(ri)
            if pos.get(x) == s and x in alive:
                continue
            for m in rd:
                p = int(m["p"])
                if p == x:
                    continue
                typ = self.pieces[p]["type"]
                a, b = m["from"], m["to"]
                if b == s:
                    return Answer(self.label(p))
                if typ != "N" and self.pattern_ok(typ, a, b):
                    cells = self.between(a, b) or []
                    if s in cells:
                        return Answer(self.label(p))
        return Answer("no piece would be blocked")

    def _skipped_attack(self, q: str) -> Answer:
        refs = self.refs(q)
        m = re.search(rf"skipped its ({ORD_RE}) move", q)
        k = ORD[m.group(1)]
        i = refs[0]
        if k >= len(self.moves_of[i]):
            return Answer(None, confidence="unsupported")
        ri, _mv = self.moves_of[i][k]
        pos, alive = self._board_after(ri, skip=(i, k))
        return Answer(str(self.attack_count(i, pos, alive)))

    def _board_after(self, round_i: int, skip: tuple[int, int] | None = None):
        """Positions + alive set after round_i, optionally skipping piece i's k-th move."""
        pos = dict(self.start)
        alive = set(range(len(self.pieces)))
        for ri in range(round_i + 1):
            rd = self.rounds[ri]
            movers = {int(m["p"]) for m in rd}
            pre = {j: pos[j] for j in alive}
            for m in rd:
                p = int(m["p"])
                if skip and p == skip[0] and ri == self.moves_of[p][skip[1]][0]:
                    continue
                for j in list(alive):
                    if j != p and j not in movers and pre.get(j) == m["to"]:
                        alive.discard(j)
                pos[p] = m["to"]
        return pos, alive

    def _never_moved(self, q: str) -> Answer:
        refs = self.refs(q)
        if not refs:
            return Answer(None, confidence="unsupported")
        x = refs[0]
        pos = dict(self.start)
        alive = set(range(len(self.pieces)))
        for rd in self.rounds:
            movers = {int(m["p"]) for m in rd} - {x}
            pre = {j: pos[j] for j in alive}
            for m in rd:
                p = int(m["p"])
                if p == x:
                    continue
                for j in list(alive):
                    if j != p and j not in movers and pre.get(j) == m["to"]:
                        alive.discard(j)
                pos[p] = m["to"]
        return Answer(str(len(alive)))

    def _board_before_round(self, round_i: int):
        pos = dict(self.start)
        alive = set(range(len(self.pieces)))
        for ri in range(round_i):
            rd = self.rounds[ri]
            movers = {int(m["p"]) for m in rd}
            pre = {j: pos[j] for j in alive}
            for m in rd:
                p = int(m["p"])
                for j in list(alive):
                    if j != p and j not in movers and pre.get(j) == m["to"]:
                        alive.discard(j)
                pos[p] = m["to"]
        return pos, alive

    def _stayed_put(self, q: str) -> Answer:
        refs = self.refs(q)
        if not refs or self.illegal_move is None:
            return Answer(None, confidence="unsupported")
        ri, p, mv = self.illegal_move
        later = [m for r2, m in self.moves_of[p] if r2 > ri]
        if not later:
            return Answer(None, confidence="unsupported")
        nxt = later[0]
        nr = next(r2 for r2, m in self.moves_of[p] if m is nxt)
        a, b = mv["from"], nxt["to"]
        typ = self.pieces[p]["type"]
        if not self.pattern_ok(typ, a, b):
            return Answer("illegal: a path its type can never take")
        pos, alive = self._board_before_round(nr)
        occupied = {j: pos[j] for j in alive if j != p}
        if typ != "N":
            cells = self.between(a, b) or []
            if any(c in occupied.values() or c in self.holes for c in cells):
                return Answer("illegal: it slides through another piece")
        if b in occupied.values():
            return Answer("illegal: it lands on an occupied square")
        return Answer("a legal move")

    def _extra_hole(self, q: str) -> Answer:
        m = re.search(r"suppose square ([a-ha-h][1-8]) had also been cut out", q)
        if not m:
            return Answer(None, confidence="unsupported")
        h = m.group(1).upper()
        for rd in self.rounds:
            for mv in rd:
                p = int(mv["p"])
                typ = self.pieces[p]["type"]
                if mv["to"] == h:
                    return Answer(self.label(p))
                if typ != "N" and self.pattern_ok(typ, mv["from"], mv["to"]):
                    if h in (self.between(mv["from"], mv["to"]) or []):
                        return Answer(self.label(p))
        return Answer("no piece would be blocked")

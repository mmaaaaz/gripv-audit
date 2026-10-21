"""tile_alignment_shuffle — replay of the board's move recipe.

The recipe pins the whole animation:

* ``n``  — the board is ``n x n`` (n = 3..6).
* ``designs`` — the tile *kinds*, each ``{kind, color, false}``; ``tiles`` maps a
  tile id (0 .. n*n-1) to its design index, so a tile never changes kind.
* ``start[r][c]`` — the tile id sitting in row ``r`` + 1, column ``c`` + 1 at
  frame 0 (the labels in the video are 1-based).
* ``moves`` — one entry per "round of tiles changing places".  ``pairs`` is the
  *simultaneous* cell permutation of that round: every ``[src, dst]`` edge means
  the tile standing in ``src`` is standing in ``dst`` when the round ends.  A
  2-cycle is written twice (a->b and b->a), a 3-cycle once round the loop, a
  slide once round the whole row/column, so the edge list is always complete
  and consistent.  A tile "moves to a new cell" in a round iff its cell changes.
* timing — 24 fps; move *i* (0-based) starts at frame
  ``intro + i*(move+pause)`` and comes to rest at frame
  ``intro + i*(move+pause) + move``.  The on-screen clock is ``frame/24``
  **truncated** to one decimal (calibrated on 1,686 clock keys: truncation at
  the rest frame is 100 %, rounding is 50 %).
* a "complete row, column or diagonal made of one kind of tile" is a whole
  line whose tiles all share one design (colour + shape/symbol).  ``line`` says
  whether the final board has one, ``line_move`` is the **1-based** number of
  the first move after which one exists; when ``line`` is true the final board
  has exactly one such line, so "where/which kind" questions are well defined.
* counterfactuals ("suppose the k-th move had not happened") delete that move
  from the recipe and replay the remaining moves with the same cell
  permutations; the tiles sitting in those cells at that moment are whatever
  the earlier moves left there.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0

ORDINALS = {
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
    "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11, "12th": 12,
}

SYMBOLS = set("&%#@?$ΣΩ")

ROW_COL = re.compile(r"row (\d+), column (\d+)")
CELL_TEXT = re.compile(r"row (\d+), column (\d+)")


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _clock(frame: float) -> str:
    """On-screen timer: frame/24 s truncated to one decimal."""
    return f"{math.floor(frame / FPS * 10.0 + 1e-9) / 10.0:.1f} s"


def _norm_kind(kind: str) -> str:
    k = str(kind).strip()
    if k in ("ω", "Ω"):
        return "Ω"
    if k in ("σ", "Σ"):
        return "Σ"
    return k.lower()


def _design_text(design: dict) -> str:
    kind = str(design["kind"])
    color = str(design["color"])
    if kind in SYMBOLS:
        return f"{color} {kind} symbol"
    return f"{color} {_norm_kind(kind)}"


def _parse_design(text: str) -> tuple[str, str]:
    """'black & symbol' / 'red Ω symbol' / 'green diamond' -> (colour, kind)."""
    t = text.lower().replace("ω", "ω").strip()
    t = re.sub(r"\s+symbol$", "", t)
    parts = t.split()
    color = parts[0]
    kind = " ".join(parts[1:])
    return color, _norm_kind(kind)


class TileAlignmentShuffle:
    """Replay one ``tile_alignment_shuffle`` clip from its generator recipe."""

    def __init__(self, params: dict):
        self.p = params
        self.n = int(params["n"])
        self.designs = params["designs"]
        self.tiles = list(params["tiles"])
        self.start = [list(row) for row in params["start"]]
        self.moves = params["moves"]
        self.timing = params["timing"]
        self.n_frames = int(params["n_frames"])

        # cell permutation (src -> dst) of every move
        self.perms = [
            {tuple(a): tuple(b) for a, b in m["pairs"]} for m in self.moves
        ]
        # board state after each move; grids[0] is the start board
        self.grids = self._run(self.perms)
        self.cell_of = [self._cell_index(g) for g in self.grids]
        # tiles that changed cell in each move (index i = move i, 0-based)
        self.moved = []
        for i in range(len(self.moves)):
            before, after = self.cell_of[i], self.cell_of[i + 1]
            self.moved.append({t for t in before if before[t] != after[t]})
        self.move_count = {}
        self.first_move = {}
        for tid in range(self.n * self.n):
            seq = [self.cell_of[i][tid] for i in range(len(self.grids))]
            self.move_count[tid] = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            self.first_move[tid] = next(
                (i + 1 for i in range(len(seq) - 1) if seq[i + 1] != seq[i]), None
            )

    # ------------------------------------------------------------------ board
    def _run(self, perms):
        grid = [list(row) for row in self.start]
        out = [grid]
        for perm in perms:
            nxt = [[None] * self.n for _ in range(self.n)]
            for r in range(self.n):
                for c in range(self.n):
                    dst = perm.get((r, c), (r, c))
                    nxt[dst[0]][dst[1]] = grid[r][c]
            grid = nxt
            out.append(grid)
        return out

    @staticmethod
    def _cell_index(grid):
        return {grid[r][c]: (r, c) for r in range(len(grid)) for c in range(len(grid))}

    def design_id_at(self, grid, r, c) -> int:
        return self.tiles[grid[r][c]]

    def line_of(self, grid):
        """The complete line on ``grid`` (unique when present) or None.

        Returns ``(kind, index)`` with kind ``row`` / ``col`` / ``diag``.
        """
        n = self.n
        for r in range(n):
            if len({self.tiles[grid[r][c]] for c in range(n)}) == 1:
                return ("row", r)
        for c in range(n):
            if len({self.tiles[grid[r][c]] for r in range(n)}) == 1:
                return ("col", c)
        if len({self.tiles[grid[i][i]] for i in range(n)}) == 1:
            return ("diag", 0)
        if len({self.tiles[grid[i][n - 1 - i]] for i in range(n)}) == 1:
            return ("diag", 1)
        return None

    def lines_count(self, grid) -> int:
        n = self.n
        total = 0
        for r in range(n):
            if len({self.tiles[grid[r][c]] for c in range(n)}) == 1:
                total += 1
        for c in range(n):
            if len({self.tiles[grid[r][c]] for r in range(n)}) == 1:
                total += 1
        if len({self.tiles[grid[i][i]] for i in range(n)}) == 1:
            total += 1
        if len({self.tiles[grid[i][n - 1 - i]] for i in range(n)}) == 1:
            total += 1
        return total

    def line_design(self, grid):
        line = self.line_of(grid)
        if line is None:
            return None
        kind, idx = line
        if kind == "row":
            return self.designs[self.tiles[grid[idx][0]]]
        if kind == "col":
            return self.designs[self.tiles[grid[0][idx]]]
        if idx == 0:
            return self.designs[self.tiles[grid[0][0]]]
        return self.designs[self.tiles[grid[0][self.n - 1]]]

    def line_cells(self, line):
        kind, idx = line
        n = self.n
        if kind == "row":
            return [(idx, c) for c in range(n)]
        if kind == "col":
            return [(r, idx) for r in range(n)]
        if idx == 0:
            return [(i, i) for i in range(n)]
        return [(i, n - 1 - i) for i in range(n)]

    def cell_text(self, cell) -> str:
        return f"row {cell[0] + 1}, column {cell[1] + 1}"

    def frame_end(self, move_index: int) -> float:
        """Frame on which move ``move_index`` (0-based) comes to rest."""
        tm = self.timing
        return float(tm["intro"] + move_index * (tm["move"] + tm["pause"]) + tm["move"])

    # ------------------------------------------------------------- dispatch
    def answer(self, question: str, options: list[str]) -> Answer:
        q = " ".join(str(question).lower().split())

        if q.startswith("suppose the "):
            return self._counterfactual(q, options)
        if q.startswith("right after which move"):
            return self._line_move(options)
        if q.startswith("what does the clock read when a complete"):
            return self._line_clock()
        if q.startswith("what does the clock read when the tile"):
            return self._rest_clock(q)
        if q.startswith("at the very start, what kind"):
            return self._kind_at(q, self.grids[0])
        if q.startswith("at the end, what kind"):
            return self._kind_at(q, self.grids[-1])
        if q.startswith("at the end, where is the complete"):
            return self._line_location()
        if q.startswith("at the end, which kind"):
            return self._line_kind()
        if q.startswith("the complete row, column or diagonal made of one kind of tile at the end"):
            return self._line_start_overlap()
        if q.startswith("how many different kinds"):
            return self._distinct_kinds()
        if q.startswith("how many tiles match no other"):
            return self._unique_tiles()
        if q.startswith("how many different cells does the tile"):
            return self._cells_visited(q)
        if q.startswith("how many times does the tile"):
            return self._times_moved(q)
        if q.startswith("how many moves happen"):
            return Answer(str(len(self.moves)), str(len(self.moves)), "exact")
        if q.startswith("how many tiles never move"):
            return self._never_moved()
        if q.startswith("how many tiles move at least once but end in the same cell"):
            return self._moved_and_returned()
        if q.startswith("how many ") and "tiles are on the board" in q:
            return self._count_kind(q, options)
        if q.startswith("where does the tile"):
            return self._end_cell(q)
        if q.startswith("where is the tile that starts") and "right after" in q:
            return self._cell_after_move(q)
        if q.startswith("the tile that ends in"):
            return self._start_cell_of_end_tile(q)
        if q.startswith("which tile moves first"):
            return self._first_move_cmp(q, options)
        if q.startswith("which tile moves to a new cell more times"):
            return self._move_count_cmp(q, options)
        if q.startswith("which tile moves to a new cell the most times"):
            return self._most_moves()
        if q.startswith("the first time the tile"):
            if "trades places" in q and "which cell did that other tile start" in q:
                return self._first_partner_cell(q)
            if "what kind of tile moves into the cell it leaves" in q:
                return self._first_partner_kind(q)
        if q.startswith("if the ") and "how many complete rows" in q:
            return self._recolor_lines(q)
        return Answer(None, None, "unsupported", f"unhandled: {question[:90]}")

    # --------------------------------------------------------------- helpers
    def _tile_at(self, q):
        m = ROW_COL.search(q)
        if not m:
            return None
        r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
        if not (0 <= r < self.n and 0 <= c < self.n):
            return None
        return self.start[r][c]

    def _two_tiles(self, q):
        cells = ROW_COL.findall(q)
        if len(cells) < 2:
            return None
        out = []
        for r, c in cells[:2]:
            r, c = int(r) - 1, int(c) - 1
            if not (0 <= r < self.n and 0 <= c < self.n):
                return None
            out.append(self.start[r][c])
        return out

    def _move_number(self, q) -> int | None:
        m = re.search(r"the (\d+(?:st|nd|rd|th)) move", q)
        if not m:
            return None
        return ORDINALS.get(m.group(1))

    # -------------------------------------------------------------- families
    def _kind_at(self, q, grid):
        m = ROW_COL.search(q)
        if not m:
            return Answer(None, None, "unsupported", "no cell")
        r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
        if not (0 <= r < self.n and 0 <= c < self.n):
            return Answer(None, None, "unsupported", "cell off board")
        return Answer(_design_text(self.designs[self.tiles[grid[r][c]]]),
                      _design_text(self.designs[self.tiles[grid[r][c]]]))

    def _end_cell(self, q):
        tid = self._tile_at(q)
        if tid is None:
            return Answer(None, None, "unsupported", "no cell")
        return Answer(self.cell_text(self.cell_of[-1][tid]), None)

    def _start_cell_of_end_tile(self, q):
        m = ROW_COL.search(q)
        if not m:
            return Answer(None, None, "unsupported", "no cell")
        r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
        tid = self.grids[-1][r][c]
        start = self.cell_of[0][tid]
        return Answer(self.cell_text(start), None)

    def _cells_visited(self, q):
        tid = self._tile_at(q)
        if tid is None:
            return Answer(None, None, "unsupported", "no cell")
        seen = {self.cell_of[i][tid] for i in range(len(self.grids))}
        return Answer(str(len(seen)), None)

    def _times_moved(self, q):
        tid = self._tile_at(q)
        if tid is None:
            return Answer(None, None, "unsupported", "no cell")
        return Answer(str(self.move_count[tid]), None)

    def _never_moved(self):
        n = sum(1 for tid in range(self.n * self.n) if self.move_count[tid] == 0)
        return Answer(str(n), None)

    def _moved_and_returned(self):
        n = sum(
            1 for tid in range(self.n * self.n)
            if self.move_count[tid] > 0 and self.cell_of[0][tid] == self.cell_of[-1][tid]
        )
        return Answer(str(n), None)

    def _distinct_kinds(self):
        ds = {self.tiles[t] for t in range(self.n * self.n)}
        return Answer(str(len(ds)), None)

    def _unique_tiles(self):
        import collections
        c = collections.Counter(self.tiles[t] for t in range(self.n * self.n))
        n = sum(1 for t in range(self.n * self.n) if c[self.tiles[t]] == 1)
        return Answer(str(n), None)

    def _count_kind(self, q, options):
        m = re.search(r"how many (.+?) tiles are on the board", q)
        if not m:
            return Answer(None, None, "unsupported", "no kind")
        color, kind = _parse_design(m.group(1))
        count = 0
        for t in range(self.n * self.n):
            d = self.designs[self.tiles[t]]
            if str(d["color"]).lower() == color and _norm_kind(d["kind"]) == kind:
                count += 1
        return Answer(str(count), None)

    def _first_move_cmp(self, q, options):
        pair = self._two_tiles(q)
        if pair is None:
            return Answer(None, None, "unsupported", "no two cells")
        a, b = pair
        fa, fb = self.first_move[a], self.first_move[b]
        if fa is None and fb is None:
            return Answer("neither of them ever moves", None)
        if fa is not None and fa == fb:
            return Answer("they first move at the same time", None)
        if fa is None:
            winner = b
        elif fb is None:
            winner = a
        else:
            winner = a if fa < fb else b
        return Answer(f"the one that starts in {self.cell_text(self.cell_of[0][winner])}", None)

    def _move_count_cmp(self, q, options):
        pair = self._two_tiles(q)
        if pair is None:
            return Answer(None, None, "unsupported", "no two cells")
        a, b = pair
        ca, cb = self.move_count[a], self.move_count[b]
        if ca == 0 and cb == 0:
            return Answer("neither of them ever moves", None)
        if ca == cb:
            return Answer("they move the same number of times", None)
        winner = a if ca > cb else b
        return Answer(f"the one that starts in {self.cell_text(self.cell_of[0][winner])}", None)

    def _most_moves(self):
        best = max(range(self.n * self.n), key=lambda t: self.move_count[t])
        return Answer(self.cell_text(self.cell_of[0][best]), None)

    def _rest_clock(self, q):
        tid = self._tile_at(q)
        if tid is None:
            return Answer(None, None, "unsupported", "no cell")
        first = self.first_move[tid]
        if first is None:
            return Answer(None, None, "uncertain", "tile never moves")
        return Answer(_clock(self.frame_end(first - 1)), None)

    def _line_move(self, options):
        lm = self.p.get("line_move")
        if lm is None:
            return Answer(None, None, "unsupported", "no line ever forms")
        return Answer(f"the {_ordinal(int(lm))} move", None)

    def _line_clock(self):
        lm = self.p.get("line_move")
        if lm is None:
            return Answer(None, None, "unsupported", "no line ever forms")
        return Answer(_clock(self.frame_end(int(lm) - 1)), None)

    def _line_location(self):
        line = self.line_of(self.grids[-1])
        if line is None:
            return Answer(None, None, "unsupported", "no end line")
        kind, idx = line
        if kind == "row":
            text = f"row {idx + 1}"
        elif kind == "col":
            text = f"column {idx + 1}"
        elif idx == 0:
            text = "the diagonal from top-left to bottom-right"
        else:
            text = "the diagonal from top-right to bottom-left"
        return Answer(text, None)

    def _line_kind(self):
        d = self.line_design(self.grids[-1])
        if d is None:
            return Answer(None, None, "unsupported", "no end line")
        return Answer(_design_text(d), None)

    def _line_start_overlap(self):
        line = self.line_of(self.grids[-1])
        if line is None:
            return Answer(None, None, "unsupported", "no end line")
        cells = set(self.line_cells(line))
        count = sum(1 for cell in cells if self.cell_of[0][self.grids[-1][cell[0]][cell[1]]] in cells)
        return Answer(str(count), None)

    def _cell_after_move(self, q):
        tid = self._tile_at(q)
        k = self._move_number(q)
        if tid is None or k is None or k < 1 or k > len(self.moves):
            return Answer(None, None, "unsupported", "bad cell/move")
        return Answer(self.cell_text(self.cell_of[k][tid]), None)

    def _first_partner_cell(self, q):
        tid = self._tile_at(q)
        first = self.first_move[tid] if tid is not None else None
        if first is None:
            return Answer(None, None, "unsupported", "tile never moves")
        perm = self.perms[first - 1]
        old = self.cell_of[first - 1][tid]
        other = next((s for s, t in perm.items() if t == old), None)
        if other is None:
            return Answer(None, None, "unsupported", "no partner")
        other_tid = self.grids[first - 1][other[0]][other[1]]
        return Answer(self.cell_text(self.cell_of[0][other_tid]), None)

    def _first_partner_kind(self, q):
        tid = self._tile_at(q)
        first = self.first_move[tid] if tid is not None else None
        if first is None:
            return Answer(None, None, "unsupported", "tile never moves")
        perm = self.perms[first - 1]
        old = self.cell_of[first - 1][tid]
        other = next((s for s, t in perm.items() if t == old), None)
        if other is None:
            return Answer(None, None, "unsupported", "no partner")
        other_tid = self.grids[first - 1][other[0]][other[1]]
        return Answer(_design_text(self.designs[self.tiles[other_tid]]), None)

    # --------------------------------------------------------- counterfactuals
    def _counterfactual(self, q, options):
        k = self._move_number(q)
        if k is None or k < 1 or k > len(self.moves):
            return Answer(None, None, "unsupported", "bad move number")
        perms = self.perms[: k - 1] + self.perms[k:]
        grids = self._run(perms)
        cells = [self._cell_index(g) for g in grids]
        tid = self._tile_at(q)

        if "where would the tile" in q:
            if tid is None:
                return Answer(None, None, "unsupported", "no cell")
            return Answer(self.cell_text(cells[-1][tid]), None)
        if "how many times would the tile" in q:
            if tid is None:
                return Answer(None, None, "unsupported", "no cell")
            seq = [cells[i][tid] for i in range(len(cells))]
            n = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            return Answer(str(n), None)
        if "how many complete rows" in q:
            return Answer(str(self.lines_count(grids[-1])), None)
        return Answer(None, None, "unsupported", "counterfactual variant")

    def _recolor_lines(self, q):
        m = re.search(r"if the (.+?) tile were a (.+?) instead", q)
        if not m:
            return Answer(None, None, "unsupported", "no recolor pair")
        src_color, src_kind = _parse_design(m.group(1))
        dst_color, dst_kind = _parse_design(m.group(2))
        if dst_kind != src_kind:
            # the question always keeps the shape/symbol and changes colour
            dst_kind = src_kind
        n = self.n
        total = 0
        grid = self.grids[-1]
        design_keys = []
        for tid in range(n * n):
            d = self.designs[self.tiles[tid]]
            key = (str(d["color"]).lower(), _norm_kind(d["kind"]))
            if key == (src_color, src_kind):
                key = (dst_color, dst_kind)
            design_keys.append(key)

        def line_ok(cells):
            keys = [design_keys[grid[r][c]] for r, c in cells]
            return len(set(keys)) == 1

        for r in range(n):
            if line_ok([(r, c) for c in range(n)]):
                total += 1
        for c in range(n):
            if line_ok([(r, c) for r in range(n)]):
                total += 1
        if line_ok([(i, i) for i in range(n)]):
            total += 1
        if line_ok([(i, n - 1 - i) for i in range(n)]):
            total += 1
        return Answer(str(total), None)

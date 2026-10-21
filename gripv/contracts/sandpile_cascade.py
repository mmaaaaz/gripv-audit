"""Deterministic replay for sandpile_cascade (chip-firing on four board shapes).

The scene is fully described by `params`: board shape (`scenario`), `dims`,
threshold `T`, the initial grain counts (`init`), and the drop list
(`drops`: target cell + gap).  A cell topples while it holds at least `T`
grains; each topple sends `T // deg` grains to every neighbour (one grain to
each of four neighbours on a grid, one to each of two neighbours on a pyramid
or a T=2 ring/row, two to each of two neighbours on a T=4 ring/row) and keeps
the remainder.  Grains pushed past an edge (or into the hole) are lost.

Every quantity asked about — final counts, per-cell topple totals, cascade
sizes, exit counts, which drop triggered a given topple — is therefore
reproducible exactly from the params.
"""
from __future__ import annotations

import re

from ..engine import Answer

FPS = 24.0

ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12,
}
ORDRE = "|".join(ORD)


class Cells:
    """Board geometry: cell ids, neighbour lists, edge labels, name parsing."""

    def __init__(self, params: dict):
        self.p = params
        self.scenario = params["scenario"]
        self.T = int(params["T"])
        d = params["dims"]
        self.cells: list[tuple[int, int]] = []
        self.index: dict[tuple[int, int], int] = {}
        if self.scenario == "grid":
            self.rows, self.cols = int(d["rows"]), int(d["cols"])
            for r in range(self.rows):
                for c in range(self.cols):
                    self.index[(r, c)] = len(self.cells)
                    self.cells.append((r, c))
        elif self.scenario == "pyramid":
            self.rows, self.w0 = int(d["rows"]), int(d["w0"])
            for r in range(self.rows):
                for c in range(self.w0 + r):
                    self.index[(r, c)] = len(self.cells)
                    self.cells.append((r, c))
        elif self.scenario == "row":
            self.n = int(d["n"])
            for c in range(self.n):
                self.index[(0, c)] = len(self.cells)
                self.cells.append((0, c))
        elif self.scenario == "ring":
            self.k = int(d["k"])
            for c in range(self.k - 1):
                self.index[(0, c)] = len(self.cells)
                self.cells.append((0, c))
        else:
            raise ValueError(f"unknown scenario {self.scenario!r}")
        self.nbrs = [self._neighbours(c) for c in self.cells]

    # ------------------------------------------------------------- geometry
    def _neighbours(self, cell):
        r, c = cell
        out = []
        if self.scenario == "grid":
            for dr, dc, edge in ((-1, 0, "top"), (1, 0, "bottom"),
                                 (0, -1, "left"), (0, 1, "right")):
                out.append((self.index.get((r + dr, c + dc)), edge))
        elif self.scenario == "pyramid":
            for dr, dc, edge in ((1, 0, "down-left"), (1, 1, "down-right")):
                out.append((self.index.get((r + dr, c + dc)), edge))
        elif self.scenario == "row":
            out = [(self.index.get((0, c - 1)), "left"),
                   (self.index.get((0, c + 1)), "right")]
        elif self.scenario == "ring":
            i = self.index[cell]
            out = [(i + 1 if i + 1 < len(self.cells) else None, "hole"),
                   (i - 1 if i - 1 >= 0 else None, "hole")]
        return out

    # ---------------------------------------------------------- name lookup
    def ref(self, text: str):
        """Index of the cell named in `text`, or None."""
        t = str(text).lower()
        m = re.search(r"row (\d+),\s*column (\d+)", t)
        if m:
            return self.index.get((int(m.group(1)) - 1, int(m.group(2)) - 1))
        if "top cell" in t:
            return self.index.get((0, 0))
        m = re.search(rf"({ORDRE}) cell clockwise from the hole", t)
        if m:
            return ORD[m.group(1)] - 1
        m = re.search(rf"({ORDRE}) cell from the left in the ({ORDRE}) row", t)
        if m:
            return self.index.get((ORD[m.group(2)] - 1, ORD[m.group(1)] - 1))
        m = re.search(rf"({ORDRE}) tray from the (left|right)", t)
        if m:
            i = ORD[m.group(1)]
            return i - 1 if m.group(2) == "left" else self.n - i
        return None


class Sim:
    """Full replay: per-drop cascades, topple events, exits."""

    def __init__(self, cells: Cells, init, drops):
        self.c = cells
        self.T = cells.T
        n = len(cells.cells)
        g = list(init)
        if len(g) < n:
            g = g + [0] * (n - len(g))
        g = g[:n]
        self.start = list(g)
        self.states = [list(g)]           # state after each drop (index 0 = start)
        self.topples = [0] * n            # cumulative topple count per cell
        self.events = []                   # (cell, amt, drop_no, wave)
        self.per_drop_topples = []
        self.per_drop_exits = []
        self.exits = []                    # edge labels, in event order
        self.drop_topples_at = []          # cumulative topples of each cell per drop
        self.cum = [list(self.topples)]

        for dno, target in enumerate(drops, 1):
            g[target] += 1
            before_exit = len(self.exits)
            cnt, wave = 0, 0
            while True:
                uns = [i for i in range(n) if g[i] >= self.T]
                if not uns:
                    break
                for i in uns:
                    if g[i] < self.T:
                        continue
                    self._fire(g, i, dno, wave)
                    cnt += 1
                wave += 1
                if wave > 100000:
                    break
            self.per_drop_topples.append(cnt)
            self.per_drop_exits.append(len(self.exits) - before_exit)
            self.states.append(list(g))
            self.cum.append(list(self.topples))

    def _fire(self, g, i, dno, wave):
        amt = g[i]
        nbrs = self.c.nbrs[i]
        deg = len(nbrs)
        q = self.T // deg if self.T >= deg else 1
        g[i] = amt - q * deg
        for j, edge in nbrs:
            if j is None:
                self.exits.extend([edge] * q)
            else:
                g[j] += q
        self.topples[i] += 1
        self.events.append((i, amt, dno, wave))

    # ------------------------------------------------------------- queries
    @property
    def final(self):
        return self.states[-1]

    def total_exits(self):
        return len(self.exits)

    def drop_of_topple(self, cell, k):
        """1-based drop number in which `cell` topples for the k-th time."""
        seen = 0
        for (c, amt, dno, wave) in self.events:
            if c == cell:
                seen += 1
                if seen == k:
                    return dno
        return None

    def order_key(self, cell):
        """(drop, wave, position) of the first topple of `cell`."""
        for pos, (c, amt, dno, wave) in enumerate(self.events):
            if c == cell:
                return (dno, wave, pos)
        return None


def _ok(v):
    return Answer(str(v), confidence="exact")


class SandpileCascade:
    def __init__(self, params: dict):
        self.p = params
        self.c = Cells(params)
        self.sim = Sim(self.c, self._init(), self._drops())

    # ------------------------------------------------------------- helpers
    def _init(self):
        init = list(self.p["init"])
        if self.c.scenario == "ring" and len(init) > len(self.c.cells):
            init = init[1:]                      # init[0] is the hole
        return init

    def _drops(self):
        off = 1 if (self.c.scenario == "ring" and len(self.p["init"]) > len(self.c.cells)) else 0
        return [d["cell"] - off for d in self.p["drops"]]

    def _opt_cell(self, options, want):
        """Return the option text whose cell satisfies `want(index) -> bool`."""
        hits = []
        for o in options:
            i = self.c.ref(o)
            if i is not None and want(i):
                hits.append(str(o))
        if len(hits) == 1:
            return hits[0]
        return None

    def _num_opt(self, options, value):
        """Return the option text equal to `value`."""
        for o in options:
            m = re.search(r"-?\d+", str(o))
            if m and int(m.group()) == value:
                return str(o)
        return str(value)

    # ------------------------------------------------------------- dispatch
    def answer(self, question: str, options=None) -> Answer:
        # trailing asides such as "(Rows are counted from the top.)" would
        # otherwise break every anchor that ends in "?"
        q = re.sub(r"\([^)]*\)\s*$", "", question.lower())
        q = re.sub(r"\s+", " ", q).strip()
        opts = [str(o) for o in (options or [])]
        c, sim = self.c, self.sim

        # ---------------------------------------------------------- counts
        if "how many grains are on the board at the very start" in q:
            return _ok(sum(sim.start))
        if "how many grains are on the board at the end of the video" in q:
            return _ok(sum(sim.final))
        if "how many grains are dropped onto the board during the video" in q:
            return _ok(len(self.p["drops"]))
        if "how many cells does the board have" in q:
            return _ok(len(c.cells))
        if "how many trays does the board have" in q:
            return _ok(len(c.cells))
        if "how many grains are in" in q and "at the very start" in q:
            i = self._ref_in(q, "at the very start")
            return Answer(str(sim.start[i]), confidence="exact") if i is not None else self._unsup()
        if "how many grains are in" in q and "at the end of the video" in q:
            i = self._ref_in(q, "at the end of the video")
            return Answer(str(sim.final[i]), confidence="exact") if i is not None else self._unsup()
        if "once the cascade set off by" in q and "has settled" in q and "how many grains are in" in q:
            m = re.search(rf"the ({ORDRE}) grain to be dropped", q)
            i = self._ref_in(q, "how many grains are in")
            if m and i is not None:
                dno = ORD[m.group(1)]
                return Answer(str(sim.states[dno][i]), confidence="exact")
            return self._unsup()
        if "how many grains are in" in q and "when the clock shows" in q:
            return self._at_clock(q, opts)

        # --------------------------------------------------------- toppling
        if "how many times in total does a" in q and "topple" in q:
            return _ok(sum(sim.topples))
        if "how many different cells topple" in q or "how many different trays topple" in q:
            return _ok(sum(1 for t in sim.topples if t > 0))
        if "how many times does" in q and "topple" in q:
            m = re.search(r"how many times does (?:the )?(.+?) topple", q)
            i = c.ref(m.group(1)) if m else None
            if i is not None:
                return Answer(str(sim.topples[i]), confidence="exact")
            return self._unsup()
        if q.startswith("does the") or q.startswith("does "):
            m = re.search(r"does (?:the )?(.+?) topple \(pass", q)
            i = c.ref(m.group(1)) if m else None
            if i is not None:
                return Answer("yes" if sim.topples[i] > 0 else "no", confidence="exact")
            return self._unsup()
        if "topples more times over the whole video" in q:
            return self._cmp_two(q, opts, lambda i: sim.topples[i])
        if "which holds more grains" in q and "at the end of the video" in q:
            return self._cmp_two(q, opts, lambda i: sim.final[i])
        if "which of these" in q and "holds the most grains" in q:
            vals = [(sim.final[i], str(o)) for o in opts for i in [c.ref(o)] if i is not None]
            if not vals:
                return self._unsup()
            top = max(v for v, _ in vals)
            win = [o for v, o in vals if v == top]
            if len(win) == 1:
                return Answer(win[0], confidence="exact")
            return self._unsup()
        if "compared with the very start" in q:
            m = re.search(r"does (?:the )?(.+?) hold more grains", q)
            i = c.ref(m.group(1)) if m else None
            if i is None:
                return self._unsup()
            a, b = sim.start[i], sim.final[i]
            word = "more" if b > a else "fewer" if b < a else "the same number"
            for o in opts:
                ol = str(o).lower()
                if word.split()[0] in ol and (word != "the same number" or "same" in ol):
                    return Answer(str(o), confidence="exact")
            return Answer("more" if b > a else "fewer" if b < a else "same", confidence="exact")

        # ------------------------------------------------------ drop targets
        if "dropped grain land" in q:
            m = re.search(rf"the ({ORDRE}) (?:dropped )?grain", q)
            if m:
                k = ORD[m.group(1)]
                if k <= len(self.p["drops"]):
                    tgt = self._drops()[k - 1]
                    hit = self._opt_cell(opts, lambda j: j == tgt)
                    if hit is not None:
                        return Answer(hit, confidence="exact")
            return self._unsup()

        # --------------------------------------------------------- cascades
        if "how many topples does" in q and "grain to be dropped set off" in q:
            m = re.search(rf"the ({ORDRE}) grain to be dropped", q)
            if m:
                d = ORD[m.group(1)]
                return Answer(str(sim.per_drop_topples[d - 1]), confidence="exact")
            return self._unsup()
        if "go off the board in the cascade set off by" in q:
            m = re.search(rf"the ({ORDRE}) grain to be dropped", q)
            if m:
                d = ORD[m.group(1)]
                return Answer(str(sim.per_drop_exits[d - 1]), confidence="exact")
            return self._unsup()
        if "go into the hole in the cascade set off by" in q:
            m = re.search(rf"the ({ORDRE}) grain to be dropped", q)
            if m:
                d = ORD[m.group(1)]
                return Answer(str(sim.per_drop_exits[d - 1]), confidence="exact")
            return self._unsup()
        if "how many grains go off the board over the whole video" in q:
            return _ok(len(sim.exits))
        if "how many grains go into the hole over the whole video" in q:
            return _ok(len(sim.exits))
        if "sets off more topples" in q:
            m = re.search(rf"the ({ORDRE}) grain to be dropped or the ({ORDRE}) grain", q)
            if m:
                a, b = ORD[m.group(1)], ORD[m.group(2)]
                ta, tb = sim.per_drop_topples[a - 1], sim.per_drop_topples[b - 1]
                if ta == tb:
                    return self._unsup()
                want = "first" if ta > tb else "second"
                for o in opts:
                    om = re.search(rf"({ORDRE}) grain", str(o).lower())
                    if om and ORD[om.group(1)] == (a if ta > tb else b):
                        return Answer(str(o), confidence="exact")
                return Answer(f"the {a if ta > tb else b}", confidence="exact")
            return self._unsup()
        if "which of these dropped grains sets off the most topples" in q:
            vals = []
            for o in opts:
                om = re.search(rf"({ORDRE}) grain", str(o).lower())
                if om:
                    vals.append((sim.per_drop_topples[ORD[om.group(1)] - 1], str(o)))
            if not vals:
                return self._unsup()
            top = max(v for v, _ in vals)
            win = [o for v, o in vals if v == top]
            return Answer(win[0], confidence="exact") if len(win) == 1 else self._unsup()
        if "sets off the cascade in which" in q:
            m = re.search(rf"in which (?:the )?(.+?) topples for the ({ORDRE}) time", q)
            if m:
                i, k = c.ref(m.group(1)), ORD[m.group(2)]
                if i is not None:
                    d = sim.drop_of_topple(i, k)
                    if d is not None:
                        for o in opts:
                            om = re.search(rf"({ORDRE}) grain", str(o).lower())
                            if om and ORD[om.group(1)] == d:
                                return Answer(str(o), confidence="exact")
                        return Answer(f"the {d}", confidence="exact")
            return self._unsup()
        if "how many grains have been dropped" in q:
            m = re.search(rf"when the ({ORDRE}) (?:cell|tray) topples", q)
            if m:
                k = ORD[m.group(1)]
                hits = [dno for (cc, amt, dno, wave) in sim.events]
                if len(hits) >= k:
                    return Answer(str(hits[k - 1]), confidence="exact")
            return self._unsup()

        # ------------------------------------------------------ which gets grain
        if "which of these cells gets one of its grains" in q or \
           "which of these trays gets one of its grains" in q:
            m = re.search(r"when (?:the )?(.+?) topples for the", q)
            if m:
                i = c.ref(m.group(1))
                if i is not None:
                    nbrs = {j for j, _ in c.nbrs[i] if j is not None}
                    hit = self._opt_cell(opts, lambda j: j in nbrs)
                    if hit is not None:
                        return Answer(hit, confidence="exact")
            return self._unsup()

        # ------------------------------------------------------------- order
        if ("which cell is the" in q or "which tray is the" in q) and "to topple" in q:
            if "last to topple" in q:
                if sim.events:
                    last = sim.events[-1][0]
                    hit = self._opt_cell(opts, lambda j: j == last)
                    if hit is not None:
                        return Answer(hit, confidence="exact")
                return self._unsup()
            if " or " not in q:
                m = re.search(rf"is the ({ORDRE}) to topple", q)
                if m and len(sim.events) >= ORD[m.group(1)]:
                    kth = sim.events[ORD[m.group(1)] - 1][0]
                    hit = self._opt_cell(opts, lambda j: j == kth)
                    if hit is not None:
                        return Answer(hit, confidence="exact")
                return self._unsup()
            if "last to topple" in q:
                if sim.events:
                    last = sim.events[-1][0]
                    hit = self._opt_cell(opts, lambda j: j == last)
                    if hit is not None:
                        return Answer(hit, confidence="exact")
                return self._unsup()
            return self._cmp_two(q, opts, None, order=True)
        if "off which end of the row" in q:
            m = re.search(rf"the ({ORDRE}) grain to leave the board", q)
            if m:
                k = ORD[m.group(1)]
                if len(sim.exits) >= k:
                    side = sim.exits[k - 1]
                    for o in opts:
                        ol = str(o).lower()
                        if side in ol:
                            return Answer(str(o), confidence="exact")
                    return Answer(f"the {side} end", confidence="exact")
            return self._unsup()
        if "over which edge of the board" in q:
            m = re.search(rf"the ({ORDRE}) grain to leave the board", q)
            if m:
                k = ORD[m.group(1)]
                if len(sim.exits) >= k:
                    edge = sim.exits[k - 1]
                    for o in opts:
                        if edge in str(o).lower():
                            return Answer(str(o), confidence="exact")
                    return Answer(f"the {edge} edge", confidence="exact")
            return self._unsup()

        # ------------------------------------------------------ counterfactual
        if "suppose" in q:
            return self._counterfactual(q, opts)

        # ------------------------------------------------------------- time
        if ("time on the clock" in q or "when the clock shows" in q
                or "topples happen between" in q or "seconds pass between" in q):
            return self._unsup("clock template not pinned")
        return self._unsup()

    # ------------------------------------------------------------- helpers
    def _unsup(self, note="template not handled"):
        return Answer(None, confidence="unsupported", note=note)

    def _ref_in(self, q, marker):
        m = re.search(r"how many grains (?:are )?in (?:the )?(.+?)(?:%s|\?)" % re.escape(marker), q)
        if not m:
            m = re.search(r"how many grains are in (?:the )?(.+?)\?", q)
        return self.c.ref(m.group(1)) if m else None

    def _cmp_two(self, q, opts, val, order=False):
        """Compare two named cells: bigger value wins (or earlier topple)."""
        m = re.search(r"(?:video|grain[s]?): (.+?) or (.+?)\?$", q)
        if not m:
            m = re.search(r": (.+?) or (.+?)\?$", q)
        if not m:
            return self._unsup()
        a_txt, b_txt = m.group(1), m.group(2)
        ia, ib = self.c.ref(a_txt), self.c.ref(b_txt)
        if ia is None or ib is None:
            return self._unsup()
        if order:
            ka, kb = self.sim.order_key(ia), self.sim.order_key(ib)
            if ka is None and kb is None:
                return self._unsup()
            if ka is None:
                win = ib
            elif kb is None:
                win = ia
            elif ka < kb:
                win = ia
            elif kb < ka:
                win = ib
            else:
                return self._unsup()
        else:
            va, vb = val(ia), val(ib)
            if va == vb:
                return self._unsup()
            win = ia if va > vb else ib
        for o in opts:
            if self.c.ref(o) == win:
                return Answer(str(o), confidence="exact")
        return self._unsup()

    def _at_clock(self, q, opts):
        return self._unsup("clock template not pinned")

    def _counterfactual(self, q, opts):
        """Re-run the board with one change and report a count."""
        c = self.c
        init = list(self._init())
        drops = list(self._drops())

        # (a) one more grain dropped after the end of the video
        if "suppose one more grain were dropped onto" in q:
            m = re.search(r"dropped onto (?:the )?(.+?) after the end", q)
            i = c.ref(m.group(1)) if m else None
            if i is None:
                return self._unsup()
            s = Sim(c, self.sim.final, [i])
            if "how many topples would follow" in q:
                return Answer(str(s.per_drop_topples[0]), confidence="exact")
            if "how many grains would go off the board" in q:
                return Answer(str(s.per_drop_exits[0]), confidence="exact")
            return self._unsup()

        # (b) the kth grain never dropped
        m = re.search(rf"suppose the ({ORDRE}) grain to be dropped had never been dropped", q)
        if m:
            k = ORD[m.group(1)]
            newdrops = drops[:k - 1] + drops[k:]
            s = Sim(c, init, newdrops)
            if "would" in q and "topple at any point" in q:
                mm = re.search(r"would (?:the )?(.+?) topple at any point", q)
                i = c.ref(mm.group(1)) if mm else None
                if i is None:
                    return self._unsup()
                return Answer("yes" if s.topples[i] > 0 else "no", confidence="exact")
            if "how many topples would there be in total" in q:
                return Answer(str(sum(s.topples)), confidence="exact")
            if "how many grains would go off the board in total" in q or \
               "how many grains would go into the hole in total" in q:
                return Answer(str(len(s.exits)), confidence="exact")
            return self._unsup()

        # (c) the kth grain landed elsewhere
        m = re.search(rf"suppose the ({ORDRE}) grain to be dropped had landed on (?:the )?(.+?) instead", q)
        if m:
            k = ORD[m.group(1)]
            i = c.ref(m.group(2))
            if i is None:
                return self._unsup()
            newdrops = list(drops)
            newdrops[k - 1] = i
            s = Sim(c, init, newdrops)
            if "how many topples would that grain set off" in q:
                return Answer(str(s.per_drop_topples[k - 1]), confidence="exact")
            if "how many topples would there be in total" in q:
                return Answer(str(sum(s.topples)), confidence="exact")
            return self._unsup()

        # (d) a cell starts with one grain more / fewer
        m = re.search(rf"suppose (?:the )?(.+?) had started with one grain (more|fewer)", q)
        if m:
            i = c.ref(m.group(1))
            if i is None:
                return self._unsup()
            newinit = list(init)
            newinit[i] += 1 if m.group(2) == "more" else -1
            s = Sim(c, newinit, drops)
            if "how many topples would there be in total" in q:
                return Answer(str(sum(s.topples)), confidence="exact")
            if "how many grains would go off the board in total" in q:
                return Answer(str(len(s.exits)), confidence="exact")
            return self._unsup()

        return self._unsup()

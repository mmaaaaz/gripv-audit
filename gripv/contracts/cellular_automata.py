"""Deterministic replay for cellular_automata colour/count questions.

`history[g][r][c]` is the state grid for generation g: 1 = live colour,
2 = fading/grey, 0 = empty. Generation g is on screen from
`start_f + g*gen_f` frames. Rule inference and counterfactual first-step
questions are left unsupported unless they can be read from history.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0


class CellularAutomata:
    def __init__(self, params: dict):
        self.p = params
        self.history = params.get("history", [])
        self.timing = params.get("timing", {})
        self.style = params.get("style", {})
        self.markers = params.get("markers", [])
        self.start_f = float(self.timing.get("start_f", 0))
        self.gen_f = float(self.timing.get("gen_f", 1))
        self.n_frames = float(params.get("n_frames") or params.get("gens", len(self.history)) * self.gen_f + self.start_f)
        self.live = str(self.style.get("live_name", "teal")).lower()
        self.dying = str(self.style.get("dying_name", "grey")).lower()

    def gen_at(self, t: float) -> int:
        f = float(t) * FPS
        return max(0, min(len(self.history) - 1, 1 + int(math.floor((f - self.start_f) / self.gen_f + 1e-9))))

    def name(self, state: int) -> str:
        if state == 1:
            return self.live
        if state == 2:
            return self.dying
        return "empty"

    def count(self, g: int, colour: str) -> int:
        grid = self.history[g]
        if colour == "empty" or (colour == "black" and self.live != "black"):
            return sum(1 for row in grid for v in row if v == 0)
        target = 1 if colour == self.live else (2 if colour == self.dying else None)
        if target is None:
            return 0
        return sum(1 for row in grid for v in row if v == target)

    def cell_state(self, g: int, marker: dict) -> int:
        r, c = marker["cell"]
        return self.history[g][r][c]

    def marker_from_q(self, q: str) -> dict | None:
        for m in self.markers:
            if str(m.get("colour", "")).lower() in q and str(m.get("shape", "")).lower() in q:
                return m
        return None

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        end = len(self.history) - 1

        # marker colour at start / between times
        m = self.marker_from_q(q)
        if "at the start" in q and "which of these cells are" in q:
            mc = re.search(r"which of these cells are (.+?) at the start", q)
            ms = [x for x in self.markers if str(x.get("colour", "")).lower() in q and str(x.get("shape", "")).lower() in q]
            if mc and len(ms) == 2:
                want = mc.group(1).strip()
                hits = [x for x in ms if self.name(self.cell_state(0, x)) == want]
                if len(hits) == 2:
                    return Answer("both", confidence="exact")
                if not hits:
                    return Answer("neither", confidence="exact")
                return Answer(f"only the {hits[0]['colour']} {hits[0]['shape']}", confidence="exact")
        if m and "between" in q and "and" in q:
            ms = re.search(r"between ([\d.]+) s and ([\d.]+) s", q)
            if ms:
                a = self.name(self.cell_state(self.gen_at(float(ms.group(1))), m))
                b = self.name(self.cell_state(self.gen_at(float(ms.group(2))), m))
                if a == b:
                    return Answer(f"it stays {a}", confidence="exact")
                return Answer(f"it turns from {a} to {b}", confidence="exact")

        # colour counts
        mc = re.search(r"how many (.+?) cells are there at the start", q)
        if mc:
            return Answer(str(self.count(0, mc.group(1).strip())), confidence="exact")
        mc = re.search(r"how many (.+?) cells are there at ([\d.]+) s", q)
        if mc:
            return Answer(str(self.count(self.gen_at(float(mc.group(2))), mc.group(1).strip())), confidence="exact")

        mc = re.search(r"compared with the start, are there more or fewer (.+?) cells at the end", q)
        if mc:
            c0, c1 = self.count(0, mc.group(1).strip()), self.count(end, mc.group(1).strip())
            return Answer("more" if c1 > c0 else "fewer" if c1 < c0 else "the same number", confidence="exact")

        mc = re.search(r"when are there more (.+?) cells: at the start or at ([\d.]+) s", q)
        if mc:
            c0, c1 = self.count(0, mc.group(1).strip()), self.count(self.gen_at(float(mc.group(2))), mc.group(1).strip())
            return Answer("at the start" if c0 > c1 else "at %s s" % mc.group(2) if c1 > c0 else "they have the same number", confidence="exact")

        mc = re.search(r"at which of these moments are there the most (.+?) cells", q)
        if mc and options:
            colour = mc.group(1).strip()
            best, best_n = None, -1
            for opt in options:
                ol = str(opt).lower()
                if "the end" in ol:
                    g = end
                else:
                    mo = re.search(r"([\d.]+) s|at the start", ol)
                    if not mo:
                        continue
                    g = 0 if mo.group(1) is None else self.gen_at(float(mo.group(1)))
                n = self.count(g, colour)
                if n > best_n or (n == best_n and "start" in str(opt).lower()):
                    best, best_n = str(opt), n
            return Answer(best, confidence="exact") if best else Answer(None, confidence="unsupported")

        if "how many different cells are" in q and "at some moment" in q:
            mc = re.search(r"different cells are (.+?) at some moment", q)
            if mc:
                colour = mc.group(1).strip()
                target = 1 if colour == self.live else (2 if colour == self.dying else None)
                if target is None:
                    return Answer(None, confidence="unsupported")
                cells = set()
                for g in range(len(self.history)):
                    for r, row in enumerate(self.history[g]):
                        for c, v in enumerate(row):
                            if v == target:
                                cells.add((r, c))
                return Answer(str(len(cells)), confidence="exact")

        if "how many times does an empty cell turn" in q:
            mc = re.search(r"empty cell turn (.+?)(?:\?|$)", q)
            if mc:
                colour = mc.group(1).strip()
                target = 1 if colour == self.live else (2 if colour == self.dying else None)
                if target is None:
                    return Answer(None, confidence="unsupported")
                n = 0
                for g in range(1, len(self.history)):
                    for r, row in enumerate(self.history[g]):
                        for c, v in enumerate(row):
                            if v == target and self.history[g - 1][r][c] == 0:
                                n += 1
                return Answer(str(n), confidence="exact")

        if m and "first change colour" in q:
            for g in range(1, len(self.history)):
                if self.cell_state(g, m) != self.cell_state(0, m):
                    t = (self.start_f + (g - 1) * self.gen_f) / FPS
                    return Answer(f"{round(t, 1)} s", confidence="exact")
            return Answer(None, confidence="unsupported")

        if "first become completely empty" in q:
            for g, grid in enumerate(self.history):
                if all(v == 0 for row in grid for v in row):
                    t = (self.start_f + (g - 1) * self.gen_f) / FPS
                    return Answer(f"{round(t, 1)} s", confidence="exact")
            return Answer("never: some cells remain to the end", confidence="exact")

        return Answer(None, confidence="unsupported", note="rule/counterfactual template not handled")

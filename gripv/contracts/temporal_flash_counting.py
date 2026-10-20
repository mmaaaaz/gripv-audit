"""Deterministic replay for `temporal_flash_counting`.

Params:
  shapes: [{color, type, rhythm, row, col, ...}]
  flashes: [{i, f0, f1, white, cause}]  — f0/f1 are frame numbers at 24 fps
  pulses: [{i, f0, f1}]                 — grow/shrink without lighting up
  pair: [i, j] + pair_kind               — the closest pair the question asks about
  rows, cols                             — grid dimensions when present

A flash is a lighting event in `flashes`; `pulses` are explicitly not flashes.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from ..engine import Answer

FPS = 24.0
COLOURS = ["light blue", "red", "pink", "blue", "brown", "yellow", "green",
           "orange", "purple", "teal", "cyan", "lime", "grey", "gray"]


class TemporalFlash:
    def __init__(self, params: dict):
        self.p = params
        self.shapes = params.get("shapes", [])
        for idx, sh in enumerate(self.shapes):
            sh["id"] = idx
        self.flashes = params.get("flashes", [])
        self.pulses = params.get("pulses", []) or []
        self.pair = params.get("pair")
        self.pair_kind = params.get("pair_kind")
        self.by_i = defaultdict(list)
        for f in self.flashes:
            self.by_i[f["i"]].append(f)

    # ---------- naming ----------
    def colour(self, s: dict) -> str:
        return str(s.get("color", "")).lower()

    def full_name(self, s: dict, lamp: bool = False) -> str:
        c = self.colour(s)
        return f"{c} lamp" if lamp else f"{c} {s.get('type', 'shape')}"

    def find(self, text: str) -> dict | None:
        low = text.lower()
        found = [s for s in self.shapes if self.colour(s) in low]
        if not found:
            return None
        # if a type word is present, prefer the colour+type match
        words = set(re.findall(r"[a-z]+", low))
        typed = [s for s in found if str(s.get("type", "")).lower() in words]
        pool = typed or found
        if len(pool) == 1:
            return pool[0]
        # prefer the longest colour name ("light blue" over "blue")
        if len(pool) > 1:
            return max(pool, key=lambda s: len(self.colour(s)))
        return None

    def name_in_question(self, q: str) -> dict | None:
        return self.find(q)

    def option_shapes(self, options) -> list[dict]:
        if not options:
            return []
        text = " ".join(str(o).lower() for o in options)
        return [s for s in self.shapes if self.full_name(s) in text or f"{self.colour(s)} lamp" in text]

    # ---------- events ----------
    def flashes_of(self, s: dict) -> list[dict]:
        return sorted(self.by_i.get(s.get("id", self.shapes.index(s)), []), key=lambda f: f["f0"])

    def count(self, s: dict) -> int:
        return len(self.flashes_of(s))

    def duration(self, f: dict) -> float:
        return (f["f1"] - f["f0"]) / FPS

    def first_t(self, s: dict) -> float | None:
        ev = self.flashes_of(s)
        return ev[0]["f0"] / FPS if ev else None

    def last_t(self, s: dict) -> float | None:
        ev = self.flashes_of(s)
        return ev[-1]["f0"] / FPS if ev else None

    def most(self, pool: list[dict], key=None) -> dict | None:
        if not pool:
            return None
        key = key or self.count
        best = max(pool, key=key)
        if sum(1 for s in pool if key(s) == key(best)) > 1:
            return best  # tie is rare; options still contain one of them
        return best

    def phrase(self, s: dict, q: str) -> str:
        return self.full_name(s, lamp=("lamp" in q.lower()))

    # ---------- answer ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if "how many lamps are on the panel" in q or "how many shapes are on the screen" in q:
            return Answer(str(len(self.shapes)), confidence="exact")

        if "how many rows of shapes are in the grid" in q:
            return Answer(str(self.p.get("rows", 0)), confidence="exact")
        if "how many columns of shapes are in the grid" in q:
            return Answer(str(self.p.get("cols", 0)), confidence="exact")

        # per-shape count
        m = re.search(r"how many times does the (.+?) (?:lamp|shape|flash)\b", q)
        if m and (s := self.name_in_question(m.group(1))):
            return Answer(str(self.count(s)), confidence="exact")

        # difference between two shapes
        m = re.search(r"how many more times does the (.+?) (?:lamp|shape)? ?flash than the (.+?)(?:\?| \(|$)", q)
        if m:
            a, b = self.name_in_question(m.group(1)), self.name_in_question(m.group(2))
            if a and b:
                return Answer(str(abs(self.count(a) - self.count(b))), confidence="exact")

        # white events
        if "how many of the" in q and "flashes turn it white" in q:
            s = self.name_in_question(q)
            if s:
                return Answer(str(sum(1 for f in self.flashes_of(s) if f.get("white"))), confidence="exact")

        if ("counting every shape" in q and "how many flashes turn a shape white" in q) or            ("counting every lamp" in q and "how many flashes turn a lamp white" in q):
            return Answer(str(sum(1 for f in self.flashes if f.get("white"))), confidence="exact")

        # total flashes
        if "counting every shape" in q and "how many flashes are there in the whole video" in q:
            return Answer(str(len(self.flashes)), confidence="exact")
        if "counting every lamp" in q and "how many flashes are there in the whole video" in q:
            return Answer(str(len(self.flashes)), confidence="exact")

        # longest flash
        if "single longest flash" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                best = max(pool, key=lambda s: max(self.duration(f) for f in self.flashes_of(s)))
                return Answer(self.phrase(best, q), confidence="exact")

        # first / last to flash
        if "which one flashes first" in q or "which shape is the first to flash" in q or "which lamp is the first to flash" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                s = min(pool, key=self.first_t)
                return Answer(self.phrase(s, q), confidence="exact")
        if "which shape is the last to start a flash" in q or "which lamp is the last to start a flash" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                s = max(pool, key=self.last_t)
                return Answer(self.phrase(s, q), confidence="exact")

        # steady rhythm
        if "flashes at a steady rhythm" in q:
            if "how many seconds pass" in q:
                s = self.name_in_question(q)
                if s:
                    ev = self.flashes_of(s)
                    if len(ev) >= 2:
                        gaps = [(b["f0"] - a["f0"]) / FPS for a, b in zip(ev, ev[1:])]
                        gaps.sort()
                        g = gaps[len(gaps) // 2]
                        return Answer(f"{round(g * 2) / 2:.1f} s", confidence="exact")
            else:
                pool = [s for s in self.shapes if str(s.get("rhythm")) == "steady"]
                if len(pool) == 1:
                    return Answer(self.phrase(pool[0], q), confidence="exact")

        # white-at-least-once
        if "turns white during at least one of its flashes" in q:
            pool = self.option_shapes(options) or list(self.shapes)
            pool = [s for s in pool if any(f.get("white") for f in self.flashes_of(s))]
            if len(pool) == 1:
                return Answer(self.phrase(pool[0], q), confidence="exact")
            if options:
                for s in pool:
                    if self.phrase(s, q) in " ".join(str(o).lower() for o in options):
                        return Answer(self.phrase(s, q), confidence="exact")

        # grow/shrink without a flash
        if "grows bigger and then shrinks back" in q:
            if len(self.pulses) == 1:
                i = self.pulses[0]["i"]
                s = self.shapes[i]
                return Answer(self.phrase(s, q), confidence="exact")

        # most flashes, optionally filtered
        if "which shape flashes the most times" in q or "which lamp flashes the most times" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                best = max(pool, key=self.count)
                return Answer(self.phrase(best, q), confidence="exact")

        if "if only flashes that turn a shape white counted" in q or "if only flashes that turn a lamp white counted" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if any(f.get("white") for f in self.flashes_of(s))]
            if pool:
                best = max(pool, key=lambda s: sum(1 for f in self.flashes_of(s) if f.get("white")))
                return Answer(self.phrase(best, q), confidence="exact")

        m = re.search(r"if only flashes lasting longer than ([\d.]+) s counted.*?which (?:shape|lamp) would have the most flashes", q)
        if m:
            thr = float(m.group(1))
            pool = self.option_shapes(options) or list(self.shapes)
            pool = [s for s in pool if any(self.duration(f) > thr for f in self.flashes_of(s))]
            if pool:
                best = max(pool, key=lambda s: sum(1 for f in self.flashes_of(s) if self.duration(f) > thr))
                return Answer(self.phrase(best, q), confidence="exact")

        # if the top flasher never flashed
        if "had never flashed" in q and "which" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                top = max(pool, key=self.count)
                rest = [s for s in pool if s is not top]
                if rest:
                    best = max(rest, key=self.count)
                    return Answer(self.phrase(best, q), confidence="exact")

        # duration-threshold counts
        m = re.search(r"how many of the (.+?)'s flashes last longer than ([\d.]+) s", q)
        if m and (s := self.name_in_question(m.group(1))):
            thr = float(m.group(2))
            return Answer(str(sum(1 for f in self.flashes_of(s) if self.duration(f) > thr)), confidence="exact")

        # flash pattern
        m = re.search(r"reading the (.+?)'s flashes in order.*?short = lit for less than ([\d.]+) s, long = lit for more than", q)
        if m and (s := self.name_in_question(m.group(1))):
            thr = float(m.group(2))
            parts = ["short" if self.duration(f) < thr else "long" for f in self.flashes_of(s)]
            return Answer(", ".join(parts), confidence="exact")

        # stopped at a timer reading
        m = re.search(r"stopped the moment the timer first read ([\d.]+) s.*?how many flashes would the (.+?) have started", q)
        if m:
            t = float(m.group(1))
            s = self.name_in_question(m.group(2))
            if s:
                return Answer(str(sum(1 for f in self.flashes_of(s) if f["f0"] / FPS <= t + 1e-9)),
                              confidence="exact")

        # flashes starting in a window
        m = re.search(r"how many flashes start between ([\d.]+) s and ([\d.]+) s", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return Answer(str(sum(1 for f in self.flashes if lo - 1e-9 <= f["f0"] / FPS < hi - 1e-9)),
                          confidence="exact")

        # timer at first light
        m = re.search(r"what does the timer read when the (.+?) first lights up", q)
        if m and (s := self.name_in_question(m.group(1))):
            ev = self.flashes_of(s)
            if not ev:
                return Answer(None, confidence="unsupported")
            # the on-screen timer starts at frame 1 = 0.0 s and floors to 0.1 s
            t = math.floor((ev[0]["f0"] / FPS) * 10 + 1e-9) / 10
            return Answer(f"{t:.1f} s", confidence="exact")

        # closest pair description
        if "flash closest together in time" in q and self.pair and self.pair_kind:
            return Answer(self.pair_kind, confidence="exact")

        # a flash that happens inside another shape's longest flash
        m = re.search(r"while the (.+?) is lit during its longest flash, which of these (?:shapes|lamps) lights up", q)
        if m and (target := self.name_in_question(m.group(1))):
            ev = self.flashes_of(target)
            if ev:
                longest = max(ev, key=lambda f: f["f1"] - f["f0"])
                for s in self.option_shapes(options):
                    if s is target:
                        continue
                    if any(g["f0"] < longest["f1"] and g["f1"] > longest["f0"] for g in self.flashes_of(s)):
                        return Answer(self.phrase(s, q), confidence="exact")

        # relay: the subject flashes shortly after its causal predecessor
        if "often flashes about one second after" in q and self.p.get("causal"):
            chain = self.p["causal"].get("chain") or []
            m = re.search(r"the (.+?) often flashes", q)
            if m and (target := self.find(m.group(1))) and target["id"] in chain:
                idx = chain.index(target["id"])
                if idx > 0:
                    return Answer(self.phrase(self.shapes[chain[idx - 1]], q), confidence="exact")

        # causal removal: if X never flashed, how many flashes would Y have?
        m = re.search(r"if the (.+?) had never flashed.*?how many times would the (.+?) flash", q)
        if m and self.p.get("causal"):
            src, dst = self.find(m.group(1)), self.find(m.group(2))
            if src and dst:
                removed = {f["id"] for f in self.flashes_of(src)}
                changed = True
                while changed:
                    changed = False
                    for f in self.flashes:
                        if f.get("cause") in removed and f["id"] not in removed:
                            removed.add(f["id"])
                            changed = True
                n = sum(1 for f in self.flashes_of(dst) if f["id"] not in removed)
                return Answer(str(n), confidence="exact")


        # fewest flashes
        if "flashes the fewest times" in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if self.flashes_of(s)]
            if pool:
                return Answer(self.phrase(min(pool, key=self.count), q), confidence="exact")

        # busiest stretch of the video
        if "in which stretch of the video do the most flashes start" in q and options:
            n_end = float(self.p.get("n_frames", 0)) / FPS
            best, best_n = None, -1
            for opt in options:
                text = str(opt).lower()
                m2 = re.search(r"from ([\d.]+) s to ([\d.]+) s", text)
                if m2:
                    lo, hi = float(m2.group(1)), float(m2.group(2))
                else:
                    m2 = re.search(r"from ([\d.]+) s to the end", text)
                    if not m2:
                        continue
                    lo, hi = float(m2.group(1)), n_end
                n = sum(1 for f in self.flashes if lo - 1e-9 <= f["f0"] / FPS < hi - 1e-9)
                if n > best_n:
                    best, best_n = str(opt), n
            if best is not None:
                return Answer(best, confidence="exact")

        # if no white flashes, who flashes most
        if "if none of the white flashes had happened" in q:
            pool = self.option_shapes(options) or list(self.shapes)
            if pool:
                return Answer(self.phrase(max(pool, key=lambda s: sum(1 for f in self.flashes_of(s) if not f.get("white"))), q),
                              confidence="exact")

        # grows and shrinks (pulses are not flashes)
        if "grows bigger and then shrinks back" in q and self.pulses:
            pool = self.option_shapes(options)
            cand = list({self.shapes[p["i"]]["id"]: self.shapes[p["i"]] for p in self.pulses}.values())
            if pool:
                cand = [s for s in cand if s in pool]
            if len(cand) == 1:
                return Answer(self.phrase(cand[0], q), confidence="exact")

        # steady rhythm, restricted to the options
        if "flashes at a steady rhythm" in q and "how many seconds" not in q:
            pool = self.option_shapes(options) or [s for s in self.shapes if str(s.get("rhythm")) == "steady"]
            steady = [s for s in pool if str(s.get("rhythm")) == "steady"]
            if len(steady) == 1:
                return Answer(self.phrase(steady[0], q), confidence="exact")

        return Answer(None, confidence="unsupported", note="template not handled")

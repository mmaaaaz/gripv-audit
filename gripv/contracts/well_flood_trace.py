"""Layer-1 contract for GRIP-V ``well_flood_trace`` (Well Flood).

The clip is a grid of R x C sections, each printed with a unique "depth
value" that is also its capacity in water units.  One or two entry taps pour
at a fixed rate; water rises linearly in a section and, once a section is
completely full, it passes its whole inflow on to its adjacent, non-walled
neighbours that have a strictly lower depth value (split equally).  Clock
answers truncate to one decimal place on screen.

Replay is event-driven and reproduces the stored per-clip ``track``
(start / full times for every cell) exactly on all 2,500 clips, which is what
makes the counterfactual templates (seal a section, remove a wall, redirect
the pour, scale the rate, repair the impossible event) replayable too.
"""
from __future__ import annotations

import re

from ..engine import Answer

# ---------------------------------------------------------------- flow engine

_HORIZON_PAD = 23.0 / 24.0   # the generator records events up to end + 23 frames


def _neighbours(i: int, R: int, C: int) -> list[int]:
    r, c = divmod(i, C)
    out = []
    if r > 0:
        out.append(i - C)
    if r < R - 1:
        out.append(i + C)
    if c > 0:
        out.append(i - 1)
    if c < C - 1:
        out.append(i + 1)
    return out


class _Clip:
    """One clip's configuration plus a memoised replay."""

    def __init__(self, params: dict):
        self.p = params
        self.R = int(params["R"])
        self.C = int(params["C"])
        self.depth = [int(d) for d in params["depth"]]
        self.walls = [tuple(w) for w in (params.get("walls") or [])]
        self.taps = params.get("taps") or []
        self.violation = params.get("violation")
        self.duration = float(params["duration"])
        self.n = len(self.depth)
        self.by_depth = {d: i for i, d in enumerate(self.depth)}
        self._cache: dict[tuple, tuple] = {}

    # -- configuration helpers -------------------------------------------
    def tap(self, colour: str | None = None) -> dict | None:
        if colour is None:
            return self.taps[0] if self.taps else None
        for t in self.taps:
            if str(t.get("colour", "")).lower() == colour.lower():
                return t
        return None

    def wall_pair(self, d1: int, d2: int):
        a, b = self.by_depth[d1], self.by_depth[d2]
        for w in self.walls:
            if set(w) == {a, b}:
                return (w[0], w[1])
        return (a, b)

    def colour_phrase(self, question: str) -> str | None:
        m = re.search(r"outlined in (red|purple)", question, re.I)
        return m.group(1).lower() if m else None

    # -- replay ----------------------------------------------------------
    def sim(self, **mods) -> tuple[list, list, list]:
        key = repr(sorted(((k, repr(v)) for k, v in mods.items()), key=lambda kv: kv[0]))
        if key in self._cache:
            return self._cache[key]
        out = _run(self, mods)
        self._cache[key] = out
        return out

    def reached(self, sim) -> int:
        start = sim[0]
        return sum(1 for t in start if t is not None and t <= self.duration + 1e-9)

    def full_at_end(self, sim) -> int:
        full = sim[1]
        return sum(1 for t in full if t is not None and t <= self.duration + 1e-9)


def _segments(clip: _Clip, rate_mult: float, tap_cells: dict | None,
              drop_colour: str | None, keep_start_rate: bool):
    out = []
    for k, tp in enumerate(clip.taps):
        if drop_colour is not None and str(tp.get("colour", "")).lower() == drop_colour:
            continue
        cell = tp["cell"] if not tap_cells or k not in tap_cells else tap_cells[k]
        raw = tp["segs"]
        if keep_start_rate:
            out.append((cell, (raw[0][0] + 0.5) / 24.0, None, raw[0][2] * rate_mult))
            continue
        for s in raw:
            t0 = (s[0] + 0.5) / 24.0
            t1 = (s[1] + 0.5) / 24.0 if s[1] is not None else None
            out.append((cell, t0, t1, s[2] * rate_mult))
    return out


def _run(clip: _Clip, mods: dict):
    n = clip.n
    R, C = clip.R, clip.C
    depth = clip.depth
    wallset = {frozenset(w) for w in clip.walls}
    if mods.get("remove_wall") is not None:
        wallset.discard(frozenset(mods["remove_wall"]))
    seal = mods.get("seal")

    vi = None if mods.get("repair_violation") else clip.violation
    recips: list[list[int]] = []
    for u in range(n):
        rs = [v for v in _neighbours(u, R, C)
              if frozenset((u, v)) not in wallset and depth[v] < depth[u]
              and v != seal]
        if vi and vi.get("kind") == "uphill_flow" and vi["src"] == u:
            if vi["dst"] not in rs and vi["dst"] != seal:
                rs.append(vi["dst"])
        recips.append(rs)
    if vi and vi.get("kind") == "uphill_flow":
        src, dst = vi["src"], vi["dst"]
        if src in recips[dst]:
            recips[dst] = [v for v in recips[dst] if v != src]

    tap_events = []
    for cell, t0, t1, rate in _segments(
        clip,
        float(mods.get("rate_mult", 1.0)),
        mods.get("tap_cells"),
        mods.get("drop_colour"),
        bool(mods.get("keep_start_rate")),
    ):
        if seal is not None and cell == seal:
            continue
        tap_events.append((t0, 1, cell, rate))
        if t1 is not None:
            tap_events.append((t1, -1, cell, rate))
    if vi and vi.get("kind") == "premature_fill" and vi["cell"] != seal:
        tap_events.append(((vi["frame"] + 0.5) / 24.0, 1, vi["cell"], vi["rate"]))
    tap_events.sort()

    t_end = clip.duration + _HORIZON_PAD

    inflow = [0.0] * n
    out_rate = [0.0] * n
    level = [0.0] * n
    start: list[float | None] = [None] * n
    full: list[float | None] = [None] * n
    parent: list[int | None] = [None] * n
    is_full = [False] * n
    t = 0.0

    def distribute(u: int) -> None:
        stack = [u]
        guard = 0
        while stack:
            guard += 1
            if guard > 8 * n + 16:
                break
            w = stack.pop()
            target = inflow[w] if is_full[w] else 0.0
            delta = target - out_rate[w]
            if delta == 0.0:
                continue
            out_rate[w] = target
            rs = recips[w]
            if not rs:
                continue
            share = delta / len(rs)
            for v in rs:
                if inflow[v] <= 0.0 and start[v] is None and share > 0.0:
                    start[v] = t
                    parent[v] = w
                inflow[v] += share
                if is_full[v]:
                    stack.append(v)

    pending = 0
    while True:
        best_t, best_v = None, -1
        for v in range(n):
            if is_full[v] or inflow[v] <= 0.0 or depth[v] <= 0:
                continue
            ft = t + max(0.0, depth[v] - level[v]) / inflow[v]
            if best_t is None or ft < best_t:
                best_t, best_v = ft, v
        next_tap = tap_events[pending][0] if pending < len(tap_events) else None
        if best_t is None and next_tap is None:
            break
        use_fill = best_t is not None and (next_tap is None or best_t <= next_tap)
        t_new = best_t if use_fill else next_tap
        if t_new > t_end:
            break
        dt = t_new - t
        if dt > 0:
            for v in range(n):
                if not is_full[v] and inflow[v] > 0.0:
                    level[v] += inflow[v] * dt
            t = t_new
        if use_fill:
            v = best_v
            level[v] = depth[v]
            is_full[v] = True
            full[v] = t
            distribute(v)
        else:
            while pending < len(tap_events) and tap_events[pending][0] == t:
                _, sign, cell, rate = tap_events[pending]
                pending += 1
                if sign > 0:
                    if start[cell] is None:
                        start[cell] = t
                    inflow[cell] += rate
                else:
                    inflow[cell] -= rate
                distribute(cell)
    return start, full, parent


# ---------------------------------------------------------------- handlers

_D = re.compile(r"depth value (\d+)")
_D2 = re.compile(r"depth value (\d+) and the section with depth value (\d+)")
_DLIST = re.compile(r"depth values ([\d, ]+ and \d+)")
_CLOCK = re.compile(r"clock shows ([\d.]+) s")
_ORD = re.compile(r"reach (second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b")
_ORDINAL = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def _sec(i: int) -> str:
    return f"the section with depth value {i}"


def _opt_depth(opt: str) -> int | None:
    m = re.search(r"depth value (\d+)", opt)
    return int(m.group(1)) if m else None


class WellFloodTrace:
    def __init__(self, params: dict):
        self.p = params
        self.clip = _Clip(params)

    # -- small helpers ---------------------------------------------------
    def _base(self):
        return self.clip.sim()

    def _t(self, x):
        return None if x is None else round(float(x) + 1e-9, 1)

    @staticmethod
    def _dur(a: float, b: float) -> float:
        """Elapsed seconds as displayed: each clock is rounded to 1 dp first."""
        return round(round(b, 1) - round(a, 1), 1)

    def _first_source_dir(self, target: int) -> str | None:
        """Direction word for the neighbour that first fed `target`."""
        start, full, parent = self._base()
        p = parent[target]
        if p is None:
            # fall back: earliest-full neighbour that can legally feed target
            best, best_t = None, None
            for j in _neighbours(target, self.clip.R, self.clip.C):
                if full[j] is None:
                    continue
                if self.clip.depth[j] < self.clip.depth[target]:
                    continue
                if best_t is None or full[j] < best_t:
                    best, best_t = j, full[j]
            p = best
            if p is None:
                return None
        return self._direction(p, target)

    def _direction(self, src: int, dst: int) -> str:
        C = self.clip.C
        if src == dst + 1:
            return "right"
        if src == dst - 1:
            return "left"
        if src == dst - C:
            return "above"
        if src == dst + C:
            return "below"
        return "?"

    def _reached_count(self, sim) -> int:
        return self.clip.reached(sim)

    def _count_at(self, times, when: float) -> int:
        return sum(1 for t in times if t is not None and t <= when + 1e-9)

    def _ordered_cells(self, sim):
        start = sim[0]
        cells = [i for i in range(self.clip.n) if start[i] is not None]
        cells.sort(key=lambda i: (start[i], i))
        return cells

    # -- answer ----------------------------------------------------------
    def answer(self, question: str, options: list[str]) -> Answer:
        q = question.lower().strip()
        clip = self.clip
        base = self._base()
        start, full, parent = base
        dur = clip.duration

        # ---- "what went wrong" -----------------------------------------
        if q.startswith("one moment in a video like this"):
            v = clip.violation
            if not v:
                return Answer("nothing: every fill was physically correct")
            if v.get("kind") == "uphill_flow":
                return Answer("water flowed into a section from a neighbour with a LOWER depth value")
            return Answer("a section started filling before any full higher neighbour could feed it")

        # ---- which section breaks the rule ------------------------------
        if q.startswith("water can only enter a section"):
            v = clip.violation
            if not v:
                return Answer("none of them: every fill was physically correct")
            cell = v["dst"] if v.get("kind") == "uphill_flow" else v["cell"]
            return Answer(_sec(clip.depth[cell]))

        # ---- walls holding back water ----------------------------------
        if q.startswith("at the end of this video, how many of the thick dark walls"):
            cnt = 0
            for a, b in clip.walls:
                fa = full[a] is not None and full[a] <= dur + 1e-9
                fb = full[b] is not None and full[b] <= dur + 1e-9
                if fa and clip.depth[b] < clip.depth[a]:
                    cnt += 1
                elif fb and clip.depth[a] < clip.depth[b]:
                    cnt += 1
            return Answer(str(cnt))

        # ---- pour-change timing counts ---------------------------------
        if q.startswith("how many sections first got water after the pour stopped"):
            tp = clip.tap()
            stop = None
            if tp:
                for s in tp["segs"]:
                    if s[1] is not None:
                        stop = (s[1] + 0.5) / 24.0
            if stop is None:
                return Answer("unsupported", confidence="unsupported")
            return Answer(str(sum(1 for t in start
                                  if t is not None and stop + 1e-9 < t <= dur + 1e-9)))

        if q.startswith("how many sections first got water after the pour started pouring twice"):
            tp = clip.tap()
            change = None
            if tp and len(tp["segs"]) > 1:
                change = (tp["segs"][1][0] + 0.5) / 24.0
            if change is None:
                return Answer("unsupported", confidence="unsupported")
            return Answer(str(sum(1 for t in start
                                  if t is not None and change + 1e-9 < t <= dur + 1e-9)))

        # ---- clock snapshots -------------------------------------------
        m = _CLOCK.search(question)
        if m:
            when = float(m.group(1))
            if "completely full" in q:
                return Answer(str(self._count_at(full, when)))
            return Answer(str(self._count_at(start, when)))

        # ---- order of three sections -----------------------------------
        if q.startswith("in what order did water reach"):
            m = _DLIST.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            cells = [clip.by_depth[x] for x in nums if x in clip.by_depth]
            cells.sort(key=lambda i: (start[i] if start[i] is not None else 9e9, i))
            return Answer(", then ".join(str(clip.depth[i]) for i in cells))

        # ---- ordinal arrival -------------------------------------------
        m = _ORD.search(question)
        if m and "counting the entry section as the first" in q:
            k = _ORDINAL[m.group(1)]
            cells = self._ordered_cells(base)
            if len(cells) < k:
                return Answer(None, confidence="unsupported")
            return Answer(_sec(clip.depth[cells[k - 1]]))

        # ---- first section after the entry -----------------------------
        if "what is the depth value of the section that water reached first after the entry section" in q:
            colour = clip.colour_phrase(question)
            tp = clip.tap(colour)
            if not tp:
                return Answer(None, confidence="unsupported")
            entry = tp["cell"]
            tap_cells = {t["cell"] for t in clip.taps}
            # The entry's own pour (plus any impossible premature fill) decides
            # this; the other entry's section is not "after" this one.
            cands = [i for i in range(clip.n)
                     if start[i] is not None and i != entry and i not in tap_cells
                     and start[i] > start[entry] + 1e-9]
            if not cands:
                return Answer(None, confidence="unsupported")
            first = min(cands, key=lambda i: (start[i], i))
            return Answer(str(clip.depth[first]))

        # ---- how many already had water when the pour began ------------
        if q.startswith("how many sections already had water in them when water started pouring"):
            colour = clip.colour_phrase(question)
            tp = clip.tap(colour)
            if not tp:
                return Answer(None, confidence="unsupported")
            t0 = (tp["segs"][0][0] + 0.5) / 24.0
            return Answer(str(sum(1 for i, t in enumerate(start)
                                  if t is not None and t < t0 - 1e-9 and i != tp["cell"])))

        # ---- entry depth ------------------------------------------------
        if q.startswith("what is the depth value of the entry section"):
            colour = clip.colour_phrase(question)
            tp = clip.tap(colour)
            if not tp:
                return Answer(None, confidence="unsupported")
            return Answer(str(clip.depth[tp["cell"]]))

        # ---- first inflow direction -------------------------------------
        if q.startswith("from which neighbouring section did water first flow"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            target = clip.by_depth[int(m.group(1))]
            d = self._first_source_dir(target)
            if d is None or d == "?":
                return Answer(None, confidence="unsupported")
            return Answer(f"from the section to its {d}" if d in ("left", "right")
                          else f"from the section {d} it")

        # ---- counterfactuals --------------------------------------------
        mods = self._counterfactual(q, question)
        if mods is not None:
            sim = clip.sim(**mods) if mods else base
            if "how many distinct sections would water have reached" in q:
                return Answer(str(self._reached_count(sim)))
            if "which one would water have reached" in q:
                hit = [o for o in options
                       if (lambda d: d is not None and d in clip.by_depth
                           and sim[0][clip.by_depth[d]] is not None
                           and sim[0][clip.by_depth[d]] <= dur + 1e-9)(_opt_depth(o))]
                if len(hit) == 1:
                    return Answer(hit[0])
                return Answer(None, confidence="unsupported")
            if "would water have started to rise" in q:
                m = _D.search(question)
                if not m:
                    return Answer(None, confidence="unsupported")
                cell = clip.by_depth[int(m.group(1))]
                a = start[cell] if start[cell] is not None and start[cell] <= dur + 1e-9 else None
                b = sim[0][cell] if sim[0][cell] is not None and sim[0][cell] <= dur + 1e-9 else None
                if a is None and b is None:
                    return Answer("not at all before the end of the video")
                if a is None:
                    return Answer("earlier than shown")
                if b is None:
                    return Answer("not at all before the end of the video")
                if abs(b - a) < 0.05:
                    return Answer("at the same time as shown")
                return Answer("earlier than shown" if b < a else "later than shown")

        # ---- reach / full membership ------------------------------------
        if q.startswith("which of these sections had water reach it by the end"):
            for o in options:
                d = _opt_depth(o)
                if d is None or d not in clip.by_depth:
                    continue
                t = start[clip.by_depth[d]]
                if t is not None and t <= dur + 1e-9:
                    return Answer(o)
            return Answer(None, confidence="unsupported")

        if q.startswith("which of these sections never had water reach it"):
            for o in options:
                d = _opt_depth(o)
                if d is None or d not in clip.by_depth:
                    continue
                t = start[clip.by_depth[d]]
                if t is None or t > dur + 1e-9:
                    return Answer(o)
            return Answer(None, confidence="unsupported")

        if q.startswith("of these sections, which one did water reach first"):
            return self._extreme(options, start, dur, first=True)

        if q.startswith("of these sections, which one did water reach last"):
            return self._extreme(options, start, dur, first=False)

        if q.startswith("of these sections, which one became completely full"):
            return self._extreme(options, full, dur, first=True)

        if q.startswith("when water began to rise in the section with depth value"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            cell = clip.by_depth[int(m.group(1))]
            t0 = start[cell]
            if t0 is None:
                return Answer(None, confidence="unsupported")
            best = None
            for o in options:
                d = _opt_depth(o)
                if d is None or d not in clip.by_depth:
                    continue
                j = clip.by_depth[d]
                if full[j] is not None and full[j] <= t0 + 1e-9:
                    if best is None or full[j] > full[clip.by_depth[_opt_depth(best)]]:
                        best = o
            if best is None:
                return Answer(None, confidence="unsupported")
            return Answer(best)

        # ---- counting ----------------------------------------------------
        if q.startswith("how many distinct sections had water reach"):
            return Answer(str(self._reached_count(base)))

        if q.startswith("how many sections are completely full"):
            return Answer(str(self.clip.full_at_end(base)))

        if q.startswith("how many sections had water in them just before"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            cell = clip.by_depth[int(m.group(1))]
            t_full = full[cell]
            if t_full is None:
                return Answer(None, confidence="unsupported")
            cnt = sum(1 for i, t in enumerate(start)
                      if t is not None and t < t_full - 1e-9)
            return Answer(str(cnt))

        # ---- timing ------------------------------------------------------
        if q.startswith("at about what clock time does water begin to rise"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            t = start[clip.by_depth[int(m.group(1))]]
            if t is None or t > dur + 1e-9:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._t(t)} s")

        if q.startswith("at about what clock time does the section"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            t = full[clip.by_depth[int(m.group(1))]]
            if t is None or t > dur + 1e-9:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._t(t)} s")

        if q.startswith("about how many seconds does the section"):
            m = _D.search(question)
            if not m:
                return Answer(None, confidence="unsupported")
            cell = clip.by_depth[int(m.group(1))]
            if start[cell] is None or full[cell] is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._dur(start[cell], full[cell])} s")

        if q.startswith("about how many seconds after water reached"):
            nums = [int(x) for x in re.findall(r"depth value (\d+)", question)]
            if len(nums) < 2:
                return Answer(None, confidence="unsupported")
            a, b = clip.by_depth[nums[0]], clip.by_depth[nums[1]]
            if start[a] is None or start[b] is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._dur(start[a], start[b])} s")

        if q.startswith("about how many seconds after the section"):
            nums = [int(x) for x in re.findall(r"depth value (\d+)", question)]
            if len(nums) < 2:
                return Answer(None, confidence="unsupported")
            a, b = clip.by_depth[nums[0]], clip.by_depth[nums[1]]
            if full[a] is None or full[b] is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self._dur(full[a], full[b])} s")

        return Answer(None, confidence="unsupported")

    # -- helpers ---------------------------------------------------------
    def _extreme(self, options, times, dur, first: bool) -> Answer:
        best, best_t = None, None
        for o in options:
            d = _opt_depth(o)
            if d is None or d not in self.clip.by_depth:
                continue
            t = times[self.clip.by_depth[d]]
            if t is None or t > dur + 1e-9:
                continue
            if best_t is None or (t < best_t if first else t > best_t):
                best, best_t = o, t
        if best is None:
            return Answer(None, confidence="unsupported")
        return Answer(best)

    def _counterfactual(self, q: str, question: str) -> dict | None:
        """Map a counterfactual template to engine modifiers (None = no match)."""
        clip = self.clip
        m = _D.search(question)
        colour = clip.colour_phrase(question)

        if q.startswith("if the section with depth value") and "had been sealed" in q:
            if not m:
                return {}
            return {"seal": clip.by_depth[int(m.group(1))]}

        if q.startswith("if the thick wall between"):
            m2 = _D2.search(question)
            if not m2:
                return {}
            return {"remove_wall": clip.wall_pair(int(m2.group(1)), int(m2.group(2)))}

        if q.startswith("if the pour had gone into the section"):
            if not m:
                return {}
            cells = self._tap_cells_for(None, int(m.group(1)))
            if cells is None:
                return {}
            return {"tap_cells": cells}

        if q.startswith("if the pour into the entry section"):
            if not m or not colour:
                return {}
            cells = self._tap_cells_for(colour, int(m.group(1)))
            if cells is None:
                return {}
            return {"tap_cells": cells}

        if q.startswith("if no water had ever been poured"):
            if not colour:
                return {}
            return {"drop_colour": colour}

        if q.startswith("if the one physically impossible event"):
            if "had instead behaved correctly" in q:
                return {"repair_violation": True}
            if "had not happened" in q:
                return {"repair_violation": True}
            return {"repair_violation": True}

        if q.startswith("if the pour rate of both entries"):
            if "double" in q:
                return {"rate_mult": 2.0}
            if "half" in q:
                return {"rate_mult": 0.5}
            return {}

        if q.startswith("if the pour rate had been"):
            if "double" in q:
                return {"rate_mult": 2.0}
            if "half" in q:
                return {"rate_mult": 0.5}
            return {}

        if q.startswith("if the pour had kept going at its starting rate"):
            return {"keep_start_rate": True}

        if q.startswith("water never reached these sections"):
            if "both entries" in q:
                return {"rate_mult": 2.0}
            return {"rate_mult": 2.0}

        return None

    def _tap_cells_for(self, colour: str | None, depth_value: int):
        clip = self.clip
        target = clip.by_depth[depth_value]
        cells = {}
        if colour is None:
            if not clip.taps:
                return None
            cells[0] = target
            return cells
        for k, tp in enumerate(clip.taps):
            if str(tp.get("colour", "")).lower() == colour:
                cells[k] = target
        return cells or None

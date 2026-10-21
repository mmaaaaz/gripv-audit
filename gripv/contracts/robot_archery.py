"""Deterministic replay for `robot_archery`.

Params (all times in seconds unless stated):
  scenario: classic | big_field | stay_on | two_lives
  frames:   clip length in frames at 24 fps
  robots:   [{colour, shape, pattern}]  (5-7 robots)
  order:    starting queue, robot ids
  lives:    starting pips per robot (1, or 2 in `two_lives`)
  stay_on:  a new-record shot keeps the turn (True) instead of going to the back
  draws:    distance of every arrow from the centre, in ring units (1-based rings)
  land:     landing time of every arrow, in turn order
  turns:    the generated timeline (shooter / d / record / prev_record / prev_holder /
            lives_after / out)
  eliminated / winner: final outcome

Model (verified against `turns`, `eliminated` and `winner` on all 2,500 clips):
  queue = order; record = inf
  for each arrow k (distance draws[k]):
      shooter = queue.pop(0)
      if draws[k] < record:  record = draws[k]; holder = shooter;
                             shooter returns to the FRONT if stay_on else the BACK
      else:                  shooter loses one pip; at zero pips it is out,
                             otherwise it goes to the back
      the game ends when one robot is left
  ring = ceil(d)  (ring 1 = centre ring, ring 10 = outer edge)
  counterfactual families keep the arrow sequence (draws[k] is the k-th arrow) and
  re-simulate the queue under the modified rule.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24.0

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}


def simulate(order, draws, lives, stay_on, n_robots):
    """Replay the queue. Returns (turns, eliminated, winner)."""
    queue = list(order)
    pips = [lives] * n_robots
    record = float("inf")
    holder: int | None = None
    turns: list[dict] = []
    eliminated: list[int] = []
    for k, d in enumerate(draws):
        if len(queue) <= 1:
            break
        shooter = queue.pop(0)
        is_record = d < record - 1e-12
        prev_record, prev_holder = record, holder
        if is_record:
            record, holder = d, shooter
            if stay_on:
                queue.insert(0, shooter)
            else:
                queue.append(shooter)
            out = False
        else:
            pips[shooter] -= 1
            out = pips[shooter] <= 0
            if out:
                eliminated.append(shooter)
            else:
                queue.append(shooter)
        turns.append(
            {
                "turn": k,
                "shooter": shooter,
                "d": d,
                "record": is_record,
                "prev_record": prev_record,
                "prev_holder": prev_holder,
                "out": out,
                "pips": pips[shooter],
            }
        )
    return turns, eliminated, (queue[0] if queue else None)


class RobotArchery:
    def __init__(self, params: dict):
        self.p = params
        self.robots = params["robots"]
        self.order = params["order"]
        self.draws = params["draws"]
        self.lives = params.get("lives", 1)
        self.stay = bool(params.get("stay_on"))
        self.land = params["land"]
        self.frames = params.get("frames", 0)
        self.n = len(self.robots)
        self.turns, self.elim, self.winner = simulate(
            self.order, self.draws, self.lives, self.stay, self.n
        )

    # ---------- helpers ----------
    def colour(self, i: int) -> str:
        return self.robots[i]["colour"]

    def name(self, i: int) -> str:
        return f"the {self.robots[i]['colour']} robot"

    def idx(self, colour: str) -> int | None:
        for i, r in enumerate(self.robots):
            if r["colour"] == colour.lower():
                return i
        return None

    def shots(self, i: int) -> list[dict]:
        return [t for t in self.turns if t["shooter"] == i]

    def nth(self, i: int, ordinal: str) -> dict | None:
        ss = self.shots(i)
        if ordinal == "only":
            return ss[0] if len(ss) == 1 else None
        n = ORDINALS.get(ordinal)
        if n is None or len(ss) < n:
            return None
        return ss[n - 1]

    def land_of(self, turn: int) -> float:
        return self.land[turn]

    def elim_turn(self, i: int) -> int | None:
        for t in self.turns:
            if t["shooter"] == i and t["out"]:
                return t["turn"]
        return None

    def holder_after(self, turn: int) -> int | None:
        holder = None
        for t in self.turns:
            if t["turn"] > turn:
                break
            if t["record"]:
                holder = t["shooter"]
        return holder

    def pips_at(self, when: float) -> list[int]:
        pips = [self.lives] * self.n
        for t in self.turns:
            if self.land_of(t["turn"]) <= when + 1e-9:
                pips[t["shooter"]] = t["pips"]
        return pips

    def record_at(self, when: float) -> dict | None:
        rec = None
        for t in self.turns:
            if self.land_of(t["turn"]) <= when + 1e-9 and t["record"]:
                rec = t
            elif self.land_of(t["turn"]) > when + 1e-9:
                break
        return rec

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        q = str(question).replace("\u2019", "'").strip().lower()
        options = [str(o) for o in (options or [])]

        # ----- counts of arrows on the target -----
        if re.search(r"how many arrows are on the target at the end of the clip", q):
            end = self.frames / FPS
            return Answer(str(sum(1 for t in self.land if t <= end + 1e-9)))

        m = re.search(r"how many arrows are on the target when the clock shows ([\d.]+) s", q)
        if m:
            when = float(m.group(1))
            return Answer(str(sum(1 for t in self.land if t <= when + 1e-9)))

        # ----- outcome / order -----
        if re.search(r"which robot is the last one left in the game", q):
            return Answer(self.name(self.winner) if self.winner is not None else "none of them")

        if re.search(r"which robot shoots the first arrow", q):
            return Answer(self.name(self.turns[0]["shooter"]) if self.turns else "none of them")

        m = re.search(r"in what order are the robots eliminated", q)
        if m:
            return Answer(" \u2192 ".join(self.colour(i) for i in self.elim))

        m = re.search(r"which robot is eliminated right after the (\w+) robot", q)
        if m:
            i = self.idx(m.group(1))
            if i is None or i not in self.elim:
                return Answer(None, confidence="unsupported")
            pos = self.elim.index(i)
            if pos + 1 >= len(self.elim):
                return Answer(None, confidence="unsupported")
            return Answer(self.name(self.elim[pos + 1]))

        m = re.search(r"when the (\w+) robot is eliminated, which robot holds the record", q)
        if m:
            i = self.idx(m.group(1))
            k = self.elim_turn(i) if i is not None else None
            if k is None:
                return Answer(None, confidence="unsupported")
            holder = self.holder_after(k)
            return Answer(self.name(holder) if holder is not None else "no robot")

        # ----- per-robot shot / record counts -----
        m = re.search(r"how many arrows does the (\w+) robot shoot", q)
        if m:
            i = self.idx(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.shots(i))))

        m = re.search(r"how many new records does the (\w+) robot set", q)
        if m:
            i = self.idx(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(sum(1 for t in self.shots(i) if t["record"])))

        if re.search(r"which robot sets the most new records", q):
            counts = {i: sum(1 for t in self.shots(i) if t["record"]) for i in range(self.n)}
            best = max(counts.values()) if counts else 0
            if best == 0:
                return Answer(None, confidence="unsupported")
            winners = [i for i, c in counts.items() if c == best]
            if len(winners) != 1:
                return Answer(None, confidence="unsupported", note="tied record counts")
            return Answer(self.name(winners[0]))

        if re.search(r"how many of the shots set a new record", q):
            return Answer(str(sum(1 for t in self.turns if t["record"])))

        # ----- elimination timing / counts -----
        m = re.search(
            r"how many arrows have been shot when the (\w+) robot is eliminated", q
        )
        if m:
            i = self.idx(m.group(1))
            k = self.elim_turn(i) if i is not None else None
            if k is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(k + 1))

        m = re.search(r"at about what time on the clock is the (\w+) robot eliminated", q)
        if m:
            i = self.idx(m.group(1))
            k = self.elim_turn(i) if i is not None else None
            if k is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.land_of(k):.1f} s")

        if re.search(r"at about what time on the clock is the last new record set", q):
            recs = [t for t in self.turns if t["record"]]
            if not recs:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.land_of(recs[-1]['turn']):.1f} s")

        # ----- state at a clock time -----
        m = re.search(r"when the clock shows ([\d.]+) s, how many robots still have both of their pips", q)
        if m:
            pips = self.pips_at(float(m.group(1)))
            return Answer(str(sum(1 for x in pips if x >= self.lives)))

        m = re.search(
            r"when the clock shows ([\d.]+) s, in which ring is the record shot so far", q
        )
        if m:
            rec = self.record_at(float(m.group(1)))
            if rec is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(math.ceil(rec["d"] - 1e-9)))

        # ----- per-shot lookups (robot's Nth shot) -----
        m = re.search(r"the (\w+) robot's (\w+) shot (?:set|was) (?:a new record|not a new record)", q)
        ordinal = m.group(2) if m else None
        who = m.group(1) if m else None

        if re.search(r"whose record did it beat", q):
            i = self.idx(who) if who else None
            shot = self.nth(i, ordinal) if i is not None and ordinal else None
            if shot is None:
                return Answer(None, confidence="unsupported")
            if shot["prev_holder"] is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(shot["prev_holder"]))

        if re.search(r"by about how many rings did the record move closer", q):
            i = self.idx(who) if who else None
            shot = self.nth(i, ordinal) if i is not None else None
            if shot is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(int(round(shot["prev_record"] - shot["d"]))))

        if re.search(r"in which ring did the (\w+) robot's (\w+) shot land", q):
            m2 = re.search(r"in which ring did the (\w+) robot's (\w+) shot land", q)
            i = self.idx(m2.group(1))
            shot = self.nth(i, m2.group(2)) if i is not None else None
            if shot is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(math.ceil(shot["d"] - 1e-9)))

        m = re.search(r"which robot shoots right after the (\w+) robot's (\w+) shot", q)
        if m:
            i = self.idx(m.group(1))
            shot = self.nth(i, m.group(2)) if i is not None else None
            if shot is None or shot["turn"] + 1 >= len(self.turns):
                return Answer(None, confidence="unsupported")
            return Answer(self.name(self.turns[shot["turn"] + 1]["shooter"]))

        # ----- closest of the listed shots -----
        if re.search(r"which of these shots landed closest to the centre", q):
            best = None
            for o in options:
                m2 = re.search(r"the (\w+) robot's (\w+) shot", o.lower())
                if not m2:
                    continue
                i = self.idx(m2.group(1))
                shot = self.nth(i, m2.group(2)) if i is not None else None
                if shot is None:
                    continue
                if best is None or shot["d"] < best[0] - 1e-12:
                    best = (shot["d"], o)
            if best is None:
                return Answer(None, confidence="unsupported")
            return Answer(best[1])

        # ----- counterfactuals -----
        m = re.search(
            r"suppose the (\w+) robot and the (\w+) robot had swapped places in the starting queue", q
        )
        if m:
            a, b = self.idx(m.group(1)), self.idx(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            order = list(self.order)
            ia, ib = order.index(a), order.index(b)
            order[ia], order[ib] = order[ib], order[ia]
            _, elim, win = simulate(order, self.draws, self.lives, self.stay, self.n)
            return self._outcome(q, win, elim)

        if re.search(r"suppose a robot that sets a new record stayed on the spot and shot again", q):
            _, elim, win = simulate(self.order, self.draws, self.lives, True, self.n)
            return self._outcome(q, win, elim)

        if re.search(r"suppose a robot that sets a new record went to the back of the queue", q):
            _, elim, win = simulate(self.order, self.draws, self.lives, False, self.n)
            return self._outcome(q, win, elim)

        if re.search(r"suppose every robot were eliminated by its first shot that is not a new record", q):
            _, elim, win = simulate(self.order, self.draws, 1, self.stay, self.n)
            return self._outcome(q, win, elim)

        m = re.search(
            r"the (\w+) robot's (\w+) shot (?:set|was) (?:a new record|not a new record)\. "
            r"suppose it had landed in ring (\d+) instead",
            q,
        )
        if m:
            i = self.idx(m.group(1))
            shot = self.nth(i, m.group(2)) if i is not None else None
            if shot is None:
                return Answer(None, confidence="unsupported")
            ring = int(m.group(3))
            draws = list(self.draws)
            draws[shot["turn"]] = ring - 0.5  # mid-ring keeps ceil(d) == ring
            _, elim, win = simulate(self.order, draws, self.lives, self.stay, self.n)
            if re.search(r"how many arrows would be shot in the whole game", q):
                return Answer(str(len(simulate(self.order, draws, self.lives, self.stay, self.n)[0])))
            return self._outcome(q, win, elim)

        return Answer(None, confidence="unsupported", note="no handler for this template")

    def _outcome(self, q: str, win: int | None, elim: list[int]) -> Answer:
        if re.search(r"which robot would be eliminated first", q):
            if not elim:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(elim[0]))
        if re.search(r"which robot would win", q):
            if win is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(win))
        return Answer(None, confidence="unsupported", note="counterfactual outcome unknown")

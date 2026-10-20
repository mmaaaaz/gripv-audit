"""Deterministic replay for `robot_long_jump`.

Params:
  foul: foul line in tenths of a ruler unit (10 = 1.0)
  timing: {hop_f, rest_f, jump_f, tip_f, gap_f}; one beat = hop_f + rest_f
  robots: [{id, colour, shape, pattern, size, lane, tau, start,
            hops: [1|3, ...], fouled, jump, order,
            hop_frames: [...], jump_frame, end_frame}]

Model (verified on clip 000000):
  take-off position = start + sum(hops)          (units of 0.1)
  jump length       = jump                        (units of 0.1)
  score             = take-off + jump  (0 if fouled)
  take-off time     = jump_frame / 24
  landing time      = end_frame / 24
"""
from __future__ import annotations

import re

from ..engine import Answer

FPS = 24.0


class RobotLongJump:
    def __init__(self, params: dict):
        self.p = params
        self.robots = params.get("robots", [])
        self.foul_line = params.get("foul", 10) / 10.0
        self.beat = (params.get("timing", {}).get("hop_f", 6)
                     + params.get("timing", {}).get("rest_f", 3))
        self.n_frames = params.get("n_frames", 0)

    # ---------- helpers ----------
    def find(self, desc: str) -> int | None:
        tokens = set(re.findall(r"[a-z]+", " " + desc.lower() + " "))
        for r in self.robots:
            if r["colour"] in tokens:
                return r["id"]
        return None

    def name(self, i: int) -> str:
        return f"the {self.robots[i]['colour']} robot"

    def takeoff(self, r: dict) -> float:
        return r.get("start", 0) / 10.0 + sum(r.get("hops", [])) / 10.0

    def jump_len(self, r: dict) -> float:
        return (r.get("jump") or 0) / 10.0

    def fouled(self, r: dict) -> bool:
        return bool(r.get("fouled")) or self.takeoff(r) > self.foul_line + 1e-9

    def score(self, r: dict, fouled: dict[int, bool] | None = None) -> float:
        if fouled is not None and fouled.get(r["id"]):
            return 0.0
        if fouled is None and self.fouled(r):
            return 0.0
        return round(self.takeoff(r) + self.jump_len(r), 3)

    def landing(self, r: dict) -> int:
        return r.get("end_frame", 0)

    def takeoff_time(self, r: dict) -> int:
        return r.get("jump_frame", 0)

    def landed(self) -> list[dict]:
        # fouled robots have no jump/landing (jump=None); they never land in the pit
        return sorted(
            [r for r in self.robots
             if r.get("jump") is not None and self.landing(r) <= self.n_frames],
            key=self.landing,
        )

    def best_after(self, frame: int, fouled: dict[int, bool] | None = None) -> dict | None:
        done = [r for r in self.robots if r.get("jump") is not None and self.landing(r) <= frame]
        if not done:
            return None
        scored = [r for r in done if self.score(r, fouled) > 0]
        if not scored:
            return None
        return max(scored, key=lambda r: self.score(r, fouled))

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if re.search(r"how many robots foul", q):
            return Answer(str(sum(1 for r in self.robots if self.fouled(r))))
        if re.search(r"which robot fouls", q):
            hit = [r for r in self.robots if self.fouled(r)]
            return Answer(self.name(hit[0]["id"]) if hit else "none of them")

        m = re.search(r"which robot lands in the sand pit first: the (\w+) robot or the (\w+) robot", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            ta, tb = self.landing(self.robots[a]), self.landing(self.robots[b])
            if ta == tb:
                return Answer("they land at the same moment")
            return Answer(self.name(a if ta < tb else b))

        m = re.search(r"which robot is the (first|last) to land", q)
        if m:
            seq = self.landed()
            if not seq:
                return Answer(None, confidence="unsupported")
            r = seq[0] if m.group(1) == "first" else seq[-1]
            return Answer(self.name(r["id"]))

        m = re.search(r"which robot is the (first|last) to take off", q)
        if m:
            seq = sorted([r for r in self.robots if r.get("jump") is not None],
                         key=self.takeoff_time)
            if not seq:
                return Answer(None, confidence="unsupported")
            r = seq[0] if m.group(1) == "first" else seq[-1]
            return Answer(self.name(r["id"]))

        m = re.search(r"how many more hops does the (\w+) robot make than the (\w+) robot", q)
        if m:
            a, b = self.find(m.group(1)), self.find(m.group(2))
            if a is None or b is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(len(self.robots[a]["hops"]) - len(self.robots[b]["hops"])))

        m = re.search(r"what does the clock read when the (\w+) robot lands", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.landing(self.robots[i]) / FPS:.1f} s")

        m = re.search(r"what does the clock read at the moment the (\w+) robot takes off", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.takeoff_time(self.robots[i]) / FPS:.1f} s")

        m = re.search(r"how long is the long jump of the (\w+) robot", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.jump_len(self.robots[i]):.1f}")

        m = re.search(r"which one has the best score right after the (\w+) robot lands", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            best = self.best_after(self.landing(self.robots[i]))
            return Answer(self.name(best["id"]) if best else "no robot has a score yet")

        if re.search(r"by how much does the winner's score beat the second-best score", q):
            scored = sorted([r for r in self.robots if self.score(r) > 0], key=self.score, reverse=True)
            if len(scored) < 2:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.score(scored[0]) - self.score(scored[1]):.1f}")

        m = re.search(r"if the (\w+) robot had fouled instead of jumping, how many times would the leader have changed", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            fouled = {self.robots[i]["id"]: True}
            changes, leader = 0, None
            for r in self.landed():
                s = self.score(r, fouled)
                if s > 0 and (leader is None or s > leader):
                    if leader is not None:
                        changes += 1
                    leader = s
            return Answer(str(changes))

        m = re.search(r"if the (\w+) robot had fouled instead of jumping, which robot would have had the best score .* right after the (\w+) robot lands", q)
        if m:
            i, j = self.find(m.group(1)), self.find(m.group(2))
            if i is None or j is None:
                return Answer(None, confidence="unsupported")
            fouled = {self.robots[i]["id"]: True}
            best = self.best_after(self.landing(self.robots[j]), fouled)
            return Answer(self.name(best["id"]) if best else "no robot would have a score yet")

        m = re.search(r"how many other robots have already landed in the sand pit at the moment the (\w+) robot takes off", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            t = self.takeoff_time(self.robots[i])
            return Answer(str(sum(1 for r in self.robots if r.get("jump") is not None and self.landing(r) < t and r["id"] != i)))

        m = re.search(r"suppose the foul line had been at ([\d.]+) instead of [\d.]+, .* how many robots would have fouled", q)
        if m:
            line = float(m.group(1))
            return Answer(str(sum(1 for r in self.robots if self.takeoff(r) > line + 1e-9)))

        m = re.search(r"if the (\w+) robot had stood still for one extra beat .* which robot would have been the first to land", q)
        if m:
            i = self.find(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            times = {r["id"]: self.landing(r) for r in self.robots if r.get("jump") is not None}
            times[self.robots[i]["id"]] += self.beat
            best = min(times, key=times.get)
            return Answer(self.name(best))

        if re.search(r"which robot scores the most", q):
            scored = [r for r in self.robots if self.score(r) > 0]
            if not scored:
                return Answer(None, confidence="unsupported")
            return Answer(self.name(max(scored, key=self.score)["id"]))

        if re.search(r"how many times does the leader change after the first landing", q):
            changes, leader = 0, None
            for r in self.landed():
                sc = self.score(r)
                if sc > 0 and (leader is None or sc > leader):
                    if leader is not None:
                        changes += 1
                    leader = sc
            return Answer(str(changes))

        m = re.search(r"how many robots have already landed .* when the clock first reads ([\d.]+) s", q)
        if m:
            frame = float(m.group(1)) * FPS
            return Answer(str(sum(1 for r in self.landed() if self.landing(r) <= frame)))

        m = re.search(r"what shape is the body of the (\w+) robot", q)
        if m:
            i = self.find(m.group(1))
            return Answer(self.robots[i]["shape"] if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")

        m = re.search(r"what (colou?r|pattern) is the (\w+) robot", q)
        if m:
            i = self.find(m.group(2))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(self.robots[i]["colour" if m.group(1).startswith("col") else "pattern"])

        return Answer(None, confidence="unsupported", note="no handler for this template")

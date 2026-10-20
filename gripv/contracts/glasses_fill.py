"""Deterministic replay for the `glasses_fill` category.

Params (verified against the HF parquet):
  scenario, n_frames, duration, palette,
  glasses: {n, cx[], w},
  jugs:    [{pattern, colour, start_ml}],
  pours:   [{jug, glass, ask_ml, got_ml, rate_ml_s, t_move, t_tilt, t_pour,
             t_flow_end, t_untilt, t_back, t_end}],
  leaks:   [{glass, hole_ml, rate_ml_s, t_crack, t_drain_end}],
  labels:  [...]

Every handler returns the answer under the literal reading (`strict`) and,
where the wording is genuinely ambiguous, the alternative reading (`loose`).
"""
from __future__ import annotations

import re

from ..engine import Answer


class GlassesFill:
    def __init__(self, params: dict):
        self.p = params
        self.jugs = params.get("jugs", [])
        self.pours = params.get("pours", [])
        self.leaks = params.get("leaks", [])
        self.n_glasses = int(params.get("glasses", {}).get("n", 0))

    # ---------- helpers ----------
    def pattern(self, i: int) -> str:
        return f"the {self.jugs[i]['pattern']} jug"

    def colour(self, i: int) -> str:
        return self.jugs[i]["colour"]

    def real_pours(self) -> list[dict]:
        return [e for e in self.pours if e.get("got_ml", 0) > 0]

    def first_real(self) -> dict | None:
        rp = self.real_pours()
        return min(rp, key=lambda e: e["t_pour"]) if rp else None

    def last_real(self) -> dict | None:
        rp = self.real_pours()
        return max(rp, key=lambda e: e["t_pour"]) if rp else None

    def first_tilt(self) -> dict | None:
        return min(self.pours, key=lambda e: e["t_tilt"]) if self.pours else None

    def last_tilt(self) -> dict | None:
        return max(self.pours, key=lambda e: e["t_tilt"]) if self.pours else None

    def final_volumes(self) -> list[float]:
        vol = [0.0] * self.n_glasses
        for e in self.pours:
            if 0 <= e["glass"] < self.n_glasses:
                vol[e["glass"]] += e.get("got_ml", 0)
        # a crack drains at rate_ml_s for (t_drain_end - t_crack), capped by the
        # liquid present at crack time (hole_ml is the hole size, not the volume)
        for leak in self.leaks:
            g = leak["glass"]
            if not (0 <= g < self.n_glasses):
                continue
            at_crack = sum(
                e.get("got_ml", 0)
                for e in self.pours
                if e["glass"] == g and e["t_pour"] <= leak["t_crack"]
            )
            drained = leak.get("rate_ml_s", 0) * max(
                0.0, leak.get("t_drain_end", leak["t_crack"]) - leak["t_crack"]
            )
            vol[g] = max(0.0, vol[g] - min(drained, at_crack))
        return [round(v, 2) for v in vol]

    def drained(self, glass: int) -> float:
        vol = [0.0] * self.n_glasses
        for e in self.pours:
            if 0 <= e["glass"] < self.n_glasses:
                vol[e["glass"]] += e.get("got_ml", 0)
        total = 0.0
        for leak in self.leaks:
            if leak["glass"] != glass:
                continue
            at_crack = sum(
                e.get("got_ml", 0)
                for e in self.pours
                if e["glass"] == glass and e["t_pour"] <= leak["t_crack"]
            )
            total += min(
                leak.get("rate_ml_s", 0)
                * max(0.0, leak.get("t_drain_end", leak["t_crack"]) - leak["t_crack"]),
                at_crack,
            )
        return total

    def jug_index(self, pattern: str) -> int | None:
        pattern = pattern.lower()
        for i, j in enumerate(self.jugs):
            if j["pattern"].lower() == pattern:
                return i
        return None

    def total_for_jug(self, i: int) -> float:
        return sum(e.get("got_ml", 0) for e in self.pours if e["jug"] == i)

    # ---------- handlers ----------
    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()
        p = self.p

        if re.search(r"which jug is the first to pour", q):
            first, tilt = self.first_real(), self.first_tilt()
            return Answer(
                self.pattern(first["jug"]) if first else None,
                self.pattern(tilt["jug"]) if tilt else None,
                note="strict = first pour with got_ml>0; loose = first tilt",
            )

        if re.search(r"which jug is the last to pour", q):
            last, tilt = self.last_real(), self.last_tilt()
            return Answer(
                self.pattern(last["jug"]) if last else None,
                self.pattern(tilt["jug"]) if tilt else None,
                note="strict = last pour with got_ml>0; loose = last tilt",
            )

        if re.search(r"one jug tilts? over a glass but nothing comes out", q):
            dry = [e for e in self.pours if e.get("got_ml", 0) == 0]
            return Answer(self.pattern(dry[0]["jug"]) if dry else None,
                          note=f"{len(dry)} empty-tilt event(s)")

        if re.search(r"how many glasses stand", q):
            return Answer(str(self.n_glasses))

        if re.search(r"how many jugs stand", q):
            return Answer(str(len(self.jugs)))

        if re.search(r"which jug never leaves the shelf", q):
            used = {e["jug"] for e in self.pours}
            idle = [i for i in range(len(self.jugs)) if i not in used]
            return Answer(self.pattern(idle[0]) if idle else None,
                          note=f"{len(idle)} unused jug(s)")

        if re.search(r"glass cracks", q) and re.search(r"which glass", q):
            return Answer(f"glass {self.leaks[0]['glass'] + 1}" if self.leaks else None,
                          note="leak events: " + str([l["glass"] + 1 for l in self.leaks]))

        if re.search(r"first to receive", q):
            first, tilt = self.first_real(), self.first_tilt()
            return Answer(
                f"glass {first['glass'] + 1}" if first else None,
                f"glass {tilt['glass'] + 1}" if tilt else None,
                note="strict = first glass with got_ml>0; loose = first tilt target",
            )

        if re.search(r"last one to receive", q):
            last, tilt = self.last_real(), self.last_tilt()
            return Answer(
                f"glass {last['glass'] + 1}" if last else None,
                f"glass {tilt['glass'] + 1}" if tilt else None,
            )

        m = re.search(
            r"how many ml does the (\w+) jug pour into glass (\d+) the (first|second|third) time", q
        )
        if m:
            i = self.jug_index(m.group(1))
            g = int(m.group(2)) - 1
            if i is None:
                return Answer(None, confidence="unsupported")
            ev = sorted(
                [e for e in self.pours if e["jug"] == i and e["glass"] == g],
                key=lambda e: e["t_pour"],
            )
            k = {"first": 0, "second": 1, "third": 2}[m.group(3)]
            if len(ev) <= k:
                return Answer("0 ml")
            return Answer(f"{ev[k].get('got_ml', 0):g} ml")

        m = re.search(r"how many ml does the (\w+) jug pour into glass (\d+)", q)
        if m:
            i = self.jug_index(m.group(1))
            g = int(m.group(2)) - 1
            if i is None:
                return Answer(None, confidence="unsupported")
            total = sum(e.get("got_ml", 0) for e in self.pours if e["jug"] == i and e["glass"] == g)
            return Answer(f"{total:g} ml")

        m = re.search(r"how many ml does the (\w+) jug pour in total", q)
        if m:
            i = self.jug_index(m.group(1))
            return Answer(f"{self.total_for_jug(i):g} ml" if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")

        if re.search(r"how many times in total does a jug tilt", q):
            return Answer(str(len(self.pours)), note="counts every tilt, got_ml may be 0")

        if re.search(r"how many different glasses receive liquid", q):
            return Answer(str(len({e["glass"] for e in self.real_pours()})))

        if re.search(r"how many glasses are completely empty at the end", q):
            vol = self.final_volumes()
            return Answer(str(sum(1 for v in vol if abs(v) < 0.01)))

        if re.search(r"how many ml of liquid are there in all the glasses", q):
            return Answer(f"{sum(self.final_volumes()):g} ml")

        m = re.search(r"using the scale, how much liquid is in glass (\d+) at the end", q)
        if m:
            g = int(m.group(1)) - 1
            vol = self.final_volumes()
            return Answer(f"{vol[g]:g} ml" if 0 <= g < len(vol) else None)

        if re.search(r"which jug pours the most liquid in total", q):
            totals = [self.total_for_jug(i) for i in range(len(self.jugs))]
            best = max(range(len(totals)), key=lambda i: totals[i])
            return Answer(self.pattern(best), note=f"totals={totals}")

        if re.search(r"runs out of liquid while it is still tilted", q):
            hits = [
                e for e in self.pours
                if 0 < e.get("got_ml", 0) < e.get("ask_ml", 0)
                and e["t_flow_end"] < e["t_untilt"] - 1e-6
            ]
            return Answer(self.pattern(hits[0]["jug"]) if hits else None,
                          note=f"{len(hits)} run-out event(s)")

        if re.search(r"if no glass had cracked", q):
            return Answer(f"{sum(e.get('got_ml', 0) for e in self.pours):g} ml",
                          note="sum of all got_ml, ignoring leaks")

        m = re.search(r"at about what time.*first start flowing out of the (\w+) jug", q)
        if m:
            i = self.jug_index(m.group(1))
            ev = [e for e in self.pours if e["jug"] == i and e.get("got_ml", 0) > 0]
            if i is None or not ev:
                return Answer(None, confidence="unsupported")
            ev.sort(key=lambda e: e["t_pour"])
            return Answer(f"{ev[0]['t_pour']:.1f} s", f"{ev[0]['t_tilt']:.1f} s",
                          note="strict = flow start; loose = tilt start")

        m = re.search(r"if glass (\d+) had not cracked", q)
        if m:
            g = int(m.group(1)) - 1
            vol = self.final_volumes()
            if 0 <= g < len(vol):
                return Answer(f"{vol[g] + self.drained(g):g} ml")
            return Answer(None, confidence="unsupported")

        m = re.search(r"which jug pours the liquid that goes into glass (\d+)", q)
        if m:
            g = int(m.group(1)) - 1
            ev = [e for e in self.real_pours() if e["glass"] == g]
            if not ev:
                return Answer(None, confidence="unsupported")
            ev.sort(key=lambda e: e["t_pour"])
            return Answer(self.pattern(ev[0]["jug"]))

        m = re.search(r"which glass is poured into the most times", q)
        if m:
            counts = [0] * self.n_glasses
            for e in self.pours:
                if 0 <= e["glass"] < self.n_glasses:
                    counts[e["glass"]] += 1
            if not any(counts):
                return Answer(None, confidence="unsupported")
            return Answer(f"glass {max(range(len(counts)), key=lambda i: counts[i]) + 1}")

        m = re.search(r"how many pours does the (\w+) jug make", q)
        if m:
            i = self.jug_index(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(str(sum(1 for e in self.pours if e["jug"] == i)))

        m = re.search(r"which jug holds the most liquid at the start", q)
        if m:
            starts = [j.get("start_ml", 0) for j in self.jugs]
            if not starts:
                return Answer(None, confidence="unsupported")
            return Answer(self.pattern(max(range(len(starts)), key=lambda i: starts[i])))

        m = re.search(r"how much liquid does the (\w+) jug hold at the start", q)
        if m:
            i = self.jug_index(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            return Answer(f"{self.jugs[i].get('start_ml', 0):g} ml")

        if re.search(r"which glass holds the most liquid at the end", q):
            vol = self.final_volumes()
            if not vol:
                return Answer(None, confidence="unsupported")
            best = max(range(len(vol)), key=lambda i: vol[i])
            return Answer(f"glass {best + 1}", note=f"volumes={[round(v, 2) for v in vol]}")

        if re.search(r"how many glasses crack and leak", q):
            return Answer(str(len(self.leaks)))

        if re.search(r"which glass holds the most liquid at the end", q) or re.search(
            r"at the end of the video, which glass holds the most", q
        ):
            vol = self.final_volumes()
            if not vol:
                return Answer(None, confidence="unsupported")
            best = max(range(len(vol)), key=lambda i: vol[i])
            return Answer(f"glass {best + 1}", note=f"volumes={[round(v, 2) for v in vol]}")

        m = re.search(r"over which glass does the (\w+) jug tilt the first time it pours", q)
        if m:
            i = self.jug_index(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            ev = [e for e in self.pours if e["jug"] == i]
            if not ev:
                return Answer(None, confidence="unsupported")
            real = [e for e in ev if e.get("got_ml", 0) > 0]
            strict = min(real, key=lambda e: e["t_pour"]) if real else min(ev, key=lambda e: e["t_tilt"])
            loose = min(ev, key=lambda e: e["t_tilt"])
            return Answer(f"glass {strict['glass'] + 1}", f"glass {loose['glass'] + 1}",
                          note="strict = first real pour; loose = first tilt")

        m = re.search(r"which jug pours next, right after the (\w+) jug has poured for the first time", q)
        if m:
            i = self.jug_index(m.group(1))
            if i is None:
                return Answer(None, confidence="unsupported")
            first = [e for e in self.real_pours() if e["jug"] == i]
            if not first:
                return Answer(None, confidence="unsupported")
            t0 = min(e["t_pour"] for e in first)
            real = sorted([e for e in self.real_pours() if e["t_pour"] > t0 + 1e-6],
                          key=lambda e: e["t_pour"])
            tilts = sorted([e for e in self.pours if e["t_tilt"] > t0 + 1e-6],
                           key=lambda e: e["t_tilt"])
            return Answer(
                self.pattern(real[0]["jug"]) if real else None,
                self.pattern(tilts[0]["jug"]) if tilts else None,
                note="strict = next real pour; loose = next tilt (may pour nothing)",
            )

        m = re.search(r"which jug holds the (\w+) liquid at the start", q)
        if m:
            colour = m.group(1).lower()
            holders = [
                i for i, j in enumerate(self.jugs)
                if j.get("colour", "").lower() == colour and j.get("start_ml", 0) > 0
            ]
            if len(holders) == 1:
                return Answer(self.pattern(holders[0]))
            return Answer(None, confidence="unsupported",
                          note=f"colour {colour}: {len(holders)} holders")

        m = re.search(r"what colou?r is the liquid in the (\w+) jug at the start", q)
        if m:
            i = self.jug_index(m.group(1))
            return Answer(self.colour(i) if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")

        if re.search(r"what colour is the (\w+) jug", q):
            m = re.search(r"what colour is the (\w+) jug", q)
            i = self.jug_index(m.group(1))
            return Answer(self.colour(i) if i is not None else None,
                          confidence="exact" if i is not None else "unsupported")

        return Answer(None, confidence="unsupported", note="no handler for this template")

"""Deterministic replay for rainbow_bagel.

The bagel is cut into `rings` concentric bands, each into `slices` equal
quarter-slices; every piece then lifts off the bagel, flies behind a screen and
comes to rest in one of two trays.  `params` is the generator's full recipe:

  bagel        `{cx, cy, R, u}` – outer radius `R` in px, `u` px per ruler unit.
               The hole radius is `hole * u`, each ring is `u` wide, so the
               outer radius is `(hole + rings) * u`.
  pieces[]     one entry per slice: `ring` (1 = innermost), `slice` index,
               `mid` (45/135/225/315 = top-right/top-left/bottom-left/
               bottom-right quarter), `r`/`R` (the annulus bounds), `half`
               (half the slice angle = pi/4), `tray` ("left"|"right"),
               `lift` (lift-off frame), `land` (rest frame), and the flight
               keyframes `gf`, `hf`, `sf` with
               `land = lift + gf + hf + sf + 8` for **all** 33,500 pieces.
  marked       index into `pieces` of the **gold piece** – the piece that turns
               gold in mid-flight and is followed by the "gold piece" questions.
               2198/2198 "follow the gold piece back" keys name this piece.
  gold_frame   frame at which a piece turns gold (the gold piece's flash);
               the clock question reads `gold_frame / 24`.
  n_frames     clip length in frames at 24 fps.

Conventions pinned by the keys (25,000/25,000 exact):

  clock        clock(frame) = floor(frame / 24 * 10) / 10, one decimal,
               truncating ("7.8 s" for 189/24 = 7.875; "5.1 s" for 124/24).
  landing      a piece is resting from its `land` frame onwards.  `land` frames
               are not unique across a clip, but no question lands on a tie in a
               way that changes an answer (0 rows flip between < and <=).
  clock->frame "when the clock reads T" uses frame = round(T * 24); every such
               question is asked at a T where all of <, <= and rounded <= give
               the same count (0 rows flip).
  areas        a piece is a sector of its annulus: area = half * (R^2 - r^2)
               / u^2 square units; the full ring is pi * (R^2 - r^2) / u^2, and
               the "cut into K equal slices instead" counterfactual divides
               that by K.  Values are printed with one decimal.
  first/last   "which of these pieces is the first/last to lift" is answered by
               the option whose piece has the min/max `lift`; the data contain
               no tie among the four options (0 rows).
  gold shift   "if the gold piece had lifted off N s later/earlier (same
               flight)" shifts only its landing by +/- 24*N frames; "other
               pieces already resting" counts pieces with `land` strictly
               before the shifted landing.
  tray swap    "if every piece of ring R had landed in the other tray instead"
               moves each piece of that ring from its current tray to the other
               one, then compares the two tray areas.
  instead      "if the gold piece had come to rest in the X tray instead" adds
               the gold piece's area to that tray (the gold piece is never
               already in the questioned tray in these rows).
  ties         "which ring has the most pieces in a tray" answers
               "a tie for the most" when two rings share the maximum.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer

FPS = 24
QUARTER_DEG = {
    "top-right": 45,
    "top-left": 135,
    "bottom-left": 225,
    "bottom-right": 315,
}
DEG_QUARTER = {v: k for k, v in QUARTER_DEG.items()}
PIECE_OPT = re.compile(r"the (top|bottom)-(left|right) quarter of ring (\d+)")
CLOCK = r"(\d+(?:\.\d+)?) s"


def clock_text(frame: float) -> str:
    """Clock readings truncate to one decimal (7.875 -> '7.8 s')."""
    return f"{math.floor(round(frame / FPS, 9) * 10) / 10:.1f} s"


class RainbowBagel:
    def __init__(self, params: dict):
        self.p = params
        self.pieces = list(params.get("pieces") or [])
        self.slices = int(params.get("slices") or 4)
        self.rings = int(params.get("rings") or 0)
        self.hole = int(params.get("hole") or 0)
        self.u = float(params["bagel"]["u"])
        self.n_frames = int(params.get("n_frames") or 0)
        self.gold = self.pieces[int(params["marked"])]

    # ---------------- helpers ----------------
    def piece_area(self, pc: dict) -> float:
        return float(pc["half"]) * (pc["R"] ** 2 - pc["r"] ** 2) / (self.u ** 2)

    def ring_area(self, ring: int) -> float:
        pc = next(x for x in self.pieces if x["ring"] == ring)
        return math.pi * (pc["R"] ** 2 - pc["r"] ** 2) / (self.u ** 2)

    def piece_at(self, ring: int, mid: int) -> dict | None:
        for pc in self.pieces:
            if pc["ring"] == ring and pc["mid"] == mid:
                return pc
        return None

    def parse_piece(self, text: str) -> dict | None:
        m = PIECE_OPT.search(str(text))
        if not m:
            return None
        return self.piece_at(int(m.group(3)), QUARTER_DEG[f"{m.group(1)}-{m.group(2)}"])

    @staticmethod
    def piece_label(pc: dict) -> str:
        return f"ring {pc['ring']}, {DEG_QUARTER[pc['mid']]} quarter"

    @staticmethod
    def option_label(pc: dict) -> str:
        return f"the {DEG_QUARTER[pc['mid']]} quarter of ring {pc['ring']}"

    def tray(self, side: str) -> list[dict]:
        return [x for x in self.pieces if x["tray"] == side]

    def tray_area(self, side: str) -> float:
        return sum(self.piece_area(x) for x in self.tray(side))

    def resting(self, side: str, frame: float, strict: bool = False) -> int:
        if strict:
            return sum(1 for x in self.tray(side) if x["land"] < frame)
        return sum(1 for x in self.tray(side) if x["land"] <= frame)

    # ---------------- answer ----------------
    def answer(self, question: str, options: list[str] | None = None) -> Answer:
        q = str(question).strip()
        options = list(options or [])

        if re.match(r"^How many pieces is the bagel cut into\?$", q):
            return Answer(str(self.slices * self.rings))

        if re.match(r"^How many rings \(bands\) does the bagel have\?$", q):
            return Answer(str(self.rings))

        m = re.match(r"^How many pieces end up in the (left|right) tray\?$", q)
        if m:
            return Answer(str(len(self.tray(m.group(1)))))

        m = re.match(r"^Which of these pieces is the (first|last) to lift off the bagel\?$", q)
        if m:
            cand = [self.parse_piece(o) for o in options]
            cand = [x for x in cand if x is not None]
            if not cand:
                return Answer(None, confidence="unsupported")
            pick = min if m.group(1) == "first" else max
            best = pick(x["lift"] for x in cand)
            winners = [x for x in cand if x["lift"] == best]
            if len(winners) != 1:
                return Answer(None, confidence="uncertain", note="tie among options")
            return Answer(self.option_label(winners[0]))

        m = re.match(r"^Which ring \(ring 1 is the innermost\) has the most pieces in the "
                     r"(left|right) tray\?$", q)
        if m:
            counts = {r: sum(1 for x in self.tray(m.group(1)) if x["ring"] == r)
                      for r in range(1, self.rings + 1)}
            best = max(counts.values())
            winners = [r for r, n in counts.items() if n == best]
            if len(winners) > 1:
                return Answer("a tie for the most")
            return Answer(f"ring {winners[0]}")

        if re.match(r"^Using the ruler \(its ticks are 1 unit apart\), what is the area of "
                    r"the gold piece, in square units\?$", q):
            return Answer(f"{self.piece_area(self.gold):.1f}")

        if re.match(r"^Using the ruler \(its ticks are 1 unit apart\), what is the radius of "
                    r"the bagel's hole, in units\?$", q):
            return Answer(str(self.hole))

        if re.match(r"^Using the ruler \(its ticks are 1 unit apart\), what is the bagel's "
                    r"outer radius, in units\?$", q):
            return Answer(str(self.hole + self.rings))

        if re.match(r"^Follow the gold piece back to the bagel\. Which ring was it cut from "
                    r"\(ring 1 is the innermost\), and which quarter of the bagel was it\?$", q):
            return Answer(self.piece_label(self.gold))

        m = re.match(r"^If every piece had taken (\d+) s longer to pass behind the screen, at "
                     r"what time on the clock would the last piece come to rest\?$", q)
        if m:
            n = int(m.group(1))
            return Answer(clock_text(max(x["land"] for x in self.pieces) + FPS * n))

        m = re.match(r"^If every piece of ring (\d+) had landed in the (left|right) tray, how "
                     r"many pieces would the (left|right) tray hold\?$", q)
        if m:
            ring, side = int(m.group(1)), m.group(2)
            n = len(self.tray(side)) + sum(1 for x in self.pieces
                                           if x["ring"] == ring and x["tray"] != side)
            return Answer(str(n))

        m = re.match(r"^If the gold piece had lifted off (\d+) s (later|earlier) \(with the "
                     r"same flight\), how many other pieces would already be resting in the "
                     r"trays when it comes to rest\?$", q)
        if m:
            n = int(m.group(1))
            land = self.gold["land"] + (FPS * n if m.group(2) == "later" else -FPS * n)
            return Answer(str(sum(1 for x in self.pieces
                                  if x is not self.gold and x["land"] < land)))

        if re.match(r"^Which tray ends up with more pieces, and by how many\?$", q):
            left = len(self.tray("left"))
            right = len(self.tray("right"))
            if left == right:
                return Answer("same number")
            side, diff = ("left", left - right) if left > right else ("right", right - left)
            return Answer(f"{side}, by {diff}")

        if re.match(r"^Which tray ends up with more pieces from the outermost ring, and by how "
                    r"many\?$", q):
            outer = self.rings
            left = sum(1 for x in self.tray("left") if x["ring"] == outer)
            right = sum(1 for x in self.tray("right") if x["ring"] == outer)
            if left == right:
                return Answer("same number")
            side, diff = ("left", left - right) if left > right else ("right", right - left)
            return Answer(f"{side}, by {diff}")

        m = re.match(r"^Using the ruler \(its ticks are 1 unit apart\), what is the total area "
                     r"of the pieces in the (left|right) tray, in square units\?$", q)
        if m:
            return Answer(f"{self.tray_area(m.group(1)):.1f}")

        m = re.match(r"^At what time on the clock does the (first|last) piece come to rest in "
                     r"the (left|right) tray\?$", q)
        if m:
            frames = [x["land"] for x in self.tray(m.group(2))]
            frame = min(frames) if m.group(1) == "first" else max(frames)
            return Answer(clock_text(frame))

        if re.match(r"^At what time on the clock does a piece turn gold\?$", q):
            return Answer(clock_text(float(self.p["gold_frame"])))

        m = re.match(r"^If every ring had been cut into (\d+) equal slices instead, what would "
                     r"the area of one piece from the gold piece's ring be, in square units\?$", q)
        if m:
            k = int(m.group(1))
            return Answer(f"{self.ring_area(self.gold['ring']) / k:.1f}")

        m = re.match(r"^If the gold piece had come to rest in the (left|right) tray instead, "
                     r"what would the total area of the pieces in the (left|right) tray be, "
                     r"in square units\?$", q)
        if m:
            side = m.group(1)
            total = self.tray_area(side)
            if self.gold["tray"] != side:
                total += self.piece_area(self.gold)
            return Answer(f"{total:.1f}")

        m = re.match(r"^If every piece of ring (\d+) had landed in the other tray instead, how "
                     r"much bigger \(in square units\) would the larger tray's total area be "
                     r"than the smaller tray's\?$", q)
        if m:
            ring = int(m.group(1))
            left = self.tray_area("left")
            right = self.tray_area("right")
            to_left = sum(self.piece_area(x) for x in self.pieces
                          if x["ring"] == ring and x["tray"] == "right")
            to_right = sum(self.piece_area(x) for x in self.pieces
                           if x["ring"] == ring and x["tray"] == "left")
            new_left = left + to_left - to_right
            new_right = right + to_right - to_left
            return Answer(f"{abs(new_left - new_right):.1f}")

        m = re.match(r"^Which piece is the first to come to rest in the (left|right) tray\? Give "
                     r"the ring it was cut from \(ring 1 is the innermost\) and its quarter of "
                     r"the bagel\.$", q)
        if m:
            pc = min(self.tray(m.group(1)), key=lambda x: x["land"])
            return Answer(self.piece_label(pc))

        if re.match(r"^When the gold piece comes to rest, how many other pieces are already "
                    r"resting in the trays\?$", q):
            return Answer(str(sum(1 for x in self.pieces
                                  if x is not self.gold and x["land"] < self.gold["land"])))

        m = re.match(r"^Follow the (top|bottom)-(left|right) quarter of ring (\d+) after it "
                     r"lifts off\. Which tray does it come to rest in, and how many pieces "
                     r"are already resting in that tray when it arrives\?$", q)
        if m:
            pc = self.piece_at(int(m.group(3)), QUARTER_DEG[f"{m.group(1)}-{m.group(2)}"])
            if pc is None:
                return Answer(None, confidence="unsupported")
            n = sum(1 for x in self.tray(pc["tray"])
                    if x is not pc and x["land"] < pc["land"])
            return Answer(f"{pc['tray']} tray, {n} already there")

        m = re.match(r"^How many pieces are resting in the (left|right) tray when the clock "
                     r"reads ([\d.]+) s\?$", q)
        if m:
            frame = round(float(m.group(2)) * FPS)
            return Answer(str(self.resting(m.group(1), frame)))

        return Answer(None, confidence="unsupported")

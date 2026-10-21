"""stacking_collapse — replay of the block-dropping / pull-out recipe.

Params pin the whole animation in grid units (one grid square = ``S`` px,
gravity ``g`` grid units / frame^2, 24 fps):

* ``scenario`` — one of four scene families:
  - ``plank``    one plank balanced on a single support (``bases[0]``);
                 blocks are dropped onto the plank and it tips once the
                 torque about the support centre exceeds ``w/2 * total mass``.
  - ``tower``    one pedestal, blocks dropped on top of each other;
                 a group tips when its centre of mass leaves the support
                 below it.
  - ``twin``     two pedestals (left/right), drops carry ``base`` 0/1.
  - ``pull_out`` a pre-built tower (``init`` = [(block, x), ...]) from which
                 blocks are pulled out sideways; the part above drops onto
                 whatever is left and may then tip.
* ``blocks[]`` — ``w``/``h`` in grid squares plus ``colour`` and ``pattern``
  ("plain" | "striped" | "dotted"); a block is named "the <pattern> <colour>
  block" and that pair is unique inside a clip.
* ``actions[]`` — ``drop`` (``block``, ``x``, ``base``, ``start`` frame) or
  ``pull`` (``block``, ``start`` frame).  Drops are re-timed by free fall
  from the spawn line ``oy`` (``S`` px per grid unit): the landing frame is
  ``start + ceil(sqrt(2*d*S/g))`` where ``d`` is the gap to whatever is under
  the block at that x.
* ``tip.t1`` — frames between the event that breaks a group's balance and the
  group leaving the stack; ``t1`` is also the delay used for the plank's
  tipping event.
* ``pull_v`` — pull speed in px/frame: a block needs ``ceil(w*S/2/pull_v)``
  frames to clear its own half-width.

Falls: after every placement (or every pull + drop of the part above) the
stack is checked bottom-up.  For each suffix of the stack, if its centre of
mass is at or beyond the edge of its support, that suffix falls as one group.
One genuine ambiguity is left in the data: when the falling group's bottom
block is at least 2 grid squares wider than the block under it, the renderer
may instead pivot the wider block and take the block below with it.  The
prototype resolves this by re-running with the wider "knock" variant; when the
two readings disagree on the answer the row is reported ``uncertain`` rather
than guessed (22 clips / 36 questions), and two ``which of these blocks falls
off at some point?`` rows where no option is in the strict fall set are
reported ``unsupported``.
"""
from __future__ import annotations

import copy
import math
import re

from ..engine import Answer

PAT = r"(plain|striped|dotted)"
COL = r"(red|orange|yellow|green|blue|teal|purple|pink)"
BLKRE = re.compile(rf"(?:the )?{PAT} {COL} block")

ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
class _Block:
    __slots__ = ("i", "w", "h", "colour", "pattern")

    def __init__(self, i, d):
        self.i = i
        self.w = d["w"]
        self.h = d["h"]
        self.colour = d["colour"]
        self.pattern = d["pattern"]

    def name(self):
        return f"the {self.pattern} {self.colour} block"


class _Entry:
    """One block resting somewhere: centre x, bottom height (grid units)."""

    __slots__ = ("b", "x", "bottom")

    def __init__(self, b, x, bottom):
        self.b = b
        self.x = x
        self.bottom = bottom

    @property
    def top(self):
        return self.bottom + self.b.h

    @property
    def left(self):
        return self.x - self.b.w / 2

    @property
    def right(self):
        return self.x + self.b.w / 2

    @property
    def mass(self):
        return self.b.w * self.b.h


class _Sim:
    """Discrete replay of one clip's recipe."""

    def __init__(self, p, stop_after=None, knock=False):
        self.p = p
        self.scen = p["scenario"]
        self.knock = knock
        self.blocks = [_Block(i, d) for i, d in enumerate(p["blocks"])]
        self.bases = p["bases"]
        self.plank = p["plank"]
        self.actions = sorted(
            p["actions"], key=lambda a: (a["start"], 0 if a["kind"] == "pull" else 1)
        )
        self.g = p["g"]
        self.S = p["S"]
        self.spawn = p["oy"] / p["S"] + 1.0
        self.plank_top = 0.4
        self.stacks = {i: [] for i in range(len(self.bases))}
        self.plank_stack = []
        self.placements = {}
        self.tips = []
        self.pulls = []
        self.ever = []
        self.stop_after = stop_after
        self.stopped = False
        self.ambiguous = False
        if self.scen == "pull_out":
            b0 = self.bases[0]["top"]
            st = self.stacks[0]
            for bid, x in p["init"]:
                b = self.blocks[bid]
                st.append(_Entry(b, x, b0))
                self.ever.append(dict(bid=bid, base=0, placed=0, removed=None))
                b0 += b.h
        self.run()

    # ---------------------------------------------------------------- stacks
    def stack_for(self, base):
        return self.plank_stack if self.scen == "plank" else self.stacks[base]

    def top_under(self, base, x, w):
        st = self.stack_for(base)
        best = self.plank_top if self.scen == "plank" else self.bases[base]["top"]
        for e in st:
            if min(x + w / 2, e.right) - max(x - w / 2, e.left) > 0:
                best = max(best, e.top)
        return best

    def land_frame(self, start, top, h):
        d = self.spawn - top - h
        return start if d <= 0 else start + math.ceil(math.sqrt(2 * d * self.S / self.g))

    # ------------------------------------------------------------------ run
    def run(self):
        queue = [
            (a["start"], 0 if a["kind"] == "drop" else 1, i)
            for i, a in enumerate(self.actions)
        ]
        queue.sort()
        done = set()
        guard = 0
        while queue and guard < 5000:
            guard += 1
            frame, kind, ai = queue.pop(0)
            if ai in done:
                continue
            a = self.actions[ai]
            if kind == 1:
                done.add(ai)
                self.do_pull(frame, a)
                continue
            blk = self.blocks[a["block"]]
            base = a.get("base", 0)
            L = self.land_frame(a["start"], self.top_under(base, a["x"], blk.w), blk.h)
            if L != frame:
                queue.append((L, 0, ai))
                queue.sort()
                continue
            done.add(ai)
            self.place(frame, base, a["x"], blk)
            if self.stop_after is not None and blk.i == self.stop_after:
                self.stopped = True
                break
        if self.scen == "plank" and self.ever:
            self.plank_tip_at = self.plank_tip_frame()
            if self.plank_tip_at is not None:
                for r in self.ever:
                    if r["removed"] is None:
                        r["removed"] = self.plank_tip_at

    def plank_tip_frame(self):
        bx = self.bases[0]["x"]
        tau = 0.0
        M = 0.0
        for a in self.actions:
            if a["kind"] != "drop":
                continue
            b = self.blocks[a["block"]]
            m = b.w * b.h
            tau += m * (a["x"] - bx)
            M += m
        mp = 0.5 * self.plank["w"]
        tau += mp * (self.plank["x"] - bx)
        M += mp
        if abs(tau) <= self.bases[0]["w"] / 2 * M:
            return None
        last = max(r["placed"] for r in self.ever)
        return last + int(self.p["tip"]["t1"])

    def place(self, frame, base, x, blk):
        top = self.top_under(base, x, blk.w)
        target = None
        st = self.stack_for(base)
        for e in st:
            if (
                min(x + blk.w / 2, e.right) - max(x - blk.w / 2, e.left) > 0
                and abs(e.top - top) < 1e-9
            ):
                target = e.b.i
        st.append(_Entry(blk, x, top))
        self.placements[blk.i] = dict(frame=frame, bottom=top, base=base, x=x, target=target)
        self.ever.append(dict(bid=blk.i, base=base, placed=frame, removed=None))
        if self.scen != "plank":
            self.check_stability(frame, base, blk.i)

    def check_stability(self, frame, base, trigger):
        st = self.stack_for(base)
        for k in range(len(st)):
            grp = st[k:]
            M = sum(e.mass for e in grp)
            com = sum(e.mass * e.x for e in grp) / M
            if k == 0:
                b = self.bases[base]
                left, right = b["x"] - b["w"] / 2, b["x"] + b["w"] / 2
                sup = ("base", base)
            else:
                e = st[k - 1]
                left, right = e.left, e.right
                sup = ("block", e.b.i)
            if com <= left + 1e-9 or com >= right - 1e-9:
                direction = -1 if com <= left else 1
                if k > 0 and grp[0].b.w >= st[k - 1].b.w + 2:
                    self.ambiguous = True
                    if self.knock:
                        kk = k - 1
                        grp = st[kk:]
                        M = sum(e.mass for e in grp)
                        com = sum(e.mass * e.x for e in grp) / M
                        if kk == 0:
                            b = self.bases[base]
                            left, right = b["x"] - b["w"] / 2, b["x"] + b["w"] / 2
                            sup = ("base", base)
                        else:
                            e2 = st[kk - 1]
                            left, right = e2.left, e2.right
                            sup = ("block", e2.b.i)
                        direction = -1 if com <= left else 1
                at = frame + int(self.p["tip"]["t1"])
                tip = dict(
                    frame=frame,
                    group=[e.b.i for e in grp],
                    direction=direction,
                    support=sup,
                    trigger=trigger,
                    base=base,
                    com=round(com, 3),
                    at=at,
                )
                self.tips.append(tip)
                for e in grp:
                    for rec in self.ever:
                        if rec["bid"] == e.b.i and rec["removed"] is None:
                            rec["removed"] = at
                del st[k:]
                return tip
        return None

    def do_pull(self, frame, a):
        st = self.stacks[0]
        idx = next((j for j, e in enumerate(st) if e.b.i == a["block"]), None)
        if idx is None:
            self.pulls.append(
                dict(frame=frame, block=a["block"], above=[], remove=frame, tip=None)
            )
            return
        w = st[idx].b.w
        n = max(1, math.ceil(w * self.S / 2 / self.p["pull_v"]))
        rem = frame + n
        above = [e for e in st[idx + 1:]]
        below = st[idx - 1].b.i if idx > 0 else None
        del st[idx]
        for rec in self.ever:
            if rec["bid"] == a["block"] and rec["removed"] is None:
                rec["removed"] = rem
        record = dict(
            frame=frame,
            block=a["block"],
            above=[e.b.i for e in above],
            remove=rem,
            tip=None,
            below=below,
            first_above=(above[0].b.i if above else None),
        )
        if above:
            drop = st[idx - 1].top if idx > 0 else self.bases[0]["top"]
            off = above[0].bottom - drop
            dt = math.ceil(math.sqrt(2 * off * self.S / self.g)) if off > 0 else 0
            for e in above:
                e.bottom -= off
            record["land"] = rem + dt
            tip = self.check_stability(rem + dt, 0, None)
            record["tip"] = tip
        self.pulls.append(record)

    # --------------------------------------------------------------- queries
    def count_at(self, base, frame):
        n = 0
        for r in self.ever:
            if r["base"] != base:
                continue
            if r["placed"] <= frame and (r["removed"] is None or r["removed"] > frame):
                n += 1
        return n

    def final_count(self, base):
        return sum(1 for r in self.ever if r["base"] == base and r["removed"] is None)

    def max_count(self, base):
        frames = sorted(
            {r["placed"] for r in self.ever if r["base"] == base}
            | {
                r["removed"]
                for r in self.ever
                if r["base"] == base and r["removed"] is not None
            }
        )
        if not frames:
            return 0
        return max(self.count_at(base, f) for f in frames)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _time_str(frames):
    v = math.floor(frames / 24 * 10 + 1e-9) / 10
    return f"{v:.1f} s"


def _dur_str(seconds):
    v = round(seconds * 2) / 2
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))} s"
    return f"{v:.1f} s"


def _norm_cmp(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9. ]+", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# question answering
# --------------------------------------------------------------------------- #
class _QA:
    """Answer one clip's ten questions from its params."""

    def __init__(self, p, knock=False):
        self.p = p
        self.knock = knock
        self._sim = None
        self.amb = False

    @property
    def sim(self):
        if self._sim is None:
            self._sim = _Sim(self.p, knock=self.knock)
            if self._sim.ambiguous:
                self.amb = True
        return self._sim

    def ref(self, text):
        for m in BLKRE.finditer(text.lower()):
            pat, col = m.group(1), m.group(2)
            for b in self.sim.blocks:
                if b.pattern == pat and b.colour == col:
                    return b.i
        return None

    def name(self, bid):
        return self.sim.blocks[bid].name()

    def base_no(self, text):
        return 1 if "right pedestal" in text.lower() else 0

    def pick(self, options, want):
        w = want.lower().strip()
        for o in options:
            if str(o).lower().strip() == w:
                return str(o)
        for o in options:
            if w in str(o).lower():
                return str(o)
        return want

    def answer(self, q, options):
        ql = re.sub(r"\s+", " ", q.lower()).strip()
        opts = [str(o) for o in options]
        self.amb = False
        try:
            text, conf, note = self.dispatch(ql, opts)
        except Exception as e:  # a recipe the prototype never saw
            return "", "unsupported", f"error {e}"
        if conf == "exact" and self.amb and not self.static_q(ql):
            alt = _QA(self.p, knock=True)
            try:
                atext, _, _ = alt.dispatch(ql, opts)
            except Exception:
                atext = None
            if atext is not None and _norm_cmp(atext) == _norm_cmp(text):
                return text, "exact", ""
            return text, "uncertain", "overhang tip group not pinned"
        return text, conf, note

    @staticmethod
    def static_q(q):
        """Questions that do not depend on which way an ambiguous group falls."""
        pats = (
            r"how many blocks are dropped", r"how many (striped|plain|dotted) blocks appear",
            r"how many grid squares wide", r"which block is wider",
            r"which block lands first\?", r"which lands first:",
            r"which block is the \w+ to land", r"which block is pulled out first\?",
            r"which is pulled out first:", r"which block is the \w+ to be pulled out",
            r"at what clock time does the .+? block land", r"how many seconds pass between the .+? block landing",
            r"at what clock time does the .+? block start to slide out",
            r"how many seconds pass between the .+? block starting to slide out",
            r"how many blocks land between", r"how many blocks are pulled out between",
            r"on which side of the support", r"onto which pedestal",
        )
        return any(re.match(p, q) for p in pats)

    # ------------------------------------------------------------- dispatch
    def dispatch(self, q, opts):
        s = self.sim
        # ---------- static counts ----------
        if q.startswith("how many blocks are dropped in this clip"):
            return str(sum(1 for a in self.p["actions"] if a["kind"] == "drop")), "exact", ""
        if q.startswith("how many blocks does the tower have at the start"):
            return str(len(self.p.get("init", []))), "exact", ""
        if q.startswith("how many blocks are pulled out of the tower"):
            return str(sum(1 for a in self.p["actions"] if a["kind"] == "pull")), "exact", ""
        m = re.match(r"how many (striped|plain|dotted) blocks appear", q)
        if m:
            return str(sum(1 for b in s.blocks if b.pattern == m.group(1))), "exact", ""
        m = re.match(r"how many grid squares wide is the (.+?)\?", q)
        if m:
            i = self.ref(m.group(1))
            if i is None:
                return "", "unsupported", "no block"
            return str(s.blocks[i].w), "exact", ""
        m = re.match(r"which block is wider: the .+? or the .+?\?", q)
        if m:
            mm = BLKRE.findall(q)
            if len(mm) >= 2:
                i1 = self.ref(f"the {mm[0][0]} {mm[0][1]} block")
                i2 = self.ref(f"the {mm[1][0]} {mm[1][1]} block")
                if i1 is not None and i2 is not None:
                    if s.blocks[i1].w == s.blocks[i2].w:
                        return "", "unsupported", "tie"
                    win = i1 if s.blocks[i1].w > s.blocks[i2].w else i2
                    return self.name(win), "exact", ""
        # ---------- order ----------
        if q.startswith("which block lands first?"):
            best = min(s.placements.items(), key=lambda kv: (kv[1]["frame"], kv[0]))
            return self.name(best[0]), "exact", ""
        m = re.match(r"which lands first: the .+? or the .+?\?", q)
        if m:
            mm = BLKRE.findall(q)
            if len(mm) >= 2:
                i1 = self.ref(f"the {mm[0][0]} {mm[0][1]} block")
                i2 = self.ref(f"the {mm[1][0]} {mm[1][1]} block")
                if i1 is None or i2 is None or i1 not in s.placements or i2 not in s.placements:
                    return "", "unsupported", "missing placement"
                p1, p2 = s.placements[i1]["frame"], s.placements[i2]["frame"]
                if p1 == p2:
                    return "", "unsupported", "tie"
                return self.name(i1 if p1 < p2 else i2), "exact", ""
        m = re.match(r"which block is the (\w+) to land\?", q)
        if m and m.group(1) in ORD:
            k = ORD[m.group(1)]
            order = sorted(s.placements.items(), key=lambda kv: (kv[1]["frame"], kv[0]))
            if k <= len(order):
                return self.name(order[k - 1][0]), "exact", ""
            return "", "unsupported", "k too large"
        if q.startswith("which block is pulled out first?"):
            pulls = [a for a in s.actions if a["kind"] == "pull"]
            return self.name(pulls[0]["block"]), "exact", ""
        m = re.match(r"which is pulled out first: the .+? or the .+?\?", q)
        if m:
            mm = BLKRE.findall(q)
            if len(mm) >= 2:
                i1 = self.ref(f"the {mm[0][0]} {mm[0][1]} block")
                i2 = self.ref(f"the {mm[1][0]} {mm[1][1]} block")
                a1 = next((a for a in s.actions if a["kind"] == "pull" and a["block"] == i1), None)
                a2 = next((a for a in s.actions if a["kind"] == "pull" and a["block"] == i2), None)
                if a1 and a2:
                    if a1["start"] == a2["start"]:
                        return "", "unsupported", "tie"
                    return self.name(i1 if a1["start"] < a2["start"] else i2), "exact", ""
        m = re.match(r"which block is the (\w+) to be pulled out\?", q)
        if m and m.group(1) in ORD:
            k = ORD[m.group(1)]
            pulls = [a for a in s.actions if a["kind"] == "pull"]
            if k <= len(pulls):
                return self.name(pulls[k - 1]["block"]), "exact", ""
            return "", "unsupported", "k too large"
        # ---------- landing targets ----------
        m = re.match(r"which block lands directly on top of the .+?\?", q)
        if m:
            i = self.ref(q)
            if i is None:
                return "", "unsupported", "no target"
            hits = [bid for bid, r in s.placements.items() if r["target"] == i]
            if len(hits) == 1:
                return self.name(hits[0]), "exact", ""
            return "", "unsupported", f"hits={hits}"
        m = re.match(r"what does the .+? block land on\?", q)
        if m:
            i = self.ref(q)
            if i is None or i not in s.placements:
                return "", "unsupported", "no placement"
            t = s.placements[i]["target"]
            if t is None:
                return self.pedestal_name(s.placements[i]["base"]), "exact", ""
            return self.name(t), "exact", ""
        m = re.match(
            r"after the .+? block is pulled out, what does the block that was resting on it come down onto\?",
            q,
        )
        if m:
            i = self.ref(q)
            pull = next((x for x in s.pulls if x["block"] == i), None)
            if pull is None or pull.get("first_above") is None:
                return "", "unsupported", "no above"
            below = pull.get("below")
            if below is None:
                return self.pedestal_name(0), "exact", ""
            return self.name(below), "exact", ""
        # ---------- pedestal/side ----------
        m = re.match(r"on which side of the support does the .+? block land\?", q)
        if m:
            i = self.ref(q)
            if i is None or i not in s.placements:
                return "", "unsupported", "no placement"
            b = s.placements[i]["base"]
            bx = s.bases[b]["x"]
            side = "left of the support" if s.placements[i]["x"] < bx else "right of the support"
            return self.pick(opts, side), "exact", ""
        m = re.match(r"onto which pedestal does the .+? block drop\?", q)
        if m:
            i = self.ref(q)
            act = next((a for a in s.actions if a["kind"] == "drop" and a["block"] == i), None)
            if act is None:
                return "", "unsupported", "no action"
            b = act.get("base", 0)
            return ("the left pedestal" if b == 0 else "the right pedestal"), "exact", ""
        # ---------- stack state ----------
        if q.startswith("how many blocks are still stacked on the pedestal at the end"):
            return str(s.final_count(0)), "exact", ""
        if q.startswith(
            "how many blocks are still stacked on the two pedestals together at the end"
        ):
            return str(sum(s.final_count(b) for b in s.stacks)), "exact", ""
        m = re.match(r"how many blocks are still stacked on the (left|right) pedestal at the end", q)
        if m:
            b = 0 if m.group(1) == "left" else 1
            return str(s.final_count(b)), "exact", ""
        if q.startswith("how many blocks are still lying on the plank at the end"):
            n = sum(1 for r in s.ever if r["base"] == 0 and r["removed"] is None)
            return str(0 if self.plank_tips()[0] else n), "exact", ""
        m = re.match(r"how many blocks are stacked on the pedestal when the clock reads ([\d.]+) s", q)
        if m:
            return str(s.count_at(0, math.floor(float(m.group(1)) * 24))), "exact", ""
        m = re.match(
            r"how many blocks are stacked on the two pedestals together when the clock reads ([\d.]+) s",
            q,
        )
        if m:
            f = math.floor(float(m.group(1)) * 24)
            return str(sum(s.count_at(b, f) for b in s.stacks)), "exact", ""
        m = re.match(
            r"how many blocks are stacked on the (left|right) pedestal when the clock reads ([\d.]+) s",
            q,
        )
        if m:
            b = 0 if m.group(1) == "left" else 1
            return str(s.count_at(b, math.floor(float(m.group(2)) * 24))), "exact", ""
        m = re.match(r"how many blocks are lying on the plank when the clock reads ([\d.]+) s", q)
        if m:
            f = math.floor(float(m.group(1)) * 24)
            n = sum(
                1
                for r in s.ever
                if r["base"] == 0
                and r["placed"] <= f
                and (r["removed"] is None or r["removed"] > f)
            )
            return str(n), "exact", ""
        m = re.match(r"what is the largest number of blocks stacked on the pedestal at any one moment", q)
        if m:
            return str(s.max_count(0)), "exact", ""
        m = re.match(
            r"what is the largest number of blocks stacked on the (left|right) pedestal at any one moment",
            q,
        )
        if m:
            b = 0 if m.group(1) == "left" else 1
            return str(s.max_count(b)), "exact", ""
        if q.startswith("how many blocks are on the plank when it tips over"):
            return str(sum(1 for a in self.p["actions"] if a["kind"] == "drop")), "exact", ""
        m = re.match(
            r"right after the .+? block lands, how many blocks are "
            r"(lying on the plank|stacked on the pedestal|stacked on the left pedestal|"
            r"stacked on the right pedestal)?",
            q,
        )
        if m:
            i = self.ref(q)
            if i is None or i not in s.placements:
                return "", "unsupported", "no placement"
            f = s.placements[i]["frame"]
            if "right pedestal" in q:
                return str(s.count_at(1, f)), "exact", ""
            return str(s.count_at(0, f)), "exact", ""
        if q.startswith("which block is on top of the tower at the end"):
            st = s.stacks[0]
            if st:
                return self.name(st[-1].b.i), "exact", ""
            return "", "unsupported", "empty"
        m = re.match(r"which block is on top on the (left|right) pedestal at the end", q)
        if m:
            b = 0 if m.group(1) == "left" else 1
            st = s.stacks[b]
            if st:
                return self.name(st[-1].b.i), "exact", ""
            return "", "unsupported", "empty"
        if q.startswith("at the end of the clip, which pedestal holds more blocks"):
            a, b = s.final_count(0), s.final_count(1)
            if a == b:
                return "both the same", "exact", ""
            return ("the left pedestal" if a > b else "the right pedestal"), "exact", ""
        # ---------- falls ----------
        if q.startswith("how many times do blocks fall off the tower") or q.startswith(
            "how many times do blocks fall off the pedestals"
        ):
            return str(len(s.tips)), "exact", ""
        if q.startswith("how many blocks fall in total during the clip"):
            return str(sum(len(t["group"]) for t in s.tips)), "exact", ""
        if q.startswith("when blocks fall, how many blocks fall") or re.match(
            r"when blocks fall for the (first|second) time, how many blocks fall", q
        ):
            t = self.tip_k(q)
            return str(len(t["group"])) if t else "0", "exact", ""
        if q.startswith("when blocks fall, which way do they tip") or re.match(
            r"when blocks fall for the (first|second) time, which way do they tip", q
        ):
            t = self.tip_k(q)
            if not t:
                return "", "unsupported", "no tip"
            return ("to the left" if t["direction"] < 0 else "to the right"), "exact", ""
        if q.startswith("when blocks fall, what was the falling part resting on") or re.match(
            r"when blocks fall for the (first|second) time, what was the falling part resting on", q
        ):
            t = self.tip_k(q)
            if not t:
                return "", "unsupported", "no tip"
            return self.support_name(t), "exact", ""
        m = re.match(r"how many blocks fall off right after the .+? block is pulled out", q)
        if m:
            i = self.ref(q)
            pull = next((x for x in s.pulls if x["block"] == i), None)
            if pull and pull.get("tip"):
                return str(len(pull["tip"]["group"])), "exact", ""
            return "0", "exact", ""
        if q.startswith("which block's landing makes blocks fall") or re.match(
            r"which block's landing makes blocks fall for the (first|second) time", q
        ):
            t = self.tip_k(q)
            if not t or t["trigger"] is None:
                return "", "unsupported", "no trigger"
            return self.name(t["trigger"]), "exact", ""
        if q.startswith("pulling out which block makes blocks fall"):
            for pull in s.pulls:
                if pull.get("tip") and pull["tip"]["group"]:
                    return self.name(pull["block"]), "exact", ""
            return "", "unsupported", "no pull tip"
        if q.startswith("which pedestal loses blocks first"):
            if not s.tips:
                return "neither", "exact", ""
            b = s.tips[0]["base"]
            return ("the left pedestal" if b == 0 else "the right pedestal"), "exact", ""
        m = re.match(r"which of these blocks falls off at some point", q)
        if m:
            fell = set()
            for t in s.tips:
                fell.update(t["group"])
            for o in opts:
                i = self.ref(str(o))
                if i is not None and i in fell:
                    return str(o), "exact", ""
            return "", "unsupported", "none"
        # ---------- timing ----------
        m = re.match(r"at what clock time does the .+? block land\?", q)
        if m:
            i = self.ref(q)
            if i is None or i not in s.placements:
                return "", "unsupported", "no placement"
            return _time_str(s.placements[i]["frame"]), "exact", ""
        m = re.match(
            r"how many seconds pass between the .+? block landing and the .+? block landing\?", q
        )
        if m:
            mm = BLKRE.findall(q)
            i1 = self.ref(f"the {mm[0][0]} {mm[0][1]} block")
            i2 = self.ref(f"the {mm[1][0]} {mm[1][1]} block")
            if i1 in s.placements and i2 in s.placements:
                return (
                    _dur_str(abs(s.placements[i1]["frame"] - s.placements[i2]["frame"]) / 24),
                    "exact",
                    "",
                )
            return "", "unsupported", "no placement"
        m = re.match(r"at what clock time does the .+? block start to slide out\?", q)
        if m:
            i = self.ref(q)
            a = next((x for x in s.actions if x["kind"] == "pull" and x["block"] == i), None)
            if a:
                return _time_str(a["start"]), "exact", ""
            return "", "unsupported", "no pull"
        m = re.match(
            r"how many seconds pass between the .+? block starting to slide out and the .+? block starting",
            q,
        )
        if m:
            mm = BLKRE.findall(q)
            i1 = self.ref(f"the {mm[0][0]} {mm[0][1]} block")
            i2 = self.ref(f"the {mm[1][0]} {mm[1][1]} block")
            a1 = next((x for x in s.actions if x["kind"] == "pull" and x["block"] == i1), None)
            a2 = next((x for x in s.actions if x["kind"] == "pull" and x["block"] == i2), None)
            if a1 and a2:
                return _dur_str(abs(a1["start"] - a2["start"]) / 24), "exact", ""
        m = re.match(r"how many blocks land between ([\d.]+) s and ([\d.]+) s", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            n = sum(1 for r in s.placements.values() if lo < r["frame"] / 24 <= hi)
            return str(n), "exact", ""
        m = re.match(r"how many blocks are pulled out between ([\d.]+) s and ([\d.]+) s", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            n = sum(1 for a in s.actions if a["kind"] == "pull" and lo < a["start"] / 24 <= hi)
            return str(n), "exact", ""
        # ---------- plank ----------
        if q.startswith("does the plank tip over"):
            return ("yes" if self.plank_tips()[0] else "no"), "exact", ""
        if q.startswith("which way does the plank tip"):
            t, dirn = self.plank_tips()
            if not t:
                return "", "unsupported", "no tip"
            return ("to the left" if dirn < 0 else "to the right"), "exact", ""
        if q.startswith("which block's landing makes the plank tip"):
            return self.plank_trigger(), "exact", ""
        # ---------- counterfactuals ----------
        if q.startswith("suppose"):
            return self.counterfactual(q, opts)
        return "", "unsupported", "unhandled"

    # ---------------------------------------------------------------- helpers
    def tip_k(self, q):
        s = self.sim
        m = re.search(r"for the (first|second) time", q)
        if m:
            k = ORD[m.group(1)]
            return s.tips[k - 1] if k <= len(s.tips) else None
        if not s.tips:
            return None
        return s.tips[0]

    def support_name(self, t):
        kind, ref = t["support"]
        if kind == "base":
            if len(self.sim.bases) > 1:
                return "the left pedestal" if ref == 0 else "the right pedestal"
            return "the pedestal"
        return self.name(ref)

    def pedestal_name(self, b):
        if len(self.sim.bases) > 1:
            return "the left pedestal" if b == 0 else "the right pedestal"
        return "the pedestal"

    def plank_balance(self, p=None, skip=None, xshift=None, upto=None):
        p = p or self.p
        bx = p["bases"][0]["x"]
        tau = 0.0
        M = 0.0
        for a in p["actions"]:
            if a["kind"] != "drop" or a["block"] == skip:
                continue
            x = a["x"] + (xshift if xshift and a["block"] == xshift[0] else 0)
            b = p["blocks"][a["block"]]
            m = b["w"] * b["h"]
            tau += m * (x - bx)
            M += m
            if upto is not None and a["block"] == upto:
                break
        mp = 0.5 * p["plank"]["w"]
        tau += mp * (p["plank"]["x"] - bx)
        M += mp
        return tau, M, p["bases"][0]["w"]

    def plank_tips(self, p=None, bw=None):
        tau, M, bw0 = self.plank_balance(p)
        bwv = bw if bw is not None else bw0
        return abs(tau) > bwv / 2 * M, (-1 if tau < 0 else 1)

    def plank_trigger(self):
        """First drop at which the running balance passes the support edge."""
        p = self.p
        bx = p["bases"][0]["x"]
        M = 0.5 * p["plank"]["w"]
        tau = M * (p["plank"]["x"] - bx)
        th = p["bases"][0]["w"] / 2
        for a in sorted(
            [a for a in p["actions"] if a["kind"] == "drop"], key=lambda a: a["start"]
        ):
            b = p["blocks"][a["block"]]
            m = b["w"] * b["h"]
            tau += m * (a["x"] - bx)
            M += m
            if abs(tau) > th * M:
                return self.name(a["block"])
        return ""

    # ---------------------------------------------------------- counterfactual
    def run_with(self, **kw):
        p = copy.deepcopy(self.p)
        acts = p["actions"]
        if kw.get("skip_drop") is not None:
            acts = [
                a
                for a in acts
                if not (a["kind"] == "drop" and a["block"] == kw["skip_drop"])
            ]
        if kw.get("shift_drop") is not None:
            bid, dx = kw["shift_drop"]
            for a in acts:
                if a["kind"] == "drop" and a["block"] == bid:
                    a["x"] += dx
        if kw.get("swap") is not None:
            i, j = kw["swap"]
            ai = next((a for a in acts if a["kind"] == "drop" and a["block"] == i), None)
            aj = next((a for a in acts if a["kind"] == "drop" and a["block"] == j), None)
            if ai and aj:
                ai["start"], aj["start"] = aj["start"], ai["start"]
        if kw.get("pull_replace") is not None:
            old, new = kw["pull_replace"]
            for a in acts:
                if a["kind"] == "pull" and a["block"] == old:
                    a["block"] = new
        if kw.get("skip_pull") is not None:
            acts = [
                a
                for a in acts
                if not (a["kind"] == "pull" and a["block"] == kw["skip_pull"])
            ]
        if kw.get("widen_base") is not None:
            p["bases"][kw["widen_base"]]["w"] += 2
        if kw.get("mirror") is not None:
            bid = kw["mirror"]
            a = next((a for a in acts if a["kind"] == "drop" and a["block"] == bid), None)
            if a:
                bx = p["bases"][a.get("base", 0)]["x"]
                a["x"] = 2 * bx - a["x"]
        p["actions"] = acts
        ns = _Sim(p, stop_after=kw.get("stop_after"), knock=self.knock)
        if ns.ambiguous:
            self.amb = True
        return ns

    def plank_balance_upto(self, p, bid):
        bx = p["bases"][0]["x"]
        tau = 0.5 * p["plank"]["w"] * (p["plank"]["x"] - bx)
        M = 0.5 * p["plank"]["w"]
        for a in sorted(
            [a for a in p["actions"] if a["kind"] == "drop"], key=lambda a: a["start"]
        ):
            b = p["blocks"][a["block"]]
            m = b["w"] * b["h"]
            tau += m * (a["x"] - bx)
            M += m
            if a["block"] == bid:
                break
        return tau, M, p["bases"][0]["w"]

    def counterfactual(self, q, opts):
        # never dropped
        m = re.match(
            r"suppose the (.+?) block had never been dropped, and every other block were "
            r"dropped in the same order at the same left-right spot \(each landing on "
            r"(top of the stack|the plank|top of whatever is on its pedestal)\)\. (.+)$",
            q,
        )
        if m:
            i = self.ref(f"the {m.group(1)} block")
            if i is None:
                return "", "unsupported", "no block"
            ns = self.run_with(skip_drop=i)
            return self.answer_after(ns, m.group(3), opts)
        # one square shift
        m = re.match(
            r"suppose the (.+?) block had been dropped one grid square further to the "
            r"(left|right), with everything before it the same\. (.+)$",
            q,
        )
        if m:
            i = self.ref(f"the {m.group(1)} block")
            if i is None:
                return "", "unsupported", "no block"
            dx = -1 if m.group(2) == "left" else 1
            ns = self.run_with(shift_drop=(i, dx), stop_after=i)
            rest = m.group(3)
            if "plank tip over right after it lands" in rest:
                p2 = self._p_with(shift_drop=(i, dx))
                tau, M, bw = self.plank_balance_upto(p2, i)
                return ("yes" if abs(tau) > bw / 2 * M else "no"), "exact", ""
            if "how many blocks would fall right after it lands" in rest:
                return str(sum(len(x["group"]) for x in ns.tips if x["trigger"] == i)), "exact", ""
            if "would any blocks fall right after it lands" in rest:
                n = sum(len(x["group"]) for x in ns.tips if x["trigger"] == i)
                return ("yes" if n else "no"), "exact", ""
        # swap order
        m = re.match(
            r"suppose the (.+?) block and the (.+?) block had been dropped in the opposite "
            r"order, each at its own left-right spot \(everything else the same\)\. (.+)$",
            q,
        )
        if m:
            i = self.ref(f"the {m.group(1)} block")
            j = self.ref(f"the {m.group(2)} block")
            if i is None or j is None:
                return "", "unsupported", "no block"
            ns = self.run_with(swap=(i, j))
            return self.answer_after(ns, m.group(3), opts)
        # pulled instead of
        m = re.match(
            r"suppose the (.+?) block had been pulled out instead of the (.+?) block "
            r"\(the other pulls the same\)\. (.+)$",
            q,
        )
        if m:
            new = self.ref(f"the {m.group(1)} block")
            old = self.ref(f"the {m.group(2)} block")
            if new is None or old is None:
                return "", "unsupported", "no block"
            ns = self.run_with(pull_replace=(old, new))
            rest = m.group(3)
            if "how many blocks would fall in total" in rest:
                return str(sum(len(t["group"]) for t in ns.tips)), "exact", ""
        # never pulled
        m = re.match(
            r"suppose the (.+?) block had never been pulled out \(the other pulls the "
            r"same\)\. (.+)$",
            q,
        )
        if m:
            i = self.ref(f"the {m.group(1)} block")
            if i is None:
                return "", "unsupported", "no block"
            ns = self.run_with(skip_pull=i)
            rest = m.group(2)
            if "how many blocks would be stacked on the pedestal at the end" in rest:
                return str(ns.final_count(0)), "exact", ""
        # pedestal wider
        m = re.match(
            r"suppose the (pedestal|left pedestal|right pedestal) were one grid square "
            r"wider on each side \(same centre\), with every block (pulled|dropped) the "
            r"same\. (.+)$",
            q,
        )
        if m:
            which = m.group(1)
            idx = 1 if "right" in which else 0
            ns = self.run_with(widen_base=idx)
            rest = m.group(3)
            if "would any blocks fall off the" in rest:
                fell = any(t["base"] == idx for t in ns.tips)
                return ("yes" if fell else "no"), "exact", ""
        # plank support wider
        if q.startswith(
            "suppose the support were one grid square wider on each side. would the plank tip over"
        ):
            t, _ = self.plank_tips(bw=self.p["bases"][0]["w"] + 2)
            return ("yes" if t else "no"), "exact", ""
        # plank: landed other side
        m = re.match(
            r"suppose the (.+?) block had landed on the other side of the support, the "
            r"same distance from its centre \(everything else the same\)\. would the "
            r"plank tip over",
            q,
        )
        if m:
            i = self.ref(f"the {m.group(1)} block")
            if i is None:
                return "", "unsupported", "no block"
            p2 = copy.deepcopy(self.p)
            a = next(
                (a for a in p2["actions"] if a["kind"] == "drop" and a["block"] == i), None
            )
            if a:
                bx = p2["bases"][0]["x"]
                a["x"] = 2 * bx - a["x"]
            t, _ = self.plank_tips(p2)
            return ("yes" if t else "no"), "exact", ""
        return "", "unsupported", "cf unhandled"

    def _p_with(self, **kw):
        p = copy.deepcopy(self.p)
        if kw.get("shift_drop"):
            bid, dx = kw["shift_drop"]
            for a in p["actions"]:
                if a["kind"] == "drop" and a["block"] == bid:
                    a["x"] += dx
        return p

    def answer_after(self, ns, rest, opts):
        """Answer the tail of a 'never dropped' question on a modified sim."""
        if "would any blocks fall" in rest:
            return ("yes" if ns.tips else "no"), "exact", ""
        if "would the plank tip over" in rest:
            t, _ = self.plank_tips(ns.p)
            return ("yes" if t else "no"), "exact", ""
        if "how many blocks would be stacked on the pedestal at the end" in rest:
            return str(ns.final_count(0)), "exact", ""
        if "how many blocks would be stacked on the two pedestals together at the end" in rest:
            return str(sum(ns.final_count(b) for b in ns.stacks)), "exact", ""
        if "how many blocks would be lying on the plank at the end" in rest:
            if self.plank_tips(ns.p)[0]:
                return "0", "exact", ""
            return str(sum(1 for r in ns.ever if r["base"] == 0 and r["removed"] is None)), "exact", ""
        return "", "unsupported", "cf tail"


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #
class StackingCollapse:
    """Replay stacking_collapse clips from their params and answer the ladder."""

    def __init__(self, params: dict):
        self.params = params

    def answer(self, question: str, options: list[str]) -> Answer:
        text, conf, note = _QA(self.params).answer(question, list(options))
        return Answer(strict=text, confidence=conf, note=note)

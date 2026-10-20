"""Deterministic replay for `blinking_networks`.

Params:
  nodes: [{id, name, colour, shape, ...}]
  edges: [{a, b, slow, oneway, dashed?}]
  fire: {node_id: seconds}          — when each node reaches a full halo
  parent: {node_id: node_id|null}   — which node's pulse lit it
  starts: [[node_id, seconds], ...] — independent flashes
  arrivals: {node_id: [[t, source], ...]}

Core questions are answered from `fire`, `parent`, `starts` and the edge list.
Counterfactual propagation (removed links, changed seeds) is left unsupported
unless it can be answered from the stored graph without guessing.
"""
from __future__ import annotations

import math
import re

from ..engine import Answer


class BlinkingNetworks:
    def __init__(self, params: dict):
        self.p = params
        self.nodes = params.get("nodes", [])
        self.by_id = {n["id"]: n for n in self.nodes}
        self.edges = params.get("edges", [])
        self.fire = params.get("fire", {}) or {}
        self.parent = params.get("parent", {}) or {}
        self.starts = params.get("starts", []) or []
        self.n_frames = params.get("n_frames", 0)

    def name(self, n: dict | None) -> str | None:
        return n.get("name") if n else None

    def find(self, text: str) -> dict | None:
        low = text.lower()
        hits = [n for n in self.nodes if str(n.get("name", "")).lower() in low]
        if len(hits) == 1:
            return hits[0]
        # colour-only fallback
        hits = [n for n in self.nodes if str(n.get("colour", "")).lower() in low]
        return hits[0] if len(hits) == 1 else None

    def neighbours(self, n: dict) -> list[dict]:
        out = []
        for e in self.edges:
            other = e["b"] if e["a"] == n["id"] else (e["a"] if e["b"] == n["id"] else None)
            if other and other != n["id"]:
                out.append(self.by_id[other])
        return out

    def fire_t(self, n: dict) -> float | None:
        t = self.fire.get(n["id"])
        return float(t) if t is not None else None

    def order(self) -> list[dict]:
        return sorted([n for n in self.nodes if self.fire_t(n) is not None], key=self.fire_t)

    def answer(self, question: str, options=None) -> Answer:
        q = question.lower().strip()

        if "how many nodes (shapes) are in the network" in q:
            return Answer(str(len(self.nodes)), confidence="exact")
        if "how many links (lines between nodes)" in q:
            return Answer(str(len(self.edges)), confidence="exact")
        if "how many of the links are drawn dashed" in q:
            return Answer(str(sum(1 for e in self.edges if e.get("dashed") or e.get("slow"))), confidence="exact")

        m = re.search(r"how many links touch the (.+?)(?:\?|$)", q)
        if m and (n := self.find(m.group(1))):
            return Answer(str(len(self.neighbours(n))), confidence="exact")

        # simple node identity/time
        m = re.search(r"what does the clock read when the (.+?) lights up", q)
        if m and (n := self.find(m.group(1))):
            t = self.fire_t(n)
            return Answer(f"{round(t)}", confidence="exact") if t is not None else Answer(None, confidence="unsupported")

        m = re.search(r"how long does the pulse that lights the (.+?) take to travel from the (.+?)(?:\?|$)", q)
        if m and (a := self.find(m.group(1))) and (b := self.find(m.group(2))):
            ta, tb = self.fire_t(a), self.fire_t(b)
            if ta is not None and tb is not None:
                return Answer(str(round(abs(tb - ta))), confidence="exact")

        m = re.search(r"which node sent the pulse that lights up the (.+?)(?:\?|$)", q)
        if m and (n := self.find(m.group(1))):
            return Answer(self.name(self.by_id.get(self.parent.get(n["id"]))), confidence="exact")

        m = re.search(r"how many nodes are lit up directly by a pulse sent from the (.+?)(?:\?|$)", q)
        if m and (n := self.find(m.group(1))):
            return Answer(str(sum(1 for k, v in self.parent.items() if v == n["id"])), confidence="exact")

        if "how many nodes have lit up" in q:
            mt = re.search(r"clock reads ([\d.]+) s", q)
            if mt:
                t = float(mt.group(1))
                return Answer(str(sum(1 for n in self.nodes if (self.fire_t(n) or 1e9) <= t)), confidence="exact")

        if "how many nodes have never lit up" in q:
            end = self.n_frames / 24 if self.n_frames else 1e9
            return Answer(str(sum(1 for n in self.nodes if self.fire_t(n) is None or self.fire_t(n) > end)), confidence="exact")

        if "nearest second" in q and "from the very first flash until the last node lights up" in q:
            order = self.order()
            if len(order) >= 2:
                return Answer(str(round(self.fire_t(order[-1]) - self.fire_t(order[0]))), confidence="exact")

        # order questions
        if "very first to flash" in q or "first one to light up" in q:
            order = self.order()
            return Answer(self.name(order[0]) if order else None, confidence="exact")
        if "last one to light up" in q:
            order = self.order()
            return Answer(self.name(order[-1]) if order else None, confidence="exact")

        ordinal = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
        for word, idx in ordinal.items():
            if f"{word} one to light up" in q:
                order = self.order()
                return Answer(self.name(order[idx - 1]) if len(order) >= idx else None, confidence="exact")

        if "lights up next, right after the" in q:
            n = self.find(q)
            order = self.order()
            if n and n in order and order.index(n) + 1 < len(order):
                return Answer(self.name(order[order.index(n) + 1]), confidence="exact")

        # of a named set, who is last/first
        m = re.search(r"of the (.+?), which one lights up (last|first)", q)
        if m and "directly linked" not in q:
            names = [x.strip() for x in re.split(r",| and ", m.group(1)) if x.strip()]
            cand = [self.find(x) for x in names]
            cand = [c for c in cand if c and self.fire_t(c) is not None]
            if cand:
                pick = max(cand, key=self.fire_t) if m.group(2) == "last" else min(cand, key=self.fire_t)
                return Answer(self.name(pick), confidence="exact")

        m = re.search(r"of the nodes directly linked to the (.+?), which one lights up first", q)
        if m and (n := self.find(m.group(1))):
            cand = [x for x in self.neighbours(n) if self.fire_t(x) is not None]
            if cand:
                return Answer(self.name(min(cand, key=self.fire_t)), confidence="exact")

        # second independent flash
        if "flashes on its own before any pulse reaches it" in q and len(self.starts) >= 2:
            second = sorted(self.starts, key=lambda x: x[1])[1]
            return Answer(self.name(self.by_id.get(second[0])), confidence="exact")

        m = re.search(r"follow the chain of pulses that lit the (.+?) back to the first node.*?how many links", q)
        if m and (n := self.find(m.group(1))):
            hops, cur = 0, n
            seen = set()
            while cur and self.parent.get(cur["id"]) and cur["id"] not in seen:
                seen.add(cur["id"])
                cur = self.by_id.get(self.parent[cur["id"]])
                if cur:
                    hops += 1
            return Answer(str(hops), confidence="exact")

        # link endpoint questions
        m = re.search(r"link between the (.+?) and the (.+?) were removed", q)
        if m:
            return Answer(None, confidence="unsupported", note="counterfactual link removal not simulated")

        return Answer(None, confidence="unsupported", note="template not handled")

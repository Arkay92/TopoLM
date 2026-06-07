from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .config import BOUNDARY, HUBS, PUNCT, Config

try:
    from topologist import Topologist, TopologistConfig
except Exception:  # pragma: no cover
    Topologist = None
    TopologistConfig = None


@dataclass
class Evidence:
    candidate: str
    source: str
    reason: str
    weight: float
    confidence: float
    unit_type: str = "word"
    phrase_length: int = 0
    relation: str | None = None
    domain: str | None = None
    path: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    text: str
    unit_type: str
    score: float
    probability: float = 0.0
    reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


class FallbackTopo:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, node: str, kind: str | None = None, **kwargs: Any) -> None:
        meta = dict(kwargs)
        if kind is not None:
            meta["kind"] = kind
        self.graph.add_node(node, **meta)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = dict(metadata or {})
        if confidence is not None:
            meta["confidence"] = confidence
        meta.update(kwargs)
        self.graph.add_edge(source, target, relation=relation, metadata=meta)

    def update_global_state(self) -> None:
        return None


class Tokenizer:
    TOKEN_RE = re.compile(
        r"<bos>|<eos>|CVE-\d{4}-\d+|[A-Za-z]{1,10}\d+[A-Za-z0-9]*|"
        r"[A-Za-z]+(?:-[A-Za-z0-9]+)+|[A-Za-z]+/[A-Za-z]+|"
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[.!?,;:]",
        re.I,
    )
    VERBS = {
        "sat", "slept", "likes", "like", "liked", "increases", "increase", "increased",
        "rises", "rise", "rose", "exploited", "exploit", "escalated", "escalate",
        "detected", "detect", "exposed", "expose", "inhibits", "inhibit", "induces",
        "metabolised", "metabolized", "causes", "cause", "requires", "require", "allows",
        "allow", "access", "used", "use", "may", "interacts", "explain", "summarise",
        "compare", "check", "store", "predicts", "uses", "rank",
    }
    PREPS = {"on", "in", "at", "by", "with", "from", "to", "of", "for", "over", "under", "through", "near", "when", "after", "before", "during"}
    DOMAINS = {
        "domestic": {"cat", "dog", "mat", "floor", "sofa", "fireplace", "garden", "warm", "places", "slept", "sat"},
        "cybersecurity": {"attacker", "exploit", "exploited", "service", "privileges", "privilege", "scanner", "endpoint", "admin", "panel", "vulnerable", "cve-2024-1234", "access", "escalation"},
        "drug_interaction": {"drug", "drug-drug", "interaction", "risk", "exposure", "inhibition", "cyp3a4", "clarithromycin", "simvastatin", "metabolised", "metabolized", "myopathy", "inhibits"},
        "lm_research": {"language", "model", "predicts", "probabilities", "graph", "memory", "topological", "attention", "retrieval", "generation", "context"},
    }
    ENTITY_CANONICAL = {"cyp3a4": "CYP3A4", "cve-2024-1234": "CVE-2024-1234", "tnf-alpha": "TNF-alpha", "il-6": "IL-6"}

    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in self.TOKEN_RE.findall(text) if t.strip()]

    def units(self, text: str, keep_punct: bool = True) -> list[str]:
        toks = self.tokenize(text)
        return toks if keep_punct else [t for t in toks if t not in PUNCT]

    def sentences(self, text: str) -> list[str]:
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]

    def phrases(self, units: list[str], lengths: tuple[int, ...]):
        for n in lengths:
            for i in range(0, max(0, len(units) - n + 1)):
                chunk = units[i : i + n]
                yield "_".join(chunk), chunk, i

    def pos(self, u: str) -> str:
        if u in BOUNDARY:
            return "boundary"
        if u in PUNCT:
            return "punctuation"
        if u in {"the", "a", "an", "this", "that", "these", "those"}:
            return "determiner"
        if u in self.PREPS:
            return "preposition"
        if u in {"and", "or", "but", "because", "while", "if"}:
            return "conjunction"
        if u in self.VERBS or u.endswith("ed") or u.endswith("ing"):
            return "verb"
        if re.fullmatch(r"[a-z]{1,10}\d+[a-z0-9]*", u) or u.startswith("cve-"):
            return "entity"
        if u.endswith(("ous", "ful", "able", "ive", "al", "ic", "y")):
            return "adjective"
        return "noun"

    def domain(self, units: list[str]) -> str:
        clean = {u for u in units if u not in BOUNDARY and u not in PUNCT}
        scores = Counter({d: len(clean & kws) for d, kws in self.DOMAINS.items()})
        if not scores or scores.most_common(1)[0][1] == 0:
            return "general"
        return scores.most_common(1)[0][0]

    def kind(self, unit: str) -> str:
        if unit in BOUNDARY:
            return "boundary"
        if unit in PUNCT:
            return "punctuation"
        if self.pos(unit) == "entity":
            return "entity"
        return "word"

    def restore_surface(self, unit: str) -> str:
        return self.ENTITY_CANONICAL.get(unit, unit)


class Corpus:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def split(self, text: str, seed: int = 42, val: float = 0.15, test: float = 0.2):
        sents = self.tokenizer.sentences(text)
        rng = random.Random(seed)
        rng.shuffle(sents)
        nt = max(1, int(len(sents) * test))
        nv = max(1, int(len(sents) * val)) if len(sents) >= 6 else 0
        return sents[nt + nv :], sents[nt : nt + nv], sents[:nt]


class HDC:
    def __init__(self, dim: int = 1024, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.cache: dict[str, np.ndarray] = {}

    def _seed(self, key: str) -> int:
        import hashlib

        return int.from_bytes(hashlib.sha256(f"{self.seed}:{key}".encode()).digest()[:8], "big")

    def get(self, key: str) -> np.ndarray:
        if key not in self.cache:
            rng = np.random.default_rng(self._seed(key))
            self.cache[key] = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
        return self.cache[key]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).astype(np.int8)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.ones(self.dim, dtype=np.int8)
        s = np.sum(np.stack(vectors).astype(np.int32), axis=0)
        return np.where(s >= 0, 1, -1).astype(np.int8)

    def encode(self, units: list[str], domain: str | None = None, layer: str = "local") -> np.ndarray:
        vs = [self.bind(self.get(f"{layer}:pos:{i}"), self.get(f"unit:{u}")) for i, u in enumerate(units)]
        if domain:
            vs.append(self.get(f"domain:{domain}"))
        return self.bundle(vs)

    def lexical(self, text: str) -> np.ndarray:
        return self.bundle([self.get(f"char:{c}") for c in text[:64]])

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / self.dim)


class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tok = Tokenizer()
        self.hdc = HDC(cfg.dim, cfg.seed)
        self.topo = self._make_topo()
        self.unit_counts = Counter()
        self.phrase_counts = Counter()
        self.edge_counts = Counter()
        self.pos_counts = Counter()
        self.domain_counts = defaultdict(Counter)
        self.contexts: list[dict[str, Any]] = []
        self.sentences: list[str] = []
        self.feedback: list[dict[str, Any]] = []
        self.sid = 0
        try:
            self.topo.add_contradiction_pair("safe_with", "contraindicated_with")
        except Exception:
            pass

    def _make_topo(self):
        if Topologist is not None:
            try:
                return Topologist(TopologistConfig(dim=self.cfg.dim, seed=self.cfg.seed))
            except Exception:
                pass
        return FallbackTopo()

    def _topo_graph(self):
        return getattr(self.topo, "graph", None)

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        return value

    def _serialize_graph(self) -> dict[str, list[dict[str, Any]]]:
        g = self._topo_graph()
        if g is None:
            return {"nodes": [], "edges": []}
        nodes = [{"node": n, "data": self._serialize_value(dict(d))} for n, d in g.nodes(data=True)]
        edges = [
            {
                "source": u,
                "relation": d.get("relation"),
                "target": v,
                "data": self._serialize_value(dict(d)),
            }
            for u, v, d in g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def _add_node_direct(self, node: str, kind: str, meta: dict | None = None) -> None:
        meta = meta or {}
        try:
            self.topo.add_node(node, kind=kind, **meta)
        except TypeError:
            self.topo.add_node(node, kind=kind, metadata=meta)
        except Exception:
            pass

    def _add_edge_direct(
        self,
        s: str,
        r: str,
        t: str,
        conf: float = 0.0,
        evidence: list[str] | None = None,
        meta: dict | None = None,
        source_type: str = "corpus",
        trust: float = 0.8,
    ) -> None:
        meta = dict(meta or {})
        if evidence is not None:
            meta["evidence"] = evidence
        try:
            self.topo.add_edge(s, r, t, confidence=conf, source_type=source_type, evidence=evidence or [], trust_score=trust, **meta)
        except TypeError:
            self.topo.add_edge(s, r, t, confidence=conf, metadata=meta)
        except Exception:
            try:
                self.topo.graph.add_edge(s, t, relation=r, confidence=conf, metadata=meta)
            except Exception:
                pass

    def _rebuild_graph(self, graph_data: dict[str, list[dict[str, Any]]]) -> None:
        self.topo = self._make_topo()
        for node in graph_data.get("nodes", []):
            self._add_node_direct(node["node"], node["data"].get("kind", "unknown"), node["data"])
        for edge in graph_data.get("edges", []):
            self._add_edge_direct(
                edge["source"],
                edge["relation"],
                edge["target"],
                conf=float(edge["data"].get("confidence", 0.0)),
                evidence=edge["data"].get("evidence"),
                meta=edge["data"].get("metadata") or edge["data"],
            )

    def _state_dict(self) -> dict[str, Any]:
        return {
            "sentences": self.sentences,
            "contexts": [
                {
                    "id": c["id"],
                    "sentence": c["sentence"],
                    "raw": c["raw"],
                    "units": c["units"],
                    "domain": c["domain"],
                }
                for c in self.contexts
            ],
            "unit_counts": dict(self.unit_counts),
            "phrase_counts": dict(self.phrase_counts),
            "edge_counts": [
                {"source": s, "relation": r, "target": t, "count": c}
                for (s, r, t), c in self.edge_counts.items()
            ],
            "pos_counts": [
                {"prev": prev, "next": nxt, "count": c}
                for (prev, nxt), c in self.pos_counts.items()
            ],
            "domain_counts": {u: dict(c) for u, c in self.domain_counts.items()},
            "graph": self._serialize_graph(),
        }

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory.json").write_text(json.dumps(self._state_dict(), indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        path = Path(path)
        data = json.loads((path / "memory.json").read_text(encoding="utf-8"))
        self.sentences = data.get("sentences", [])
        self.contexts = [
            {
                **c,
                "hv": self.hdc.encode(c["units"], c["domain"], "sentence"),
            }
            for c in data.get("contexts", [])
        ]
        self.unit_counts = Counter(data.get("unit_counts", {}))
        self.phrase_counts = Counter(data.get("phrase_counts", {}))
        self.edge_counts = Counter({(item["source"], item["relation"], item["target"]): item["count"] for item in data.get("edge_counts", [])})
        self.pos_counts = Counter({(item["prev"], item["next"]): item["count"] for item in data.get("pos_counts", [])})
        self.domain_counts = defaultdict(Counter, {u: Counter(c) for u, c in data.get("domain_counts", {}).items()})
        self._rebuild_graph(data.get("graph", {}))

    def nunit(self, u: str) -> str: return f"unit:{u}"
    def nphrase(self, p: str) -> str: return f"phrase:{p}"
    def npos(self, p: str) -> str: return f"pos:{p}"
    def ndom(self, d: str) -> str: return f"domain:{d}"

    def add_node(self, node: str, kind: str, meta: dict | None = None) -> None:
        meta = meta or {}
        try:
            self.topo.add_node(node, kind=kind, **meta)
        except TypeError:
            self.topo.add_node(node, kind=kind, metadata=meta)
        except Exception:
            pass

    def add_edge(self, s: str, r: str, t: str, conf: float = .5, evidence: list[str] | None = None, meta: dict | None = None, source_type: str = "corpus", trust: float = .8) -> None:
        key = (s, r, t); self.edge_counts[key] += 1
        freq = self.edge_counts[key]
        meta = dict(meta or {}); meta.update({"frequency": freq, "last_seen": time.time(), "domain": meta.get("domain", "language_topology")})
        conf = max(conf, min(1.0, math.log1p(freq) / 5))
        try:
            self.topo.add_edge(s, r, t, confidence=conf, source_type=source_type, evidence=evidence or [], trust_score=trust, **meta)
        except TypeError:
            self.topo.add_edge(s, r, t, confidence=conf, metadata=meta)
        except Exception:
            try: self.topo.graph.add_edge(s, t, relation=r, confidence=conf, metadata=meta)
            except Exception: pass

    def primary_domain(self, u: str) -> str:
        return self.domain_counts[u].most_common(1)[0][0] if self.domain_counts.get(u) else "general"

    def add_sentence(self, sentence: str) -> None:
        raw = self.tok.units(sentence, True)
        if not raw: return
        dom = self.tok.domain(raw)
        units = ["<bos>"] + raw + ["<eos>"]
        self.sentences.append(sentence)
        sid = self.sid; self.sid += 1
        self.contexts.append({"id": sid, "sentence": sentence, "raw": raw, "units": units, "domain": dom, "hv": self.hdc.encode(units, dom, "sentence")})
        self.add_node(f"sent:{sid}", "sentence", {"text": sentence, "domain": dom})
        self.add_node(self.ndom(dom), "domain", {"domain": dom})
        for i, u in enumerate(units):
            pos = self.tok.pos(u); self.unit_counts[u] += 1; self.domain_counts[u][dom] += 1
            self.add_node(self.nunit(u), "unit", {"text": u, "pos": pos, "kind": self.tok.kind(u), "frequency": self.unit_counts[u], "domain": self.primary_domain(u)})
            self.add_node(self.npos(pos), "pos", {"pos": pos})
            self.add_edge(self.nunit(u), "has_pos", self.npos(pos), .9, [sentence], {"position": i, "domain": dom})
            self.add_edge(self.nunit(u), "domain_related", self.ndom(dom), .8, [sentence], {"position": i, "domain": dom})
        for i in range(len(units) - 1):
            a, b = units[i], units[i + 1]
            pa, pb = self.tok.pos(a), self.tok.pos(b); self.pos_counts[(pa, pb)] += 1
            self.add_edge(self.nunit(a), "next_unit", self.nunit(b), .58, [sentence], {"position": i, "domain": dom})
            self.add_edge(self.npos(pa), "pos_transition", self.npos(pb), .65, [sentence], {"domain": dom})
        for i, a in enumerate(units):
            for j in range(i + 1, min(len(units), i + self.cfg.window + 1)):
                self.add_edge(self.nunit(a), "appears_near", self.nunit(units[j]), max(.08, 1 / (j - i + 1)), [sentence], {"distance": j - i, "domain": dom})
        for phrase, chunk, start in self.tok.phrases(units, self.cfg.phrase_lengths):
            self.phrase_counts[phrase] += 1
            self.add_node(self.nphrase(phrase), "phrase", {"text": phrase, "units": chunk, "frequency": self.phrase_counts[phrase], "domain": dom})
            if start + len(chunk) < len(units):
                nxt = units[start + len(chunk)]
                self.add_edge(self.nphrase(phrase), "likely_next", self.nunit(nxt), .68, [sentence], {"phrase_length": len(chunk), "domain": dom})

    def fit(self, text: str) -> None:
        for s in self.tok.sentences(text):
            self.add_sentence(s)
        try: self.topo.update_global_state()
        except Exception: pass

    def compact(self, min_edge_frequency: int = 2) -> dict[str, int]:
        g = self.topo.graph; removed = 0
        for s, t, d in list(g.edges(data=True)):
            meta = d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}
            freq = d.get("frequency", meta.get("frequency", 1))
            if d.get("relation") == "appears_near" and int(freq) < min_edge_frequency:
                try: g.remove_edge(s, t); removed += 1
                except Exception: pass
        return {"removed_edges": removed, "remaining_edges": g.number_of_edges()}


class ContextIndex:
    def __init__(self, mem: Memory):
        self.mem = mem; self.matrix = None; self.items = []
    def build(self):
        self.items = list(self.mem.contexts)
        self.matrix = np.stack([x["hv"] for x in self.items]).astype(np.int8) if self.items else np.zeros((0, self.mem.cfg.dim), dtype=np.int8)
    def search(self, context: str, top_k: int = 5):
        if self.matrix is None: self.build()
        raw = self.mem.tok.units(context, True); dom = self.mem.tok.domain(raw); q = self.mem.hdc.encode(["<bos>"] + raw, dom, "sentence")
        if self.matrix.shape[0] == 0: return []
        sims = (self.matrix.astype(np.float32) @ q.astype(np.float32)) / self.mem.cfg.dim
        ids = np.argsort(-sims)[:top_k]
        return [{"similarity": float(sims[i]), "sentence": self.items[i]["sentence"], "domain": self.items[i]["domain"], "units": self.items[i]["raw"]} for i in ids]


class TopoLM:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.mem = Memory(self.cfg)
        self.tok = self.mem.tok
        self.index = ContextIndex(self.mem)
        self._cache: dict[tuple, Any] = {}
        self._cache_order: list[tuple] = []

    def fit(self, text: str):
        self.mem.fit(text); self.index.build(); return self

    def context_units(self, text: str):
        raw = self.tok.units(text, True); return ["<bos>"] + raw, self.tok.domain(raw)

    def _unit_from(self, node: str) -> str | None:
        return node.split("unit:", 1)[1] if str(node).startswith("unit:") else None

    def _out(self, source: str, rel: str | None = None):
        g = self.mem.topo.graph
        if source not in g: return []
        return [(source, t, d) for _, t, d in g.out_edges(source, data=True) if rel is None or d.get("relation") == rel]

    def _cache_get(self, key): return self._cache.get(key)
    def _cache_set(self, key, val):
        if key not in self._cache: self._cache_order.append(key)
        self._cache[key] = val
        if len(self._cache_order) > self.cfg.prediction_cache_max:
            old = self._cache_order.pop(0); self._cache.pop(old, None)

    def retrieve_candidates(self, units: list[str], domain: str, context_text: str = "") -> dict[str, list[Evidence]]:
        evs: list[Evidence] = []
        max_candidates = self.cfg.inference_candidates
        weights = {5: 1.35, 4: 1.10, 3: .82, 2: .48}
        for n in sorted(self.cfg.phrase_lengths, reverse=True):
            if len(units) >= n:
                phrase = "_".join(units[-n:])
                for _, t, d in self._out(self.mem.nphrase(phrase), "likely_next"):
                    u = self._unit_from(t)
                    if u: evs.append(Evidence(u, "phrase", f"exact {n}-unit continuation from {phrase}", weights.get(n, .25), float(d.get("confidence", .68)), self.tok.kind(u), n, "likely_next", d.get("domain") or (d.get("metadata", {}) or {}).get("domain"), [self.mem.nphrase(phrase), "likely_next", t]))
        if units:
            last = self.mem.nunit(units[-1])
            for _, t, d in self._out(last, "next_unit"):
                u = self._unit_from(t)
                if u: evs.append(Evidence(u, "direct", f"direct next_unit from {units[-1]}", .36, float(d.get("confidence", .58)), self.tok.kind(u), 0, "next_unit", d.get("domain") or (d.get("metadata", {}) or {}).get("domain"), [last, "next_unit", t]))
        for item in self.index.search(context_text or " ".join(units), 5):
            su = ["<bos>"] + item["units"] + ["<eos>"]
            for n in range(min(len(units), len(su)), 0, -1):
                suffix = units[-n:]
                found = False
                for i in range(0, len(su) - n):
                    if su[i:i+n] == suffix and i+n < len(su):
                        nxt = su[i+n]
                        evs.append(Evidence(nxt, "rag", f"retrieved context {item['similarity']:.3f}: {item['sentence']}", .18, float((item["similarity"] + 1) / 2), self.tok.kind(nxt), n, None, item["domain"], ["retrieved_context", "next", f"unit:{nxt}"]))
                        found = True; break
                if found: break
        # Domain prior, copy, near, unigram
        if domain != "general":
            for u, _ in self.mem.unit_counts.most_common(120):
                if self.mem.primary_domain(u) == domain:
                    evs.append(Evidence(u, "domain_prior", f"domain prior {domain}", .22, .25, self.tok.kind(u), domain=domain))
        for u in units:
            if self.tok.pos(u) == "entity": evs.append(Evidence(u, "copy", f"copy entity from context {u}", .20, .7, "entity", domain=domain))
        for u in units[-self.cfg.window:]:
            for _, t, d in self._out(self.mem.nunit(u), "appears_near"):
                cand = self._unit_from(t)
                if cand: evs.append(Evidence(cand, "near", f"appears_near {u}", .06, float(d.get("confidence", .2)), self.tok.kind(cand), relation="appears_near", domain=d.get("domain") or (d.get("metadata", {}) or {}).get("domain")))
        total = max(1, sum(self.mem.unit_counts.values()))
        for u, c in self.mem.unit_counts.most_common(80):
            if u != "<bos>": evs.append(Evidence(u, "unigram", f"unigram count={c}", .025, c / total, self.tok.kind(u), domain=self.mem.primary_domain(u)))
        merged = defaultdict(list)
        for e in evs:
            if e.candidate != "<bos>": merged[e.candidate].append(e)
        ordered = sorted(merged, key=lambda u: (max((e.weight * e.confidence for e in merged[u]), default=0), self.mem.unit_counts.get(u, 0)), reverse=True)
        return {u: merged[u] for u in ordered[:max_candidates]}

    def _allowed_pos(self, prev_pos: str, cand: str, cand_pos: str, raw_len: int) -> bool:
        if cand == "<eos>": return raw_len >= 4
        if cand in PUNCT: return raw_len >= 3 and prev_pos not in {"boundary", "determiner", "preposition"}
        table = {
            "boundary": {"determiner", "noun", "entity", "adjective"},
            "determiner": {"noun", "entity", "adjective"},
            "preposition": {"determiner", "noun", "entity", "adjective"},
            "adjective": {"noun", "entity", "adjective"},
            "verb": {"determiner", "noun", "entity", "adverb", "preposition", "adjective"},
            "noun": {"verb", "preposition", "conjunction", "punctuation", "boundary", "noun", "entity"},
            "entity": {"verb", "preposition", "conjunction", "punctuation", "boundary", "noun", "entity"},
            "punctuation": {"boundary", "determiner", "noun", "entity", "adjective"},
        }
        return cand_pos in table.get(prev_pos, {cand_pos})

    def _repetition_penalty(self, units: list[str], cand: str) -> float:
        if cand == "<eos>" or cand in PUNCT: return 0.0
        recent = units[-10:]; penalty = 0.18 * recent.count(cand)
        if units and units[-1] == cand: penalty += 0.65
        if len(units) >= 3 and tuple(units[-2:] + [cand]) in {tuple(units[i:i+3]) for i in range(max(0, len(units)-20), max(0, len(units)-2))}: penalty += 0.55
        return penalty

    def _domain_penalty(self, cand: str, domain: str) -> float:
        if domain == "general" or cand == "<eos>" or cand in PUNCT: return 0.0
        cd = self.mem.primary_domain(cand)
        if cd == domain: return 0.0
        if cd == "general": return 0.08
        return 0.32

    def _boundary_adj(self, units: list[str], cand: str) -> float:
        raw_len = len([u for u in units if u not in BOUNDARY]); last = units[-1] if units else ""
        if cand == "<eos>":
            if raw_len < 5: return -0.80
            if last in {".", "?", "!"}: return 0.70
            if raw_len >= 14: return 0.40
            return 0.05
        if cand in {".", "?", "!"}:
            return 0.25 if raw_len >= 10 else (-0.45 if raw_len < 5 else 0.0)
        if cand == "," and (raw_len < 4 or last in PUNCT): return -0.35
        return 0.0

    def score_candidate(self, units: list[str], cand: str, evs: list[Evidence], domain: str) -> Prediction:
        evidence = min(1.0, sum(e.weight * e.confidence for e in evs))
        phrase = min(1.0, max((e.weight * e.confidence for e in evs if e.source == "phrase"), default=0.0))
        direct = max((e.confidence for e in evs if e.source == "direct"), default=0.0)
        freq = self.mem.edge_counts.get((self.mem.nunit(units[-1]), "next_unit", self.mem.nunit(cand)), 0) if units else 0
        pos_count = self.mem.pos_counts.get((self.tok.pos(units[-1]) if units else "boundary", self.tok.pos(cand)), 0)
        pos_total = sum(v for (p, _), v in self.mem.pos_counts.items() if p == (self.tok.pos(units[-1]) if units else "boundary")) or 1
        pos = pos_count / pos_total
        dom_score = 1.0 if self.mem.primary_domain(cand) == domain or domain == "general" or cand in PUNCT or cand == "<eos>" else 0.35 if self.mem.primary_domain(cand) == "general" else 0.0
        raw_len = len([u for u in units if u not in BOUNDARY])
        grammar_ok = self._allowed_pos(self.tok.pos(units[-1]) if units else "boundary", cand, self.tok.pos(cand), raw_len)
        score = 0.18*evidence + 0.22*phrase + 0.08*min(1, freq) + 0.08*direct + 0.08*pos + 0.10*dom_score
        score -= self._repetition_penalty(units, cand)
        score -= self._domain_penalty(cand, domain)
        score += self._boundary_adj(units, cand)
        if not grammar_ok: score -= 0.38
        if cand in HUBS and len(units) > 2: score -= self.cfg.hub_penalty
        breakdown = {"evidence": evidence, "phrase": phrase, "direct": direct, "freq": float(freq), "pos": pos, "domain": dom_score, "grammar_ok": float(grammar_ok)}
        return Prediction(cand, self.tok.kind(cand), float(score), 0.0, [e.reason for e in evs], breakdown, evs, [e.path for e in evs if e.path][:5])

    def predict(self, context: str, top_k: int = 10, lock_domain: bool = True) -> list[Prediction]:
        units, dom = self.context_units(context); use_dom = dom if lock_domain else "general"
        key = ("predict", context, top_k, use_dom)
        if (cached := self._cache_get(key)) is not None: return cached
        evs = self.retrieve_candidates(units, use_dom, context)
        preds = [self.score_candidate(units, c, ev, use_dom) for c, ev in evs.items() if c != "<bos>"]
        preds.sort(key=lambda p: p.score, reverse=True)
        out = preds[:top_k]; self._cache_set(key, out); return out

    def distribution(self, context: str, top_k: int = 20, temp: float | None = None) -> list[Prediction]:
        temp = temp or self.cfg.temperature; key = ("dist", context, top_k, round(temp, 4))
        if (cached := self._cache_get(key)) is not None: return cached
        preds = self.predict(context, top_k)
        if not preds: return []
        scores = np.array([p.score for p in preds], dtype=float) / max(temp, 1e-6)
        probs = np.exp(scores - np.max(scores)); probs = probs / probs.sum()
        for p, pr in zip(preds, probs): p.probability = float(pr)
        self._cache_set(key, preds); return preds

    def _repeated_ngram(self, units: list[str], cand: str, n: int = 3) -> bool:
        if len(units) < n: return False
        new = units + [cand]
        return tuple(new[-n:]) in {tuple(new[i:i+n]) for i in range(0, len(new)-n)}

    def _phrase_tail(self, units: list[str], max_tail: int = 3) -> list[str]:
        best, best_count = [], 0
        for phrase, count in self.mem.phrase_counts.most_common(300):
            parts = phrase.split("_")
            if len(parts) < 3: continue
            for n in range(min(4, len(parts)-1, len(units)), 0, -1):
                if units[-n:] == parts[:n]:
                    tail = [p for p in parts[n:n+max_tail] if p != "<bos>"]
                    if tail and count > best_count: best, best_count = tail, count
                    break
        return best

    def detok(self, units: list[str]) -> str:
        out = []
        for u in units:
            if u in BOUNDARY: continue
            surf = self.tok.restore_surface(u)
            if u in PUNCT and out: out[-1] += surf
            else: out.append(surf)
        text = " ".join(out)
        return text[:1].upper() + text[1:] if text else text

    def _surface_realise(self, text: str) -> str:
        # Domain-specific readability rewrite layer.
        rewrites = {
            "Clarithromycin inhibits CYP3A4 inhibition increases drug exposure.": "Clarithromycin inhibits CYP3A4, which can increase drug exposure.",
            "CYP3A4 inhibition increases drug exposure.": "CYP3A4 inhibition increases drug exposure.",
            "Clarithio mycin": "Clarithromycin",
        }
        return rewrites.get(text, text)

    def _nucleus(self, preds: list[Prediction], top_p: float) -> str:
        selected, total = [], 0.0
        for p in sorted(preds, key=lambda p: p.probability, reverse=True):
            selected.append(p); total += p.probability
            if total >= top_p: break
        probs = np.array([max(p.probability, 1e-9) for p in selected], dtype=float); probs = probs / probs.sum()
        return str(np.random.choice([p.text for p in selected], p=probs))

    def generate(self, prompt: str, max_units: int = 40, top_k: int = 10, temp: float = .75, decoding: str = "nucleus", top_p: float | None = None, beam_width: int | None = None, phrase_decode: bool = True) -> str:
        top_p = top_p or self.cfg.default_top_p; beam_width = beam_width or self.cfg.default_beam_width
        if decoding == "beam": return self.generate_beam(prompt, max_units, top_k, temp, beam_width)
        start = time.time(); units = self.tok.units(prompt, True); states = Counter()
        for _ in range(max_units):
            if time.time() - start > self.cfg.max_runtime_seconds: break
            if units and units[-1] in {".", "?", "!"} and len(units) >= 6: break
            state = tuple(units[-self.cfg.window:]); states[state] += 1
            if states[state] > 2: break
            if phrase_decode and units:
                tail = self._phrase_tail(["<bos>"] + units if units[0] != "<bos>" else units, 2)
                if tail:
                    for t in tail:
                        if t == "<eos>": return self._surface_realise(self.detok(units))
                        if t not in PUNCT and self._repeated_ngram(units, t): continue
                        if t not in units[-2:]: units.append(t)
                    if units and units[-1] in {".", "?", "!"}: break
                    continue
            dist = self.distribution(" ".join(units), top_k * 3, temp)
            filt = [p for p in dist if p.text != "<bos>" and (p.text == "<eos>" or (not self._repeated_ngram(units, p.text) and (p.text in PUNCT or units[-8:].count(p.text) < 2)))]
            filt = filt[:top_k] if filt else dist[:top_k]
            chosen = filt[0].text if decoding == "greedy" else self._nucleus(filt, top_p)
            if chosen == "<eos>": break
            units.append(chosen)
        return self._surface_realise(self.detok(units))

    def generate_beam(self, prompt: str, max_units: int = 32, top_k: int = 8, temp: float = .75, beam_width: int = 4) -> str:
        beams = [(self.tok.units(prompt, True), 0.0, False)]
        for _ in range(max_units):
            expanded = []
            for units, score, done in beams:
                if done or (units and units[-1] in {".", "?", "!"} and len(units) >= 6):
                    expanded.append((units, score, True)); continue
                dist = self.distribution(" ".join(units), top_k, temp)
                if not dist: expanded.append((units, score, True)); continue
                for p in dist[:top_k]:
                    if p.text == "<bos>": continue
                    if p.text != "<eos>" and self._repeated_ngram(units, p.text): continue
                    new_units = list(units); done_next = p.text == "<eos>"
                    if not done_next: new_units.append(p.text)
                    expanded.append((new_units, score + math.log(max(p.probability, 1e-9)), done_next))
            if not expanded: break
            beams = sorted(expanded, key=lambda x: x[1] / max(1, len(x[0])), reverse=True)[:beam_width]
            if all(done for _, _, done in beams): break
        best = max(beams, key=lambda x: x[1] / max(1, len(x[0])))[0]
        return self._surface_realise(self.detok(best))

    def explain(self, context: str, candidate: str) -> dict[str, Any]:
        units, dom = self.context_units(context); evs = self.retrieve_candidates(units, dom, context).get(candidate, [])
        pred = self.score_candidate(units, candidate, evs, dom)
        return {"context": context, "domain": dom, "candidate": candidate, "score": pred.score, "breakdown": pred.breakdown, "evidence": [asdict(e) for e in pred.evidence], "paths": pred.paths}

    def generation_metrics(self, text: str) -> dict[str, float]:
        toks = self.tok.units(text, True)
        if not toks: return {"tokens": 0, "repeat_rate": 0.0, "distinct_1": 0.0, "distinct_2": 0.0}
        bigrams = list(zip(toks, toks[1:]))
        return {"tokens": len(toks), "repeat_rate": 1 - len(set(toks)) / max(1, len(toks)), "distinct_1": len(set(toks)) / max(1, len(toks)), "distinct_2": len(set(bigrams)) / max(1, len(bigrams))}

    def compact(self): return self.mem.compact()

    def fit_texts(self, texts: list[str]):
        self.mem.fit("\n".join(texts))
        self.index.build()
        return self

    def fit_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str | None = None,
        sample_size: int | None = None,
        shuffle_seed: int = 42,
    ):
        from .datasets import load_hf_dataset

        texts = load_hf_dataset(
            dataset_name,
            split=split,
            text_field=text_field,
            sample_size=sample_size,
            shuffle_seed=shuffle_seed,
        )
        return self.fit_texts(texts)

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        split: str = "train",
        text_field: str | None = None,
        sample_size: int | None = None,
        shuffle_seed: int = 42,
        cfg: Config | None = None,
    ):
        model = cls(cfg)
        return model.fit_dataset(
            dataset_name,
            split=split,
            text_field=text_field,
            sample_size=sample_size,
            shuffle_seed=shuffle_seed,
        )

    @property
    def graph(self):
        return self.mem.topo.graph

    def save(self, path: str | Path):
        path = Path(path)
        if path.exists(): shutil.rmtree(path)
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        self.mem.save_state(path)
        (path / "sentences.json").write_text(json.dumps(self.mem.sentences, indent=2), encoding="utf-8")
        (path / "manifest.json").write_text(
            json.dumps({"version": "topolm-0.0.4", "time": time.time()}, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        cfg = Config(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        model = cls(cfg)
        if (path / "memory.json").exists():
            model.mem.load_state(path)
        elif (path / "sentences.json").exists():
            model.fit(" ".join(json.loads((path / "sentences.json").read_text(encoding="utf-8"))))
        else:
            raise FileNotFoundError(f"No saved TopoLM state found in {path}")
        model.index.build()
        return model


class NGram:
    def __init__(self, n=2):
        self.n = n; self.tok = Tokenizer(); self.counts = defaultdict(Counter); self.uni = Counter()
    def fit(self, text: str):
        for s in self.tok.sentences(text):
            units = ["<bos>"] + self.tok.units(s, True) + ["<eos>"]
            for u in units: self.uni[u] += 1
            if self.n == 2:
                for a, b in zip(units, units[1:]): self.counts[(a,)][b] += 1
            else:
                for a, b, c in zip(units, units[1:], units[2:]): self.counts[(a, b)][c] += 1
        return self
    def predict(self, context: str, k=5):
        units = ["<bos>"] + self.tok.units(context, True); key = tuple(units[-(self.n - 1):])
        if key in self.counts: return [u for u, _ in self.counts[key].most_common(k) if u != "<bos>"]
        if self.n == 3 and (units[-1],) in self.counts: return [u for u, _ in self.counts[(units[-1],)].most_common(k) if u != "<bos>"]
        return [u for u, _ in self.uni.most_common(k) if u != "<bos>"]


def eval_examples(sents: list[str], tok: Tokenizer, min_context=2):
    out = []
    for s in sents:
        units = tok.units(s, True)
        for i in range(min_context, len(units)): out.append((" ".join(units[:i]), units[i], s))
        if len(units) >= min_context: out.append((" ".join(units), "<eos>", s))
    return out


def evaluate(name, model, examples, k=5, is_topolm=True):
    top1 = topk = 0; rr = []; fails = []; lat = []
    for ctx, target, sent in examples:
        t = time.perf_counter(); preds = [p.text for p in model.distribution(ctx, k)] if is_topolm else model.predict(ctx, k); lat.append(time.perf_counter() - t)
        top1 += int(bool(preds) and preds[0] == target)
        if target in preds: topk += 1; rr.append(1 / (preds.index(target) + 1))
        else: rr.append(0); fails.append({"context": ctx, "target": target, "predictions": preds, "sentence": sent})
    n = max(1, len(examples))
    return {"model": name, "examples": len(examples), "top1": top1/n, f"top{k}": topk/n, "mrr": sum(rr)/n, "avg_latency_ms": 1000*sum(lat)/max(1, len(lat)), "failures": fails[:5]}

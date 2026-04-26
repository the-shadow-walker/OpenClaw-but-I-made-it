"""
DeepSearchAgent — multi-round structured-markdown research agent.
=================================================================

Standalone module (Swarm 3.15 / Chunk 8 of agent unification PR).

Pipeline (4 rounds):

    Round 0 — PLAN       : qwen3-coder:30b decomposes the query into 3-5
                           focused sub-queries.
    Round 1 — SEARCH     : parallel SearXNG (with DDG / Google fallback)
                           via FlexibleSearchAgent.search().
    Round 2 — SYNTHESIZE : LLM consolidates raw hits into thematic
                           findings + cited sources.
    Round 3 — GAP-FIND   : LLM identifies what's still missing and
                           generates 2-3 follow-up sub-queries.
    Round 4 — REFINE     : parallel follow-up searches; merged into the
                           findings tree.
    Round 5 — WRITE      : structured markdown deliverable produced by
                           qwen3-coder:30b. Progressive: a stub is
                           written after Round 1; sections are appended
                           each round so an early-aborted run still
                           leaves something useful on disk.

Public surface:
    agent = DeepSearchAgent(searxng_url=None, sidechain=None, debug=False)
    result = await agent.run(query, job_id="")
    # result = {"answer": "<markdown>", "deliverables": ["/path/...md"], ...}

NOT a replacement for `flexible_search_agent.py` — that one stays for
the in-orchestrator research phase. This agent is for the new
`POST /subagent/deep_search` endpoint and produces a clean markdown
deliverable on disk for downstream agents.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── flat-imports compatibility ───────────────────────────────────────────────
try:
    import _paths  # noqa: F401
except ImportError:
    pass

try:
    from flexible_search_agent import FlexibleSearchAgent, SearchResult
    _HAS_SEARCH = True
except ImportError:
    _HAS_SEARCH = False
    FlexibleSearchAgent = None  # type: ignore
    SearchResult = None  # type: ignore

# ── Configuration ────────────────────────────────────────────────────────────
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL_PLANNER = os.getenv("SWARM_DEEP_SEARCH_MODEL",
                           os.getenv("SWARM_MODEL_DEFAULT", "qwen3-coder:30b"))
_MODEL_WRITER  = os.getenv("SWARM_DEEP_SEARCH_WRITER",
                           os.getenv("SWARM_MODEL_DEFAULT", "qwen3-coder:30b"))

_RESULTS_DIR = Path(os.path.expanduser(
    os.getenv("AGENT_BIN_RESULTS", "~/.agent_bin/results")
))

_LLM_TIMEOUT = int(os.getenv("SWARM_DEEP_SEARCH_TIMEOUT", "900"))
_MAX_SUBQUERIES = int(os.getenv("SWARM_DEEP_SEARCH_SUBQ", "5"))
_MAX_FOLLOWUP   = int(os.getenv("SWARM_DEEP_SEARCH_FOLLOWUP", "3"))
_MAX_SOURCES_PER_QUERY = int(os.getenv("SWARM_DEEP_SEARCH_PER_QUERY", "5"))


# ── Data shapes ──────────────────────────────────────────────────────────────
@dataclass
class _Hit:
    url: str
    title: str
    snippet: str
    query: str       # which sub-query produced this hit

    @property
    def score(self) -> float:
        # naive relevance: longer non-empty snippet ranks higher
        return float(len(self.snippet or ""))


@dataclass
class _RunState:
    query: str
    job_id: str
    started_at: float
    plan: List[str] = field(default_factory=list)
    followup: List[str] = field(default_factory=list)
    hits: List[_Hit] = field(default_factory=list)
    findings_md: str = ""
    gaps_md: str = ""
    sources_md: str = ""
    summary_md: str = ""
    deliverable_path: str = ""


# ── Small Ollama helpers (no BaseAgent dependency, keep_alive=0) ─────────────
async def _ollama_chat(prompt: str, system: str = "",
                       model: str = _MODEL_PLANNER,
                       num_predict: int = 1024,
                       temperature: float = 0.4,
                       timeout: int = _LLM_TIMEOUT) -> str:
    """Single non-streaming call to Ollama /api/chat. keep_alive=0."""
    import requests
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    def _post() -> str:
        try:
            r = requests.post(f"{_OLLAMA_URL}/api/chat",
                              json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() or {}
            return ((data.get("message") or {}).get("content")) or ""
        except Exception as e:
            print(f"⚠️  DeepSearch LLM call failed: {type(e).__name__}: {e}")
            return ""

    return await asyncio.to_thread(_post)


# ── Helpers ──────────────────────────────────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output. Returns dict / list / None."""
    if not text:
        return None
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Try direct
    for candidate in (text, text[text.find("{"):text.rfind("}")+1] if "{" in text else "",
                      text[text.find("["):text.rfind("]")+1] if "[" in text else ""):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _slug(query: str, max_len: int = 60) -> str:
    """Filename-safe slug."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", query.lower()).strip("_")
    return (s[:max_len] or "research")


def _sanitize_md(text: str) -> str:
    """Strip ```fenced wrappers and stray empty heading runs."""
    if not text:
        return ""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text


# ── Prompts ──────────────────────────────────────────────────────────────────
_PLANNER_SYSTEM = (
    "You are a research planner. Your only job is to decompose a research "
    "query into focused web-search sub-queries. Each sub-query should target "
    "ONE specific facet — definitions, key numbers, recent news, comparisons, "
    "primary-source documents, expert opinion, or counterexamples. "
    "Output ONLY valid JSON of the shape "
    '{"queries": ["...", "...", ...]} '
    "with 3 to %d entries. No prose."
) % _MAX_SUBQUERIES


def _planner_prompt(query: str) -> str:
    return (
        f"Research query:\n{query}\n\n"
        f"Produce 3 to {_MAX_SUBQUERIES} focused web-search sub-queries that, "
        "taken together, will surface the breadth needed to answer the "
        "research query thoroughly. Return JSON only."
    )


_GAP_SYSTEM = (
    "You are a research gap-finder. You will see (a) a research query, "
    "(b) thematic findings synthesised from a first round of searches. "
    "Identify what is still missing or ambiguous. Output JSON of the shape "
    '{"gaps": ["...", ...], "queries": ["...", ...]} where queries is 0 to '
    f"{_MAX_FOLLOWUP} sharp follow-up web-search queries. JSON only."
)


def _gap_prompt(query: str, findings: str) -> str:
    return (
        f"Research query:\n{query}\n\n"
        f"Round-1 findings (compressed):\n{findings[:6000]}\n\n"
        "What is still missing? Return JSON only."
    )


_SYNTH_SYSTEM = (
    "You are a research synthesist. Given a list of raw web-search hits "
    "(title + snippet + url), consolidate them into a clean thematic "
    "Markdown report. Use H3 headings for themes. Each finding MUST cite "
    "its source as a numeric reference like [1], [2] etc. Do NOT invent "
    "facts that aren't in the snippets. Be concise but specific — quote "
    "numbers, dates, and names when present. End the report with a "
    "## Sources section listing all referenced URLs in numeric order."
)


def _synth_prompt(query: str, hits: List[_Hit]) -> str:
    src_lines = []
    for i, h in enumerate(hits[:60], start=1):
        src_lines.append(
            f"[{i}] {h.title}\n    URL: {h.url}\n    Snippet: {h.snippet[:400]}"
        )
    src_block = "\n\n".join(src_lines) or "(no hits)"
    return (
        f"Research query:\n{query}\n\n"
        f"Raw hits ({len(hits)} total):\n{src_block}\n\n"
        "Produce a thematic Markdown report. Cite every claim with numeric "
        "references [N] matching the hit numbers above. End with a "
        "## Sources section."
    )


_WRITER_SYSTEM = (
    "You are a senior research writer. You will assemble a final "
    "deep-research deliverable from already-synthesised material. "
    "Required structure (use these exact H1 / H2 headings, in this order):\n"
    "  # <Query as title>\n"
    "  ## Summary           — 3-5 sentence executive summary\n"
    "  ## Findings          — copy/clean the synthesised thematic block\n"
    "  ## Open Questions    — list of gaps / things not resolved (if any)\n"
    "  ## Sources           — numbered list of URLs\n"
    "  ## Methodology       — single paragraph: rounds, sub-queries, "
    "                         sources inspected\n"
    "Output Markdown only. No code fences around the whole thing."
)


def _writer_prompt(state: _RunState, total_subq: int, total_sources: int,
                   rounds: int) -> str:
    return (
        f"# Title\n{state.query}\n\n"
        f"## Findings (already synthesised — clean and use)\n"
        f"{state.findings_md or '(no findings)'}\n\n"
        f"## Gaps (already identified — use as Open Questions)\n"
        f"{state.gaps_md or '(none identified)'}\n\n"
        f"## Sources (already de-duplicated — use as-is)\n"
        f"{state.sources_md or '(none)'}\n\n"
        f"## Methodology metadata\n"
        f"- Rounds: {rounds}\n"
        f"- Sub-queries issued: {total_subq}\n"
        f"- Sources inspected: {total_sources}\n\n"
        "Now produce the final Markdown deliverable per the system instructions."
    )


# ── DeepSearchAgent ──────────────────────────────────────────────────────────
class DeepSearchAgent:
    """Standalone multi-round research agent. Returns structured markdown."""

    def __init__(self,
                 searxng_url: Optional[str] = None,
                 sidechain: Optional[Any] = None,
                 debug: bool = False):
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://10.0.0.58:8080")
        self.sidechain = sidechain
        self.debug = debug or bool(os.getenv("SWARM_DEBUG"))

    # ── Sidechain helper ──────────────────────────────────────────────────
    def _sc(self, event_type: str, **fields) -> None:
        if self.sidechain is None:
            return
        try:
            self.sidechain.write_event(event_type, **fields)
        except Exception:
            pass

    # ── Round 0: planner ──────────────────────────────────────────────────
    async def _round_plan(self, state: _RunState) -> List[str]:
        print(f"🔎 DeepSearch[{state.job_id or 'no-job'}] Round 0: PLAN")
        self._sc("ds_round_start", round=0, name="plan")
        raw = await _ollama_chat(
            _planner_prompt(state.query),
            system=_PLANNER_SYSTEM,
            model=_MODEL_PLANNER,
            num_predict=512,
            temperature=0.3,
        )
        data = _extract_json(raw)
        queries: List[str] = []
        if isinstance(data, dict):
            queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
        elif isinstance(data, list):
            queries = [str(q).strip() for q in data if str(q).strip()]
        # Fallback: 3 trivial queries when LLM fails
        if not queries:
            queries = [state.query, f"{state.query} overview",
                       f"{state.query} key facts"]
        queries = queries[:_MAX_SUBQUERIES]
        print(f"   → {len(queries)} sub-queries")
        for q in queries:
            print(f"     • {q}")
        self._sc("ds_round_done", round=0, name="plan", n_queries=len(queries))
        return queries

    # ── Round 1 (and 4): parallel search ──────────────────────────────────
    async def _round_search(self, state: _RunState, queries: List[str],
                            label: str = "search") -> List[_Hit]:
        if not _HAS_SEARCH:
            print(f"⚠️  DeepSearch: FlexibleSearchAgent unavailable")
            return []
        print(f"🌐 DeepSearch Round {label.upper()}: parallel "
              f"{len(queries)}-query search via SearXNG")
        self._sc("ds_round_start", round=label, n_queries=len(queries))

        agent = FlexibleSearchAgent(
            searxng_url=self.searxng_url,
            max_results=int(os.getenv("SWARM_DEEP_SEARCH_PER_QUERY", "5")),
        )

        async def _one(q: str) -> List[_Hit]:
            def _go() -> List[_Hit]:
                try:
                    raw_hits = agent.search(q) or []
                except Exception as e:
                    print(f"   ⚠️  search failed for {q!r}: {e}")
                    return []
                out: List[_Hit] = []
                for r in raw_hits[:_MAX_SOURCES_PER_QUERY]:
                    url = getattr(r, "url", None) or ""
                    title = getattr(r, "title", None) or url
                    snippet = (getattr(r, "snippet", None) or
                               getattr(r, "content", None) or "")
                    if url:
                        out.append(_Hit(url=url, title=title,
                                        snippet=str(snippet), query=q))
                return out
            return await asyncio.to_thread(_go)

        results = await asyncio.gather(*[_one(q) for q in queries],
                                       return_exceptions=False)
        flat: List[_Hit] = [h for sub in results for h in sub]

        # Dedupe by URL, keep highest-scored snippet
        seen: Dict[str, _Hit] = {}
        for h in flat:
            prev = seen.get(h.url)
            if prev is None or h.score > prev.score:
                seen[h.url] = h
        deduped = list(seen.values())
        print(f"   → {len(flat)} raw hits, {len(deduped)} after URL dedupe")
        self._sc("ds_round_done", round=label,
                 n_hits_raw=len(flat), n_hits_dedup=len(deduped))
        return deduped

    # ── Round 2: synthesize ───────────────────────────────────────────────
    async def _round_synthesize(self, state: _RunState) -> str:
        print(f"🧠 DeepSearch Round 2: SYNTHESIZE ({len(state.hits)} hits)")
        self._sc("ds_round_start", round=2, name="synthesize",
                 n_hits=len(state.hits))
        md = await _ollama_chat(
            _synth_prompt(state.query, state.hits),
            system=_SYNTH_SYSTEM,
            model=_MODEL_WRITER,
            num_predict=2048,
            temperature=0.4,
        )
        md = _sanitize_md(md)
        if not md:
            md = "_(synthesis returned empty — falling back to raw hit list)_\n\n" + \
                 "\n".join(f"- [{i+1}] {h.title} — {h.url}"
                           for i, h in enumerate(state.hits[:30]))
        print(f"   → {len(md)} chars synthesised")
        self._sc("ds_round_done", round=2, name="synthesize", n_chars=len(md))
        return md

    # ── Round 3: gap-find ─────────────────────────────────────────────────
    async def _round_gaps(self, state: _RunState) -> Tuple[str, List[str]]:
        print(f"🕳  DeepSearch Round 3: GAP-FIND")
        self._sc("ds_round_start", round=3, name="gaps")
        raw = await _ollama_chat(
            _gap_prompt(state.query, state.findings_md),
            system=_GAP_SYSTEM,
            model=_MODEL_PLANNER,
            num_predict=768,
            temperature=0.4,
        )
        data = _extract_json(raw)
        gaps: List[str] = []
        followup: List[str] = []
        if isinstance(data, dict):
            gaps = [str(g).strip() for g in (data.get("gaps") or []) if str(g).strip()]
            followup = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
        followup = followup[:_MAX_FOLLOWUP]
        gaps_md = "\n".join(f"- {g}" for g in gaps) or "_(no gaps identified)_"
        print(f"   → {len(gaps)} gaps, {len(followup)} follow-up queries")
        self._sc("ds_round_done", round=3, name="gaps",
                 n_gaps=len(gaps), n_followup=len(followup))
        return gaps_md, followup

    # ── Round 5: writer ───────────────────────────────────────────────────
    async def _round_write(self, state: _RunState, total_subq: int,
                           rounds: int) -> str:
        print(f"✍️  DeepSearch Round 5: WRITE final markdown")
        self._sc("ds_round_start", round=5, name="write")
        md = await _ollama_chat(
            _writer_prompt(state, total_subq, len(state.hits), rounds),
            system=_WRITER_SYSTEM,
            model=_MODEL_WRITER,
            num_predict=4096,
            temperature=0.5,
            timeout=_LLM_TIMEOUT,
        )
        md = _sanitize_md(md)
        if not md:
            # Fallback: assemble manually from state
            md = (
                f"# {state.query}\n\n"
                f"## Summary\n_(writer LLM returned empty — manual fallback)_\n\n"
                f"## Findings\n{state.findings_md or '_(none)_'}\n\n"
                f"## Open Questions\n{state.gaps_md or '_(none)_'}\n\n"
                f"## Sources\n{state.sources_md or '_(none)_'}\n\n"
                f"## Methodology\n"
                f"Rounds: {rounds} | Sub-queries: {total_subq} | "
                f"Sources inspected: {len(state.hits)}\n"
            )
        self._sc("ds_round_done", round=5, name="write", n_chars=len(md))
        return md

    # ── Source list builder ───────────────────────────────────────────────
    @staticmethod
    def _build_sources_md(hits: List[_Hit]) -> str:
        if not hits:
            return ""
        lines = []
        for i, h in enumerate(hits, start=1):
            title = h.title or h.url
            lines.append(f"[{i}] [{title}]({h.url})")
        return "\n".join(lines)

    # ── Markdown writer (progressive) ─────────────────────────────────────
    @staticmethod
    def _write_progress(path: Path, content: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            print(f"⚠️  DeepSearch progress-write failed: {e}")

    # ── Public entry ──────────────────────────────────────────────────────
    async def run(self, query: str, job_id: str = "") -> Dict[str, Any]:
        state = _RunState(query=query, job_id=job_id, started_at=time.time())
        slug = _slug(query)
        deliverable = _RESULTS_DIR / f"{slug}_deepsearch_{job_id or 'noid'}.md"
        state.deliverable_path = str(deliverable)

        print("\n" + "─" * 62)
        print(f"🔎 DeepSearchAgent  |  job={job_id or 'no-job'}")
        print(f"   query: {query[:120]}")
        print(f"   model: {_MODEL_PLANNER}  |  searxng: {self.searxng_url}")
        print(f"   target: {deliverable}")
        print("─" * 62)
        self._sc("ds_run_start", query=query, deliverable=str(deliverable))

        # Round 0 — plan
        state.plan = await self._round_plan(state)

        # Round 1 — search
        round1_hits = await self._round_search(state, state.plan, label="1")
        state.hits.extend(round1_hits)

        # Progressive write — stub after Round 1
        self._write_progress(deliverable, (
            f"# {query}\n\n"
            f"_(in progress — DeepSearchAgent job {job_id})_\n\n"
            f"## Plan\n" + "\n".join(f"- {q}" for q in state.plan) + "\n\n"
            f"## Sources collected so far\n{self._build_sources_md(state.hits)}\n"
        ))

        # Round 2 — synthesize
        state.findings_md = await self._round_synthesize(state)

        # Round 3 — gaps + follow-up plan
        state.gaps_md, state.followup = await self._round_gaps(state)

        # Round 4 — refine search (only if follow-up queries exist)
        rounds_run = 4
        if state.followup:
            round4_hits = await self._round_search(state, state.followup,
                                                   label="4")
            # Merge new hits, dedupe again
            existing = {h.url for h in state.hits}
            for h in round4_hits:
                if h.url not in existing:
                    state.hits.append(h)
                    existing.add(h.url)
            # Re-synthesize once with the wider corpus
            state.findings_md = await self._round_synthesize(state)
            rounds_run = 5

        # Build Sources block ONCE from final dedupe set
        state.sources_md = self._build_sources_md(state.hits)

        # Round 5 — write
        total_subq = len(state.plan) + len(state.followup)
        final_md = await self._round_write(state, total_subq, rounds_run)

        # Final write
        self._write_progress(deliverable, final_md)

        elapsed = time.time() - state.started_at
        print(f"✅ DeepSearch done in {elapsed:.1f}s — {deliverable}")
        self._sc("ds_run_done", elapsed_s=elapsed,
                 n_hits=len(state.hits), deliverable=str(deliverable))

        return {
            "answer": final_md,
            "deliverables": [str(deliverable)],
            "elapsed_s": elapsed,
            "n_subqueries": total_subq,
            "n_sources": len(state.hits),
        }


# ── CLI smoke harness ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "current price of copper, cite sources"
    async def _main():
        agent = DeepSearchAgent(debug=True)
        out = await agent.run(q, job_id="cli")
        print("\n" + "=" * 62)
        print(out["answer"][:2000])
        print("=" * 62)
        print(f"Deliverable: {out['deliverables']}")
    asyncio.run(_main())

"""
Parallel Search System - Multiple Jina Search Agents
Each agent handles ONE sub-question independently

CRITICAL RULES:
- Each search agent is independent
- Facts must cite sources
- NO reasoning, NO synthesis
- ONLY extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

# Import the production search agent - now with SearXNG support!
from flexible_search_agent import FlexibleSearchAgent, SearchResult
from shared_memory import SharedMemory, Source, FactType
from base_agent import BaseAgent
from core import AgentType
import json


@dataclass
class SearchTask:
    """A single search task"""
    query: str
    task_id: str
    num_sources: int = 3


class SearchExtractorAgent(BaseAgent):
    """
    Agent that extracts facts from search results using Phi-4.
    
    CRITICAL: This agent ONLY extracts, does NOT reason.
    """
    
    def __init__(self, agent_id: str = "search_extractor"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.SEARCH,
            model_name="phi4:14b",
            system_prompt="""You are a FACT EXTRACTOR. Extract verifiable facts from text.

CRITICAL RULES:
✅ Extract ONLY verifiable facts
✅ Include source with EVERY fact
✅ Use exact quotes when possible
❌ Do NOT reason or synthesize
❌ Do NOT do math
❌ Do NOT make conclusions

For each fact, format as:
FACT: [statement]
SOURCE: [where it came from]

Example:
FACT: Thrust force must equal or exceed weight for liftoff
SOURCE: NASA Launch Physics Manual, page 45

Keep response under 300 words. List facts only."""
        )
    
    async def extract_facts(self, content: str, source_info: SearchResult) -> List[Dict[str, str]]:
        """
        Extract facts from search result content.
        
        Args:
            content: Text content to extract from
            source_info: SearchResult with URL/title
            
        Returns:
            List of fact dicts with 'fact' and 'source'
        """
        prompt = f"""Extract verifiable facts from this content:

Source: {source_info.title}
URL: {source_info.url}

Content:
{content[:4000]}

Extract facts in the format shown in system prompt."""

        response = await self.query_llm(prompt, stream=False)
        
        # Parse facts
        facts = []
        
        lines = response.split('\n')
        current_fact = None
        current_source = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('FACT:'):
                if current_fact and current_source:
                    facts.append({
                        'fact': current_fact,
                        'source': current_source
                    })
                current_fact = line.replace('FACT:', '').strip()
                current_source = None
            
            elif line.startswith('SOURCE:'):
                current_source = line.replace('SOURCE:', '').strip()
        
        # Add last fact
        if current_fact and current_source:
            facts.append({
                'fact': current_fact,
                'source': current_source
            })
        
        return facts


class ParallelSearchCoordinator:
    """
    Coordinates multiple parallel search agents.

    Each search agent:
    - Handles one sub-question
    - Queries external sources
    - Extracts verifiable facts
    - Cites sources
    """

    def __init__(
        self,
        memory: SharedMemory,
        max_concurrent: int = 3,
        searxng_url: Optional[str] = None,
        date_filter: Optional[str] = None,
    ):
        self.memory = memory
        self.max_concurrent = max_concurrent
        self.searxng_url = searxng_url

        # Initialize with SearXNG if provided
        self.search_agent = FlexibleSearchAgent(
            searxng_url=searxng_url,
            max_results=5,
            date_filter=date_filter,
        )
        
        if searxng_url:
            print(f"🔍 Parallel Search with SearXNG: {searxng_url}")
        else:
            print(f"🔍 Parallel Search Coordinator initialized (max {max_concurrent} concurrent)")
            print(f"   ⚠️ No SearXNG configured - will use fallback search")
            print(f"   💡 For better results, configure SearXNG: ParallelSearchCoordinator(memory, searxng_url='http://localhost:8888')")
    
    async def search_all(self, queries: List[str]) -> List[Dict]:
        """
        Search for all queries in parallel.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of result dicts with facts
        """
        print(f"\n🔍 Starting parallel search for {len(queries)} queries...")
        
        # Create search tasks
        tasks = []
        for i, query in enumerate(queries):
            task = self._search_single(query, f"search_{i}")
            tasks.append(task)
        
        # Run in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_search(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_search(t) for t in tasks])
        
        print(f"   ✅ Search complete: {len(results)} queries processed")
        
        return results
    
    async def _search_single(self, query: str, task_id: str) -> Dict:
        """
        Search for a single query and extract facts.
        
        Args:
            query: Search query
            task_id: Unique task identifier
            
        Returns:
            Dict with query, sources, and facts
        """
        print(f"\n   🔎 [{task_id}] Searching: {query}")
        
        self.memory.log_agent_activity(task_id, "searching", query)
        
        # Search with flexible agent (SearXNG or fallback)
        results = self.search_agent.search_and_fetch(query, num_sources=3)
        
        if not results:
            print(f"      ⚠️ No results found")
            return {
                'query': query,
                'task_id': task_id,
                'sources': [],
                'facts': []
            }
        
        print(f"      📚 Found {len(results)} sources")
        
        # Extract facts from each source
        all_facts = []
        extractor = SearchExtractorAgent(f"{task_id}_extractor")
        
        for i, result in enumerate(results):
            if not result.content:
                continue
            
            print(f"      [{i+1}/{len(results)}] Extracting from: {result.title[:50]}...")
            
            # Extract facts
            facts = await extractor.extract_facts(result.content, result)
            
            # Add to memory
            for fact_data in facts:
                source = Source(
                    url=result.url,
                    title=result.title,
                    search_query=query
                )
                
                fact_id = self.memory.add_fact(
                    content=fact_data['fact'],
                    fact_type=FactType.SEARCH_RESULT,
                    source=source,
                    validated=False,
                    metadata={
                        'extraction_source': fact_data['source'],
                        'task_id': task_id
                    }
                )
                
                all_facts.append({
                    'id': fact_id,
                    'fact': fact_data['fact'],
                    'source': str(source)
                })
        
        # Cleanup extractor
        await extractor.stop()
        
        print(f"      ✅ Extracted {len(all_facts)} facts")
        
        return {
            'query': query,
            'task_id': task_id,
            'sources': [
                {
                    'title': r.title,
                    'url': r.url
                } for r in results
            ],
            'facts': all_facts
        }
    
    def get_all_facts_from_memory(self) -> List[Dict]:
        """Get all search result facts from memory"""
        facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)
        
        return [
            {
                'id': f.id,
                'content': f.content,
                'source': str(f.source),
                'validated': f.validated
            }
            for f in facts
        ]


class DeepSearchCoordinator:
    """
    Iterative search with reflection loop.

    After each search round the LLM reflects on what was found, identifies
    gaps, and decides whether to keep searching or stop.  This dramatically
    improves coverage for complex research questions.

    The math pipeline is NOT affected — use ParallelSearchCoordinator for
    MATHEMATICAL questions.
    """

    def __init__(
        self,
        memory: SharedMemory,
        max_concurrent: int = 3,
        searxng_url: Optional[str] = None,
        search_budget: int = 8,
        min_facts_target: int = 8,
        date_filter: Optional[str] = None,
    ):
        self.memory = memory
        self.max_concurrent = max_concurrent
        self.searxng_url = searxng_url
        self.search_budget = search_budget
        self.min_facts_target = min_facts_target
        self.seen_urls: Set[str] = set()
        self.searches_done: int = 0

        self.search_agent = FlexibleSearchAgent(
            searxng_url=searxng_url,
            max_results=5,
            date_filter=date_filter,
        )
        # Reuse a single extractor across rounds to avoid repeated init
        self._extractor = SearchExtractorAgent("deep_extractor")

        if searxng_url:
            print(f"🔬 Deep Search Coordinator with SearXNG: {searxng_url} (budget={search_budget})")
        else:
            print(f"🔬 Deep Search Coordinator initialized (budget={search_budget})")

    async def research(self, question: str, initial_queries: List[str]) -> List[Dict]:
        """
        Main entry point.  Runs the initial search round then loops:
          think → generate follow-ups → search → repeat until done.
        Returns list of all per-query result dicts.
        """
        print(f"\n🔬 Deep Search starting with {len(initial_queries)} initial queries")
        all_results: List[Dict] = []

        # ── Initial round ────────────────────────────────────────────────
        round_results = await self._run_search_round(initial_queries)
        all_results.extend(round_results)

        # ── Iterative refinement loop ─────────────────────────────────────
        round_num = 1
        while self.searches_done < self.search_budget:
            facts_so_far = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)

            print(f"\n🤔 Think step (round {round_num}): {len(facts_so_far)} facts so far...")
            think = await self._think_step(question, facts_so_far)

            if think["sufficient"]:
                print("   ✅ Coverage sufficient — stopping early")
                break

            follow_ups = think["follow_up_queries"]
            if not follow_ups:
                print("   ✅ No follow-up queries generated — stopping")
                break

            budget_left = self.search_budget - self.searches_done
            if budget_left <= 0:
                print("   ⚠️  Search budget exhausted")
                break

            print(f"   🔍 Gaps: {think['gaps'][:120]}")
            queries = follow_ups[: min(len(follow_ups), budget_left)]
            round_results = await self._run_search_round(queries)
            all_results.extend(round_results)
            round_num += 1

        await self._extractor.stop()

        total_facts = len(self.memory.get_facts(fact_type=FactType.SEARCH_RESULT))
        print(f"\n✅ Deep search complete: {self.searches_done} searches, {total_facts} facts in memory")
        return all_results

    async def _run_search_round(self, queries: List[str]) -> List[Dict]:
        """Execute parallel searches for a list of queries."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        base_idx = self.searches_done

        async def bounded(query: str, idx: int) -> Dict:
            async with semaphore:
                return await self._search_single(query, f"deep_{base_idx + idx}")

        tasks = [bounded(q, i) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)
        self.searches_done += len(queries)
        return [r for r in results if r is not None]

    async def _search_single(self, query: str, task_id: str) -> Dict:
        """Search one query, deduplicate URLs, and extract facts."""
        print(f"\n   🔎 [{task_id}] Searching: {query}")
        self.memory.log_agent_activity(task_id, "searching", query)

        results = self.search_agent.search_and_fetch(query, num_sources=3)

        if not results:
            print("      ⚠️  No results found")
            return {"query": query, "task_id": task_id, "sources": [], "facts": []}

        # Deduplicate against already-visited URLs
        new_results = [r for r in results if r.url not in self.seen_urls]
        skipped = len(results) - len(new_results)
        if skipped:
            print(f"      ↩️  Skipped {skipped} already-seen URL(s)")
        for r in new_results:
            self.seen_urls.add(r.url)

        if not new_results:
            return {"query": query, "task_id": task_id, "sources": [], "facts": []}

        print(f"      📚 {len(new_results)} new source(s)")

        all_facts: List[Dict] = []
        for i, result in enumerate(new_results):
            if not result.content:
                continue
            print(f"      [{i+1}/{len(new_results)}] Extracting: {result.title[:50]}...")
            facts = await self._extractor.extract_facts(result.content, result)
            for fact_data in facts:
                source = Source(
                    url=result.url,
                    title=result.title,
                    search_query=query
                )
                fact_id = self.memory.add_fact(
                    content=fact_data["fact"],
                    fact_type=FactType.SEARCH_RESULT,
                    source=source,
                    validated=False,
                    metadata={"extraction_source": fact_data["source"], "task_id": task_id}
                )
                all_facts.append({"id": fact_id, "fact": fact_data["fact"], "source": str(source)})

        print(f"      ✅ {len(all_facts)} facts extracted")
        return {
            "query": query,
            "task_id": task_id,
            "sources": [{"title": r.title, "url": r.url} for r in new_results],
            "facts": all_facts,
        }

    async def _think_step(self, question: str, facts_so_far: list) -> Dict:
        """
        LLM reflection: review facts, identify gaps, decide to continue/stop.
        Returns: {sufficient: bool, gaps: str, follow_up_queries: List[str]}
        """
        facts_summary = "\n".join(
            f"- {f.content[:150]}" for f in facts_so_far[:20]
        )

        prompt = f"""You are a research coordinator reviewing search progress.

Original question: {question}

Facts found so far ({len(facts_so_far)} total):
{facts_summary}

Answer these questions:
1. What key information has been found?
2. What important aspects are still missing or underexplored?
3. Is coverage sufficient to answer the question? (yes/no)
4. If no, list 2-3 specific follow-up search queries to fill the gaps.

Output JSON only:
{{
  "sufficient": true,
  "gaps": "description of what is missing",
  "follow_up_queries": ["query1", "query2"]
}}"""

        try:
            response = await self._llm_query(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "sufficient": bool(data.get("sufficient", False)),
                    "gaps": str(data.get("gaps", "")),
                    "follow_up_queries": [q for q in data.get("follow_up_queries", []) if q],
                }
        except Exception as e:
            print(f"      ⚠️  Think step parse error: {e}")

        # Fallback: treat as sufficient so we don't loop forever on LLM errors
        return {"sufficient": True, "gaps": "", "follow_up_queries": []}

    async def _llm_query(self, prompt: str) -> str:
        """Query the LLM using a temporary BaseAgent instance."""
        try:
            agent = BaseAgent(
                agent_id="deep_think",
                agent_type=AgentType.WORKER,
                model_name="phi4:14b",
                system_prompt="You are a research coordinator. Output JSON only when asked.",
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  LLM error in deep think: {e}")
            return ""


# Quick test
if __name__ == "__main__":
    import asyncio
    from shared_memory import SharedMemory
    
    async def test_parallel_search():
        print("Testing Parallel Search System...\n")
        
        memory = SharedMemory()
        coordinator = ParallelSearchCoordinator(memory, max_concurrent=2)
        
        # Test queries
        queries = [
            "thrust force required for liftoff physics",
            "weight calculation lbm to lbf conversion",
            "Earth gravity acceleration constant"
        ]
        
        results = await coordinator.search_all(queries)
        
        print("\n" + "="*70)
        print("📊 Search Results Summary:")
        print("="*70)
        
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Sources: {len(result['sources'])}")
            print(f"Facts extracted: {len(result['facts'])}")
            
            if result['facts']:
                print("Sample facts:")
                for fact in result['facts'][:3]:
                    print(f"   • {fact['fact'][:80]}...")
        
        print("\n" + "="*70)
        memory.print_state()
    
    # Run test
    # Uncomment to test (requires internet connection)
    # asyncio.run(test_parallel_search())
    
    print("\n✅ Parallel Search System ready!")
    print("   Uncomment test to run live search test")

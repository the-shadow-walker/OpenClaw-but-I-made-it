"""
Writer Agent - Qwen2.5-14B
Converts verified results into human language

CRITICAL RULES:
- NO new facts
- NO math
- NO sources (already cited)
- ONLY explanation of verified results
"""

import sys
import os
import re as _re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
from shared_memory import SharedMemory, ComputedResult, Fact, FactType
from datetime import datetime
from typing import List, Dict, Optional


class WriterAgent(BaseAgent):
    """
    Writer Agent using Qwen2.5-14B
    
    Job:
    - Convert verified results into clear language
    - Explain the solution process
    - Make it understandable
    
    Does NOT:
    - Generate new facts
    - Do calculations
    - Add citations (already done)
    - Change results
    """
    
    def __init__(self, memory: SharedMemory, agent_id: str = "writer"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.WORKER,
            model_name="qwen2.5:14b",  # Good at clear writing
            system_prompt="""You are a TECHNICAL WRITER. Explain verified results clearly.

Your ONLY job:
1. Take verified computational results
2. Explain them in clear, natural language
3. Make the solution process understandable
4. Keep it concise and accurate

STRICT RULES:
❌ Do NOT generate new facts
❌ Do NOT do calculations
❌ Do NOT cite sources (already done)
❌ Do NOT add assumptions
✅ DO explain results clearly
✅ DO describe the solution process
✅ DO make it easy to understand

Format your response as a clear answer to the user's question.
Start with the direct answer, then explain how it was determined.

Keep response under 400 words. Be clear and concise."""
        )
        
        self.memory = memory
    
    async def write_answer(self, original_question: str) -> str:
        """
        Write final answer based on verified results.
        
        Args:
            original_question: User's original question
            
        Returns:
            Human-readable answer
        """
        print(f"\n✍️ [{self.agent_id}] Writing final answer...")
        
        # Gather all verified information
        context = self._prepare_context(original_question)
        
        if not context['has_results']:
            print("   ⚠️ No results to write about")
            return "Unable to compute answer - insufficient information."
        
        prompt = f"""Write a clear answer to this question based on verified results:

QUESTION:
{original_question}

VERIFIED RESULTS:
{context['results_summary']}

COMPUTATION PROCESS:
{context['process_summary']}

ASSUMPTIONS:
{context['assumptions_summary']}

Write a clear, direct answer explaining these results."""

        answer = await self.query_llm(prompt, stream=True)
        
        print(f"   ✅ Answer written ({len(answer)} chars)")

        return answer

    async def write_research_answer(self, question: str) -> str:
        """
        Write a research-synthesized answer for theoretical questions.

        Reads validated facts first; falls back to SEARCH_RESULT facts if none
        are validated. Formats up to 20 facts with source attribution and
        calls the LLM with a synthesis prompt.

        Args:
            question: User's original question

        Returns:
            Synthesized answer string
        """
        print(f"\n✍️ [{self.agent_id}] Writing research answer...")

        # Prefer validated facts; fall back to raw search results
        facts = self.memory.get_facts(validated_only=True)
        if not facts:
            facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)

        if not facts:
            print("   ⚠️ No facts available for research answer")
            return "Unable to answer - no research results available."

        # Format up to 20 facts with source attribution
        fact_lines = []
        for i, fact in enumerate(facts[:20], 1):
            source_str = str(fact.source) if fact.source else "Unknown source"
            fact_lines.append(f"[{i}] {fact.content}\n    Source: {source_str}")

        facts_block = "\n\n".join(fact_lines)

        prompt = f"""Synthesize a comprehensive, well-structured answer to the following question based on the research findings below.

QUESTION:
{question}

RESEARCH FINDINGS:
{facts_block}

Write a clear, informative answer that:
1. Directly addresses the question
2. Integrates key findings from the research
3. Notes areas of agreement or uncertainty across sources
4. Cites sources where relevant (use [N] notation)

Keep the response under 600 words."""

        answer = await self.query_llm(prompt, stream=True)

        print(f"   ✅ Research answer written ({len(answer)} chars)")
        return answer

    async def write_deep_markdown(self, question: str) -> str:
        """
        Generate a detailed, richly formatted Markdown research report.

        Structure:
          # <question>          (metadata header)
          ---
          ## Summary            (LLM-written)
          ## Key Findings       (LLM-written, grouped by topic with ### subsections)
          ## Discussion         (LLM-written — uncertainties, gaps)
          ---
          ## Sources            (programmatic numbered list with URLs)
          ### Search Queries    (programmatic)
        """
        print(f"\n✍️ [{self.agent_id}] Generating deep Markdown report...")

        # ── Collect facts & unique sources ───────────────────────────────
        facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)
        if not facts:
            facts = self.memory.get_facts()

        # Build ordered unique-source list (preserves first-seen order)
        sources = []
        seen_urls: set = set()
        for fact in facts:
            if fact.source and fact.source.url and fact.source.url not in seen_urls:
                sources.append(fact.source)
                seen_urls.add(fact.source.url)

        # Source index for inline citation numbers
        source_idx = {src.url: i + 1 for i, src in enumerate(sources)}

        # ── Build facts block for LLM ─────────────────────────────────────
        fact_lines = []
        for fact in facts[:30]:
            cite = ""
            if fact.source and fact.source.url and fact.source.url in source_idx:
                cite = f" [{source_idx[fact.source.url]}]"
            fact_lines.append(f"- {fact.content}{cite}")
        facts_block = "\n".join(fact_lines) if fact_lines else "(no facts collected)"

        # Validated / assumed context
        assumptions_text = (
            "\n".join(f"- {a}" for a in self.memory.assumptions)
            if self.memory.assumptions else "None"
        )
        contradictions_text = (
            "\n".join(
                f"- Fact {c['fact_1']} vs {c['fact_2']}: {c['reason']}"
                for c in self.memory.contradictions
            )
            if self.memory.contradictions else "None detected"
        )

        # ── LLM prompt ────────────────────────────────────────────────────
        prompt = f"""Write a comprehensive Markdown research report body for the question below.

QUESTION: {question}

RESEARCH FINDINGS (with citation numbers in brackets):
{facts_block}

ASSUMPTIONS:
{assumptions_text}

CONTRADICTIONS / CONFLICTS:
{contradictions_text}

Instructions:
- Write a **## Summary** section first (2-3 informative paragraphs).
- Then **## Key Findings** with 3-6 subsections using **### [Topic Name]** headers.
  Group related facts together under each subsection.
  Under each subsection add a blockquote (> ...) callout with the single most important insight.
  Use inline citation numbers like [1] or [2] where relevant.
- Then **## Discussion** — note uncertainties, conflicting information, or important gaps.
- Use bold text for key terms, bullet lists for multi-point items, and tables where it helps clarity.
- Be detailed and thorough, not brief. Aim for depth.
- Do NOT include a document title — it will be added automatically.
- Do NOT include a ## Sources section — it will be appended automatically."""

        body = await self.query_llm(prompt, stream=True)

        # ── Assemble the full document ────────────────────────────────────
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M")

        lines: List[str] = [
            f"# {question}",
            "",
            f"> **Generated:** {date_str} &nbsp;|&nbsp; "
            f"**Sources:** {len(sources)} &nbsp;|&nbsp; "
            f"**Facts collected:** {len(facts)}",
            "",
            "---",
            "",
            body.strip(),
            "",
            "---",
            "",
        ]

        # Sources appendix
        if sources:
            lines.append("## Sources")
            lines.append("")
            for i, src in enumerate(sources, 1):
                title = src.title or src.url
                url = src.url
                query_note = f" *(query: {src.search_query})*" if src.search_query else ""
                lines.append(f"{i}. [{title}]({url}){query_note}")
            lines.append("")

        # Search queries appendix
        search_queries = sorted({
            f.source.search_query
            for f in facts
            if f.source and f.source.search_query
        })
        if search_queries:
            lines.append("### Search Queries Used")
            lines.append("")
            for q in search_queries:
                lines.append(f"- `{q}`")
            lines.append("")

        print(f"   ✅ Markdown report generated ({sum(len(l) for l in lines)} chars)")
        return "\n".join(lines)

    def _prepare_context(self, question: str) -> Dict:
        """Prepare context for writing"""
        
        # Get computed results
        results = self.memory.get_computed_results()
        
        # Get validated facts
        facts = self.memory.get_facts(validated_only=True)
        
        # Get equations
        equations = self.memory.get_equations(validated_only=True)
        
        # Get assumptions
        assumptions = self.memory.assumptions
        
        # Build summaries
        results_summary = self._format_results(results)
        process_summary = self._format_process(equations, results)
        assumptions_summary = self._format_assumptions(assumptions)
        
        return {
            'has_results': len(results) > 0,
            'results_summary': results_summary,
            'process_summary': process_summary,
            'assumptions_summary': assumptions_summary,
            'num_results': len(results),
            'num_facts': len(facts),
            'num_equations': len(equations)
        }
    
    def _format_results(self, results: List[ComputedResult]) -> str:
        """Format computed results for context"""
        
        if not results:
            return "No results computed"
        
        lines = []
        for result in results:
            var_def = self.memory.get_variable(result.variable)
            meaning = var_def.meaning if var_def else result.variable
            
            lines.append(
                f"{meaning} ({result.variable}) = {result.value} {result.unit}"
            )
        
        return '\n'.join(lines)
    
    def _format_process(self, equations: List, results: List[ComputedResult]) -> str:
        """Format computation process for context"""
        
        lines = []
        
        if equations:
            lines.append("Equations used:")
            for eq in equations:
                lines.append(f"  - {eq.symbolic_form}")
        
        if results:
            lines.append("\nCalculations:")
            for result in results:
                if result.numeric_substitution:
                    lines.append(f"  - {result.numeric_substitution}")
                    lines.append(f"    = {result.value} {result.unit}")
        
        return '\n'.join(lines) if lines else "No process to describe"
    
    def _format_assumptions(self, assumptions: List[str]) -> str:
        """Format assumptions for context"""
        
        if not assumptions:
            return "No assumptions made"
        
        return '\n'.join(f"  - {a}" for a in assumptions)
    
    async def write_summary(self) -> str:
        """
        Write a summary of the entire solution process.
        
        Returns:
            Summary text
        """
        print(f"\n✍️ [{self.agent_id}] Writing process summary...")
        
        # Get memory state
        facts = self.memory.get_facts(validated_only=True)
        variables = self.memory.variables
        equations = self.memory.get_equations(validated_only=True)
        results = self.memory.get_computed_results()
        
        prompt = f"""Write a summary of the research and computation process:

VALIDATED FACTS: {len(facts)}
VARIABLES IDENTIFIED: {len(variables)}
EQUATIONS USED: {len(equations)}
RESULTS COMPUTED: {len(results)}

Key Results:
{self._format_results(results)}

Process:
{self._format_process(equations, results)}

Write a brief summary (under 200 words) explaining how the answer was determined."""

        summary = await self.query_llm(prompt, stream=False)
        
        return summary
    
    async def explain_result(self, variable: str) -> str:
        """
        Explain a specific result in detail.
        
        Args:
            variable: Variable to explain (e.g., "F")
            
        Returns:
            Explanation text
        """
        print(f"\n✍️ [{self.agent_id}] Explaining result for {variable}...")
        
        # Find the result
        result = None
        for r in self.memory.get_computed_results():
            if r.variable == variable:
                result = r
                break
        
        if not result:
            return f"No result found for {variable}"
        
        # Get variable definition
        var_def = self.memory.get_variable(variable)
        
        # Get equation used
        equation = None
        for eq in self.memory.get_equations():
            if eq.symbolic_form == result.method:
                equation = eq
                break
        
        prompt = f"""Explain this computation result in detail:

VARIABLE: {variable}
MEANING: {var_def.meaning if var_def else 'Unknown'}
RESULT: {result.value} {result.unit}

METHOD: {result.method}
CALCULATION: {result.numeric_substitution or 'N/A'}

Write a clear explanation (under 150 words) of what this result means and how it was computed."""

        explanation = await self.query_llm(prompt, stream=False)
        
        return explanation


# Quick test
if __name__ == "__main__":
    import asyncio
    from shared_memory import SharedMemory, Source, FactType
    
    async def test_writer():
        print("Testing Writer Agent...\n")
        
        memory = SharedMemory()
        
        # Setup a complete solution
        memory.add_open_question("How much thrust is needed to lift a 5000 kg object?")
        memory.add_assumption("Assuming Earth gravity (9.81 m/s²)")
        
        # Add variables
        memory.add_variable("F", "thrust force", "N")
        memory.add_variable("m", "mass", "kg", value=5000.0)
        memory.add_variable("g", "gravity", "m/s^2", value=9.81)
        
        # Add equation
        eq_id = memory.add_equation(
            symbolic_form="F = m * g",
            variables_used=["F", "m", "g"],
            domain="physics"
        )
        memory.validate_equation(eq_id)
        
        # Add result
        memory.add_computed_result(
            variable="F",
            value=49050.0,
            unit="N",
            method="F = m * g",
            symbolic_form="F = m * g",
            numeric_substitution="F = 5000 * 9.81"
        )
        
        # Add some facts
        source = Source(agent_id="consensus")
        memory.add_fact(
            "Thrust must equal weight for liftoff",
            FactType.VALIDATED_FACT,
            source,
            validated=True
        )
        
        # Test writer
        writer = WriterAgent(memory)
        
        question = "How much thrust force is needed to lift a 5000 kg object on Earth?"
        
        answer = await writer.write_answer(question)
        
        print("\n" + "="*70)
        print("📝 FINAL ANSWER:")
        print("="*70)
        print(answer)
        
        print("\n" + "="*70)
        summary = await writer.write_summary()
        print("\n📊 PROCESS SUMMARY:")
        print("="*70)
        print(summary)
        
        print("\n" + "="*70)
        explanation = await writer.explain_result("F")
        print("\n💡 DETAILED EXPLANATION:")
        print("="*70)
        print(explanation)
    
    asyncio.run(test_writer())

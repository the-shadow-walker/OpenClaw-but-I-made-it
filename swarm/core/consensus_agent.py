"""
Consensus/Validation Agent - Phi-4
Compares facts across sources, detects contradictions

CRITICAL RULES:
- NO new facts
- NO math
- NO computation
- ONLY validation and contradiction detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
from shared_memory import SharedMemory, FactType, Fact
import json
import re
from typing import List, Dict, Optional


class ConsensusAgent(BaseAgent):
    """
    Consensus/Validation Agent using Phi-4
    
    Job:
    - Compare facts across sources
    - Detect contradictions
    - Flag uncertainty
    - Decide what's reliable
    
    Does NOT:
    - Generate new facts
    - Do math
    - Make computations
    """
    
    def __init__(self, memory: SharedMemory, agent_id: str = "consensus"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.WORKER,
            model_name="phi4:14b",
            system_prompt="""You are a FACT VALIDATOR. Compare facts and detect contradictions.

Your ONLY job:
1. Review facts from different sources
2. Identify agreements (multiple sources say same thing)
3. Detect contradictions (sources disagree)
4. Flag uncertainties (only one source, or unclear)
5. Mark facts as validated or rejected

STRICT RULES:
❌ Do NOT generate new facts
❌ Do NOT do calculations
❌ Do NOT make assumptions
✅ DO compare existing facts
✅ DO note source agreement/disagreement
✅ DO flag contradictions clearly

Output ONLY JSON:
{
  "validated_facts": [
    {"fact_id": "abc123", "reason": "Confirmed by 3 sources", "confidence": "high"}
  ],
  "rejected_facts": [
    {"fact_id": "def456", "reason": "Contradicted by NASA source"}
  ],
  "contradictions": [
    {
      "fact_ids": ["abc", "def"],
      "issue": "Source A says X, Source B says Y",
      "resolution": "NASA source more authoritative"
    }
  ],
  "uncertain_facts": [
    {"fact_id": "ghi789", "reason": "Only one source, needs verification"}
  ]
}

Keep response under 300 words."""
        )
        
        self.memory = memory
    
    async def validate_all_facts(self) -> Dict:
        """
        Validate all unvalidated facts in memory.
        
        Returns:
            Dict with validation results
        """
        print(f"\n🔍 [{self.agent_id}] Validating facts...")
        
        # Get all search result facts
        facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT, validated_only=False)
        
        if not facts:
            print("   ℹ️ No facts to validate")
            return {
                'validated_facts': [],
                'rejected_facts': [],
                'contradictions': [],
                'uncertain_facts': []
            }
        
        print(f"   📊 Validating {len(facts)} facts...")
        
        # Prepare fact summary for LLM
        fact_summary = self._prepare_fact_summary(facts)
        
        prompt = f"""Review and validate these facts from search results:

{fact_summary}

Compare facts across sources. Look for:
- Agreement (multiple sources confirm)
- Contradictions (sources disagree)
- Uncertainties (single source, unclear)

Provide validation in JSON format."""

        response = await self.query_llm(prompt, stream=False)
        
        # Parse JSON
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            validation = json.loads(response.strip())
            
            # Apply validation to memory
            self._apply_validation(validation)
            
            return validation
            
        except Exception as e:
            print(f"   ⚠️ Failed to parse validation: {e}")
            print(f"   Raw response: {response[:200]}")
            
            # Fallback: validate all facts with low confidence
            return {
                'validated_facts': [
                    {'fact_id': f.id, 'reason': 'Fallback validation', 'confidence': 'low'}
                    for f in facts
                ],
                'rejected_facts': [],
                'contradictions': [],
                'uncertain_facts': []
            }
    
    def _prepare_fact_summary(self, facts: List[Fact]) -> str:
        """Prepare a summary of facts for LLM review"""
        summary_lines = []
        
        for fact in facts:
            summary_lines.append(f"ID: {fact.id}")
            summary_lines.append(f"FACT: {fact.content}")
            summary_lines.append(f"SOURCE: {fact.source}")
            summary_lines.append("")
        
        return '\n'.join(summary_lines)
    
    def _apply_validation(self, validation: Dict):
        """Apply validation results to memory"""
        
        # Validate facts
        for item in validation.get('validated_facts', []):
            fact_id = item.get('fact_id')
            reason = item.get('reason', 'Validated by consensus')
            confidence = item.get('confidence', 'medium')
            
            if self.memory.validate_fact(fact_id, self.agent_id):
                print(f"   ✅ Validated {fact_id}: {reason} ({confidence} confidence)")
        
        # Reject facts
        for item in validation.get('rejected_facts', []):
            fact_id = item.get('fact_id')
            reason = item.get('reason', 'Rejected by consensus')
            
            self.memory.reject_fact(fact_id, reason, self.agent_id)
        
        # Flag contradictions
        for item in validation.get('contradictions', []):
            fact_ids = item.get('fact_ids', [])
            issue = item.get('issue', 'Contradiction detected')
            
            if len(fact_ids) >= 2:
                self.memory.flag_contradiction(fact_ids[0], fact_ids[1], issue)
        
        # Log uncertain facts
        for item in validation.get('uncertain_facts', []):
            fact_id = item.get('fact_id')
            reason = item.get('reason', 'Uncertain')
            
            print(f"   ⚠️ Uncertain {fact_id}: {reason}")
    
    async def check_consistency(self, fact_ids: List[str]) -> Dict:
        """
        Check if specific facts are consistent with each other.
        
        Args:
            fact_ids: List of fact IDs to check
            
        Returns:
            Dict with consistency results
        """
        print(f"\n🔍 [{self.agent_id}] Checking consistency of {len(fact_ids)} facts...")
        
        # Get facts
        facts = [f for f in self.memory.facts if f.id in fact_ids]
        
        if len(facts) < 2:
            return {
                'consistent': True,
                'reason': 'Not enough facts to compare'
            }
        
        fact_summary = self._prepare_fact_summary(facts)
        
        prompt = f"""Check if these facts are consistent with each other:

{fact_summary}

Are there any contradictions? Are they logically consistent?

Respond with JSON:
{{
  "consistent": true/false,
  "reason": "explanation",
  "contradictions": ["list of issues if any"]
}}"""

        response = await self.query_llm(prompt, stream=False)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response.strip())
            
            if not result.get('consistent', True):
                print(f"   ⚠️ Inconsistency detected: {result.get('reason')}")
            else:
                print(f"   ✅ Facts are consistent")
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Failed to check consistency: {e}")
            return {
                'consistent': True,
                'reason': 'Could not parse consistency check'
            }
    
    async def resolve_contradiction(self, fact_id_1: str, fact_id_2: str) -> Dict:
        """
        Attempt to resolve a contradiction between two facts.
        
        Args:
            fact_id_1: First fact ID
            fact_id_2: Second fact ID
            
        Returns:
            Dict with resolution
        """
        print(f"\n🔍 [{self.agent_id}] Resolving contradiction: {fact_id_1} vs {fact_id_2}")
        
        # Get facts
        fact1 = None
        fact2 = None
        
        for f in self.memory.facts:
            if f.id == fact_id_1:
                fact1 = f
            elif f.id == fact_id_2:
                fact2 = f
        
        if not fact1 or not fact2:
            return {
                'resolved': False,
                'reason': 'One or both facts not found'
            }
        
        prompt = f"""Resolve this contradiction:

FACT 1 ({fact_id_1}):
{fact1.content}
Source: {fact1.source}

FACT 2 ({fact_id_2}):
{fact2.content}
Source: {fact2.source}

Which is more reliable? Can both be true? Is there a misunderstanding?

Respond with JSON:
{{
  "resolved": true/false,
  "resolution": "explanation",
  "prefer_fact": "{fact_id_1}" or "{fact_id_2}" or "both" or "neither",
  "reason": "why this resolution"
}}"""

        response = await self.query_llm(prompt, stream=False)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            resolution = json.loads(response.strip())
            
            # Apply resolution
            prefer = resolution.get('prefer_fact')
            if prefer == fact_id_1:
                self.memory.reject_fact(fact_id_2, f"Contradicts {fact_id_1}", self.agent_id)
                print(f"   ✅ Resolved: Preferring {fact_id_1}")
            elif prefer == fact_id_2:
                self.memory.reject_fact(fact_id_1, f"Contradicts {fact_id_2}", self.agent_id)
                print(f"   ✅ Resolved: Preferring {fact_id_2}")
            elif prefer == "neither":
                self.memory.reject_fact(fact_id_1, "Part of unresolved contradiction", self.agent_id)
                self.memory.reject_fact(fact_id_2, "Part of unresolved contradiction", self.agent_id)
                print(f"   ⚠️ Both facts rejected")
            else:
                print(f"   ✅ Both facts accepted (not actually contradictory)")
            
            return resolution
            
        except Exception as e:
            print(f"   ⚠️ Failed to resolve: {e}")
            return {
                'resolved': False,
                'reason': str(e)
            }


# Quick test
if __name__ == "__main__":
    import asyncio
    from shared_memory import SharedMemory, Source, FactType
    
    async def test_consensus():
        print("Testing Consensus Agent...\n")
        
        memory = SharedMemory()
        
        # Add some test facts
        source1 = Source(url="https://nasa.gov", title="NASA")
        source2 = Source(url="https://physics.org", title="Physics Org")
        source3 = Source(url="https://random.com", title="Random Blog")
        
        # Add facts that agree
        memory.add_fact(
            "Thrust must equal weight for liftoff",
            FactType.SEARCH_RESULT,
            source1
        )
        
        memory.add_fact(
            "For liftoff, thrust force must equal or exceed weight",
            FactType.SEARCH_RESULT,
            source2
        )
        
        # Add contradicting fact
        memory.add_fact(
            "Thrust must be twice the weight for liftoff",
            FactType.SEARCH_RESULT,
            source3
        )
        
        # Add uncertain fact
        memory.add_fact(
            "Liftoff requires consideration of air resistance",
            FactType.SEARCH_RESULT,
            source3
        )
        
        # Test validation
        consensus = ConsensusAgent(memory)
        
        validation = await consensus.validate_all_facts()
        
        print("\n" + "="*70)
        print("📊 Validation Results:")
        print("="*70)
        print(json.dumps(validation, indent=2))
        
        print("\n" + "="*70)
        memory.print_state()
    
    asyncio.run(test_consensus())

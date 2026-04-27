"""
Independent Verifier Agent
Uses Phi-4 14B - assumes answer is WRONG until proven right
RECOMPUTES from scratch, does NOT restate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
import json
import re


class VerifierAgent(BaseAgent):
    """
    Independent verification agent using Phi-4.
    
    CORE PRINCIPLE: Assume answer is WRONG
    
    Process:
    1. Read question (NOT the answer yet)
    2. Solve independently
    3. Compare with provided answer
    4. PASS only if both match
    
    This is NOT a "checker" - it's a second solver.
    """
    
    def __init__(self, agent_id: str = "verifier"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.WORKER,
            model_name=os.getenv("SWARM_MODEL_DEFAULT", "batiai/qwen3.6-27b:iq4"),  # Unified model
            system_prompt="""You are an INDEPENDENT VERIFIER. Your job is to RECOMPUTE, not review.

CRITICAL MINDSET: Assume the provided answer is WRONG until you prove it's right.

Process:
1. Read the question
2. Solve it independently (ignore the provided answer)
3. Compare YOUR answer with the provided answer
4. Return PASS or FAIL

RULES FOR PHYSICS/MATH:
✅ Recompute from scratch
✅ Check unit conversions (especially lbm vs lbf)
✅ Verify dimensional consistency  
✅ State YOUR assumptions
❌ Do NOT trust the provided answer
❌ Do NOT restate their work
❌ Do NOT be polite - be CORRECT

For US customary units:
- gc = 32.174 lbm·ft/(lbf·s²)
- weight_lbf = mass_lbm × (g/gc) = mass_lbm × 1.0 on Earth
- If answer says "5000 lbm × 32.174 = 161,870 lbf" → FAIL (missing gc)

Output format (JSON):
{
  "verdict": "PASS" | "FAIL",
  "your_answer": "your computed result",
  "their_answer": "what they claimed",
  "reason": "why it passes or fails",
  "assumptions": ["your assumptions"]
}

Keep response under 150 words."""
        )
    
    async def verify(self, question: str, provided_answer: str) -> dict:
        """
        Independently verify an answer.
        
        Args:
            question: Original question
            provided_answer: Answer to verify
            
        Returns:
            dict with verdict
        """
        prompt = f"""Independently solve this question, then compare with the provided answer.

Question: {question}

Provided Answer (assume this is WRONG):
{provided_answer}

Recompute independently and return verdict.
Respond ONLY with JSON."""
        
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
            
            data = json.loads(response.strip())
            
            # Ensure verdict is present
            if "verdict" not in data:
                data["verdict"] = "FAIL"
                data["reason"] = "Could not determine verdict"
            
            return data
            
        except Exception as e:
            print(f"⚠️ Failed to parse verification: {e}")
            return {
                "verdict": "FAIL",
                "your_answer": "parsing error",
                "their_answer": provided_answer,
                "reason": f"Could not parse verifier response: {e}",
                "assumptions": []
            }
    
    async def quick_check(self, text: str) -> str:
        """
        Quick sanity check of text.
        Returns: "APPROVED" or "NEEDS_FIX: reason"
        
        This is for the critic role.
        """
        prompt = f"""Quick check: Is this answer correct?

{text}

Reply in ONE sentence:
"APPROVED" OR "NEEDS_FIX: [specific issue]"

Your response:"""
        
        response = await self.query_llm(prompt, stream=False)
        
        # Clean up response
        response = response.strip()
        if len(response) > 200:
            response = response[:200]
        
        return response


# Test placeholder
if __name__ == "__main__":
    print("Verifier Agent loaded.")
    print("Use verify(question, answer) to check answers.")

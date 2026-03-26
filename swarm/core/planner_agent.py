"""
Planner Agent - Phi-4
Breaks user question into answerable sub-questions

CRITICAL RULES:
- NO facts
- NO math
- NO conclusions
- ONLY questions and task structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
from shared_memory import SharedMemory, Source, FactType
import json
import re
from datetime import datetime
from typing import List, Dict, Optional


class PlannerAgent(BaseAgent):
    """
    Planner Agent using Phi-4
    
    Job:
    - Break user question into sub-questions
    - Identify what needs to be searched
    - Determine if math/physics is required
    - Write assumptions
    
    Does NOT:
    - State facts
    - Do calculations
    - Draw conclusions
    """
    
    def __init__(self, memory: SharedMemory, agent_id: str = "planner"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.WORKER,
            model_name="phi4:14b",
            system_prompt="""You are a PLANNER. Break down questions into research tasks.

Your ONLY job:
1. Analyze the question
2. Break it into 3-5 specific sub-questions
3. Identify what must be searched/researched
4. Note if physics/math calculations are needed
5. State any assumptions

STRICT RULES:
❌ Do NOT state facts ("weight = mg") 
❌ Do NOT do math ("5000 × 9.81 = 49,050")
❌ Do NOT draw conclusions
✅ DO ask questions ("What is the weight of X?")
✅ DO identify unknowns ("Need to find: force required")
✅ DO note assumptions ("Assuming Earth gravity")

Output ONLY JSON:
{
  "sub_questions": [
    "What is the weight of the object?",
    "What force is needed to lift it?"
  ],
  "search_needed": ["weight calculation formula", "liftoff physics"],
  "needs_math": true,
  "needs_physics": true,
  "assumptions": ["Assuming Earth gravity", "Ignoring air resistance"],
  "unknowns": ["thrust force required"],
  "domain": "physics"
}

Keep response under 200 words."""
        )
        
        self.memory = memory
    
    async def plan(self, question: str) -> Dict:
        """
        Break down a question into a plan.
        
        Args:
            question: User's question
            
        Returns:
            Dict with sub_questions, search_needed, etc.
        """
        print(f"\n📋 [{self.agent_id}] Planning for: {question[:60]}...")
        
        self.memory.log_agent_activity(self.agent_id, "planning", question)
        
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""Break down this question into a research plan.

Today's date: {today}
When generating search queries, include the current year ({datetime.now().year}) where recency matters.

Question: {question}

Provide your plan in JSON format."""

        response = await self.query_llm(prompt, stream=False)
        
        # Parse JSON
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            plan = json.loads(response.strip())
            
            # Validate structure
            if 'sub_questions' not in plan:
                raise ValueError("Missing sub_questions")
            
            # Log to memory
            self._log_plan_to_memory(question, plan)
            
            return plan
            
        except Exception as e:
            print(f"   ⚠️ Failed to parse plan: {e}")
            print(f"   Raw response: {response[:200]}")
            
            # Fallback plan
            plan = {
                'sub_questions': [question],
                'search_needed': [question],
                'needs_math': True,
                'needs_physics': True,
                'assumptions': [],
                'unknowns': [],
                'domain': 'general'
            }
            
            self._log_plan_to_memory(question, plan)
            return plan
    
    def _log_plan_to_memory(self, question: str, plan: Dict):
        """Log the plan to shared memory"""
        
        # Add original question
        self.memory.add_open_question(question)
        
        # Add sub-questions
        for sub_q in plan.get('sub_questions', []):
            self.memory.add_open_question(sub_q)
            print(f"   ❓ {sub_q}")
        
        # Add assumptions
        for assumption in plan.get('assumptions', []):
            self.memory.add_assumption(assumption)
        
        # Add search queries as facts
        for search_query in plan.get('search_needed', []):
            self.memory.add_fact(
                content=f"Need to search: {search_query}",
                fact_type=FactType.OPEN_QUESTION,
                source=Source(agent_id=self.agent_id)
            )
        
        print(f"   📊 Plan: {len(plan['sub_questions'])} questions, "
              f"{len(plan.get('search_needed', []))} searches, "
              f"Math: {plan.get('needs_math', False)}")
    
    async def refine_plan(self, current_plan: Dict, new_info: str) -> Dict:
        """
        Refine the plan based on new information.
        
        Args:
            current_plan: Current plan dict
            new_info: New information to incorporate
            
        Returns:
            Updated plan
        """
        print(f"\n🔄 [{self.agent_id}] Refining plan with new info...")
        
        prompt = f"""Refine this research plan based on new information:

Current Plan:
{json.dumps(current_plan, indent=2)}

New Information:
{new_info}

Should the plan change? Add/remove questions? Update assumptions?

Provide refined plan in JSON format."""

        response = await self.query_llm(prompt, stream=False)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            refined_plan = json.loads(response.strip())
            
            print(f"   ✅ Plan refined")
            return refined_plan
            
        except Exception as e:
            print(f"   ⚠️ Refinement failed, keeping current plan: {e}")
            return current_plan


# Quick test
if __name__ == "__main__":
    import asyncio
    from shared_memory import SharedMemory
    
    async def test_planner():
        print("Testing Planner Agent...\n")
        
        memory = SharedMemory()
        planner = PlannerAgent(memory)
        
        # Test question
        question = "How much thrust force is needed to lift a 5000 lbm tungsten cube off the ground?"
        
        plan = await planner.plan(question)
        
        print("\n📋 Plan Generated:")
        print(json.dumps(plan, indent=2))
        
        print("\n" + "="*70)
        memory.print_state()
        
        # Test refinement
        print("\n" + "="*70)
        new_info = "Weight of object is 5000 lbf on Earth"
        refined = await planner.refine_plan(plan, new_info)
        
        print("\n📋 Refined Plan:")
        print(json.dumps(refined, indent=2))
    
    asyncio.run(test_planner())

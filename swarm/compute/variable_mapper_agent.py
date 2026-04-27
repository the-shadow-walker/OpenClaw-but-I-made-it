"""
Variable Mapper Agent - Phi-4
Identifies variables needed to solve the problem

CRITICAL RULES:
- Maps symbols to meanings
- Defines units
- Identifies known vs unknown
- NO equations yet
- NO math
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
from shared_memory import SharedMemory, Source, FactType
import json
import re
from typing import List, Dict, Optional


class VariableMapperAgent(BaseAgent):
    """
    Variable Mapper Agent using Phi-4
    
    Job:
    - Identify variables needed
    - Map symbols to meanings
    - Define units
    - Mark known vs unknown
    
    Does NOT:
    - Create equations
    - Do calculations
    """
    
    def __init__(self, memory: SharedMemory, agent_id: str = "variable_mapper"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.WORKER,
            model_name=os.getenv("SWARM_MODEL_DEFAULT", "batiai/qwen3.6-27b:iq4"),
            system_prompt="""You are a VARIABLE MAPPER. Identify and define variables for physics/math problems.

Your ONLY job:
1. Identify what variables are needed
2. Assign symbols (F, m, g, etc.)
3. Define what each symbol means
4. Specify units
5. Mark which are known vs unknown

STRICT RULES:
❌ Do NOT write equations ("F = m × g")
❌ Do NOT do calculations
❌ Do NOT solve anything
✅ DO map symbols to meanings ("F = thrust force")
✅ DO specify units ("F in Newtons")
✅ DO identify knowns and unknowns

Example output:
{
  "variables": {
    "F": {
      "meaning": "thrust force required",
      "unit": "N",
      "known": false,
      "notes": "This is what we're solving for"
    },
    "m": {
      "meaning": "mass of object",
      "unit": "kg",
      "known": true,
      "value": 5000,
      "notes": "Given in problem as 5000 lbm, needs conversion"
    },
    "g": {
      "meaning": "gravitational acceleration",
      "unit": "m/s^2",
      "known": true,
      "value": 9.81,
      "notes": "Standard Earth gravity"
    }
  },
  "unit_system": "SI",
  "conversions_needed": ["5000 lbm to kg"]
}

Keep response under 250 words. Output ONLY JSON."""
        )
        
        self.memory = memory
    
    async def map_variables(self, validated_facts: Optional[List[str]] = None) -> Dict:
        """
        Map variables based on validated facts and the problem.
        
        Args:
            validated_facts: List of fact IDs to consider (if None, use all validated)
            
        Returns:
            Dict with variable mappings
        """
        print(f"\n🔤 [{self.agent_id}] Mapping variables...")
        
        # Get validated facts
        if validated_facts:
            facts = [f for f in self.memory.facts if f.id in validated_facts]
        else:
            facts = self.memory.get_facts(validated_only=True)
        
        # Get open questions (to understand what we're solving)
        questions = self.memory.open_questions
        assumptions = self.memory.assumptions
        
        # Prepare context
        context = self._prepare_context(facts, questions, assumptions)
        
        prompt = f"""Based on this problem context, identify and map all variables needed:

{context}

Provide variable mapping in JSON format as shown in system prompt."""

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
            
            mapping = json.loads(response.strip())
            
            # Add variables to memory
            self._add_variables_to_memory(mapping)
            
            return mapping
            
        except Exception as e:
            print(f"   ⚠️ Failed to parse variable mapping: {e}")
            print(f"   Raw response: {response[:200]}")
            
            # Return empty mapping
            return {
                'variables': {},
                'unit_system': 'SI',
                'conversions_needed': []
            }
    
    def _prepare_context(self, facts: List, questions: List[str], assumptions: List[str]) -> str:
        """Prepare context for variable mapping"""
        lines = []
        
        # CRITICAL: Include original question with numeric values!
        if self.memory.original_question:
            lines.append("ORIGINAL QUESTION:")
            lines.append(f"  {self.memory.original_question}")
            lines.append("")
        
        lines.append("PROBLEM QUESTIONS:")
        for q in questions:
            lines.append(f"  - {q}")
        
        lines.append("\nVALIDATED FACTS:")
        for fact in facts[:10]:  # Limit to 10 most relevant
            lines.append(f"  - {fact.content}")
        
        if assumptions:
            lines.append("\nASSUMPTIONS:")
            for a in assumptions:
                lines.append(f"  - {a}")
        
        return '\n'.join(lines)
    
    def _add_variables_to_memory(self, mapping: Dict):
        """Add variables to shared memory"""
        
        variables = mapping.get('variables', {})
        
        for symbol, var_info in variables.items():
            meaning = var_info.get('meaning', '')
            unit = var_info.get('unit')
            value = var_info.get('value')
            
            self.memory.add_variable(
                symbol=symbol,
                meaning=meaning,
                unit=unit,
                value=value
            )
        
        # Log conversions needed
        conversions = mapping.get('conversions_needed', [])
        if conversions:
            print(f"   ⚠️ Unit conversions needed:")
            for conv in conversions:
                print(f"      - {conv}")
                
                # Add as open question
                self.memory.add_open_question(f"Convert: {conv}")
        
        # CRITICAL FIX: If no variables have values, try to extract from original question
        has_values = any(var_info.get('value') is not None for var_info in variables.values())
        
        if not has_values and self.memory.original_question:
            print(f"   ⚠️ LLM failed to extract values, trying regex fallback...")
            self._extract_values_from_question()
    
    def _extract_values_from_question(self):
        """
        Fallback: Extract numeric values directly from the original question using regex.
        This is a deterministic fallback when the LLM fails.
        """
        import re
        
        question = self.memory.original_question
        
        # Pattern to match numbers with units like "10000 N", "5000 kg", "9.81 m/s^2"
        # This matches: number (with optional comma/decimal) followed by optional space and unit
        pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*([A-Za-z]+(?:/[A-Za-z]+)?(?:\^?\d+)?)?'
        
        matches = re.findall(pattern, question)
        
        print(f"      [DEBUG] Found {len(matches)} number+unit pairs in question")
        
        # Common physics variable patterns in questions
        value_patterns = [
            (r'(?:crane|lift).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(N|newton)', 'F_crane', 'crane lifting force'),
            (r'(?:object|weighs?).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(N|newton|kg|pound)', 'F_weight', 'object weight'),
            (r'(?:mass).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(kg|pound|lbm)', 'm', 'mass'),
            (r'(?:gravity|g).*?(\d+(?:\.\d+)?)\s*(m/s|ft/s)', 'g', 'gravitational acceleration'),
        ]
        
        extracted_count = 0
        for pattern, var_symbol, var_meaning in value_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(',', '')
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                try:
                    value = float(value_str)
                    
                    # Update the variable if it exists
                    if var_symbol in self.memory.variables:
                        var = self.memory.variables[var_symbol]
                        var.value = value
                        var.known = True
                        if unit and not var.unit:
                            var.unit = unit
                        print(f"      ✓ Extracted: {var_symbol} = {value} {unit or var.unit}")
                        extracted_count += 1
                    else:
                        # Create new variable
                        self.memory.add_variable(
                            symbol=var_symbol,
                            meaning=var_meaning,
                            unit=unit or 'dimensionless',
                            value=value
                        )
                        print(f"      ✓ Created: {var_symbol} = {value} {unit}")
                        extracted_count += 1
                        
                except ValueError:
                    pass
        
        if extracted_count > 0:
            print(f"   ✓ Fallback extraction found {extracted_count} values")
        else:
            print(f"   ⚠️ Fallback extraction found no values")
    
    async def validate_variable_consistency(self) -> Dict:
        """
        Check if variable definitions are dimensionally consistent.
        
        Returns:
            Dict with consistency check results
        """
        print(f"\n🔍 [{self.agent_id}] Checking variable consistency...")
        
        variables = self.memory.variables
        
        if not variables:
            return {
                'consistent': True,
                'issues': []
            }
        
        # Prepare variable summary
        var_summary = []
        for symbol, var in variables.items():
            var_summary.append(
                f"{symbol}: {var.meaning} [{var.unit or 'dimensionless'}] "
                f"= {var.value if var.known else 'unknown'}"
            )
        
        prompt = f"""Check if these variable definitions are dimensionally consistent:

{chr(10).join(var_summary)}

Are units compatible? Any missing dimensions? Any conflicts?

Respond with JSON:
{{
  "consistent": true/false,
  "issues": ["list of any dimensional issues"],
  "suggestions": ["any corrections needed"]
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
                print(f"   ⚠️ Dimensional issues found:")
                for issue in result.get('issues', []):
                    print(f"      - {issue}")
            else:
                print(f"   ✅ Variables are dimensionally consistent")
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Failed to validate consistency: {e}")
            return {
                'consistent': True,
                'issues': []
            }
    
    async def suggest_missing_variables(self, domain: str = "physics") -> List[Dict]:
        """
        Suggest variables that might be missing.
        
        Args:
            domain: Domain of problem (physics, math, chemistry)
            
        Returns:
            List of suggested variables
        """
        print(f"\n💡 [{self.agent_id}] Checking for missing variables...")
        
        current_vars = list(self.memory.variables.keys())
        questions = self.memory.open_questions
        
        prompt = f"""Given this {domain} problem and current variables, are any important variables missing?

Current variables: {', '.join(current_vars)}

Problem questions:
{chr(10).join(questions)}

Respond with JSON:
{{
  "missing_variables": [
    {{
      "symbol": "a",
      "meaning": "acceleration",
      "unit": "m/s^2",
      "why_needed": "Required if object is accelerating"
    }}
  ]
}}"""

        response = await self.query_llm(prompt, stream=False)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response.strip())
            
            missing = result.get('missing_variables', [])
            
            if missing:
                print(f"   💡 Suggested {len(missing)} additional variables:")
                for var in missing:
                    print(f"      - {var['symbol']}: {var['meaning']} ({var['why_needed']})")
            else:
                print(f"   ✅ No missing variables detected")
            
            return missing
            
        except Exception as e:
            print(f"   ⚠️ Failed to suggest variables: {e}")
            return []


# Quick test
if __name__ == "__main__":
    import asyncio
    from shared_memory import SharedMemory, Source, FactType
    
    async def test_variable_mapper():
        print("Testing Variable Mapper Agent...\n")
        
        memory = SharedMemory()
        
        # Add some context
        memory.add_open_question("How much thrust is needed to lift a 5000 lbm tungsten cube?")
        memory.add_assumption("Assuming Earth gravity")
        
        source = Source(agent_id="planner")
        memory.add_fact(
            "Mass of object is 5000 lbm",
            FactType.VALIDATED_FACT,
            source,
            validated=True
        )
        
        memory.add_fact(
            "Thrust must equal weight for liftoff",
            FactType.VALIDATED_FACT,
            source,
            validated=True
        )
        
        # Test variable mapping
        mapper = VariableMapperAgent(memory)
        
        mapping = await mapper.map_variables()
        
        print("\n" + "="*70)
        print("🔤 Variable Mapping:")
        print("="*70)
        print(json.dumps(mapping, indent=2))
        
        print("\n" + "="*70)
        memory.print_state()
        
        # Test consistency check
        print("\n" + "="*70)
        consistency = await mapper.validate_variable_consistency()
        print("\n📊 Consistency Check:")
        print(json.dumps(consistency, indent=2))
        
        # Test missing variable detection
        print("\n" + "="*70)
        missing = await mapper.suggest_missing_variables("physics")
        print("\n💡 Missing Variables:")
        print(json.dumps(missing, indent=2))
    
    asyncio.run(test_variable_mapper())

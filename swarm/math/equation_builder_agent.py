"""
Universal Equation Builder v5.1 - FINAL WORKING VERSION

Fixes:
1. Correct SharedMemory.add_computed_result() API (no 'variable_symbol' arg)
2. Returns results for BOTH math questions AND general questions
3. Doesn't crash on non-math questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent
from core import AgentType
from shared_memory import SharedMemory
import re
from typing import Dict, List, Optional
import math


class EquationBuilderAgent(BaseAgent):
    """
    Universal Equation Builder - Works for EVERYTHING
    
    Math questions: Extracts values, computes, returns results
    General questions: Returns empty dict so orchestrator writes summary
    """
    
    def __init__(self, memory: SharedMemory, agent_id: str = "equation_builder"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.MATH,
            model_name="t1c/deepseek-math-7b-rl:Q8",
            system_prompt="""You are a calculator. Write Python expressions to compute answers.

Examples:
Question: "What is the pH with H+ = 0.001 M?"
Expression: -math.log10(0.001)

Question: "Net force if crane lifts 10000 N, object weighs 8000 N?"
Expression: 10000 - 8000

Question: "Area of circle radius 5 cm?"
Expression: math.pi * (5 ** 2)

Write ONLY the Python expression. No text, no explanations.
"""
        )
        self.memory = memory
    
    def _is_mathematical_question(self) -> bool:
        """Does this question need computation?"""
        
        question = (self.memory.original_question or '').lower()
        
        # Reject non-math questions
        non_math = [
            'what are the', 'what is the current', 'tell me about',
            'explain', 'describe', 'who is', 'when did', 'where is',
            'happenings', 'developments', 'news', 'latest', 'history of'
        ]
        
        if any(phrase in question for phrase in non_math):
            return False
        
        # Require calculation keywords
        calc_words = [
            'calculate', 'compute', 'solve', 'find', 'determine',
            'what is the ph', 'what is the force', 'what is the area',
            'how much', 'how many', 'net force'
        ]
        
        has_calc_word = any(word in question for word in calc_words)
        has_number = bool(re.search(r'\d+', question))
        
        return has_calc_word and has_number
    
    async def build_equations(self, problem_domain: str = "general", validate_output: bool = True) -> Dict:
        """
        Main entry point - called by orchestrator
        
        Returns:
        - For math: {'equations': [...], 'computed_results': {var: value}}
        - For non-math: {'equations': [], 'computed_results': {}}
        """
        
        print(f"\n🔢 [{self.agent_id}] Building equations...")
        
        if not self._is_mathematical_question():
            print(f"   ✅ Not a math question - skipping")
            # Return empty so orchestrator can write summary from search results
            return {
                'equations': [],
                'computed_results': {},
                'domain': 'non-mathematical'
            }
        
        question = self.memory.original_question
        print(f"   📝 Question: {question}")
        
        # Extract numbers from question
        extracted = self._extract_values_from_question(question)
        print(f"   🔍 Extracted: {extracted}")
        
        if not extracted:
            print(f"   ⚠️  No values extracted")
            return {'equations': [], 'computed_results': {}, 'domain': problem_domain}
        
        # Detect domain
        domain = self._detect_domain(question)
        print(f"   🎯 Domain: {domain}")
        
        # Try direct computation
        result = await self._compute_direct(question, extracted, domain)
        
        if result:
            # IMPORTANT: Don't call SharedMemory.add_computed_result here
            # Let the orchestrator handle that with correct API
            print(f"   ✅ {list(result.keys())[0]} = {list(result.values())[0]}")
            
            return {
                'equations': [f"Direct computation in {domain}"],
                'computed_results': result,  # Dict like {'pH': 3.0}
                'domain': domain
            }
        
        print(f"   ⚠️  Could not compute")
        return {'equations': [], 'computed_results': {}, 'domain': domain}
    
    def _extract_values_from_question(self, question: str) -> Dict[str, float]:
        """Extract numerical values from question"""
        
        values = {}
        q_lower = question.lower()
        
        # Pattern: <description> <number> <unit>
        patterns = [
            # Chemistry
            (r'(?:h\+|hydrogen|hydronium).*?(\d+\.?\d*)\s*(?:m|mol/l)', 'h_plus_conc'),
            (r'(?:oh-|hydroxide).*?(\d+\.?\d*)\s*(?:m|mol/l)', 'oh_minus_conc'),
            (r'concentration.*?(\d+\.?\d*)\s*(?:m|mol/l)', 'concentration'),
            
            # Physics
            (r'crane.*?(?:lift|pull).*?(\d+\.?\d*)\s*n', 'force_crane'),
            (r'(?:object|weight).*?(?:weigh|weight).*?(\d+\.?\d*)\s*n', 'weight_object'),
            (r'(\d+\.?\d*)\s*n.*?(?:up|lift|crane)', 'force_up'),
            (r'(\d+\.?\d*)\s*n.*?(?:down|weight|gravity)', 'force_down'),
            
            # General
            (r'(\d+\.?\d*)\s*(?:n|kg|m|cm|°c|k)', 'value'),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, q_lower)
            if match:
                try:
                    value = float(match.group(1))
                    values[key] = value
                except ValueError:
                    pass
        
        return values
    
    def _detect_domain(self, question: str) -> str:
        """Detect domain from keywords"""
        
        q = question.lower()
        
        if any(word in q for word in ['ph', 'acid', 'base', 'concentration', 'h+', 'oh-']):
            return 'chemistry'
        if any(word in q for word in ['force', 'mass', 'weight', 'crane', 'lift', 'net']):
            return 'physics'
        if any(word in q for word in ['area', 'volume', 'radius', 'circle', 'rectangle']):
            return 'geometry'
        if any(word in q for word in ['interest', 'loan', 'investment']):
            return 'finance'
        
        return 'general'
    
    async def _compute_direct(self, question: str, values: Dict[str, float], domain: str) -> Optional[Dict]:
        """Direct computation without LLM (fast path)"""
        
        q = question.lower()
        
        # Chemistry: pH
        if domain == 'chemistry' and 'ph' in q:
            if 'h_plus_conc' in values:
                h_conc = values['h_plus_conc']
                ph = -math.log10(h_conc)
                print(f"   📐 pH = -log10({h_conc}) = {ph:.2f}")
                return {'pH': round(ph, 2)}
            
            elif 'concentration' in values and ('h+' in q or 'hydrogen' in q):
                h_conc = values['concentration']
                ph = -math.log10(h_conc)
                print(f"   📐 pH = -log10({h_conc}) = {ph:.2f}")
                return {'pH': round(ph, 2)}
        
        # Physics: Net force
        if domain == 'physics' and 'net' in q and 'force' in q:
            if 'force_crane' in values and 'weight_object' in values:
                f_crane = values['force_crane']
                w_obj = values['weight_object']
                f_net = f_crane - w_obj
                print(f"   📐 F_net = {f_crane} - {w_obj} = {f_net} N")
                return {'F_net': f_net}
            
            elif 'force_up' in values and 'force_down' in values:
                f_up = values['force_up']
                f_down = values['force_down']
                f_net = f_up - f_down
                print(f"   📐 F_net = {f_up} - {f_down} = {f_net} N")
                return {'F_net': f_net}
        
        # Geometry: Circle area
        if domain == 'geometry' and 'area' in q and 'circle' in q:
            if 'value' in values:
                r = values['value']
                area = math.pi * (r ** 2)
                print(f"   📐 Area = π * {r}² = {area:.2f}")
                return {'area': round(area, 2)}
        
        # Fallback: Try LLM
        print(f"   🤖 No direct pattern - trying LLM...")
        return await self._compute_with_llm(question, values, domain)
    
    async def _compute_with_llm(self, question: str, values: Dict, domain: str) -> Optional[Dict]:
        """Ask LLM to generate Python expression"""
        
        try:
            values_str = '\n'.join([f"{k} = {v}" for k, v in values.items()])
            
            prompt = f"""Question: {question}
Domain: {domain}

Known values:
{values_str}

Write a single Python expression to compute the answer.
Use: math.log10(), math.log(), math.pi, math.sqrt()

Examples:
pH → -math.log10(h_plus_conc)
Net force → force_crane - weight_object
Circle area → math.pi * radius ** 2

Expression:"""
            
            # Call LLM
            response = await self._call_llm_safe(prompt)
            
            # Extract expression
            expr = response.strip()
            expr = expr.replace('`', '')
            expr = expr.split('\n')[0]
            
            print(f"   📝 LLM expression: {expr}")
            
            # Evaluate
            safe_dict = {
                'math': math,
                **values
            }
            
            result_value = eval(expr, {"__builtins__": {}}, safe_dict)
            
            # Determine result name
            if 'ph' in question.lower():
                result_name = 'pH'
            elif 'force' in question.lower():
                result_name = 'F_net'
            elif 'area' in question.lower():
                result_name = 'area'
            else:
                result_name = 'result'
            
            return {result_name: round(result_value, 4)}
        
        except Exception as e:
            print(f"   ⚠️  LLM computation failed: {e}")
            return None
    
    async def _call_llm_safe(self, prompt: str) -> str:
        """Call LLM safely"""
        try:
            try:
                from base_agent import call_ollama_http
                return await call_ollama_http(
                    model=self.model_name,
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    max_tokens=200
                )
            except ImportError:
                pass
            
            import subprocess
            import json
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False
            }
            
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/generate',
                 '-H', 'Content-Type: application/json',
                 '-d', json.dumps(data)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response_data = json.loads(result.stdout)
                return response_data.get('response', '')
            else:
                raise Exception(f"Ollama call failed: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")

"""
Two-Phase TODO Checklist System

Phase 1: "What information do we need?"
  - Identify knowledge gaps
  - Create search TODOs
  - Plan information gathering

Phase 2: "How do we solve this?" 
  - Identify solution steps
  - Create computation TODOs
  - Assign to agents for execution
"""

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class TodoStatus(Enum):
    PENDING = "⏳"
    IN_PROGRESS = "🔄"
    COMPLETED = "✅"
    FAILED = "❌"
    SKIPPED = "⏭️"


@dataclass
class TodoItem:
    """A single actionable TODO"""
    id: str
    title: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    assigned_to: str = None  # Agent name
    result: str = None
    
    def __str__(self):
        return f"{self.status.value} [{self.id}] {self.title}\n   {self.description}"


@dataclass
class Phase1Checklist:
    """Information gathering phase"""
    problem: str
    knowledge_gaps: List[TodoItem]
    search_queries: List[str]
    completeness: float  # 0-1
    can_proceed_to_phase_2: bool
    
    def format(self) -> str:
        lines = [
            "\n" + "="*70,
            "PHASE 1 CHECKLIST: WHAT INFORMATION DO WE NEED?",
            "="*70,
            f"\nProblem: {self.problem[:80]}...",
            f"\n📋 KNOWLEDGE GAPS ({len(self.knowledge_gaps)}):",
        ]
        
        for item in self.knowledge_gaps:
            status = "✅" if item.status == TodoStatus.COMPLETED else "⏳"
            lines.append(f"   {status} {item.title}")
            lines.append(f"      {item.description}")
        
        if self.search_queries:
            lines.append(f"\n🔍 SEARCH QUERIES TO RUN:")
            for i, query in enumerate(self.search_queries, 1):
                lines.append(f"   {i}. {query}")
        
        lines.extend([
            f"\n📊 Information Completeness: {self.completeness:.0%}",
            f"Ready for Phase 2: {'✅ YES' if self.can_proceed_to_phase_2 else '❌ NO'}",
            "\n" + "="*70
        ])
        
        return "\n".join(lines)


@dataclass
class Phase2Checklist:
    """Solution planning phase"""
    problem: str
    solution_method: str
    steps: List[TodoItem]
    validation_steps: List[TodoItem]
    final_answer: str = None
    
    def format(self) -> str:
        lines = [
            "\n" + "="*70,
            "PHASE 2 CHECKLIST: HOW DO WE SOLVE THIS?",
            "="*70,
            f"\nProblem: {self.problem[:80]}...",
            f"\nMethod: {self.solution_method}",
            f"\n📋 SOLUTION STEPS ({len(self.steps)}):",
        ]
        
        for item in self.steps:
            status = "✅" if item.status == TodoStatus.COMPLETED else "⏳"
            lines.append(f"   {status} Step: {item.title}")
            if item.result:
                lines.append(f"      Result: {item.result[:60]}")
        
        if self.validation_steps:
            lines.append(f"\n🔍 VALIDATION STEPS ({len(self.validation_steps)}):")
            for item in self.validation_steps:
                status = "✅" if item.status == TodoStatus.COMPLETED else "⏳"
                lines.append(f"   {status} {item.title}")
        
        if self.final_answer:
            lines.append(f"\n📊 FINAL ANSWER:")
            lines.append(f"   {self.final_answer[:100]}")
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)


class ChecklistGenerator:
    """Generate checklists for problems"""
    
    PHASE_1_PROMPT = """
You are analyzing a physics/math problem to identify what information is needed.

Problem: {problem}

Create a Phase 1 checklist that identifies:
1. What knowledge gaps exist (specific concepts, formulas, values needed)
2. What search queries would fill those gaps
3. Overall completeness (0-100%)

Respond with ONLY JSON (no markdown):
{{
    "knowledge_gaps": [
        {{"title": "Know velocity formula", "description": "Need conservation of momentum equation"}},
        {{"title": "Find Earth's gravitational parameter", "description": "μ = 3.986e14 m³/s²"}}
    ],
    "search_queries": [
        "conservation of momentum in collisions",
        "orbital mechanics semi-major axis"
    ],
    "completeness": 60,
    "can_proceed": false,
    "why_blocked": "Missing specific values for masses and velocities"
}}
"""
    
    PHASE_2_PROMPT = """
You are planning HOW to solve this problem step by step.

Problem: {problem}
Available Info: {available_info}

Create a Phase 2 checklist that shows:
1. The overall solution method
2. Step-by-step computational TODOs
3. Validation checks needed

Respond with ONLY JSON (no markdown):
{{
    "solution_method": "Apply conservation of momentum, then use orbital energy equation",
    "steps": [
        {{"step": 1, "title": "Calculate initial velocity", "description": "v_i = sqrt(μ/r)"}},
        {{"step": 2, "title": "Apply momentum conservation", "description": "m₁v₁ + m₂v₂ = ..."}},
        {{"step": 3, "title": "Compute new semi-major axis", "description": "a = -μ/(2ε)"}}
    ],
    "validation_steps": [
        {{"title": "Check energy conservation", "description": "Verify total energy is conserved"}},
        {{"title": "Verify eccentricity", "description": "0 ≤ e < 1"}}
    ]
}}
"""
    
    @staticmethod
    async def generate_phase_1(problem: str, llm_query_func) -> Phase1Checklist:
        """Generate Phase 1 checklist"""
        
        prompt = ChecklistGenerator.PHASE_1_PROMPT.format(problem=problem)
        
        try:
            response = await llm_query_func(
                prompt=prompt,
                system_prompt="You are a physics problem analyst. Respond ONLY with JSON."
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Convert to TodoItems
            gaps = [
                TodoItem(
                    id=f"gap_{i}",
                    title=gap.get('title', ''),
                    description=gap.get('description', '')
                )
                for i, gap in enumerate(data.get('knowledge_gaps', []))
            ]
            
            return Phase1Checklist(
                problem=problem,
                knowledge_gaps=gaps,
                search_queries=data.get('search_queries', []),
                completeness=data.get('completeness', 0) / 100.0,
                can_proceed_to_phase_2=data.get('can_proceed', False)
            )
        
        except Exception as e:
            print(f"⚠️  Phase 1 generation failed: {e}")
            return Phase1Checklist(
                problem=problem,
                knowledge_gaps=[],
                search_queries=[],
                completeness=0,
                can_proceed_to_phase_2=False
            )
    
    @staticmethod
    async def generate_phase_2(
        problem: str,
        available_info: str,
        llm_query_func
    ) -> Phase2Checklist:
        """Generate Phase 2 checklist"""
        
        prompt = ChecklistGenerator.PHASE_2_PROMPT.format(
            problem=problem,
            available_info=available_info
        )
        
        try:
            response = await llm_query_func(
                prompt=prompt,
                system_prompt="You are a solution planner. Respond ONLY with JSON."
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Convert to TodoItems
            steps = [
                TodoItem(
                    id=f"step_{step.get('step', i)}",
                    title=step.get('title', ''),
                    description=step.get('description', ''),
                    assigned_to="compute_agent"
                )
                for i, step in enumerate(data.get('steps', []))
            ]
            
            validation = [
                TodoItem(
                    id=f"validate_{i}",
                    title=val.get('title', ''),
                    description=val.get('description', ''),
                    assigned_to="validate_agent"
                )
                for i, val in enumerate(data.get('validation_steps', []))
            ]
            
            return Phase2Checklist(
                problem=problem,
                solution_method=data.get('solution_method', ''),
                steps=steps,
                validation_steps=validation
            )
        
        except Exception as e:
            print(f"⚠️  Phase 2 generation failed: {e}")
            return Phase2Checklist(
                problem=problem,
                solution_method='',
                steps=[],
                validation_steps=[]
            )


class AgentTaskAssigner:
    """Assigns TODOs to agents for execution"""
    
    @staticmethod
    def assign_phase_1_tasks(checklist: Phase1Checklist) -> Dict[str, List[TodoItem]]:
        """Assign Phase 1 tasks to agents"""
        
        tasks = {
            'search_agent': [],
            'research_agent': [],
        }
        
        # Assign search queries
        for query in checklist.search_queries:
            task = TodoItem(
                id=f"search_{len(tasks['search_agent'])}",
                title=f"Search: {query}",
                description=f"Find information about: {query}",
                assigned_to="search_agent"
            )
            tasks['search_agent'].append(task)
        
        # Assign knowledge gap filling
        for gap in checklist.knowledge_gaps:
            task = TodoItem(
                id=f"research_{gap.id}",
                title=f"Find: {gap.title}",
                description=gap.description,
                assigned_to="research_agent"
            )
            tasks['research_agent'].append(task)
        
        return tasks
    
    @staticmethod
    def assign_phase_2_tasks(checklist: Phase2Checklist) -> Dict[str, List[TodoItem]]:
        """Assign Phase 2 tasks to agents"""
        
        tasks = {
            'compute_agent': [],
            'validate_agent': [],
            'answer_agent': []
        }
        
        # Solution steps go to compute agent
        for step in checklist.steps:
            step.assigned_to = 'compute_agent'
            tasks['compute_agent'].append(step)
        
        # Validation goes to validate agent
        for val_step in checklist.validation_steps:
            val_step.assigned_to = 'validate_agent'
            tasks['validate_agent'].append(val_step)
        
        # Final answer assembly goes to answer agent
        answer_task = TodoItem(
            id="final_answer",
            title="Assemble final answer",
            description="Combine all computed values into clear final answer",
            assigned_to="answer_agent"
        )
        tasks['answer_agent'].append(answer_task)
        
        return tasks


if __name__ == "__main__":
    print("Two-Phase Checklist System")
    print("="*70)
    print("\nCapabilities:")
    print("  • Generate Phase 1 checklist (what info needed)")
    print("  • Generate Phase 2 checklist (how to solve)")
    print("  • Assign tasks to specific agents")
    print("  • Track TODO status and results")

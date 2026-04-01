"""
Question Classifier

Determines:
1. Is this theoretical (explain/conceptual) or mathematical (compute/solve)?
2. What variables are given vs what we're solving for?
3. Do we have enough info to solve it?
4. What steps are needed?
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class QuestionType(Enum):
    THEORETICAL = "theoretical"        # Explain concepts, describe, discuss
    MATHEMATICAL = "mathematical"      # Calculate, compute, solve, find value
    HYBRID = "hybrid"                  # Explain AND calculate
    ENGINEERING_DESIGN = "engineering_design"  # Design/size/configure a multi-variable system
    UNKNOWN = "unknown"                # Can't classify


@dataclass
class ClassificationResult:
    question_type: QuestionType
    is_solvable: bool
    given_variables: List[str]
    unknown_variables: List[str]
    equations_needed: List[str]
    steps: List[str]
    confidence: float
    reasoning: str
    critical_missing: List[str] = field(default_factory=list)
    domain: str = ""
    dependency_order: List[str] = field(default_factory=list)
    variable_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Format: {"delta_v": {"unit": "m/s", "description": "total velocity change", "known": false},
    #           "mass":   {"unit": "kg",  "description": "initial wet mass", "known": true, "value": 50000}}
    self_contained: bool = False  # True when MATHEMATICAL and all needed values are in the question


class QuestionClassifier:
    """
    Classifies questions to determine appropriate solving strategy
    """
    
    # Keywords indicating theoretical questions
    THEORETICAL_KEYWORDS = {
        'explain', 'describe', 'what is', 'define', 'discuss', 'how does',
        'why', 'compare', 'contrast', 'analyze', 'interpret', 'summarize',
        'outline', 'review', 'discuss', 'evaluate', 'state', 'list', 'identify'
    }
    
    # Keywords indicating mathematical questions
    MATHEMATICAL_KEYWORDS = {
        'calculate', 'compute', 'solve', 'find', 'determine', 'derive',
        'prove', 'verify', 'show', 'simplify', 'evaluate', 'approximate',
        'estimate', 'differentiate', 'integrate', 'minimize', 'maximize'
    }
    
    CLASSIFICATION_PROMPT = """
Analyze this question carefully. Classify it and identify what's needed to solve it.

Question: {question}

Use 'engineering_design' when the question asks to DESIGN, SIZE, or CONFIGURE a system
where multiple interdependent design parameters must be CHOSEN (not just computed from given
values). Key signals: requires selecting propellant type, structural fractions, motor specs,
material grades, system architecture etc. — not just plugging numbers into equations.
Examples: "design a rocket", "size an electric motor drive", "specify a heat exchanger".
For engineering_design questions also provide 'domain' and 'dependency_order'.

CRITICAL CLASSIFICATION RULES (these override your intuition):
1. "theoretical" means PURELY conceptual — no specific numerical values present, no request
   to compute any number. Example: "Explain qualitatively how orbital precession arises."
2. If the question contains SPECIFIC NUMERICAL VALUES for variables (e.g., "m = 2 kg",
   "L = 3 kg·m²/s", "V(r) = -5/r + 3r²", any coefficient with a number) → NEVER classify
   as "theoretical". Use "mathematical" or "hybrid".
3. If ANY sub-task says "evaluate", "compute", "calculate", "solve for", "find numerically",
   "evaluate numerically", or asks to compare specific numerical frequencies/values →
   classify as "mathematical" (if mostly compute) or "hybrid" (if mix of compute + explain).
4. Multi-part questions with BOTH numerical computation tasks AND conceptual explanation
   tasks → ALWAYS "hybrid", never "theoretical".
5. The presence of "explain", "describe", or "qualitatively" in ONE sub-task does NOT make
   the whole question theoretical if OTHER sub-tasks require numerical computation.

Respond with ONLY valid JSON (no markdown, no explanation):
{{
    "question_type": "theoretical" | "mathematical" | "hybrid" | "engineering_design" | "unknown",
    "is_solvable": true/false,
    "why_solvable": "explanation of solvability",
    "given_variables": ["var1", "var2", ...],
    "unknown_variables": ["what_we_solve_for", ...],
    "critical_missing": ["if any parameters are completely missing"],
    "equations_needed": ["Tsiolkovsky equation", "orbital energy equation", ...],
    "domain": "rocket | motor | structure | thermal | power | fluid | ballistics | orbital_mechanics | aerodynamics | aircraft | rotorcraft | automotive | robotics | controls | electronics | rf | optics | acoustics | chemical | combustion | gas_turbine | steam_turbine | refrigeration | heat_exchanger | cryogenics | nuclear | solar_pv | wind_turbine | hvac | civil | geotechnical | marine | materials | composites | manufacturing | welding | gearbox | bearing | vibration | fatigue | impact | pneumatics | hydraulics | vacuum | semiconductor | battery_chem | nuclear_power | geophysics | biomedical | mining | other",
    "dependency_order": ["payload_mass", "delta_v", "mass_ratio", "GTOW"],
    "variable_schema": {{
        "delta_v":  {{"unit": "m/s", "description": "total velocity change required", "known": false}},
        "mass_wet": {{"unit": "kg",  "description": "initial wet mass (with propellant)", "known": true, "value": 50000}},
        "isp":      {{"unit": "s",   "description": "specific impulse", "known": true, "value": 350}}
    }},
    "steps": [
        "Step 1: describe what to do",
        "Step 2: describe what to do",
        ...
    ],
    "confidence": 0.0-1.0,
    "reasoning": "why you classified it this way"
}}
"""
    
    @staticmethod
    async def classify(question: str, llm_query_func) -> ClassificationResult:
        """
        Classify a question using LLM
        
        Args:
            question: The question to classify
            llm_query_func: Async function to call LLM
        
        Returns:
            ClassificationResult with all analysis
        """
        
        prompt = QuestionClassifier.CLASSIFICATION_PROMPT.format(question=question)
        
        try:
            response = await llm_query_func(
                prompt=prompt,
                system_prompt="You are a physics/math question analyzer. Respond ONLY with valid JSON."
            )
            
            import json
            import re
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Parse question type
            qtype_str = data.get('question_type', 'unknown').lower()
            if qtype_str == 'theoretical':
                qtype = QuestionType.THEORETICAL
            elif qtype_str == 'mathematical':
                qtype = QuestionType.MATHEMATICAL
            elif qtype_str == 'hybrid':
                qtype = QuestionType.HYBRID
            elif qtype_str == 'engineering_design':
                qtype = QuestionType.ENGINEERING_DESIGN
            else:
                qtype = QuestionType.UNKNOWN

            # Parse variable_schema — ensure all values are dicts
            raw_schema = data.get('variable_schema', {})
            variable_schema: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw_schema, dict):
                for var_name, meta in raw_schema.items():
                    if isinstance(meta, dict):
                        variable_schema[var_name] = meta

            # Self-contained: MATHEMATICAL, solvable, no missing values, has known vars in schema
            critical_missing = data.get('critical_missing', [])
            known_vars = [v for v, m in variable_schema.items() if m.get('known', False)]
            self_contained = (
                qtype == QuestionType.MATHEMATICAL
                and data.get('is_solvable', False)
                and len(critical_missing) == 0
                and len(known_vars) > 0
            )

            return ClassificationResult(
                question_type=qtype,
                is_solvable=data.get('is_solvable', True),
                given_variables=data.get('given_variables', []),
                unknown_variables=data.get('unknown_variables', []),
                equations_needed=data.get('equations_needed', []),
                steps=data.get('steps', []),
                confidence=data.get('confidence', 0.5),
                reasoning=data.get('reasoning', ''),
                critical_missing=critical_missing,
                domain=data.get('domain', ''),
                dependency_order=data.get('dependency_order', []),
                variable_schema=variable_schema,
                self_contained=self_contained,
            )
        
        except Exception as e:
            print(f"⚠️ Classification failed: {e}")
            return ClassificationResult(
                question_type=QuestionType.UNKNOWN,
                is_solvable=False,
                given_variables=[],
                unknown_variables=[],
                equations_needed=[],
                steps=[],
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    @staticmethod
    def classify_simple(question: str) -> QuestionType:
        """
        Quick heuristic classification (no LLM)
        
        Useful for initial classification before full analysis
        """
        
        q = question.lower()

        # Check for engineering design verbs paired with domain keywords
        _design_verbs = {'design', 'size', 'specify', 'configure', 'dimension', 'develop'}
        _domain_keywords = {
            # original
            'rocket', 'launch vehicle', 'spacecraft', 'motor', 'drive', 'inverter',
            'heat exchanger', 'beam', 'truss', 'pipeline', 'battery', 'power system',
            'propulsion', 'cooling', 'frame', 'structure',
            # ballistics / FRC
            'turret', 'projectile', 'trajectory', 'shooter', 'ballistic', 'frc',
            'first robotics', 'shoot on the move', 'lead angle', 'gun', 'cannon',
            # orbital
            'orbit', 'orbital', 'satellite', 'hohmann', 'delta-v', 'delta_v',
            # aero / aircraft
            'airfoil', 'aircraft', 'airplane', 'aerodynamic', 'lift', 'wing',
            'drone', 'helicopter', 'rotorcraft', 'quadcopter', 'uav', 'multirotor',
            # automotive
            'vehicle', 'car', 'truck', 'drivetrain', 'electric vehicle', 'ev',
            # robotics / controls
            'robot', 'pid', 'control system', 'controller', 'feedback loop',
            'servo', 'actuator', 'kinematics',
            # electronics / rf
            'circuit', 'pcb', 'antenna', 'wireless', 'amplifier', 'filter',
            'microcontroller', 'transistor',
            # optics / acoustics
            'laser', 'lens', 'optical system', 'speaker', 'noise', 'acoustic',
            # thermo / energy
            'turbine', 'compressor', 'refrigeration', 'hvac', 'heat pump',
            'solar panel', 'wind turbine', 'nuclear reactor',
            # civil / geo
            'foundation', 'concrete', 'column', 'dam', 'slope', 'embankment',
            # marine / materials
            'ship', 'propeller', 'hull', 'composite', 'carbon fiber', 'weld',
            # mechanical components
            'gearbox', 'gear', 'bearing', 'fatigue', 'vibration', 'impact',
        }
        has_design_verb = any(v in q for v in _design_verbs)
        has_domain = any(d in q for d in _domain_keywords)
        if has_design_verb and has_domain:
            return QuestionType.ENGINEERING_DESIGN

        # Count keyword matches
        theoretical_count = sum(1 for kw in QuestionClassifier.THEORETICAL_KEYWORDS if kw in q)
        mathematical_count = sum(1 for kw in QuestionClassifier.MATHEMATICAL_KEYWORDS if kw in q)
        
        # Check for numbers/values (indicates mathematical)
        has_numbers = bool(re.search(r'\d+', q))
        has_units = bool(re.search(r'(kg|m|s|°|km|m/s|N|J|W|Hz)', q))
        
        if mathematical_count > theoretical_count and (has_numbers or has_units):
            return QuestionType.MATHEMATICAL
        elif theoretical_count > mathematical_count:
            return QuestionType.THEORETICAL
        elif mathematical_count > 0 and theoretical_count > 0:
            return QuestionType.HYBRID
        else:
            return QuestionType.UNKNOWN
    
    @staticmethod
    def format_classification(result: ClassificationResult) -> str:
        """Format classification result for display"""
        
        lines = [
            "\n" + "="*70,
            "QUESTION CLASSIFICATION",
            "="*70,
            f"\n🎯 Type: {result.question_type.value.upper()}",
            f"   Confidence: {result.confidence:.0%}",
            f"\n📝 Reasoning:\n   {result.reasoning}",
        ]
        
        if result.is_solvable:
            lines.extend([
                f"\n✅ Status: SOLVABLE",
                f"\n📋 Given Variables ({len(result.given_variables)}):",
            ])
            for var in result.given_variables:
                lines.append(f"   • {var}")
            
            lines.append(f"\n❓ Unknown Variables ({len(result.unknown_variables)}):")
            for var in result.unknown_variables:
                lines.append(f"   • {var}")
            
            if result.equations_needed:
                lines.append(f"\n📐 Equations Needed:")
                for eq in result.equations_needed:
                    lines.append(f"   • {eq}")
            
            lines.append(f"\n📋 Solution Steps:")
            for i, step in enumerate(result.steps, 1):
                lines.append(f"   {i}. {step}")
        else:
            lines.extend([
                f"\n❌ Status: NOT SOLVABLE",
                f"   Reason: Missing critical information"
            ])
            
            if result.critical_missing:
                lines.append(f"\n   Missing:")
                for missing in result.critical_missing:
                    lines.append(f"   • {missing}")
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("Question Classifier Module")
    print("="*70)
    print("\nCapabilities:")
    print("  • Classify questions as theoretical/mathematical/hybrid")
    print("  • Extract given vs unknown variables")
    print("  • Identify required equations")
    print("  • Determine if question is solvable")
    print("  • Plan solution steps")

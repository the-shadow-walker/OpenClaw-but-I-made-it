"""
Swarm 3.0 - Shared Memory System
Append-only fact store with complete provenance tracking

Core Principle: All knowledge is written, nothing trusted from memory
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import uuid


class FactType(Enum):
    """Types of facts that can be stored"""
    SEARCH_RESULT = "search_result"
    VALIDATED_FACT = "validated_fact"
    ASSUMPTION = "assumption"
    VARIABLE_DEF = "variable_definition"
    EQUATION = "equation"
    COMPUTED_RESULT = "computed_result"
    OPEN_QUESTION = "open_question"
    REJECTION = "rejection"


@dataclass
class Source:
    """Source citation for a fact"""
    url: Optional[str] = None
    title: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    search_query: Optional[str] = None
    
    def __str__(self):
        if self.url:
            return f"{self.title or 'Source'}: {self.url}"
        elif self.agent_id:
            return f"Agent {self.agent_id} at {self.timestamp}"
        return "Unknown source"


@dataclass
class Fact:
    """A single fact in shared memory"""
    id: str
    fact_type: FactType
    content: str
    source: Source
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validated: bool = False
    contradicts: List[str] = field(default_factory=list)  # IDs of contradicting facts
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self):
        d = asdict(self)
        # Convert enum to string for JSON serialization
        if 'fact_type' in d and hasattr(d['fact_type'], 'value'):
            d['fact_type'] = d['fact_type'].value
        return d


@dataclass
class Variable:
    """Variable definition for equations"""
    symbol: str
    meaning: str
    unit: Optional[str] = None
    value: Optional[float] = None
    known: bool = False
    source_fact_id: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Equation:
    """Symbolic equation (no numbers)"""
    id: str
    symbolic_form: str  # "F = m * a"
    variables_used: List[str]  # ["F", "m", "a"]
    domain: str = "physics"  # physics, math, chemistry, etc.
    source_fact_id: Optional[str] = None
    validated: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ComputedResult:
    """Result from Python computation (NOT from LLM)"""
    id: str
    variable: str
    value: float
    unit: str
    method: str  # Which equation/calculation
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    symbolic_form: Optional[str] = None
    numeric_substitution: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self):
        return asdict(self)


class SharedMemory:
    """
    Append-only shared memory for Swarm 3.0
    
    Design Rules:
    - Facts are never deleted, only marked as rejected
    - Every fact has a source
    - LLMs write facts, Python computes results
    - Verification is separate from generation
    """
    
    def __init__(self):
        # Core stores (append-only)
        self.facts: List[Fact] = []
        self.variables: Dict[str, Variable] = {}
        self.equations: List[Equation] = []
        self.computed_results: List[ComputedResult] = []
        
        # Original question (for variable extraction)
        self.original_question: str = ""
        
        # Tracking
        self.open_questions: List[str] = []
        self.assumptions: List[str] = []
        self.contradictions: List[Dict[str, str]] = []
        
        # Agent status
        self.agent_activity: List[Dict[str, str]] = []
        
        print("📦 Shared Memory initialized (Swarm 3.0)")
    
    # ==================== FACTS ====================
    
    def add_fact(
        self,
        content: str,
        fact_type: FactType,
        source: Source,
        validated: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a fact to shared memory"""
        fact = Fact(
            id=str(uuid.uuid4())[:8],
            fact_type=fact_type,
            content=content,
            source=source,
            validated=validated,
            metadata=metadata or {}
        )
        
        self.facts.append(fact)
        print(f"   📝 Added fact {fact.id}: {content[:60]}...")
        
        return fact.id
    
    def get_facts(
        self,
        fact_type: Optional[FactType] = None,
        validated_only: bool = False
    ) -> List[Fact]:
        """Retrieve facts with optional filtering"""
        results = self.facts
        
        if fact_type:
            results = [f for f in results if f.fact_type == fact_type]
        
        if validated_only:
            results = [f for f in results if f.validated]
        
        return results
    
    def validate_fact(self, fact_id: str, validator_agent: str):
        """Mark a fact as validated"""
        for fact in self.facts:
            if fact.id == fact_id:
                fact.validated = True
                fact.metadata['validated_by'] = validator_agent
                fact.metadata['validated_at'] = datetime.now().isoformat()
                print(f"   ✅ Validated fact {fact_id}")
                return True
        return False
    
    def reject_fact(self, fact_id: str, reason: str, rejector_agent: str):
        """Mark a fact as rejected (don't delete, just flag)"""
        rejection_fact = Fact(
            id=str(uuid.uuid4())[:8],
            fact_type=FactType.REJECTION,
            content=f"Rejected {fact_id}: {reason}",
            source=Source(agent_id=rejector_agent),
            validated=True
        )
        
        self.facts.append(rejection_fact)
        print(f"   ❌ Rejected fact {fact_id}: {reason}")
    
    def flag_contradiction(self, fact_id_1: str, fact_id_2: str, reason: str):
        """Flag two facts as contradicting"""
        # Update both facts
        for fact in self.facts:
            if fact.id == fact_id_1:
                fact.contradicts.append(fact_id_2)
            elif fact.id == fact_id_2:
                fact.contradicts.append(fact_id_1)
        
        # Record contradiction
        self.contradictions.append({
            'fact_1': fact_id_1,
            'fact_2': fact_id_2,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"   ⚠️ Contradiction: {fact_id_1} ↔ {fact_id_2}")
    
    # ==================== VARIABLES ====================
    
    def add_variable(
        self,
        symbol: str,
        meaning: str,
        unit: Optional[str] = None,
        value: Optional[float] = None,
        source_fact_id: Optional[str] = None
    ) -> Variable:
        """Define a variable"""
        var = Variable(
            symbol=symbol,
            meaning=meaning,
            unit=unit,
            value=value,
            known=value is not None,
            source_fact_id=source_fact_id
        )
        
        self.variables[symbol] = var
        print(f"   🔤 Variable: {symbol} = {meaning} ({unit or 'dimensionless'})")
        
        return var
    
    def set_variable_value(self, symbol: str, value: float, unit: str):
        """Set the value of a variable (from facts, not computation)"""
        if symbol in self.variables:
            self.variables[symbol].value = value
            self.variables[symbol].unit = unit
            self.variables[symbol].known = True
            print(f"   📊 {symbol} = {value} {unit}")
    
    def get_variable(self, symbol: str) -> Optional[Variable]:
        """Get variable definition"""
        return self.variables.get(symbol)
    
    def get_known_variables(self) -> Dict[str, Variable]:
        """Get all variables with known values"""
        return {s: v for s, v in self.variables.items() if v.known}
    
    def get_unknown_variables(self) -> Dict[str, Variable]:
        """Get all variables we need to solve for"""
        return {s: v for s, v in self.variables.items() if not v.known}
    
    # ==================== EQUATIONS ====================
    
    def add_equation(
        self,
        symbolic_form: str,
        variables_used: List[str],
        domain: str = "physics",
        source_fact_id: Optional[str] = None
    ) -> str:
        """Add a symbolic equation (NO numbers)"""
        eq = Equation(
            id=str(uuid.uuid4())[:8],
            symbolic_form=symbolic_form,
            variables_used=variables_used,
            domain=domain,
            source_fact_id=source_fact_id
        )
        
        self.equations.append(eq)
        print(f"   🔢 Equation: {symbolic_form}")
        
        return eq.id
    
    def validate_equation(self, equation_id: str):
        """Mark equation as validated"""
        for eq in self.equations:
            if eq.id == equation_id:
                eq.validated = True
                print(f"   ✅ Validated equation {equation_id}")
                return True
        return False
    
    def get_equations(self, validated_only: bool = False) -> List[Equation]:
        """Get equations"""
        if validated_only:
            return [eq for eq in self.equations if eq.validated]
        return self.equations
    
    # ==================== COMPUTED RESULTS ====================
    
    def add_computed_result(
        self,
        variable: str,
        value: float,
        unit: str,
        method: str,
        symbolic_form: Optional[str] = None,
        numeric_substitution: Optional[str] = None
    ) -> str:
        """Add a result from Python computation (NOT LLM)"""
        result = ComputedResult(
            id=str(uuid.uuid4())[:8],
            variable=variable,
            value=value,
            unit=unit,
            method=method,
            symbolic_form=symbolic_form,
            numeric_substitution=numeric_substitution
        )
        
        self.computed_results.append(result)
        print(f"   🎯 Computed: {variable} = {value} {unit}")
        
        # Also update variable if it exists
        if variable in self.variables:
            self.variables[variable].value = value
            self.variables[variable].unit = unit
            self.variables[variable].known = True
        
        return result.id
    
    def get_computed_results(self) -> List[ComputedResult]:
        """Get all computed results"""
        return self.computed_results
    
    # ==================== QUESTIONS & ASSUMPTIONS ====================
    
    def add_open_question(self, question: str):
        """Add a question that needs to be answered"""
        self.open_questions.append(question)
        print(f"   ❓ Open question: {question}")
    
    def add_assumption(self, assumption: str):
        """Add an assumption being made"""
        self.assumptions.append(assumption)
        print(f"   💭 Assumption: {assumption}")
    
    # ==================== AGENT ACTIVITY ====================
    
    def log_agent_activity(self, agent_id: str, action: str, details: str = ""):
        """Log agent activity"""
        self.agent_activity.append({
            'agent_id': agent_id,
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    # ==================== EXPORT & DEBUG ====================
    
    def to_dict(self) -> Dict:
        """Export entire memory as dict"""
        return {
            'facts': [f.to_dict() for f in self.facts],
            'variables': {k: v.to_dict() for k, v in self.variables.items()},
            'equations': [eq.to_dict() for eq in self.equations],
            'computed_results': [r.to_dict() for r in self.computed_results],
            'open_questions': self.open_questions,
            'assumptions': self.assumptions,
            'contradictions': self.contradictions,
            'agent_activity': self.agent_activity
        }
    
    def to_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 Saved memory to {filepath}")
    
    def summary(self) -> str:
        """Get a summary of current memory state"""
        validated_facts = len([f for f in self.facts if f.validated])
        total_facts = len(self.facts)
        
        return f"""
📊 Shared Memory Summary:
   Facts: {total_facts} ({validated_facts} validated)
   Variables: {len(self.variables)} ({len(self.get_known_variables())} known)
   Equations: {len(self.equations)}
   Computed Results: {len(self.computed_results)}
   Open Questions: {len(self.open_questions)}
   Assumptions: {len(self.assumptions)}
   Contradictions: {len(self.contradictions)}
        """.strip()
    
    def print_state(self):
        """Print current state for debugging"""
        print("\n" + "="*70)
        print(self.summary())
        print("="*70)
        
        if self.facts:
            print("\n📝 Facts:")
            for fact in self.facts[-5:]:  # Last 5
                status = "✅" if fact.validated else "⏳"
                print(f"   {status} [{fact.fact_type.value}] {fact.content[:60]}...")
        
        if self.variables:
            print("\n🔤 Variables:")
            for symbol, var in self.variables.items():
                status = f"{var.value} {var.unit}" if var.known else "unknown"
                print(f"   {symbol}: {var.meaning} = {status}")
        
        if self.equations:
            print("\n🔢 Equations:")
            for eq in self.equations:
                status = "✅" if eq.validated else "⏳"
                print(f"   {status} {eq.symbolic_form}")
        
        if self.computed_results:
            print("\n🎯 Computed Results:")
            for result in self.computed_results:
                print(f"   {result.variable} = {result.value} {result.unit}")
        
        print()


# Quick test
if __name__ == "__main__":
    print("Testing Shared Memory System...\n")
    
    memory = SharedMemory()
    
    # Add some facts
    source1 = Source(
        url="https://physics.com/article",
        title="Physics Article",
        search_query="thrust force"
    )
    
    fact_id = memory.add_fact(
        content="Thrust force must equal weight for liftoff",
        fact_type=FactType.SEARCH_RESULT,
        source=source1
    )
    
    memory.validate_fact(fact_id, "consensus_agent")
    
    # Add variables
    memory.add_variable("F", "thrust force", "N")
    memory.add_variable("m", "mass", "kg", value=5000.0)
    memory.add_variable("g", "gravity", "m/s^2", value=9.81)
    
    # Add equation
    memory.add_equation(
        symbolic_form="F = m * g",
        variables_used=["F", "m", "g"],
        domain="physics"
    )
    
    # Add computed result
    memory.add_computed_result(
        variable="F",
        value=49050.0,
        unit="N",
        method="F = m * g",
        symbolic_form="F = m * g",
        numeric_substitution="F = 5000 * 9.81"
    )
    
    # Add question and assumption
    memory.add_open_question("What is the required thrust?")
    memory.add_assumption("Assuming Earth gravity (g = 9.81 m/s^2)")
    
    # Print state
    memory.print_state()
    
    # Save to JSON
    memory.to_json("/tmp/memory_test.json")
    
    print("\n✅ Shared Memory test complete!")

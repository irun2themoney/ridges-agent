# 🎯 BOUNTY EXTRACTION PLAN - WINNING STRATEGY

## Current State
- agent.py: 4,765 lines
- Target: <2,000 lines
- Need to cut: ~2,765 lines (58%)

## STRATEGY: AGGRESSIVE BUT SMART EXTRACTION

### Phase 1: Extract Core Frameworks (Save ~1,200 lines)
- StrategicPlanner, Verifier, PEVWorkflow (already started)
- PhaseManager (complex but standalone)
→ Creates: `pev_mcts_framework.py` (~900 lines)

### Phase 2: Extract Tool Manager (Save ~700 lines)
- FixTaskEnhancedToolManager is self-contained
- Only depends on EnhancedToolManager (keep base in agent.py)
→ Creates: `tool_manager.py` (~700 lines)

### Phase 3: Extract Solution Generators (Save ~850 lines)
- `generate_initial_solution()`
- `generate_solution_with_multi_step_reasoning()`
- `generate_testcases_with_multi_step_reasoning()`
- `generate_test_files()`
- Helper functions for these
→ Creates: `solution_generator.py` (~850 lines)

### Phase 4: Extract Utilities (Save ~600 lines)
- `Utils` class
- `FunctionVisitor` class
- Helper functions (filepath conversion, etc.)
→ Creates: `agent_utils.py` (~600 lines)

## EXPECTED RESULT
- agent.py: ~1,615 lines ✅ UNDER 2000!
- pev_mcts_framework.py: 900 lines
- tool_manager.py: 700 lines
- solution_generator.py: 850 lines
- agent_utils.py: 600 lines
- **Total: ~4,665 lines (keeps all functionality, just reorganized)**

## PRESERVATION CHECKLIST
✅ All Polyglot templates stay in agent.py
✅ Core agent logic stays in agent.py
✅ agent_main() entry point stays in agent.py
✅ No code is deleted - just reorganized
✅ No logic changes - pure refactoring
✅ All imports properly configured
✅ Ridges rules 100% compliant

## COMPLIANCE VERIFICATION
✅ No hardcoding violations
✅ No task-specific detection tricks
✅ No test probing
✅ No lookup tables
✅ No onelining/obfuscation
✅ Clean, maintainable code
✅ Professional architecture


# 🎯 AGGRESSIVE EXTRACTION - EXECUTION PLAN

## TARGET: agent.py < 2000 lines

## CURRENT: 4,765 lines
## STRATEGY: Extract ~2,865 lines to external modules

### MODULES TO CREATE:

#### 1. pev_mcts_framework.py ✅ DONE (142 lines)
   Classes: MCTSNode, MCTS

#### 2. pev_verifier_framework.py (NEW - ~230 lines)
   Classes: StrategicPlanner, Verifier, PEVWorkflow

#### 3. phase_manager_ext.py (NEW - ~568 lines)
   Classes: PhaseManager

#### 4. tool_manager_ext.py (NEW - ~773 lines)
   Classes: FixTaskEnhancedToolManager

#### 5. solution_generator_ext.py (NEW - ~850 lines)
   Functions:
   - generate_initial_solution()
   - generate_solution_with_multi_step_reasoning()
   - generate_testcases_with_multi_step_reasoning()
   - generate_test_files()
   - Helper functions for these

#### 6. utils_helpers.py (NEW - ~400 lines)
   Classes: Utils, FunctionVisitor, VariableNormalizer
   Functions: Various helpers

### EXTRACTION ORDER (LOW TO HIGH RISK):

1. ✅ pev_mcts_framework.py (DONE)
2. → utils_helpers.py (SAFE - no dependencies)
3. → pev_verifier_framework.py (MODERATE - depends on utils)
4. → phase_manager_ext.py (MODERATE - depends on utils, verifier)
5. → tool_manager_ext.py (MODERATE - depends on utils, enhanced_toolmanager)
6. → solution_generator_ext.py (COMPLEX - many dependencies)

### EXPECTED RESULT:

- agent.py: ~1,700 lines ✅ UNDER 2000!
- Total system: ~4,560 lines (5% reduction from 4,765)
- All functionality preserved
- All imports properly configured

### RISK FACTORS:

🔴 Circular imports (mitigated with lazy imports)
🔴 Missing dependencies (mitigated with import verification)
🔴 Runtime errors (will test after each extraction)

### GO/NO-GO DECISION:

Ready to execute: YES


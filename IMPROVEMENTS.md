# Agent Improvement Strategy

Based on current performance analysis, here are strategic improvements to increase success rates:

## Current Performance

- **Polyglot**: 33/33 (100%) ✅ - Templates work perfectly
- **SWE-bench**: Variable (2/19 to 79/80 depending on problem)
- **Main issues**: 
  - Missing multi-file changes
  - Incomplete fixes
  - Not understanding problem context deeply enough

## Improvement Strategies

### 1. Enhanced File Discovery for SWE-bench

**Current**: Only includes 3 test files + 2 source files  
**Improvement**: Smarter file selection

```python
# Better file selection strategy:
# 1. Find files mentioned in problem statement
# 2. Find files that import/are imported by mentioned files
# 3. Find files with similar names to mentioned files
# 4. Prioritize smaller files that are likely to be changed
# 5. Include more context (up to 5-10 files, larger excerpts)
```

### 2. Better Context Understanding

**Improvements**:
- Parse problem statement to extract file names and function names
- Include import graph analysis to find related files
- Add codebase structure analysis (which files call which functions)
- Include diff context if previous attempts exist

### 3. Multi-Step Reasoning for Complex Problems

**Current**: Single-shot inference  
**Improvement**: Chain-of-thought reasoning

```python
# For complex SWE-bench problems:
# 1. First pass: Analyze problem and identify affected files
# 2. Second pass: Understand the relationships between files
# 3. Third pass: Generate fixes for each affected file
# 4. Fourth pass: Validate the complete solution
```

### 4. Better Prompt Engineering

**Add**:
- Few-shot examples of successful fixes
- Problem type classification (bug fix, feature addition, refactor)
- Explicit guidance on what to look for based on problem type
- Examples of correct multi-file patches

### 5. Enhanced Retry Logic

**Current**: 2 retries with same prompt  
**Improvement**: Adaptive retry strategies

```python
# Retry strategies:
# 1. First retry: Same prompt (current behavior)
# 2. Second retry: Simplified prompt focused on single file
# 3. Third retry: Decomposed approach (identify files first, then fix)
# 4. Fourth retry: Look at tests to understand expected behavior
```

### 6. Test-Driven Understanding

**Improvement**: Analyze test files more deeply

```python
# For SWE-bench:
# 1. Parse test files to understand what's expected
# 2. Identify test functions that are failing
# 3. Extract expected behavior from test assertions
# 4. Use test expectations to guide fix generation
```

### 7. Codebase Navigation

**Improvement**: Better file discovery using semantic analysis

```python
# Strategies:
# 1. Find files that contain keywords from problem statement
# 2. Follow import chains to find related modules
# 3. Search for function/class names mentioned in problem
# 4. Use file paths to understand module structure
```

### 8. Temperature & Model Selection

**Current**: Fixed temperature 0.8, single model  
**Improvement**: Adaptive based on problem type

```python
# SWE-bench (complex): Lower temperature (0.3-0.5) for more deterministic fixes
# Polyglot (algorithmic): Current temperature (0.8) works well
# Could also try different models for different problem types
```

### 9. Patch Validation Before Return

**Improvement**: Validate generated patches

```python
# Before returning patch:
# 1. Check syntax of all modified files
# 2. Verify diff format is correct
# 3. Check that all mentioned files in problem statement are included
# 4. Verify no syntax errors in generated code
```

### 10. Incremental Approach for Large Changes

**Improvement**: Break down large problems

```python
# For problems requiring many file changes:
# 1. First: Identify all files that need changes
# 2. Second: Generate fixes one file at a time
# 3. Third: Combine all fixes into single patch
# 4. Fourth: Validate complete solution
```

## Priority Order

**High Priority (Immediate Impact)**:
1. ✅ Enhanced file discovery (more relevant files)
2. ✅ Better context understanding (parse problem statement)
3. ✅ Test-driven understanding (analyze tests more deeply)

**Medium Priority (Good ROI)**:
4. Improved retry logic with different strategies
5. Better prompt engineering with examples
6. Multi-step reasoning for complex problems

**Lower Priority (Nice to Have)**:
7. Codebase navigation with semantic search
8. Adaptive temperature/model selection
9. Patch validation
10. Incremental approach for large changes

## Implementation Plan

### Phase 1: Quick Wins
1. Increase file context (5-10 files instead of 3-5)
2. Parse problem statement for file/function names
3. Include more test file content

### Phase 2: Better Understanding
4. Analyze imports to find related files
5. Extract test expectations to guide fixes
6. Add retry strategies with different approaches

### Phase 3: Advanced Features
7. Multi-step reasoning chains
8. Few-shot examples in prompts
9. Semantic codebase navigation


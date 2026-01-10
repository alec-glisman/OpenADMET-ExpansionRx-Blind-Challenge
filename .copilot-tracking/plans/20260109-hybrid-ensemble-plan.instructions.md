---
applyTo: ".copilot-tracking/changes/20260109-hybrid-ensemble-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Hybrid Ensemble Model for Minimum MA-RAE

## Overview

Create a hybrid ensemble that combines predictions from 5 January 2026 submissions using per-task optimal model selection to achieve minimum MA-RAE on the blind challenge.

## Objectives

- Select optimal model predictions per task based on leaderboard rank analysis
- Calculate expected performance improvements per task and overall MA-RAE
- Implement simple task-best selection (recommended over differential evolution)
- Generate final hybrid blind predictions for submission

## Research Summary

### Project Files

- [SUBMISSIONS.md](SUBMISSIONS.md) - Complete per-task metrics for all 5 January 2026 submissions
- [merge_task_weighted_predictions.py](configs/3-hpo-ensemble-production/2_task_weighted_ensemble/merge_task_weighted_predictions.py) - Existing ensemble infrastructure

### External References

- #file:../research/20260109-hybrid-ensemble-strategy-research.md - Comprehensive analysis and implementation plan

## Why Task-Best Selection Over Differential Evolution

**Recommendation: Use simple task-best selection, NOT differential evolution.**

### Arguments Against Differential Evolution

1. **Blind Challenge = No Optimization Target**
   - DE requires ground truth to optimize weights
   - Blind test set has no labels - we cannot compute MAE to minimize
   - Any "optimization" would be on training/validation data, risking overfitting

2. **Test Set Performance ≠ Blind Set Performance**
   - Optimizing weights on test split may not transfer to blind molecules
   - Blind set may have different chemical space distribution
   - Overfitted weights could hurt blind performance

3. **Single Best Model Per Task is Already Validated**
   - Leaderboard ranks come from actual blind predictions
   - Best model per task is proven to work on this exact blind set
   - No speculation about weight combinations

4. **Occam's Razor**
   - Simpler approach with fewer assumptions
   - No hyperparameters (weight bounds, population size, generations)
   - Deterministic and reproducible

5. **DE Adds Complexity Without Clear Benefit**
   - Weighted averaging assumes models make independent errors
   - If Jan-05 has rank 15 on LogD, adding Jan-08 (rank 88) dilutes predictions
   - Why blend in worse predictions?

### When Differential Evolution WOULD Be Useful

- If you have labeled validation data from same distribution as blind set
- For model selection during training (not final submission)
- When models are more similar in performance (not 15 vs 88 rank gap)

## Implementation Checklist

### [ ] Phase 1: Analysis and Documentation

- [ ] Task 1.1: Extract per-task MAE values from all 5 submissions
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 10-60)

- [ ] Task 1.2: Calculate expected performance improvement per task
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 62-120)

- [ ] Task 1.3: Calculate expected MA-RAE improvement
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 122-160)

### [ ] Phase 2: Implementation

- [ ] Task 2.1: Create hybrid ensemble script
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 162-250)

- [ ] Task 2.2: Generate merged blind predictions
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 252-300)

### [ ] Phase 3: Validation

- [ ] Task 3.1: Verify prediction file format matches submission requirements
  - Details: .copilot-tracking/details/20260109-hybrid-ensemble-details.md (Lines 302-330)

## Dependencies

- pandas
- numpy
- Existing submission CSV files in assets/submissions/

## Success Criteria

- Expected per-task MAE values documented with improvement estimates
- Hybrid blind_predictions.csv generated
- Ready for challenge submission

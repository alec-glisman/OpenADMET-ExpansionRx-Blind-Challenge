---
mode: agent
model: Claude Sonnet 4
---

<!-- markdownlint-disable-file -->

# Implementation Prompt: Sampler Harmonization

## Task Overview

Implement a unified `JointSampler` that combines task-aware oversampling (for sparse multi-task data) with quality-aware curriculum learning into a single configurable sampler with multiplicative weight composition.

## Implementation Instructions

### Step 1: Create Changes Tracking File

You WILL create `20251216-sampler-harmonization-changes.md` in #file:../changes/ if it does not exist.

### Step 2: Execute Implementation

You WILL follow #file:../../.github/instructions/task-implementation.instructions.md
You WILL systematically implement #file:../plans/20251216-sampler-harmonization-plan.instructions.md task-by-task
You WILL follow ALL project standards and conventions

**CRITICAL**: If ${input:phaseStop:true} is true, you WILL stop after each Phase for user review.
**CRITICAL**: If ${input:taskStop:false} is true, you WILL stop after each Task for user review.

### Implementation Phases Summary

1. **Phase 1: Configuration Schema** - Create `JointSamplingConfig` and `TaskOversamplingConfig` dataclasses
2. **Phase 2: JointSampler Implementation** - Implement the unified sampler with multiplicative weight composition
3. **Phase 3: Model Integration** - Update `ChempropModel` to use JointSampler when configured
4. **Phase 4: Testing** - Comprehensive unit tests for all scenarios
5. **Phase 5: Documentation** - Example config and API documentation
6. **Phase 6: HPO Integration and MLflow Logging** - Add HPO search space and weight statistics logging
7. **Phase 7: Configuration Migration** - Migrate all YAML configs to new schema with automated script

### Key Implementation Notes

- JointSampler computes weights as: `w_joint[i] = w_task[i] × w_quality[i]`
- Task weights use inverse-power formula: `w_task ∝ task_count^(-α)`
- For multi-label samples: use **rarest task** (smallest label count) as primary task for weight computation
- Curriculum weights come from `CurriculumState.sampling_probs()`
- Weights are recomputed each iteration to capture phase transitions
- Seed behavior: `increment_seed_per_epoch=True` (default) increments seed each epoch for variety
- Alpha validation: warn if alpha outside [0, 1] range
- Weight statistics logging: log entropy, effective samples, min/max to MLflow
- Backward compatibility is critical - existing configs must work unchanged

### Files to Create/Modify

**New Files:**
- `src/admet/model/chemprop/joint_sampler.py` - JointSampler class
- `tests/test_joint_sampler.py` - Comprehensive tests
- `configs/0-experiment/ensemble_joint_sampling.yaml` - Example config

**Modified Files:**
- `src/admet/model/chemprop/config.py` - Add config dataclasses
- `src/admet/model/chemprop/model.py` - Integrate joint sampler

### Step 3: Cleanup

When ALL Phases are checked off (`[x]`) and completed you WILL do the following:

1. You WILL provide a markdown style link and a summary of all changes from #file:../changes/20251216-sampler-harmonization-changes.md to the user:

   - You WILL keep the overall summary brief
   - You WILL add spacing around any lists
   - You MUST wrap any reference to a file in a markdown style link

2. You WILL provide markdown style links to .copilot-tracking/plans/20251216-sampler-harmonization-plan.instructions.md, .copilot-tracking/details/20251216-sampler-harmonization-details.md, and .copilot-tracking/research/20251216-sampler-harmonization-research.md documents. You WILL recommend cleaning these files up as well.

3. **MANDATORY**: You WILL attempt to delete .copilot-tracking/prompts/implement-sampler-harmonization.prompt.md

## Success Criteria

- [ ] Changes tracking file created
- [ ] All plan items implemented with working code
- [ ] All detailed specifications satisfied
- [ ] Project conventions followed (PEP 8, type hints, docstrings)
- [ ] Changes file updated continuously
- [ ] All tests pass (`pytest tests/test_joint_sampler.py -v`)
- [ ] Backward compatibility verified with existing tests

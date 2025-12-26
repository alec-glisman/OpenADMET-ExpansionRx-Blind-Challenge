---
mode: agent
model: Claude Sonnet 4
---

<!-- markdownlint-disable-file -->

# Implementation Prompt: CheMeleon-Chemprop FFN Integration

## Implementation Instructions

### Step 1: Create Changes Tracking File

You WILL create `20251226-chemeleon-ffn-integration-changes.md` in #file:../changes/ if it does not exist.

### Step 2: Execute Implementation

You WILL follow #file:../../.github/instructions/task-implementation.instructions.md
You WILL systematically implement #file:../plans/20251226-chemeleon-ffn-integration-plan.instructions.md task-by-task
You WILL follow ALL project standards and conventions

**CRITICAL**: If ${input:phaseStop:true} is true, you WILL stop after each Phase for user review.
**CRITICAL**: If ${input:taskStop:false} is true, you WILL stop after each Task for user review.

### Step 3: Cleanup

When ALL Phases are checked off (`[x]`) and completed you WILL do the following:

1. You WILL provide a markdown style link and a summary of all changes from #file:../changes/20251226-chemeleon-ffn-integration-changes.md to the user:

   - You WILL keep the overall summary brief
   - You WILL add spacing around any lists
   - You MUST wrap any reference to a file in a markdown style link

2. You WILL provide markdown style links to .copilot-tracking/plans/20251226-chemeleon-ffn-integration-plan.instructions.md, .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md, and .copilot-tracking/research/20251226-chemeleon-ffn-integration-plan.md documents. You WILL recommend cleaning these files up as well.

3. **MANDATORY**: You WILL attempt to delete .copilot-tracking/prompts/implement-chemeleon-ffn-integration.prompt.md

## Implementation Notes

### Phase Priority

Execute phases in this order due to dependencies:

1. **Phase 1**: FFN Factory (no dependencies)
2. **Phase 2**: Config Updates (no dependencies)
3. **Phase 3**: CheMeleon Model (depends on 1, 2)
4. **Phase 4**: Chemprop Refactor (depends on 1)
5. **Phase 5**: HPO Support (can run parallel after Phase 2)
6. **Phase 6**: Config Files (depends on 2, 5)
7. **Phase 7**: Documentation (depends on 3)

### Key Files to Create

- `src/admet/model/ffn_factory.py` - Shared FFN factory
- `tests/test_ffn_factory.py` - Factory tests
- `src/admet/model/chemeleon/hpo_config.py` - HPO config
- `src/admet/model/chemeleon/hpo_search_space.py` - Search space builder
- `src/admet/model/chemeleon/hpo.py` - HPO runner
- `configs/0-experiment/chemeleon.yaml` - Example config
- `configs/1-hpo-single/hpo_chemeleon.yaml` - HPO config

### Key Files to Modify

- `src/admet/model/config.py` - Add FFN params to ChemeleonModelParams
- `src/admet/model/chemeleon/model.py` - Use FFN factory
- `src/admet/model/chemprop/model.py` - Refactor to use FFN factory
- `tests/test_chemeleon_model.py` - Add FFN type tests
- `README.md` - Update model table
- `docs/guide/modeling.rst` - Add CheMeleon FFN docs
- `MODEL_CARD.md` - Update architecture section

### Testing Strategy

After each implementation phase, run:

```bash
# Run affected tests
pytest tests/test_ffn_factory.py tests/test_chemeleon_model.py -v

# Run full test suite before Phase 7
pytest tests/ -v --tb=short
```

## Success Criteria

- [ ] Changes tracking file created
- [ ] All plan items implemented with working code
- [ ] All detailed specifications satisfied
- [ ] Project conventions followed
- [ ] Changes file updated continuously
- [ ] All tests pass

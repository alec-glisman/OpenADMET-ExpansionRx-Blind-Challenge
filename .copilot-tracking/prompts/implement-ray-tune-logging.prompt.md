---
mode: agent
model: Claude Sonnet 4
---

<!-- markdownlint-disable-file -->

# Implementation Prompt: Ray Tune Output Logging to MLflow Artifacts

## User Decisions (Finalized)

The following design decisions have been confirmed by the user:

| Decision | Value | Notes |
|----------|-------|-------|
| Default verbosity | `0` | Quiet mode, only progress indicators |
| Max total logs GB | `1` | Per experiment, enforced by truncation |
| CLI flags | Yes | `--logging-verbose N`, `--no-logging` |
| Fail on upload error | `True` | Immediate failure, no retries |
| Testing approach | Progressive | Tests between implementation phases |
| Config updates | Aggressive | ALL 200+ YAML files get logging section |
| Documentation | Rich | Comprehensive guide with troubleshooting |
| Performance benchmarks | Yes | Compression overhead, memory usage |

## Implementation Instructions

### Step 1: Create Changes Tracking File

You WILL create `20250104-ray-tune-logging-changes.md` in #file:../changes/ if it does not exist.

### Step 2: Execute Implementation

You WILL follow #file:../../.github/instructions/task-implementation.instructions.md
You WILL systematically implement #file:../plans/20250104-ray-tune-logging-plan.instructions.md task-by-task
You WILL follow ALL project standards and conventions

**CRITICAL**: If ${input:phaseStop:true} is true, you WILL stop after each Phase for user review.
**CRITICAL**: If ${input:taskStop:false} is true, you WILL stop after each Task for user review.

### Step 3: Cleanup

When ALL Phases are checked off (`[x]`) and completed you WILL do the following:

1. You WILL provide a markdown style link and a summary of all changes from #file:../changes/20250104-ray-tune-logging-changes.md to the user:

   - You WILL keep the overall summary brief
   - You WILL add spacing around any lists
   - You MUST wrap any reference to a file in a markdown style link

2. You WILL provide markdown style links to .copilot-tracking/plans/20250104-ray-tune-logging-plan.instructions.md, .copilot-tracking/details/20250104-ray-tune-logging-details.md, and .copilot-tracking/research/20250104-ray-tune-logging-research.md documents. You WILL recommend cleaning these files up as well.

3. **MANDATORY**: You WILL attempt to delete .copilot-tracking/prompts/implement-ray-tune-logging.prompt.md

## Implementation Notes

### Phase Priority

Execute phases in this order due to dependencies:

1. **Phase 1**: Core Logging Infrastructure (no dependencies)
2. **Phase 2**: Configuration Schema (no dependencies)
3. **Phase 3**: CLI Integration (depends on 2)
4. **Phase 4**: HPO Integration (depends on 1, 2, 3)
5. **Phase 5**: Ensemble Integration (depends on 1, 2, 3)
6. **Phase 6**: Progressive Testing (depends on 1-5)
7. **Phase 7**: Configuration Updates - ALL 200+ YAML files (depends on 2)
8. **Phase 8**: Rich Documentation (depends on all)

### Key Files to Create

- `src/admet/util/ray_logging.py` - Core logging utilities
- `tests/test_ray_logging.py` - Unit and integration tests with benchmarks
- `docs/guide/logging.rst` - Rich documentation with troubleshooting
- `scripts/add_logging_to_configs.py` - Batch config update script

### Key Files to Modify

- `src/admet/model/config.py` - Add RayLoggingConfig with user defaults
- `src/admet/model/chemprop/hpo_config.py` - Add logging field
- `src/admet/model/chemprop/hpo.py` - Integrate logging
- `src/admet/model/chemeleon/hpo.py` - Integrate logging
- `src/admet/model/chemprop/ensemble.py` - Integrate logging
- `src/admet/cli/model.py` - Add `--logging-verbose`, `--no-logging` flags
- `configs/**/*.yaml` - ALL 200+ YAML files get logging section

### Critical Implementation Notes

**DO NOT** use `sys.stdout`/`sys.stderr` redirection to capture Ray worker output.
Ray workers run in separate processes and will NOT be captured this way.

**DO** use:
- Ray environment variables (`RAY_LOG_TO_DRIVER`, `RAY_LOGGING_LEVEL`)
- Custom `ProgressReporter` for terminal output
- Post-run log collection from trial directories
- Ray Tune callbacks for artifact upload
- `fail_on_upload_error=True` for immediate failures (user decision)
- `max_total_logs_gb=1` for disk space management (user decision)

### Testing Strategy

After each implementation phase, run:

```bash
# Run affected tests
pytest tests/test_ray_logging.py -v

# Run HPO tests to ensure no regressions
pytest tests/test_hpo.py -v --tb=short

# Run full test suite before Phase 8
pytest tests/ -v --tb=short -x
```

### Config Update Strategy (Phase 7)

Use batch Python script to update ALL 200+ YAML files:

```bash
# Run the batch update script
python scripts/add_logging_to_configs.py

# Validate all YAML files parse correctly
python -c "
import yaml
from pathlib import Path
for yaml_file in Path('configs').rglob('*.yaml'):
    with open(yaml_file) as f:
        yaml.safe_load(f)
    print(f'✓ {yaml_file}')
"
```

## Success Criteria

- [ ] Changes tracking file created
- [ ] All plan items implemented with working code
- [ ] All detailed specifications satisfied
- [ ] Project conventions followed
- [ ] Changes file updated continuously
- [ ] All tests pass (unit + integration + benchmarks)
- [ ] HPO runs with minimal terminal output (verbose=0)
- [ ] Logs uploaded to MLflow artifacts
- [ ] CLI flags work: `--logging-verbose`, `--no-logging`
- [ ] Max 1 GB log limit enforced
- [ ] Fail-fast on upload errors
- [ ] ALL 200+ YAML config files updated
- [ ] Rich documentation with troubleshooting guide

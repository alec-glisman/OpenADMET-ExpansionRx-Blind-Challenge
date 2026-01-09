# Create Pull Request

Create GitHub pull requests using gh CLI.

## Workflow

1. Ensure changes are committed
2. Push branch: `git push -u origin <branch>`
3. Create PR with gh CLI

## Create PR

```bash
gh pr create --title "feat: description" --body "$(cat <<'EOF'
## Summary
- Change 1
- Change 2

## Test plan
- [ ] Run pytest tests
- [ ] Verify HPO workflow
- [ ] Check MLflow logging

Generated with Claude Code
EOF
)"
```

## PR Templates

**Feature:**
```
## Summary
Brief description of the feature.

## Changes
- Added X
- Modified Y

## Test plan
- [ ] Unit tests pass
- [ ] Integration tests pass
```

**Bug Fix:**
```
## Problem
Description of the bug.

## Solution
How it was fixed.

## Test plan
- [ ] Regression test added
- [ ] Existing tests pass
```

## View/Manage PRs

```bash
gh pr list
gh pr view <number>
gh pr merge <number>
```

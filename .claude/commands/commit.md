# Git Commit Workflow

Create conventional commits following project standards.

## Commit Format

```
<type>: <description>

[optional body]

Co-Authored-By: Claude <assistant@anthropic.com>
```

## Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code restructuring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

## Workflow

1. Check status: `git status`
2. Review changes: `git diff`
3. Stage files: `git add <files>`
4. Commit with message

## Pre-commit Hooks

Hooks run automatically: black, isort, flake8, pylint, mypy, pytest

Skip specific hooks if needed:
```bash
SKIP=pytest,mypy git commit -m "feat: add feature"
```

## Examples

```bash
git commit -m "feat: add weight decay regularization"
git commit -m "fix: resolve CUDA memory leak in HPO"
git commit -m "test: add curriculum learning tests"
```

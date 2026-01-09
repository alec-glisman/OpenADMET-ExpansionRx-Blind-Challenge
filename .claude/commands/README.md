# Claude Skill Files

This directory contains Claude Code skill files for common workflows in this ADMET ML pipeline.

## How to Use

Invoke a skill by typing its name with a forward slash in Claude Code:

```
/test
/hpo
/ml-researcher
```

Some skills accept arguments:

```
/test-single tests/model/test_ffn_factory.py
```

## Available Skills

### Development Workflow

| Skill | Description |
|-------|-------------|
| `/test` | Run full pytest suite with markers |
| `/test-single` | Run specific test file or function |
| `/lint` | Run flake8, mypy, pylint checks |
| `/format` | Auto-format with black and isort |
| `/commit` | Create conventional commits |
| `/pr` | Create GitHub pull requests |

### ML Training

| Skill | Description |
|-------|-------------|
| `/train` | Train single model (chemprop, chemeleon, xgboost) |
| `/ensemble` | Train ensembles across split/fold structure |
| `/hpo` | Run hyperparameter optimization with Ray Tune |
| `/config-create` | Create new config files from templates |

### Data & Infrastructure

| Skill | Description |
|-------|-------------|
| `/data-split` | Split data with BitBirch clustering |
| `/mlflow` | Start/manage MLflow tracking server |
| `/docs` | Build Sphinx documentation |
| `/leaderboard` | Scrape and analyze challenge leaderboard |
| `/debug` | Profile training, troubleshoot issues |

### ML Strategy

| Skill | Description |
|-------|-------------|
| `/ml-researcher` | Brainstorm ideas to improve model performance |
| `/ml-engineer` | Implement ML improvements step-by-step |

## When to Use Each Skill

**Starting a new experiment:**
1. `/config-create` - Create config from template
2. `/train` or `/hpo` - Run training or optimization
3. `/mlflow` - View results in MLflow UI

**Improving model performance:**
1. `/ml-researcher` - Brainstorm improvement ideas
2. `/ml-engineer` - Implement chosen improvement
3. `/test` - Validate changes
4. `/hpo` - Run HPO to measure impact

**Before committing code:**
1. `/format` - Auto-format code
2. `/lint` - Check for issues
3. `/test` - Run test suite
4. `/commit` - Create conventional commit

**Debugging issues:**
1. `/debug` - Profiling and troubleshooting guidance
2. `/test-single` - Run specific failing test

## Syncing to Other Machines

These files are stored in `.claude/commands/` and can be synced via git:

```bash
git add .claude/commands/
git commit -m "chore: add Claude skill files"
git push
```

On another machine, pull and the skills will be available.

## Creating New Skills

Add a new `.md` file to this directory. The filename (without extension) becomes the skill name.

Example: `my-skill.md` is invoked as `/my-skill`

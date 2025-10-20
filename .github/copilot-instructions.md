# Project general coding standards (Copilot onboarding)

This file helps a coding agent see this repository for the first time and act efficiently. It summarizes what the repo does, how to build and validate changes, and where to find important files. Keep this file trusted; only search the repo if information here is incomplete or found to be incorrect.

## Goals

- Reduce the chance a generated PR is rejected due to CI, validation failures, or misbehavior.
- Minimize bash/build failures and needless exploration.
- Help the agent complete tasks faster by summarizing common commands, file locations, and pitfalls.

## Limitations

- This document must be short (no longer than ~2 pages).
- Instructions are high-level and not task-specific.

## High-level details

- Summary: Briefly describe the repository's purpose here (one or two sentences).
- Repo type and size: Note approximate size and primary language(s)/framework(s)/runtimes (e.g., Python, TypeScript, Node, Docker).
- Typical runtime/tool versions (e.g., Python 3.10, Node 18, Go 1.20) — list versions actually used.

## What to add (for the agent)

When onboarding, include these high-level details in the agent's mental model:

- A short summary of what the repository does.
- High-level repository information: size, project type, languages, frameworks, target runtimes.

## Build & validation instructions

For each scripted step (bootstrap, build, test, run, lint, etc.), document:

- Exact sequence of commands to run.
- Required versions of runtimes/build tools.
- Preconditions (e.g., environment variables, authentication).
- Postconditions (artifacts produced, server started).
- Known pitfalls, errors observed during validation, and reliable workarounds.
- Order that commands must be run in and any steps that should always be done (e.g., "always run npm install before building").
- Timeouts or commands that may be slow.

Include explicit validated examples:

- How to bootstrap the environment.
- How to build the project.
- How to run unit and integration tests.
- How to run linters and formatters.

When documenting commands, indicate which were actually run and validated. If a sequence of commands produced a reliable result, record it exactly.

## Project layout

Describe the major architectural elements and locations:

- Main entrypoints and their relative paths (e.g., src/, cmd/, app/).
- Config files for linting, compilation, testing, and preferences (e.g., package.json, pyproject.toml, .eslintrc, .github/workflows/).
- Tests location and how to run them.
- Any generated artifacts and where they live.
- Notes about non-obvious dependencies.

Document checks that run on PRs:

- List GitHub workflows or CI checks and what they validate.
- Steps to replicate those checks locally.

Give priority lists:

1. Files in the repo root.
2. README contents and key docs (CONTRIBUTING.md).
3. Key source files (main entrypoint) and important directories.
4. Configuration and pipeline files.

Include short code snippets of critical commands or minimal examples where helpful.

## Steps to follow when creating these instructions (agent checklist)

- Inventory the repository: read README.md, CONTRIBUTING.md, and other docs.
- Search for build steps and in-code hints like TODO, HACK, or Quick start notes.
- Inspect scripts and CI workflows (.github/workflows/\*\*).
- Inspect configuration files for linters, formatters, and build tools.
- For any file that matters to build/test/deploy, record the relevant command(s) and any quirks.
- Validate commands by running them where possible; record failures and workarounds.
- For each command sequence that works, document it exactly and label as validated.
- In the absence of complete information, perform targeted searches; otherwise, trust this document.

## Final guidance

- Trust these instructions to avoid unnecessary repo searches; only search when the instructions are incomplete or clearly wrong.
- Keep the file concise and practical. Update this file whenever environment, CI, or major project layout changes.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed startup failure caused by an unescaped `{old, new}` format placeholder and corrected prompt-storage documentation.
- Pinned supported Python versions to 3.11–3.13.
- Validated `--project-dir` and startup `--thread` arguments with clear errors.
- Made inline prompt IDs unique and ignored malformed version-history filenames.
- Fixed multi-model run comparison and added pass-rate deltas.
- Counted missing, unknown, or unparseable metric results as failures instead of passes.
- Validated test-case metrics before writing state files.
- Improved eval cancellation and failure-state persistence.
- Kept interactive sessions alive after concise orchestrator errors.

### Security

- Added strict prompt/test/run/config ID validation and `.evals/` path containment.
- Required registered source files to resolve inside the project after following symlinks.
- Confined deepagents filesystem access to the project directory.
- Made prompt-edit approval fail closed, including non-interactive use unless `--yes` is supplied.
- Stopped loading a global `~/.env`; only project `.env` and `.env.local` are used.

### Added

- Added an offline pytest suite covering path safety, prompt editing, approvals, metrics, comparisons, schema inference, and thread resolution.
- Added Ruff linting and basedpyright type-checking configuration.

## [0.1.0] - 2026-11-23

### Added

- Initial interactive prompt evaluation agent built on LangChain deepagents.
- Prompt registration and version history.
- Test-case management and eval configuration.
- Multi-model eval execution with latency, JSON-schema, fuzzy-match, and LLM-judge metrics.
- Eval reports, run comparison, and SQLite-backed conversation resumption.

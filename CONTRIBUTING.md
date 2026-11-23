# Contributing

Thanks for helping improve promptloop. Issues, bug reports, documentation fixes, tests, and pull requests are welcome.

## Development setup

Requirements:

- Python 3.11–3.13
- uv
- macOS or a POSIX/Linux environment

```bash
git clone https://github.com/Vela3202019/promptloop.git
cd promptloop
uv sync
```

The development dependency group is installed by default and includes pytest, Ruff, and basedpyright.

## Checks

Run the full local check suite before opening a pull request:

```bash
uv run ruff check .
uv run basedpyright src
uv run pytest -q
uv build
```

Use `uv run ruff check --fix .` for mechanically fixable lint findings. Do not apply broad formatting-only churn unless it is the purpose of the change.

Tests are offline unit tests. Do not add tests that require live model providers, network access, or real API keys. Use temporary directories and fake/no-model implementations for filesystem and orchestration behavior.

## Commit style

Use concise conventional-commit messages, for example:

- `fix: reject ambiguous thread prefixes`
- `fix(security): confine filesystem backend`
- `test: cover metric validation`
- `docs: document eval layout`
- `chore: update dependencies`

Keep unrelated refactors out of behavioral changes. Do not commit `.env`, `.env.local`, `.evals/`, virtual environments, build outputs, or credentials.

## Pull requests

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Update documentation for user-facing changes.
4. Run the checks above.
5. Fill in the pull-request template, including verification output.
6. Keep the branch current and respond to review feedback with follow-up commits rather than rewriting history unless asked.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

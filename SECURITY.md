# Security Policy

## Threat model

promptloop evaluates prompts that should be treated as untrusted input. Prompt text, test-case output, model responses, and generated reports can all contain prompt-injection attempts directed at the orchestrating agent.

Current mitigations include:

- deepagents filesystem access runs in virtual mode and is confined to the selected project directory.
- Registered prompt source files must resolve, after following symlinks, inside the project directory.
- Eval state identifiers are validated and resolved paths must remain inside `.evals/`.
- The custom `edit_prompt` flow is fail-closed: missing approval, non-interactive input, Enter, or a diff-rendering failure denies the edit.
- `--yes` intentionally bypasses interactive approval, including in non-interactive sessions. Use it only when the prompt content and agent instructions are trusted.
- Project `.env` is loaded before `.env.local`, which overrides it. A global `~/.env` is not loaded.

These controls reduce risk but do not make autonomous evaluation of hostile prompts safe. Model output can still influence subsequent agent behavior, and chat checkpoints/eval artifacts may contain sensitive content. Run promptloop only in projects whose contents and evaluated prompts you are comfortable exposing to the configured model providers. Do not pass `--yes` for untrusted prompts.

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vela3202019/promptloop/security/advisories/new). Do not open a public issue for a vulnerability.

Include:

- affected version or commit;
- reproduction steps;
- expected and actual impact;
- whether authentication, local files, credentials, or network access are affected;
- any suggested mitigation.

Do not include real API keys or other secrets in reports.

## Supported versions

promptloop is currently alpha software. Security fixes are applied to the main branch and the latest release when practical; older releases are not guaranteed to receive backports.

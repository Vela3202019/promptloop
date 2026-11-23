# Promptloop

[![CI](https://github.com/Vela3202019/promptloop/actions/workflows/ci.yml/badge.svg)](https://github.com/Vela3202019/promptloop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://github.com/Vela3202019/promptloop)

An interactive CLI agent for the full prompt-eval loop: create test cases, run evals, generate reports, and approve prompt diffs without leaving your terminal.

## The Prompt Eval Loop

Agent harnesses are getting better, but prompts still shape what they do. promptloop turns a prompt and eval intent into a repeatable loop:

<img src="https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/prompt_flow.png" alt="Prompt eval loop" style="max-width: 640px; width: 100%; height: auto;">

It saves the methodology, test cases, reports, prompt history, and chat checkpoints under `.evals/` in the target project.

```text
.evals/
  prompts/
    <prompt-id>/
      current.txt      # active prompt text
      meta.json        # source and version metadata
      history/         # v1.txt, v2.txt, ...
  test_cases/
    <prompt-id>/       # one JSON file per test case
  eval_configs/
    <prompt-id>.json   # methodology (metrics, models, judges)
  results/
    <prompt-id>/
      <run-id>/
        raw.json       # machine-readable run results
        report.md      # generated report, when requested
  chat.db              # SQLite checkpoint of conversation threads
```

Example metrics:

- `latency`: response time
- `json_schema`: validates structured output
- `fuzzy_match`: compares text similarity
- `llm_judge`: scores output with a judge prompt

## Requirements

- Python 3.11, 3.12, or 3.13
- macOS or a POSIX/Linux environment
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- `ANTHROPIC_API_KEY` for the default orchestrator, judge, and report models
- `OPENAI_API_KEY` only when you choose an OpenAI model

Copy `.env.example` to `.env` in the project being evaluated and fill in the keys you use. promptloop loads that project's `.env` first, then `.env.local` with override enabled. It does not load a global `~/.env`.

## Install and Run

```bash
git clone https://github.com/Vela3202019/promptloop.git
cd promptloop
uv sync
uv run promptloop --project-dir /path/to/your/project
```

You'll get an interactive chat. Try things like:

- *"Evaluate the prompt at `src/prompts/summarize.txt`"*
- *"Add three more test cases for edge cases"*
- *"Re-run with `openai:gpt-4o-mini` and compare to the last run"*
- *"Propose a fix for the failing JSON schema cases"*

### CLI options

| Option | Description |
| --- | --- |
| `--project-dir PATH` | Project root to evaluate. It must already exist and contain the prompts you want to eval. Defaults to the current directory. |
| `--model MODEL` | Orchestrator model. Defaults to `anthropic:claude-sonnet-4-6`. |
| `--thread ID` | Resume a saved chat thread. Exact IDs and unambiguous prefixes are accepted; unknown or ambiguous IDs exit with an error. |
| `--yes` | Automatically approve prompt edits and filesystem writes, including in non-interactive use. Use with caution. |
| `-h`, `--help` | Show help and exit. |

## Commands

| Command | Description |
| --- | --- |
| `/help` | Show help |
| `/clear` | Start a new conversation thread |
| `/threads` | List saved threads |
| `/thread <id>` | Switch to a thread in-session |
| `/quit` | Exit |

Resume past sessions with `promptloop --thread <id>`. Press **Esc** to interrupt a streaming response. If a write approval was pending when the prior session stopped, promptloop shows that approval and streams the rest of the turn before continuing.

## Quick Demo

**Stage 1: Register a prompt** — point promptloop at a prompt file and it registers it with version tracking:

![register prompt](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/01-register-prompt.gif)

**Stage 2: Add test cases** — the agent proposes test cases based on your prompt and intent; you pick what to keep:

![add test cases](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/02-add-test-cases.gif)

**Stage 3: Run the eval** — see pass/fail results per test case with metrics and latency:

![run eval](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/03-run-eval.gif)

**Stage 4: Review and update the prompt** — when cases fail, ask for a fix. promptloop reads the report, shows a colored diff in the terminal, and waits at `[y/N]` before writing anything. Enter, `n`, Escape, or a non-interactive session denies the edit unless `--yes` was passed:

![propose and update prompt](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/04-propose-and-update-prompt.gif)

**Stage 5: Next iteration** — re-run the eval on the new prompt version and keep iterating:

![next iteration](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/05-next-iteration.gif)

Every run persists the full loop — versioned prompts, test cases, eval configs, and reports — so nothing is lost between sessions:

![promptloop show result](https://github.com/Vela3202019/promptloop/releases/download/v0.1-media/promptloop.-.show.result.gif)

## How It Works

The agent combines deepagents' project-confined filesystem tools with a small set of eval-specific tools:

- `register_prompt`, `read_current_prompt`, `list_prompts`, `edit_prompt`, `show_prompt_history`
- `add_test_case`, `list_test_cases`, `delete_test_case`, `infer_json_schema`, `save_eval_config`, `get_eval_config`
- `run_eval`, `list_eval_runs`
- `generate_report`, `read_report`, `compare_runs`

deepagents also supplies filesystem tools such as `ls`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`. They run in the backend's virtual mode, which confines paths to the selected project directory. Read-only filesystem calls are ungated. `write_file` and `edit_file` pause before execution, show the target path plus a content preview or colored diff, and use the same fail-closed `[y/N]` decision as `edit_prompt`; use `edit_prompt` for registered prompts.

For more detail on the agent runtime behind this project, see [The Harness Behind Deep Agent](docs/The_Harness_Behind_Deep_Agent.md).

## Security

Treat prompts and model outputs as untrusted input. promptloop confines backend filesystem paths to the project directory and requires explicit approval for registered prompt edits and built-in filesystem writes unless `--yes` is supplied, but prompt injection can still influence agent behavior. Read the threat model and reporting instructions in [SECURITY.md](SECURITY.md).

## Development

Development uses uv, pytest, Ruff, and basedpyright. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, commit style, and the pull-request flow.

Early / experimental. Feedback and issues welcome.

## What's Next

This is a starting point. A few directions I'm thinking about:

**1. Interface** — the CLI works, but the prompt loop deserves a less friction interface. Extending toward a richer chat UI or editor integration so the loop feels more natural to run.

**2. Less human in the loop** — the current flow still relies on you to drive each stage. 

> *"It would be extremely cool to be able to write one or two lines of prompt in my harness, and have a light model iterate with me a few times writing/proposing requirements, guidelines and explanations, refining the prompt until it's ready to be sent to the actual LLM."* ---- [HN commenters](https://news.ycombinator.com/item?id=48325073):

Ideally it would be a lightweight thing(could be tool, skill, cli, chat interface or a snippet) that co-authors the prompt with you, proposes requirements, flags gaps, and tightens the spec before it ever hits your production model.

Built on LangChain [deepagents](https://github.com/langchain-ai/deepagents).

## License

MIT — see [LICENSE](LICENSE).

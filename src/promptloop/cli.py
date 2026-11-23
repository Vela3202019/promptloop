import argparse
import asyncio
import difflib
import os
import select
import sqlite3
import sys
import termios
import time
import tty
import uuid
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from prompt_toolkit import HTML
from rich.console import Console
from rich.markup import escape

from .agent import create_eval_agent
from .paths import resolve_in_evals, validate_id
from .tools import ApprovalGate

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent


def _load_env(project_dir: Path) -> None:
    """Load env files from broad defaults to project-specific overrides."""
    load_dotenv(project_dir / ".env", override=False, verbose=False)
    load_dotenv(project_dir / ".env.local", override=True, verbose=False)


console = Console()

WELCOME = """\
[bold green]promptloop[/bold green] — prompt engineering & evaluation agent
Type your message and press [bold]Enter[/bold] to send.
Commands: /help /clear /quit /threads /thread <id>
"""

HELP = """\
[bold]Commands[/bold]
  /help      — show this message
  /clear     — start a new conversation thread
  /threads   — list saved chat threads (resume with: promptloop --thread <id>)
  /thread <id> — switch to thread (in-session)
  /quit      — exit

[bold]Resuming a chat[/bold]
  Run [bold]promptloop --thread <thread_id>[/bold] to continue a previous conversation.
  Use [bold]/threads[/bold] to see available thread IDs.

[bold]Input[/bold]
  [bold]Enter[/bold] to send.
  Press [bold]Esc[/bold] while streaming to interrupt the current response.
  Multi-line paste keeps formatting for prompts, JSON examples, and markdown.

[bold]Getting started[/bold]
  Tell me a prompt file path: "evaluate the prompt at src/prompts/summarize.txt"
  Or just describe what you want to evaluate and I'll help you find it.
"""

# Increase above deepagents default of 25 to handle multi-step eval workflows
RECURSION_LIMIT = 100

CHAT_DB_NAME = "chat.db"
ESCAPE_KEY = "\x1b"


def _chat_db_path(project_dir: Path) -> Path:
    return resolve_in_evals(project_dir, CHAT_DB_NAME)


def _list_thread_ids(project_dir: Path) -> list[str]:
    """List thread IDs from the chat SQLite DB."""
    db_path = _chat_db_path(project_dir)
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        with conn:
            cur = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            )
            return [row[0] for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def _resolve_thread_id(project_dir: Path, requested_id: str) -> tuple[str | None, str | None]:
    thread_ids = _list_thread_ids(project_dir)
    if requested_id in thread_ids:
        return requested_id, None

    matches = [tid for tid in thread_ids if tid.startswith(requested_id)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"Ambiguous thread prefix '{requested_id}' matches {len(matches)} threads."
    return None, f"No saved thread found for '{requested_id}'. Use /threads to list saved threads."


def _create_prompt_session() -> "PromptSession[str]":
    """Create a PromptSession with Enter=send and formatting-preserving paste."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: "KeyPressEvent") -> None:
        event.current_buffer.validate_and_handle()

    @kb.add(Keys.BracketedPaste)
    def _bracketed_paste(event: "KeyPressEvent") -> None:
        """Preserve pasted multi-paragraph input for prompts and examples."""
        pasted_text = event.data if isinstance(event.data, str) else ""
        normalized = _normalize_user_input(pasted_text)
        if not normalized:
            return

        buffer = event.current_buffer
        if buffer.document.text_before_cursor and not buffer.document.text_before_cursor.endswith(("\n", " ")):
            normalized = f"\n{normalized}"
        buffer.insert_text(normalized)

    @kb.add("c-c")
    def _ctrl_c(event: "KeyPressEvent") -> None:
        # Ask prompt_toolkit to end the current prompt with KeyboardInterrupt
        # so callers can handle it without an asyncio traceback.
        event.app.exit(exception=KeyboardInterrupt())

    return PromptSession(key_bindings=kb, multiline=True)


async def _read_user_input(session: "PromptSession[str]", prompt: Any) -> str:
    """Read user input via prompt_toolkit. Enter sends the message."""
    return await session.prompt_async(prompt)


def _normalize_user_input(raw_input: str) -> str:
    """Normalize terminal line endings without changing user formatting."""
    return raw_input.replace("\r\n", "\n").replace("\r", "\n")


def _hide_cursor() -> None:
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _format_tool_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)
        if len(v_str) > 35:
            v_str = v_str[:32] + "…"
        parts.append(f"{k}={v_str!r}" if isinstance(v, str) else f"{k}={v_str}")
    preview = ", ".join(parts)
    if len(args) > 2:
        preview += ", …"
    return f"({preview})"


def _format_tool_result(event: dict) -> str:
    name = event.get("name", "")
    output = event.get("data", {}).get("output")
    if not output:
        return ""
    out_str = str(output)
    lines = [line for line in out_str.splitlines() if line.strip()]
    if any(x in name for x in ("read_file", "read")):
        return f"{len(lines)} lines" if lines else ""
    if any(x in name for x in ("glob", "grep", "search", "ls")):
        return f"{len(lines)} results" if lines else ""
    if out_str.strip() and len(out_str.strip()) < 60:
        return out_str.strip()
    return ""


def _render_prompt_diff(project_dir: Path, prompt_id: str, edits: list[dict]) -> str | None:
    """Print a colored unified diff before the tool writes. Returns an error string on failure."""
    try:
        current_file = resolve_in_evals(project_dir, "prompts", validate_id(prompt_id, "prompt_id"), "current.txt")
        if not current_file.exists():
            return "diff unavailable: current prompt file not found"
        current = current_file.read_text()

        proposed = current
        for index, edit in enumerate(edits, start=1):
            old, new = edit.get("old", ""), edit.get("new", "")
            if not isinstance(old, str) or not isinstance(new, str) or not old or current.count(old) != 1:
                return f"diff unavailable: edit {index} is invalid or ambiguous"
            proposed = proposed.replace(old, new, 1)

        diff_lines = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"{prompt_id} (current)",
            tofile=f"{prompt_id} (proposed)",
            n=3,
        ))
        if not diff_lines:
            return "diff unavailable: no changes detected"

        console.print()
        _print_diff_lines(diff_lines)
        console.print()
    except Exception as e:  # noqa: BLE001
        return f"diff unavailable: {e}"
    return None


def _terminal_safe(text: str) -> str:
    """Replace control characters that could manipulate the user's terminal."""
    rendered = []
    for char in text:
        if char == "\t" or char.isprintable():
            rendered.append(char)
        elif ord(char) <= 0xFF:
            rendered.append(f"\\x{ord(char):02x}")
        elif ord(char) <= 0xFFFF:
            rendered.append(f"\\u{ord(char):04x}")
        else:
            rendered.append(f"\\U{ord(char):08x}")
    return "".join(rendered)


def _print_diff_lines(diff_lines: list[str]) -> None:
    for line in diff_lines:
        line = _terminal_safe(line.rstrip("\n"))
        rendered = escape(line)
        if line.startswith(("+++", "---")):
            console.print(f"[dim]{rendered}[/dim]")
        elif line.startswith("+"):
            console.print(f"[green]{rendered}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{rendered}[/red]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{rendered}[/cyan]")
        else:
            console.print(f"[dim]{rendered}[/dim]")


def _virtual_project_path(project_dir: Path, file_path: str) -> Path:
    if not file_path:
        raise ValueError("missing file path")
    posix_path = file_path.replace("\\", "/")
    if ".." in PurePosixPath(posix_path).parts or file_path.startswith("~"):
        raise ValueError(f"path traversal is not allowed: {file_path}")
    if len(file_path) >= 2 and file_path[0].isalpha() and file_path[1] == ":":
        raise ValueError(f"Windows absolute paths are not supported: {file_path}")
    normalized = os.path.normpath(posix_path).replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    candidate = (project_dir / normalized.lstrip("/")).resolve()
    try:
        candidate.relative_to(project_dir.resolve())
    except ValueError:
        raise ValueError(f"path escapes project directory: {file_path}") from None
    return candidate


def _render_file_write_action(project_dir: Path, name: str, args: dict[str, Any]) -> str | None:
    """Render a write_file/edit_file preview. Any error means the caller must deny."""
    try:
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str):
            return "preview unavailable: missing or invalid file_path"
        display_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
        path = _virtual_project_path(project_dir, raw_path)

        if name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                return "preview unavailable: missing or invalid content"
            if path.exists():
                if not path.is_file():
                    return "preview unavailable: target exists but is not a file"
                console.print(
                    "[yellow]write_file cannot overwrite existing files; "
                    "approval will return a tool error. Use edit_file instead.[/yellow]"
                )
                current = path.read_text(encoding="utf-8")
                if current == content:
                    console.print("[dim]content is unchanged[/dim]")
                else:
                    diff_lines = list(difflib.unified_diff(
                        current.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=f"{display_path} (current)",
                        tofile=f"{display_path} (proposed)",
                        n=3,
                    ))
                    console.print()
                    _print_diff_lines(diff_lines)
                    console.print()
            else:
                console.print(f"[dim]new file ({len(content)} characters)[/dim]")
                lines = content.splitlines()
                truncated = len(content) > 1200 or len(lines) > 30
                preview = "\n".join(lines[:30])[:1200]
                sys.stdout.write("  proposed content:\n")
                if preview:
                    preview_lines = "\n".join(f"    {_terminal_safe(line)}" for line in preview.splitlines())
                    sys.stdout.write(preview_lines + "\n")
                if truncated:
                    sys.stdout.write("    …\n    [truncated preview]\n")
            return None

        if name == "edit_file":
            old_string = args.get("old_string")
            new_string = args.get("new_string")
            replace_all = args.get("replace_all", False)
            if not isinstance(old_string, str):
                return "preview unavailable: missing or invalid old_string"
            if not isinstance(new_string, str):
                return "preview unavailable: missing or invalid new_string"
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")
            if old_string == new_string:
                return "preview unavailable: old_string and new_string are identical"
            if not isinstance(replace_all, bool):
                return "preview unavailable: replace_all must be a boolean"
            if not path.exists() or not path.is_file():
                return "preview unavailable: target file not found"
            current = path.read_text(encoding="utf-8")
            count = current.count(old_string)
            if count == 0:
                return "preview unavailable: old_string was not found"
            if count != 1 and not replace_all:
                return f"preview unavailable: old_string matches {count} places"
            if replace_all:
                proposed = current.replace(old_string, new_string)
            else:
                proposed = current.replace(old_string, new_string, 1)
            diff_lines = list(difflib.unified_diff(
                current.splitlines(keepends=True),
                proposed.splitlines(keepends=True),
                fromfile=f"{display_path} (current)",
                tofile=f"{display_path} (proposed)",
                n=3,
            ))
            console.print()
            _print_diff_lines(diff_lines)
            console.print()
            return None

        return f"preview unavailable: unsupported write action '{name}'"
    except Exception as e:  # noqa: BLE001
        return f"preview unavailable: {e}"


def _interrupt_action_requests(value: Any) -> list[Any]:
    if not isinstance(value, dict):
        raise ValueError("unsupported interrupt payload")
    action_requests = value.get("action_requests")
    if not isinstance(action_requests, list) or not action_requests:
        raise ValueError("interrupt payload has no pending actions")
    return action_requests


def _combine_interrupt_responses(responses: list[tuple[str, dict[str, Any]]]) -> Any:
    if len(responses) == 1:
        return responses[0][1]
    return {interrupt_id: response for interrupt_id, response in responses}


async def _decide_file_write_interrupt(
    project_dir: Path,
    interrupt_value: Any,
    auto_approve: bool,
) -> dict[str, Any]:
    decisions = []
    for action in _interrupt_action_requests(interrupt_value):
        if not isinstance(action, dict):
            console.print("[red]invalid interrupt action; write denied[/red]")
            decisions.append({
                "type": "reject",
                "message": "The pending file-write action was malformed; the file was not changed.",
            })
            continue
        name = action.get("name")
        args = action.get("args")
        if not isinstance(name, str) or not isinstance(args, dict):
            console.print("[red]interrupt action is missing a name or arguments; write denied[/red]")
            decisions.append({
                "type": "reject",
                "message": "The pending file-write action was malformed; the file was not changed.",
            })
            continue
        raw_path = args.get("file_path")
        display_path = raw_path if isinstance(raw_path, str) else "<unknown path>"
        header = (
            f"\n[yellow]approval required[/yellow] [bold]{escape(name)}[/bold] "
            f"[cyan]{escape(_terminal_safe(display_path))}[/cyan]"
        )
        console.print(header)

        render_error = _render_file_write_action(project_dir, name, args)
        if render_error:
            console.print(f"[red]{escape(_terminal_safe(render_error))}; write denied[/red]")
            approved = False
        else:
            sys.stdout.write("  [y/N] Apply  ")
            sys.stdout.flush()
            approved = await _read_approval_key(auto_approve)

        if approved:
            console.print("[green]applying[/green]")
            decisions.append({"type": "approve"})
        else:
            console.print("[dim]cancelled[/dim]")
            decisions.append({
                "type": "reject",
                "message": f"The user declined approval for `{name}`; the file was not changed.",
            })
    return {"decisions": decisions}


async def _resume_file_write_interrupts(
    project_dir: Path,
    interrupts: Any,
    auto_approve: bool,
) -> Command:
    responses = []
    for item in interrupts:
        response = await _decide_file_write_interrupt(project_dir, item.value, auto_approve)
        responses.append((item.id, response))
    return Command(resume=_combine_interrupt_responses(responses))


async def _read_approval_key(auto_approve: bool) -> bool:
    """Read a single y/n keypress. Defaults to deny unless auto-approval is enabled."""
    if auto_approve:
        return True
    if not sys.stdin.isatty():
        return False
    fd = sys.stdin.fileno()
    original_mode = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([fd], [], [], 0.05)
            if not readable:
                await asyncio.sleep(0)
                continue
            key = os.read(fd, 1).decode("utf-8", errors="ignore").lower()
            if key == "y":
                return True
            if key in ("n", ESCAPE_KEY, "q", "\r", "\n"):
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_mode)


async def _stream_agent_input(
    agent,
    stream_input: Any,
    config: dict[str, Any],
    project_dir: Path,
    gate: ApprovalGate,
    auto_approve: bool,
) -> None:
    while True:
        status = await _stream_single_input(agent, stream_input, config, project_dir, gate, auto_approve)
        if status != "completed":
            break

        try:
            state = await agent.aget_state(config)
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)[:200] or type(exc).__name__
            console.print(f"\n[red]error reading agent state: {detail}[/red]", end="")
            break
        if not state.interrupts:
            break

        try:
            stream_input = await _resume_file_write_interrupts(project_dir, state.interrupts, auto_approve)
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)[:200] or type(exc).__name__
            console.print(f"\n[red]error processing write approval: {detail}[/red]", end="")
            break
    console.print()


async def _stream_pending_file_writes(
    agent,
    thread_id: str,
    project_dir: Path,
    gate: ApprovalGate,
    auto_approve: bool = False,
) -> bool:
    """Resume a thread's pending write approvals through the visible streaming path."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    try:
        state = await agent.aget_state(config)
        if not state.interrupts:
            return False
        console.print("[yellow]pending file-write approval from the interrupted session[/yellow]")
        stream_input = await _resume_file_write_interrupts(project_dir, state.interrupts, auto_approve)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)[:200] or type(exc).__name__
        console.print(f"[red]error processing pending write approval: {detail}[/red]")
        return True

    await _stream_agent_input(agent, stream_input, config, project_dir, gate, auto_approve)
    return True


async def _stream_response(
    agent,
    message: str,
    thread_id: str,
    project_dir: Path,
    gate: ApprovalGate,
    auto_approve: bool = False,
) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    stream_input: Any = {"messages": [HumanMessage(content=message)]}
    await _stream_agent_input(agent, stream_input, config, project_dir, gate, auto_approve)


async def _stream_single_input(
    agent,
    stream_input: Any,
    config: dict[str, Any],
    project_dir: Path,
    gate: ApprovalGate,
    auto_approve: bool = False,
) -> str:
    console.print("\n[bold green]agent[/bold green]", end=" ")
    _hide_cursor()

    # ── In-place status ticker ────────────────────────────────────────────────
    # Shows "verb… Ns" in place, updating every 300 ms.
    # Call _status_start(verb) to begin, await _status_stop() before any output.
    _st_task: asyncio.Task[None] | None = None
    _st_active = False
    _st_verb = ""
    _st_start = 0.0
    _st_rendered = 0  # visible chars currently on stdout from the ticker

    def _st_erase() -> None:
        nonlocal _st_rendered
        if _st_rendered:
            sys.stdout.write("\b" * _st_rendered + " " * _st_rendered + "\b" * _st_rendered)
            sys.stdout.flush()
            _st_rendered = 0

    async def _st_run() -> None:
        nonlocal _st_rendered
        while _st_active:
            elapsed = int(time.monotonic() - _st_start)
            if _st_verb.startswith("Drafting "):
                text = f"{_st_verb}… {elapsed}s (Esc cancels)"
            else:
                text = f"{_st_verb}… {elapsed}s" if _st_verb else f"… {elapsed}s"
            _st_erase()
            sys.stdout.write(text)
            sys.stdout.flush()
            _st_rendered = len(text)
            await asyncio.sleep(0.3)

    def _status_start(verb: str) -> None:
        nonlocal _st_task, _st_active, _st_verb, _st_start, _st_rendered
        _st_active = False
        if _st_task and not _st_task.done():
            _st_task.cancel()
        _st_erase()
        _st_verb = verb
        _st_start = time.monotonic()
        _st_active = True
        _st_rendered = 0
        _st_task = asyncio.create_task(_st_run())

    async def _status_stop() -> None:
        nonlocal _st_active, _st_task
        _st_active = False
        _st_erase()
        if _st_task and not _st_task.done():
            _st_task.cancel()
            with suppress(asyncio.CancelledError):
                await _st_task
        _st_task = None

    # ─────────────────────────────────────────────────────────────────────────

    interrupted = False
    stream_error: Exception | None = None
    stop_escape_listener = asyncio.Event()
    escape_task: asyncio.Task[bool] | None = None
    active_tool_run_ids: set[str] = set()
    streaming_text = False
    pending_tool_name: str | None = None

    async def _wait_for_escape(stop_event: asyncio.Event) -> bool:
        if not sys.stdin.isatty():
            await stop_event.wait()
            return False

        fd = sys.stdin.fileno()
        original_mode = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    await asyncio.sleep(0)
                    continue
                key = os.read(fd, 1).decode("utf-8", errors="ignore")
                if key == ESCAPE_KEY:
                    return True
            return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_mode)

    async def _consume_stream() -> None:
        nonlocal pending_tool_name, streaming_text, escape_task, stop_escape_listener

        def _is_inside_tool(event: dict[str, Any]) -> bool:
            parent_ids = event.get("parent_ids") or []
            return any(pid in active_tool_run_ids for pid in parent_ids)

        _status_start("Thinking")

        async for event in agent.astream_events(
            stream_input,
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                if _is_inside_tool(event):
                    continue
                chunk = event["data"].get("chunk")
                if not chunk:
                    continue

                # When model streams a tool call, show which tool is being called
                # before on_tool_start fires (which only fires after model finishes).
                for tc in getattr(chunk, "tool_call_chunks", None) or []:
                    if isinstance(tc, dict) and tc.get("name"):
                        tool_name = tc["name"]
                        if pending_tool_name != tool_name:
                            pending_tool_name = tool_name
                            if streaming_text:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                                streaming_text = False
                            _status_start(f"Drafting {tool_name} input")
                        break

                content = chunk.content
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if text:
                    if not streaming_text:
                        await _status_stop()
                        streaming_text = True
                    sys.stdout.write(text)
                    sys.stdout.flush()

            elif kind == "on_tool_start":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.add(run_id)
                await _status_stop()
                streaming_text = False
                name = event.get("name", "tool")
                args = event.get("data", {}).get("input") or {}

                if name == "edit_prompt":
                    prompt_id = args.get("prompt_id", "")
                    gate.expect(prompt_id)
                    prefix = "" if pending_tool_name else "\n"
                    pending_tool_name = None
                    sys.stdout.write(f"{prefix}  → {name}")
                    sys.stdout.flush()

                    diff_error = _render_prompt_diff(project_dir, prompt_id, args.get("edits", []))

                    # Pause escape listener so it doesn't race for stdin
                    stop_escape_listener.set()
                    if escape_task and not escape_task.done():
                        escape_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await escape_task

                    if diff_error:
                        console.print(f"[red]{escape(_terminal_safe(diff_error))}; edit denied[/red]")
                        approved = False
                    else:
                        sys.stdout.write("  [y/N] Apply  ")
                        sys.stdout.flush()
                        approved = await _read_approval_key(auto_approve)
                    gate.resolve(prompt_id, approved)

                    if approved:
                        console.print("[green]applying[/green]")
                    else:
                        console.print("[dim]cancelled[/dim]")

                    # Restart escape listener
                    stop_escape_listener = asyncio.Event()
                    escape_task = asyncio.create_task(_wait_for_escape(stop_escape_listener))
                    _status_start("")
                    continue

                args_preview = _format_tool_args(args)
                prefix = "" if pending_tool_name else "\n"
                pending_tool_name = None
                sys.stdout.write(f"{prefix}  → {name}{args_preview}")
                sys.stdout.flush()
                _status_start("")  # show elapsed on the tool line

            elif kind == "on_tool_end":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.discard(run_id)
                await _status_stop()
                result_summary = _format_tool_result(event)
                suffix = f" · {result_summary}" if result_summary else ""
                sys.stdout.write(f" ✓{suffix}\n")
                sys.stdout.flush()
                _status_start("Thinking")  # restart for next model step

            elif kind == "on_chat_model_end" and not _is_inside_tool(event) and streaming_text:
                streaming_text = False

    try:
        stream_task = asyncio.create_task(_consume_stream())
        escape_task = asyncio.create_task(_wait_for_escape(stop_escape_listener))
        done, _ = await asyncio.wait(
            {stream_task, escape_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if escape_task in done and escape_task.result():
            interrupted = True
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task
        elif stream_task in done:
            try:
                await stream_task
            except Exception as exc:  # noqa: BLE001
                stream_error = exc
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_escape_listener.set()
        await _status_stop()
        if escape_task is not None and not escape_task.done():
            escape_task.cancel()
            with suppress(asyncio.CancelledError):
                await escape_task
        _show_cursor()

    if stream_error is not None:
        error_lines = str(stream_error).strip().splitlines()
        detail = error_lines[0] if error_lines else type(stream_error).__name__
        if len(detail) > 200:
            detail = detail[:197] + "..."
        console.print(f"\n[red]error ({type(stream_error).__name__}): {detail}[/red]", end="")
        return "error"

    if interrupted:
        console.print("\n[yellow]interrupted[/yellow]", end="")
        return "interrupted"
    return "completed"


async def amain() -> None:
    parser = argparse.ArgumentParser(
        prog="promptloop",
        description="Interactive prompt evaluation agent",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root to evaluate. Must contain the prompts you want to eval (default: cwd)",
    )
    parser.add_argument(
        "--model",
        default="anthropic:claude-sonnet-4-6",
        help="Orchestrator model (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Automatically approve prompt edits and filesystem writes, "
            "including non-interactive use (use with caution)"
        ),
    )
    parser.add_argument(
        "--thread",
        metavar="ID",
        help="Resume a previous chat by thread ID (use /threads to list)",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.expanduser().resolve()
    if not project_dir.exists():
        parser.error(f"--project-dir does not exist: {project_dir}")
    if not project_dir.is_dir():
        parser.error(f"--project-dir is not a directory: {project_dir}")
    _load_env(project_dir)

    try:
        db_path = _chat_db_path(project_dir)
    except ValueError as e:
        parser.error(str(e))

    if args.thread:
        resolved_thread_id, error = _resolve_thread_id(project_dir, args.thread)
        if error:
            parser.error(error)
        thread_id = resolved_thread_id or args.thread
    else:
        thread_id = str(uuid.uuid4())

    # Ensure .evals exists for the already-contained chat DB path.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        gate = ApprovalGate()
        agent = create_eval_agent(project_dir, args.model, checkpointer=checkpointer, gate=gate)

        console.print(WELCOME)
        console.print(f"[dim]project: {project_dir}[/dim]")
        if args.thread:
            console.print(f"[dim]thread: {thread_id}[/dim] (resuming)")
            await _stream_pending_file_writes(agent, thread_id, project_dir, gate, auto_approve=args.yes)
        console.print()

        session = _create_prompt_session()
        prompt_template = (
            '<style fg="ansicyan"><b>you</b></style> '
            '<style fg="#6b7280">[{thread}…]: </style>'
        )

        while True:
            try:
                prompt_str = HTML(prompt_template).format(thread=thread_id[:8])
                user_input = _normalize_user_input(await _read_user_input(session, prompt_str))
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break

            stripped_input = user_input.strip()
            if not stripped_input:
                continue

            if stripped_input.startswith("/") and "\n" not in stripped_input:
                parts = stripped_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[dim]bye[/dim]")
                    break
                elif cmd == "/help":
                    console.print(HELP)
                elif cmd == "/clear":
                    thread_id = str(uuid.uuid4())
                    console.print(
                        "[dim]conversation cleared — new thread started[/dim]"
                    )
                elif cmd == "/threads":
                    thread_ids = _list_thread_ids(project_dir)
                    if not thread_ids:
                        console.print("[dim]No saved threads yet.[/dim]")
                    else:
                        console.print("[bold]Saved threads:[/bold]")
                        for tid in thread_ids:
                            marker = " ←" if tid == thread_id else ""
                            console.print(f"  {tid}{marker}")
                        console.print(
                            "[dim]Resume with: promptloop --thread <id>[/dim]"
                        )
                elif cmd == "/thread" and arg:
                    resolved_thread_id, error = _resolve_thread_id(project_dir, arg)
                    if error:
                        console.print(f"[red]{error}[/red]")
                    else:
                        thread_id = resolved_thread_id or arg
                        console.print(f"[dim]switched to thread {thread_id}[/dim]")
                        await _stream_pending_file_writes(agent, thread_id, project_dir, gate, auto_approve=args.yes)
                elif cmd == "/thread":
                    console.print("[red]Usage: /thread <thread_id>[/red]")
                else:
                    console.print(f"[red]unknown command: {cmd}[/red]")
                continue

            await _stream_response(agent, user_input, thread_id, project_dir, gate, auto_approve=args.yes)


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        # Final safety net for interrupts outside prompt input flow.
        console.print("\n[dim]bye[/dim]")

import asyncio
import difflib
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from ..paths import resolve_in_evals, resolve_in_project, validate_id


def prompt_approval_key(prompt_id: str, current: str, edits: list[dict]) -> str:
    payload = json.dumps({"prompt_id": prompt_id, "current": current, "edits": edits}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class _ApprovalWaiter:
    def __init__(self, future: asyncio.Future[bool]) -> None:
        self.future = future
        self.listener_seen = False


class ApprovalGate:
    """Awaitable approval state shared between the CLI and tools.

    The CLI calls expect() as soon as it observes edit_prompt's on_tool_start
    event, then resolve() after reading the keypress. The tool waits for that
    decision because event consumption and tool execution are concurrent.
    """

    def __init__(self, listener_timeout: float = 1.0, decision_timeout: float = 300.0) -> None:
        self._decisions: dict[str, bool] = {}
        self._expected: set[str] = set()
        self._waiters: dict[str, list[_ApprovalWaiter]] = {}
        self._listener_timeout = listener_timeout
        self._decision_timeout = decision_timeout

    def expect(self, key: str) -> None:
        self._expected.add(key)
        for waiter in self._waiters.get(key, []):
            waiter.listener_seen = True

    def resolve(self, key: str, approved: bool) -> None:
        self._expected.discard(key)
        waiters = self._waiters.get(key)
        if waiters:
            waiter = waiters.pop(0)
            if not waiters:
                self._waiters.pop(key, None)
            if not waiter.future.done():
                waiter.future.set_result(approved)
        else:
            self._decisions[key] = approved

    def drop(self, key: str) -> None:
        self._decisions.pop(key, None)

    async def consume(self, key: str) -> bool:
        if key in self._decisions:
            return self._decisions.pop(key)

        loop = asyncio.get_running_loop()
        waiter = _ApprovalWaiter(loop.create_future())
        waiter.listener_seen = key in self._expected
        self._waiters.setdefault(key, []).append(waiter)
        try:
            try:
                return await asyncio.wait_for(asyncio.shield(waiter.future), timeout=self._listener_timeout)
            except TimeoutError:
                if not waiter.listener_seen:
                    return False
                try:
                    return await asyncio.wait_for(waiter.future, timeout=self._decision_timeout)
                except TimeoutError:
                    return False
        finally:
            waiters = self._waiters.get(key)
            if waiters is not None:
                if waiter in waiters:
                    waiters.remove(waiter)
                if not waiters:
                    self._waiters.pop(key, None)
            if not waiter.future.done():
                waiter.future.cancel()


def make_prompt_tools(project_dir: Path, gate: ApprovalGate):
    prompts_dir = resolve_in_evals(project_dir, "prompts")

    def _version_number(path: Path) -> int | None:
        try:
            return int(path.stem[1:])
        except ValueError:
            return None

    def _next_version(history_dir: Path) -> int:
        versions = []
        for path in history_dir.glob("v*.txt"):
            version = _version_number(path)
            if version is not None:
                versions.append(version)
        if not versions:
            return 1
        return max(versions) + 1

    def _save_prompt(
        content: str,
        prompt_id: str,
        source_path: str | None,
        source_type: str,
        resolved_source_path: str | None = None,
    ) -> tuple[int, Path]:
        prompt_dir = _prompt_dir(prompt_id)
        history_dir = _prompt_artifact(prompt_id, "history")
        current_file = _prompt_artifact(prompt_id, "current.txt")
        meta_file = _prompt_artifact(prompt_id, "meta.json")

        prompt_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(exist_ok=True)
        version = _next_version(history_dir)
        history_file = _prompt_artifact(prompt_id, "history", f"v{version}.txt")

        current_file.write_text(content)
        history_file.write_text(content)
        meta = {
            "source_path": source_path,
            "resolved_source_path": resolved_source_path,
            "source_type": source_type,
            "prompt_id": prompt_id,
            "registered_at": datetime.now().isoformat(),
            "version": version,
        }
        meta_file.write_text(json.dumps(meta, indent=2))
        return version, prompt_dir

    def _inline_prompt_id() -> str:
        return f"inline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _prompt_dir(prompt_id: str) -> Path:
        return resolve_in_evals(project_dir, "prompts", validate_id(prompt_id, "prompt_id"))

    def _prompt_artifact(prompt_id: str, *parts: str) -> Path:
        return resolve_in_evals(project_dir, "prompts", validate_id(prompt_id, "prompt_id"), *parts)

    def _dedupe_file_prompt_id(prompt_id: str, source_path: str) -> str:
        candidate = prompt_id
        suffix = 2
        while True:
            try:
                prompt_dir = _prompt_dir(candidate)
            except ValueError:
                return candidate
            if not prompt_dir.exists():
                return candidate
            meta_file = _prompt_artifact(prompt_id, "meta.json")
            existing_source = None
            if meta_file.exists():
                try:
                    existing_source = json.loads(meta_file.read_text()).get("source_path")
                except (OSError, json.JSONDecodeError):
                    existing_source = None
            if existing_source == source_path:
                return candidate
            candidate = f"{prompt_id}_{suffix}"
            suffix += 1

    def _source_path_if_file(source: str) -> tuple[Path, Path] | None:
        if "\n" in source or len(source) > 240:
            return None

        def _is_file(path: Path) -> bool:
            try:
                return path.exists() and path.is_file()
            except OSError:
                return False

        raw_path = Path(source)
        physical_path = raw_path if raw_path.is_absolute() else project_dir / raw_path
        physical_exists = _is_file(physical_path)
        if physical_exists:
            try:
                return physical_path.absolute(), resolve_in_project(project_dir, physical_path)
            except ValueError:
                if not raw_path.is_absolute():
                    raise

        if raw_path.is_absolute():
            virtual_path = project_dir / str(raw_path).lstrip("/")
            if _is_file(virtual_path):
                return virtual_path.absolute(), resolve_in_project(project_dir, virtual_path)

        if physical_exists:
            return physical_path.absolute(), resolve_in_project(project_dir, physical_path)
        return None

    def _diff(current: str, proposed: str, prompt_id: str) -> str:
        diff_lines = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"{prompt_id} (current)",
            tofile=f"{prompt_id} (proposed)",
        ))
        return "".join(diff_lines)

    def _build_prompt_from_edits(current: str, edits: list[dict]) -> tuple[str, str | None]:
        if not edits:
            return current, "Error: no edits provided."

        validated_edits = []
        for i, edit in enumerate(edits, start=1):
            old = edit.get("old")
            new = edit.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                return current, f"Error: edit {i} must include string 'old' and 'new' values."
            if not old:
                return current, f"Error: edit {i} has an empty 'old' value."

            count = current.count(old)
            if count == 0:
                return current, f"Error: edit {i} did not match the current prompt exactly."
            if count > 1:
                return current, f"Error: edit {i} matched {count} places. Make the 'old' text more specific."
            validated_edits.append((old, new))

        proposed = current
        for old, new in validated_edits:
            proposed = proposed.replace(old, new, 1)
        return proposed, None

    def _apply_content(prompt_id: str, new_content: str) -> str:
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not found."

        try:
            current_file = _prompt_artifact(prompt_id, "current.txt")
            meta_file = _prompt_artifact(prompt_id, "meta.json")
            current = current_file.read_text()
            meta = json.loads(meta_file.read_text())
            new_version = meta["version"] + 1
            history_file = _prompt_artifact(prompt_id, "history", f"v{new_version}.txt")
        except ValueError as e:
            return f"Error: {e}; changes were not applied."

        if current == new_content:
            return "No changes detected between current and proposed content."

        source_path_value = meta.get("source_path")
        source_path = None
        if source_path_value:
            try:
                source_path = resolve_in_project(project_dir, Path(source_path_value))
            except ValueError as e:
                return f"Error: {e}; changes were not applied."
            if not source_path.exists():
                return f"Error: source file {source_path} not found; changes were not applied."
            expected_target = meta.get("resolved_source_path")
            if expected_target is not None and str(source_path) != expected_target:
                return "Error: prompt changed, re-propose the edit; changes were not applied."

        if source_path is None:
            source_note = "No source file to update for inline prompt."
        else:
            source_path.write_text(new_content)
            source_note = f"Source file updated at {source_path}."

        history_file.write_text(new_content)
        current_file.write_text(new_content)
        meta["version"] = new_version
        meta["updated_at"] = datetime.now().isoformat()
        meta_file.write_text(json.dumps(meta, indent=2))

        return f"Applied. Prompt '{prompt_id}' is now v{new_version}. {source_note}"

    @tool(parse_docstring=True)
    def register_prompt(source: str, prompt_id: str | None = None) -> str:
        """Register a prompt file or inline prompt text into the eval system.

        If source is an existing path, reads the file and records the path for
        future syncing. Otherwise, treats source as the literal prompt content.
        Call this first before any eval work.

        Args:
            source: Path to a prompt file, or literal inline prompt text.
            prompt_id: Optional short identifier. Defaults to the file stem for
                files, or an inline timestamp ID for pasted prompt text.

        Returns:
            Confirmation with the registered prompt ID and a content preview.
        """
        if prompt_id is not None and not prompt_id.strip():
            prompt_id = None

        try:
            p = _source_path_if_file(source)
        except ValueError as e:
            return f"Error: {e}"

        resolved_source_path = None
        if p is not None:
            p, resolved_path = p
            resolved_source_path = str(resolved_path)
            content = p.read_text()
            pid = prompt_id or p.stem.replace(" ", "_").lower()
            source_path = str(p)
            source_type = "file"
            source_label = f"Source: {p}"
        else:
            content = source
            if not content.strip():
                return "Error: inline prompt content is empty."
            pid = prompt_id or _inline_prompt_id()
            source_path = None
            source_type = "inline"
            source_label = "Source: inline prompt"

        if source_type == "file" and source_path is not None and prompt_id is None:
            pid = _dedupe_file_prompt_id(pid, source_path)

        try:
            version, _ = _save_prompt(content, pid, source_path, source_type, resolved_source_path)
        except ValueError as e:
            return f"Error: {e}"
        preview = content[:300] + "\n..." if len(content) > 300 else content
        return f"Registered prompt '{pid}' (v{version})\n{source_label}\n\n{preview}"

    @tool(parse_docstring=True)
    def read_current_prompt(prompt_id: str) -> str:
        """Read the current version of a registered prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            The full prompt content with version and source metadata.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not registered. Use register_prompt first."
        meta = json.loads(_prompt_artifact(prompt_id, "meta.json").read_text())
        content = (prompt_dir / "current.txt").read_text()
        source = meta.get("source_path") or "inline prompt"
        return f"Prompt '{prompt_id}' — v{meta['version']} — source: {source}\n\n{content}"

    @tool(parse_docstring=True)
    def list_prompts() -> str:
        """List all prompts registered in the eval system.

        Returns:
            A formatted list of prompt IDs with version and source path.
        """
        if not prompts_dir.exists():
            return "No prompts registered yet."
        entries = [
            d for d in prompts_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        ]
        if not entries:
            return "No prompts registered yet."
        lines = ["Registered prompts:"]
        for d in entries:
            meta = json.loads((d / "meta.json").read_text())
            source = meta.get("source_path") or "inline prompt"
            lines.append(f"  {d.name}  v{meta['version']}  →  {source}")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    async def edit_prompt(prompt_id: str, edits: list[dict]) -> str:
        """Edit a prompt with targeted find/replace changes.

        The TUI will show a diff and ask for approval before writing. If the
        user denies, the edit is cancelled and nothing is written.

        Args:
            prompt_id: The prompt identifier.
            edits: List of edit dicts with "old" and "new" string keys. Each
                "old" string must appear exactly once in the current prompt.

        Returns:
            Confirmation with the new version number, or cancellation message.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not found."
        current_file = _prompt_artifact(prompt_id, "current.txt")
        meta_file = _prompt_artifact(prompt_id, "meta.json")
        current = current_file.read_text()
        base_meta = json.loads(meta_file.read_text())
        base_version = base_meta.get("version")
        approval_key = prompt_approval_key(prompt_id, current, edits)
        try:
            new_content, error = _build_prompt_from_edits(current, edits)
            if error:
                return error
            if not await gate.consume(approval_key):
                return f"Edit to '{prompt_id}' cancelled by user."

            latest_content = current_file.read_text()
            latest_meta = json.loads(meta_file.read_text())
            identity_fields = ("prompt_id", "source_path", "source_type", "registered_at", "resolved_source_path")
            if any(latest_meta.get(field) != base_meta.get(field) for field in identity_fields):
                return "Error: prompt changed, re-propose the edit."
            latest_version = latest_meta.get("version")
            if latest_content != current or latest_version != base_version:
                new_content, error = _build_prompt_from_edits(latest_content, edits)
                if error:
                    return "Error: prompt changed, re-propose the edit."
            return _apply_content(prompt_id, new_content)
        finally:
            gate.drop(approval_key)

    @tool(parse_docstring=True)
    def show_prompt_history(prompt_id: str) -> str:
        """Show version history for a prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            List of versions with size info and current marker.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        history_dir = prompt_dir / "history"
        if not history_dir.exists():
            return f"No history found for '{prompt_id}'."

        meta = json.loads(_prompt_artifact(prompt_id, "meta.json").read_text())
        versioned_files = []
        for path in history_dir.glob("v*.txt"):
            num = _version_number(path)
            if num is not None:
                versioned_files.append((num, path))
        lines = [f"Version history for '{prompt_id}':"]
        for num, v in sorted(versioned_files):
            marker = " ← current" if num == meta["version"] else ""
            lines.append(f"  v{num}: {len(v.read_text())} chars{marker}")
        return "\n".join(lines)

    return [
        register_prompt,
        read_current_prompt,
        list_prompts,
        edit_prompt,
        show_prompt_history,
    ]

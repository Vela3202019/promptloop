import sqlite3

import pytest

from promptloop.cli import _list_thread_ids, _resolve_thread_id
from promptloop.system_prompt import SYSTEM_PROMPT


def _make_thread_db(project_dir, thread_ids):
    evals = project_dir / ".evals"
    evals.mkdir()
    with sqlite3.connect(evals / "chat.db") as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        conn.executemany("INSERT INTO checkpoints VALUES (?)", [(tid,) for tid in thread_ids])


def test_thread_listing_and_prefix_resolution(tmp_path):
    _make_thread_db(tmp_path, ["abcdef", "abcxyz", "unique-thread"])
    assert _list_thread_ids(tmp_path) == ["abcdef", "abcxyz", "unique-thread"]

    assert _resolve_thread_id(tmp_path, "unique-thread") == ("unique-thread", None)
    assert _resolve_thread_id(tmp_path, "unique")[0] == "unique-thread"

    resolved, error = _resolve_thread_id(tmp_path, "abc")
    assert resolved is None
    assert "Ambiguous thread prefix" in error

    resolved, error = _resolve_thread_id(tmp_path, "missing")
    assert resolved is None
    assert "No saved thread found" in error


def test_thread_listing_rejects_symlinked_chat_db(tmp_path):
    evals = tmp_path / ".evals"
    evals.mkdir()
    target = tmp_path / "chat-target.db"
    target.write_text("not a database")
    (evals / "chat.db").symlink_to(target)

    with pytest.raises(ValueError, match=r"path escapes \.evals"):
        _list_thread_ids(tmp_path)


def test_system_prompt_formats_without_keyerror():
    rendered = SYSTEM_PROMPT.format(project_dir="/tmp/project")
    assert "`{old, new}` pairs" in rendered
    assert "one directory per prompt" in rendered
    assert "virtual project-relative paths" in rendered
    assert "Never use the physical absolute path" in rendered

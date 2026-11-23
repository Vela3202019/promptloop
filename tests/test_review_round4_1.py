from promptloop.cli import _render_file_write_action, _render_prompt_diff


def test_oversized_file_edit_diff_fails_closed(tmp_path):
    current = "".join(f"old line {index}\n" for index in range(150))
    proposed = "".join(f"new line {index}\n" for index in range(150))
    (tmp_path / "large.txt").write_text(current)

    error = _render_file_write_action(
        tmp_path,
        "edit_file",
        {"file_path": "/large.txt", "old_string": current, "new_string": proposed},
    )

    assert error == "change too large to review; split it into smaller changes"


def test_oversized_prompt_diff_fails_closed(tmp_path):
    prompt_dir = tmp_path / ".evals/prompts/p"
    prompt_dir.mkdir(parents=True)
    current = "".join(f"old line {index}\n" for index in range(150))
    (prompt_dir / "current.txt").write_text(current)
    edits = [{"old": current, "new": "".join(f"new line {index}\n" for index in range(150))}]

    assert _render_prompt_diff(tmp_path, "p", edits) == "change too large to review; split it into smaller changes"

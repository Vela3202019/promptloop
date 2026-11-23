from promptloop.cli import _render_file_write_action


def test_new_file_preview_shows_control_characters(tmp_path, capsys):
    capsys.readouterr()
    error = _render_file_write_action(
        tmp_path,
        "write_file",
        {"file_path": "/new.txt", "content": "before\x0bafter\u2028end"},
    )
    assert error is None
    output = capsys.readouterr().out
    assert "\\x0b" in output
    assert "\\u2028" in output

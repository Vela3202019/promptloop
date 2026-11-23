from promptloop.cli import _render_file_write_action


def test_crlf_only_edit_is_previewable(tmp_path, capsys):
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")
    capsys.readouterr()
    error = _render_file_write_action(
        tmp_path,
        "edit_file",
        {"file_path": "/crlf.txt", "old_string": "alpha\r\n", "new_string": "alpha\n"},
    )
    assert error is None
    assert "normalizes all line endings to LF" in capsys.readouterr().out

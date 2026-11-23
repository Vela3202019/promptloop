from promptloop.cli import _render_file_write_action


def test_edit_preview_estimate_uses_utf8_bytes(tmp_path):
    target = tmp_path / "unicode.txt"
    target.write_text("a" * 1000)

    error = _render_file_write_action(
        tmp_path,
        "edit_file",
        {
            "file_path": "/unicode.txt",
            "old_string": "a",
            "new_string": "😀" * 50,
            "replace_all": True,
        },
    )

    assert error == "preview unavailable: edit result is too large to preview safely"

from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def _tools(project_dir):
    gate = ApprovalGate()
    return gate, {tool.name: tool for tool in make_prompt_tools(project_dir, gate)}


def test_prompt_register_edit_history_and_approval_gate(tmp_path):
    gate, tools = _tools(tmp_path)
    assert "Registered prompt 'p' (v1)" in tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    current = tmp_path / ".evals" / "prompts" / "p" / "current.txt"
    assert current.read_text() == "alpha beta"

    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}
    assert tools["edit_prompt"].invoke(edit) == "Edit to 'p' cancelled by user."
    assert current.read_text() == "alpha beta"

    gate.resolve("p", False)
    assert tools["edit_prompt"].invoke(edit) == "Edit to 'p' cancelled by user."
    assert current.read_text() == "alpha beta"

    gate.resolve("p", True)
    assert "Prompt 'p' is now v2" in tools["edit_prompt"].invoke(edit)
    assert current.read_text() == "omega beta"

    history_file = tmp_path / ".evals" / "prompts" / "p" / "history" / "vnot-a-number.txt"
    history_file.write_text("not a version")
    history = tools["show_prompt_history"].invoke({"prompt_id": "p"})
    assert "v1:" in history
    assert "v2:" in history
    assert "← current" in history


def test_inline_prompt_ids_are_unique(tmp_path):
    _, tools = _tools(tmp_path)
    first = tools["register_prompt"].invoke({"source": "first inline prompt"})
    second = tools["register_prompt"].invoke({"source": "second inline prompt"})
    assert "Registered prompt 'inline_" in first
    assert "Registered prompt 'inline_" in second
    assert len(list((tmp_path / ".evals" / "prompts").iterdir())) == 2


def test_prompt_id_traversal_is_rejected(tmp_path):
    _, tools = _tools(tmp_path)
    result = tools["register_prompt"].invoke({"source": "content", "prompt_id": "../escape"})
    assert result.startswith("Error:")
    assert not (tmp_path / ".evals" / "escape").exists()


def test_registered_source_must_stay_inside_project(tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("secret")
    try:
        link = tmp_path / "linked.txt"
        link.symlink_to(outside)
        _, tools = _tools(tmp_path)
        result = tools["register_prompt"].invoke({"source": "linked.txt", "prompt_id": "p"})
        assert result.startswith("Error: path escapes project directory")
    finally:
        outside.unlink(missing_ok=True)


def test_approved_edit_updates_source_file_inside_project(tmp_path):
    source = tmp_path / "prompt.txt"
    source.write_text("alpha beta")
    gate, tools = _tools(tmp_path)
    assert "Registered prompt 'p'" in tools["register_prompt"].invoke({"source": "prompt.txt", "prompt_id": "p"})
    gate.resolve("p", True)
    result = tools["edit_prompt"].invoke({"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]})
    assert "Source file updated" in result
    assert source.read_text() == "omega beta"

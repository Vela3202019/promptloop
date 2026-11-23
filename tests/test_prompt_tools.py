import asyncio
import json

from promptloop.tools.prompts import ApprovalGate, make_prompt_tools, prompt_approval_key


def _tools(project_dir):
    gate = ApprovalGate(listener_timeout=0.01)
    return gate, {tool.name: tool for tool in make_prompt_tools(project_dir, gate)}


def _ainvoke(tool, args):
    return asyncio.run(tool.ainvoke(args))


def _approval_key(edits):
    return prompt_approval_key("p", "alpha beta", edits)


def test_approval_gate_consume_waits_for_resolve():
    async def scenario():
        gate = ApprovalGate(listener_timeout=0.01)
        task = asyncio.create_task(gate.consume("p"))
        await asyncio.sleep(0)
        assert not task.done()
        gate.expect("p")
        await asyncio.sleep(0)
        assert not task.done()
        gate.resolve("p", True)
        assert await task is True

    asyncio.run(scenario())


def test_approval_gate_denies_without_listener():
    async def scenario():
        gate = ApprovalGate(listener_timeout=0.01)
        return await gate.consume("p")

    assert asyncio.run(scenario()) is False


def test_prompt_register_edit_history_and_approval_gate(tmp_path):
    gate, tools = _tools(tmp_path)
    assert "Registered prompt 'p' (v1)" in tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    current = tmp_path / ".evals" / "prompts" / "p" / "current.txt"
    assert current.read_text() == "alpha beta"

    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}
    assert _ainvoke(tools["edit_prompt"], edit) == "Edit to 'p' cancelled by user."
    assert current.read_text() == "alpha beta"

    gate.resolve(_approval_key(edit["edits"]), False)
    assert _ainvoke(tools["edit_prompt"], edit) == "Edit to 'p' cancelled by user."
    assert current.read_text() == "alpha beta"

    gate.resolve(approval_key, True)
    assert "Prompt 'p' is now v2" in _ainvoke(tools["edit_prompt"], edit)
    assert current.read_text() == "omega beta"

    history_file = tmp_path / ".evals" / "prompts" / "p" / "history" / "vnot-a-number.txt"
    history_file.write_text("not a version")
    history = tools["show_prompt_history"].invoke({"prompt_id": "p"})
    assert "v1:" in history
    assert "v2:" in history
    assert "← current" in history


async def _wait_for_pending_approval(gate, prompt_id):
    for _ in range(100):
        if gate._waiters:
            key = next(iter(gate._waiters))
            gate.expect(key)
            return key
        await asyncio.sleep(0)
    raise AssertionError("edit did not reach approval gate")


def _intervening_prompt_change(tmp_path, prompt_id, content):
    prompt_dir = tmp_path / ".evals" / "prompts" / prompt_id
    (prompt_dir / "current.txt").write_text(content)
    meta_file = prompt_dir / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["version"] += 1
    meta_file.write_text(json.dumps(meta))


def test_approved_edit_reapplies_after_intervening_change(tmp_path):
    gate, tools = _tools(tmp_path)
    tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}

    async def scenario():
        task = asyncio.create_task(tools["edit_prompt"].ainvoke(edit))
        approval_key = await _wait_for_pending_approval(gate, "p")
        _intervening_prompt_change(tmp_path, "p", "alpha beta gamma")
        gate.resolve(approval_key, True)
        return await task

    result = asyncio.run(scenario())

    assert "Prompt 'p' is now v3" in result
    assert (tmp_path / ".evals" / "prompts" / "p" / "current.txt").read_text() == "omega beta gamma"


def test_approved_edit_rejects_unmatched_intervening_change(tmp_path):
    gate, tools = _tools(tmp_path)
    tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}

    async def scenario():
        task = asyncio.create_task(tools["edit_prompt"].ainvoke(edit))
        approval_key = await _wait_for_pending_approval(gate, "p")
        _intervening_prompt_change(tmp_path, "p", "replaced beta")
        gate.resolve(approval_key, True)
        return await task

    result = asyncio.run(scenario())

    assert "prompt changed, re-propose" in result
    assert (tmp_path / ".evals" / "prompts" / "p" / "current.txt").read_text() == "replaced beta"


def test_approved_edit_rejects_reregistration(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("alpha beta")
    second_file.write_text("alpha beta")
    gate, tools = _tools(tmp_path)
    tools["register_prompt"].invoke({"source": "first.txt", "prompt_id": "p"})
    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}

    async def scenario():
        task = asyncio.create_task(tools["edit_prompt"].ainvoke(edit))
        approval_key = await _wait_for_pending_approval(gate, "p")
        tools["register_prompt"].invoke({"source": "second.txt", "prompt_id": "p"})
        gate.resolve(approval_key, True)
        return await task

    result = asyncio.run(scenario())

    assert "prompt changed, re-propose" in result
    assert second_file.read_text() == "alpha beta"


def test_inline_prompt_ids_are_unique(tmp_path):
    _, tools = _tools(tmp_path)
    first = tools["register_prompt"].invoke({"source": "first inline prompt"})
    second = tools["register_prompt"].invoke({"source": "second inline prompt"})
    assert "Registered prompt 'inline_" in first
    assert "Registered prompt 'inline_" in second
    assert len(list((tmp_path / ".evals" / "prompts").iterdir())) == 2


def test_duplicate_source_file_stems_get_distinct_ids(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "shared.txt").write_text("first prompt")
    (second_dir / "shared.txt").write_text("second prompt")

    _, tools = _tools(tmp_path)
    first = tools["register_prompt"].invoke({"source": "first/shared.txt"})
    second = tools["register_prompt"].invoke({"source": "second/shared.txt"})
    first_again = tools["register_prompt"].invoke({"source": "first/shared.txt"})

    assert "Registered prompt 'shared' (v1)" in first
    assert "Registered prompt 'shared_2' (v1)" in second
    assert "Registered prompt 'shared' (v2)" in first_again
    prompts_dir = tmp_path / ".evals" / "prompts"
    assert (prompts_dir / "shared" / "current.txt").read_text() == "first prompt"
    assert (prompts_dir / "shared_2" / "current.txt").read_text() == "second prompt"


def test_explicit_duplicate_id_updates_source_path(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "shared.txt").write_text("first prompt")
    (second_dir / "shared.txt").write_text("second prompt")

    _, tools = _tools(tmp_path)
    first = tools["register_prompt"].invoke({"source": "first/shared.txt"})
    moved = tools["register_prompt"].invoke({"source": "second/shared.txt", "prompt_id": "shared"})

    assert "Registered prompt 'shared' (v1)" in first
    assert "Registered prompt 'shared' (v2)" in moved
    meta = json.loads((tmp_path / ".evals" / "prompts" / "shared" / "meta.json").read_text())
    assert meta["source_path"] == str((second_dir / "shared.txt").resolve())
    assert not (tmp_path / ".evals" / "prompts" / "shared_2").exists()


def test_blank_prompt_id_auto_derives_and_dedupes(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "shared.txt").write_text("first prompt")
    (second_dir / "shared.txt").write_text("second prompt")

    _, tools = _tools(tmp_path)
    first = tools["register_prompt"].invoke({"source": "first/shared.txt"})
    second = tools["register_prompt"].invoke({"source": "second/shared.txt", "prompt_id": "   "})

    assert "Registered prompt 'shared' (v1)" in first
    assert "Registered prompt 'shared_2' (v1)" in second


def test_register_prompt_accepts_virtual_project_path(tmp_path):
    prompt_file = tmp_path / "src" / "prompts" / "virtual.txt"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("virtual path prompt")

    _, tools = _tools(tmp_path)
    result = tools["register_prompt"].invoke({"source": "/src/prompts/virtual.txt"})

    assert "Registered prompt 'virtual' (v1)" in result
    meta = json.loads((tmp_path / ".evals" / "prompts" / "virtual" / "meta.json").read_text())
    assert meta["source_path"] == str(prompt_file.resolve())


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
    edit = {"prompt_id": "p", "edits": [{"old": "alpha", "new": "omega"}]}
    gate.resolve(approval_key, True)
    result = _ainvoke(tools["edit_prompt"], edit)
    assert "Source file updated" in result
    assert source.read_text() == "omega beta"

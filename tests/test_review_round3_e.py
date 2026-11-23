import asyncio

from promptloop.tools import prompts
from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_repointed_source_symlink_rejects_edit(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha beta")
    second.write_text("alpha beta")
    link = tmp_path / "prompt.txt"
    link.symlink_to(first)
    gate = ApprovalGate(listener_timeout=0.01)
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, gate)}
    assert "Registered prompt 'p'" in tools["register_prompt"].invoke({"source": "prompt.txt", "prompt_id": "p"})
    link.unlink()
    link.symlink_to(second)

    edits = [{"old": "alpha", "new": "omega"}]
    key = prompts.prompt_approval_key("p", "alpha beta", edits)
    gate.resolve(key, True)
    result = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": edits}))

    assert "prompt changed, re-propose" in result
    assert first.read_text() == "alpha beta"
    assert second.read_text() == "alpha beta"

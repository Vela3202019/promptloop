import asyncio

from promptloop.tools import prompts
from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_unconsumed_prompt_approval_is_dropped(tmp_path):
    gate = ApprovalGate(listener_timeout=0.01)
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, gate)}
    tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    edits = [{"old": "alpha", "new": "omega"}]
    key = prompts.prompt_approval_key("p", "alpha beta", edits)

    gate.resolve(key, True)
    invalid = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": [{"old": "", "new": "x"}]}))
    assert invalid.startswith("Error:")

    result = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": edits}))
    assert result == "Edit to 'p' cancelled by user."
    assert (tmp_path / ".evals/prompts/p/current.txt").read_text() == "alpha beta"

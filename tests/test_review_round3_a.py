import asyncio

from promptloop.tools import prompts
from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_unconsumed_prompt_approval_is_dropped(tmp_path):
    gate = ApprovalGate(listener_timeout=0.01)
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, gate)}
    tools["register_prompt"].invoke({"source": "alpha beta", "prompt_id": "p"})
    edits = [{"old": "alpha", "new": "omega"}]
    invalid_edits = [{"old": "", "new": "x"}]
    invalid_key = prompts.prompt_approval_key("p", "alpha beta", invalid_edits)

    gate.resolve(invalid_key, True)
    invalid = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": invalid_edits}))
    assert invalid.startswith("Error:")
    assert invalid_key not in gate._decisions

    result = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": edits}))
    assert result == "Edit to 'p' cancelled by user."
    assert (tmp_path / ".evals/prompts/p/current.txt").read_text() == "alpha beta"

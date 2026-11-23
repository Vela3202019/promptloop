import asyncio

from promptloop.cli import (
    CHANGE_TOO_LARGE_MESSAGE,
    _decide_file_write_interrupt,
    _oversized_review_rejection,
    _render_prompt_diff,
)
from promptloop.tools.prompts import ApprovalGate, make_prompt_tools, prompt_approval_key


def _oversized_edit():
    current = "".join(f"old line {index}\n" for index in range(150))
    proposed = "".join(f"new line {index}\n" for index in range(150))
    return current, proposed


def test_oversized_file_edit_rejection_tells_model_to_split(tmp_path):
    current, proposed = _oversized_edit()
    (tmp_path / "large.txt").write_text(current)
    interrupt_value = {
        "action_requests": [
            {
                "name": "edit_file",
                "args": {"file_path": "/large.txt", "old_string": current, "new_string": proposed},
            }
        ]
    }

    result = asyncio.run(_decide_file_write_interrupt(tmp_path, interrupt_value, True))

    decision = result["decisions"][0]
    assert decision["type"] == "reject"
    assert "too large to review" in decision["message"]
    assert "split" in decision["message"]
    assert "user declined" not in decision["message"]


def test_oversized_prompt_edit_rejection_reaches_model(tmp_path):
    current, proposed = _oversized_edit()
    edits = [{"old": current, "new": proposed}]
    gate = ApprovalGate(listener_timeout=0.01)
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, gate)}
    tools["register_prompt"].invoke({"source": current, "prompt_id": "p"})

    diff_error = _render_prompt_diff(tmp_path, "p", edits)
    assert diff_error == CHANGE_TOO_LARGE_MESSAGE
    rejection = _oversized_review_rejection(diff_error)
    assert rejection is not None

    key = prompt_approval_key("p", current, edits)
    gate.resolve(key, False, rejection)
    result = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": edits}))

    assert "too large to review" in result
    assert "split" in result
    assert (tmp_path / ".evals/prompts/p/current.txt").read_text() == current

from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_prompt_registration_preflights_artifacts_before_writing(tmp_path):
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, ApprovalGate(listener_timeout=0.01))}
    tools["register_prompt"].invoke({"source": "alpha", "prompt_id": "p"})
    prompt_dir = tmp_path / ".evals/prompts/p"
    outside = tmp_path.parent / f"history-{tmp_path.name}"
    outside.mkdir()
    (prompt_dir / "history").rename(outside / "history")
    (prompt_dir / "history").symlink_to(tmp_path.parent)
    try:
        result = tools["register_prompt"].invoke({"source": "replacement", "prompt_id": "p"})
        assert result.startswith("Error:")
        assert (prompt_dir / "current.txt").read_text() == "alpha"
    finally:
        (prompt_dir / "history").unlink(missing_ok=True)
        (outside / "history").rename(prompt_dir / "history")

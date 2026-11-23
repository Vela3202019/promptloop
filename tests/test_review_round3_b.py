from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_prompt_artifact_symlink_escape_is_refused(tmp_path):
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, ApprovalGate(listener_timeout=0.01))}
    tools["register_prompt"].invoke({"source": "alpha", "prompt_id": "p"})
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("secret")
    artifact = tmp_path / ".evals/prompts/p/current.txt"
    artifact.unlink()
    artifact.symlink_to(outside)
    try:
        result = tools["register_prompt"].invoke({"source": "replacement", "prompt_id": "p"})
        assert result.startswith("Error:")
        assert outside.read_text() == "secret"
    finally:
        outside.unlink(missing_ok=True)

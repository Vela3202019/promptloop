from promptloop.tools.prompts import ApprovalGate, make_prompt_tools


def test_reregistering_suffixed_source_reuses_its_prompt_id(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "shared.txt").write_text("first prompt")
    (second_dir / "shared.txt").write_text("second prompt")
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, ApprovalGate(listener_timeout=0.01))}

    tools["register_prompt"].invoke({"source": "first/shared.txt"})
    tools["register_prompt"].invoke({"source": "second/shared.txt"})
    result = tools["register_prompt"].invoke({"source": "second/shared.txt"})

    assert "Registered prompt 'shared_2' (v2)" in result
    assert not (tmp_path / ".evals/prompts/shared_3").exists()

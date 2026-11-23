import json

from promptloop.tools.test_cases import make_test_case_tools


def test_list_test_cases_skips_symlink_escape(tmp_path):
    case_dir = tmp_path / ".evals/test_cases/p"
    case_dir.mkdir(parents=True)
    outside = tmp_path.parent / f"outside-{tmp_path.name}.json"
    outside.write_text(json.dumps({"id": "evil", "input": "secret input", "metrics": []}))
    (case_dir / "evil.json").symlink_to(outside)
    try:
        tools = {tool.name: tool for tool in make_test_case_tools(tmp_path)}
        result = tools["list_test_cases"].invoke({"prompt_id": "p"})
        assert "secret input" not in result
    finally:
        outside.unlink(missing_ok=True)

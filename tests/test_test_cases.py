import json

from promptloop.tools.test_cases import make_test_case_tools


def _tools(project_dir):
    return {tool.name: tool for tool in make_test_case_tools(project_dir)}


def test_test_case_crud_and_metric_validation(tmp_path):
    tools = _tools(tmp_path)
    invalid = tools["add_test_case"].invoke({"prompt_id": "p", "input": "x", "metrics": [{}], "test_id": "bad"})
    assert invalid == "Error: metric 1 has unknown type None."
    assert not (tmp_path / ".evals" / "test_cases" / "p" / "bad.json").exists()

    missing_schema = tools["add_test_case"].invoke({"prompt_id": "p", "input": "x", "metrics": [{"type": "json_schema"}]})
    assert "requires an object or boolean 'schema'" in missing_schema

    valid_metrics = [{"type": "fuzzy_match", "threshold": 0.9}]
    added = tools["add_test_case"].invoke({
        "prompt_id": "p",
        "input": "say hello",
        "expected_output": "hello",
        "metrics": valid_metrics,
        "test_id": "tc1",
    })
    assert added.startswith("Added test case 'tc1'")
    data = json.loads((tmp_path / ".evals" / "test_cases" / "p" / "tc1.json").read_text())
    assert data["metrics"] == valid_metrics

    listed = tools["list_test_cases"].invoke({"prompt_id": "p"})
    assert "[tc1]" in listed
    assert "fuzzy_match" in listed

    assert tools["delete_test_case"].invoke({"prompt_id": "p", "test_id": "tc1"}) == "Deleted test case 'tc1'."
    assert "not found" in tools["delete_test_case"].invoke({"prompt_id": "p", "test_id": "tc1"})


def test_test_case_and_config_traversal_are_rejected(tmp_path):
    tools = _tools(tmp_path)
    result = tools["add_test_case"].invoke({"prompt_id": "../escape", "input": "x"})
    assert result.startswith("Error: prompt_id")
    assert not (tmp_path / ".evals" / "escape").exists()

    result = tools["delete_test_case"].invoke({"prompt_id": "p", "test_id": "../target"})
    assert result.startswith("Error: test_id")

    result = tools["save_eval_config"].invoke({
        "prompt_id": "../escape",
        "default_models": ["fake"],
        "default_metrics": [{"type": "latency"}],
    })
    assert result.startswith("Error: prompt_id")


def test_infer_json_schema_valid_and_invalid_json(tmp_path):
    tools = _tools(tmp_path)
    result = tools["infer_json_schema"].invoke({"example_output": '{"name": "x", "count": 2}'})
    assert "Inferred schema" in result
    assert '"type": "object"' in result
    assert '"json_schema"' in result

    invalid = tools["infer_json_schema"].invoke({"example_output": "not json"})
    assert invalid.startswith("Error: could not parse as JSON")


def test_delete_test_case_removes_symlink_not_target(tmp_path):
    target_dir = tmp_path / ".evals" / "test_cases" / "other"
    target_dir.mkdir(parents=True)
    target = target_dir / "target.json"
    target.write_text('{"id": "target"}')
    case_dir = tmp_path / ".evals" / "test_cases" / "p"
    case_dir.mkdir(parents=True)
    link = case_dir / "linked.json"
    link.symlink_to(target)

    tools = _tools(tmp_path)
    result = tools["delete_test_case"].invoke({"prompt_id": "p", "test_id": "linked"})

    assert result == "Deleted test case 'linked'."
    assert not link.exists()
    assert target.read_text() == '{"id": "target"}'

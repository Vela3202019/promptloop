import json

from promptloop.tools.report import make_report_tools


def _write_run(project_dir, prompt_id, run_id, rows):
    run_dir = project_dir / ".evals" / "results" / prompt_id / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "raw.json").write_text(json.dumps({
        "results": rows,
        "prompt_version": 1,
        "models": ["model-a", "model-b"],
    }))


def _row(test_id, model, passed):
    return {
        "test_id": test_id,
        "model": model,
        "latency_ms": 10,
        "metrics": [{"type": "fuzzy_match", "passed": passed}],
        "error": None,
    }


def test_compare_runs_keys_results_by_test_and_model(tmp_path):
    _write_run(tmp_path, "p", "run_a", [
        _row("t1", "model-a", True),
        _row("t1", "model-b", False),
    ])
    _write_run(tmp_path, "p", "run_b", [
        _row("t1", "model-a", True),
        _row("t1", "model-b", True),
    ])

    tools = {tool.name: tool for tool in make_report_tools(tmp_path)}
    output = tools["compare_runs"].invoke({"prompt_id": "p", "run_id_a": "run_a", "run_id_b": "run_b"})

    assert "Pass rate" in output
    assert "+50%" in output
    assert "[t1] model-a" in output
    assert "[t1] model-b" in output
    assert "A: ✗  →  B: ✓" in output


def test_compare_runs_rejects_invalid_ids(tmp_path):
    tools = {tool.name: tool for tool in make_report_tools(tmp_path)}
    output = tools["compare_runs"].invoke({"prompt_id": "../p", "run_id_a": "a", "run_id_b": "b"})
    assert output.startswith("Error: prompt_id")

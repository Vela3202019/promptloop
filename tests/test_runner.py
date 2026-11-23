import json

from promptloop.tools.runner import make_runner_tools


def test_list_eval_runs_shows_status(tmp_path):
    run_dir = tmp_path / ".evals" / "results" / "p" / "run_cancelled"
    run_dir.mkdir(parents=True)
    (run_dir / "raw.json").write_text(json.dumps({
        "results": [],
        "models": ["model-a"],
        "prompt_version": 1,
        "status": "cancelled",
    }))

    tools = {tool.name: tool for tool in make_runner_tools(tmp_path)}
    output = tools["list_eval_runs"].invoke({"prompt_id": "p"})

    assert "run_cancelled" in output
    assert "[cancelled]" in output

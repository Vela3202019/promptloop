import asyncio
import json
import re
import time
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from ..metrics import evaluate_sync_metric, result_passed
from ..paths import resolve_in_evals, validate_id

DEFAULT_MAX_CONCURRENCY = 4


def make_runner_tools(project_dir: Path):

    def _get_model(model_cache: dict[str, Any], model_name: str) -> Any:
        if model_name not in model_cache:
            model_cache[model_name] = init_chat_model(model_name, temperature=0)
        return model_cache[model_name]

    async def _call_model(
        model_cache: dict[str, Any],
        model_name: str,
        system_prompt: str,
        user_input: str,
    ) -> tuple[str, float]:
        model = _get_model(model_cache, model_name)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        start = time.monotonic()
        response = await model.ainvoke(messages)
        latency_ms = (time.monotonic() - start) * 1000

        content = response.content
        if isinstance(content, str):
            return content, latency_ms
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ), latency_ms
        return str(content), latency_ms

    async def _run_llm_judge(
        model_cache: dict[str, Any],
        judge_prompt_template: str,
        actual_output: str,
        user_input: str,
        expected_output: str | None,
        judge_model: str,
    ) -> dict:
        prompt = judge_prompt_template.format(
            input=user_input,
            output=actual_output,
            expected=expected_output or "N/A",
        )
        model = _get_model(model_cache, judge_model)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else str(response.content)

        # Parse a numeric score out of the response (0-10 or 0.0-1.0).
        score = None
        m = re.search(r'\b(10|[0-9](?:\.\d+)?)(?:/10)?\b', raw)
        if m:
            raw_score = float(m.group(1))
            score = raw_score if raw_score <= 1 else raw_score / 10.0
            score = min(max(score, 0.0), 1.0)

        detail = (
            "failed: could not parse a numeric score; judge response: " + raw[:500]
            if score is None
            else raw[:600]
        )

        return {
            "type": "llm_judge",
            "passed": (score >= 0.7) if score is not None else None,
            "score": round(score, 3) if score is not None else None,
            "detail": detail,
        }

    async def _run_test_case(
        tc: dict,
        prompt_content: str,
        model_name: str,
        model_cache: dict[str, Any],
    ) -> dict:
        result: dict = {
            "test_id": tc["id"],
            "model": model_name,
            "input": tc["input"],
            "expected_output": tc.get("expected_output"),
            "actual_output": None,
            "latency_ms": None,
            "metrics": [],
            "error": None,
        }
        try:
            actual_output, latency_ms = await _call_model(
                model_cache,
                model_name,
                prompt_content,
                tc["input"],
            )
            result["actual_output"] = actual_output
            result["latency_ms"] = round(latency_ms, 1)

            default_judge_model = "anthropic:claude-sonnet-4-6"
            metrics = tc.get("metrics") or [{"type": "latency"}]

            async_tasks = []
            sync_results = []

            for metric in metrics:
                if metric.get("type") == "llm_judge":
                    async_tasks.append(_run_llm_judge(
                        model_cache,
                        metric.get("judge_prompt", "Rate the output quality 0-10.\nInput: {input}\nOutput: {output}"),
                        actual_output,
                        tc["input"],
                        tc.get("expected_output"),
                        metric.get("judge_model", default_judge_model),
                    ))
                else:
                    sync_results.append(evaluate_sync_metric(metric, actual_output, tc.get("expected_output")))

            judge_results = await asyncio.gather(*async_tasks) if async_tasks else []

            # Merge latency into the latency metric result
            for r in sync_results:
                if r["type"] == "latency":
                    r["latency_ms"] = result["latency_ms"]
                    r["detail"] = f"{result['latency_ms']:.0f}ms"

            result["metrics"] = sync_results + list(judge_results)

        except Exception as e:  # noqa: BLE001 - isolate per-test model/metric failures
            result["error"] = str(e)

        return result

    @tool(parse_docstring=True)
    async def run_eval(
        prompt_id: str,
        models: list[str] | None = None,
        test_case_ids: list[str] | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> str:
        """Run evaluation for a prompt against all (or selected) test cases.

        Runs each test case x model combination with bounded concurrency.
        Measures latency, validates JSON schema, computes fuzzy match, and
        calls LLM-as-judge as configured per test case.

        Args:
            prompt_id: The prompt identifier.
            models: Models to test, e.g. ["anthropic:claude-sonnet-4-6", "openai:gpt-4o"].
                    Falls back to eval config defaults, then claude-sonnet-4-6.
            test_case_ids: Run only these test case IDs. Runs all if omitted.
            max_concurrency: Maximum model/judge calls to run at once.

        Returns:
            Run ID and a pass/fail summary. Use generate_report() for the full analysis.
        """
        try:
            prompt_id = validate_id(prompt_id, "prompt_id")
            prompt_dir = resolve_in_evals(project_dir, "prompts", prompt_id)
            tc_dir = resolve_in_evals(project_dir, "test_cases", prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not registered."
        prompt_content = (prompt_dir / "current.txt").read_text()
        all_cases = list(tc_dir.glob("*.json")) if tc_dir.exists() else []
        if not all_cases:
            return f"Error: no test cases found for '{prompt_id}'. Add some with add_test_case."

        test_cases = []
        for f in all_cases:
            tc = json.loads(f.read_text())
            if test_case_ids is None or tc["id"] in test_case_ids:
                test_cases.append(tc)

        if not test_cases:
            return "No test cases matched the given IDs."

        # Resolve models and default metrics.
        default_metrics = [{"type": "latency"}]
        if models is None:
            config_file = resolve_in_evals(project_dir, "eval_configs", f"{prompt_id}.json")
            if config_file.exists():
                config_data = json.loads(config_file.read_text())
                resolved_models: list[str] = config_data.get("default_models", ["anthropic:claude-sonnet-4-6"])
                default_metrics = config_data.get("default_metrics") or default_metrics
            else:
                resolved_models = ["anthropic:claude-sonnet-4-6"]
        else:
            resolved_models = models
            config_file = resolve_in_evals(project_dir, "eval_configs", f"{prompt_id}.json")
            if config_file.exists():
                config_data = json.loads(config_file.read_text())
                default_metrics = config_data.get("default_metrics") or default_metrics

        for tc in test_cases:
            if not tc.get("metrics"):
                tc["metrics"] = default_metrics

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        run_dir = resolve_in_evals(project_dir, "results", prompt_id, validate_id(run_id, "run_id"))
        run_dir.mkdir(parents=True, exist_ok=True)

        prompt_version = json.loads((prompt_dir / "meta.json").read_text()).get("version")
        started_at = datetime.now().isoformat()
        try:
            requested_concurrency = int(max_concurrency or DEFAULT_MAX_CONCURRENCY)
        except (TypeError, ValueError):
            requested_concurrency = DEFAULT_MAX_CONCURRENCY

        model_cache: dict[str, Any] = {}
        concurrency = max(1, min(requested_concurrency, len(test_cases) * len(resolved_models)))
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict] = []
        total_count = len(test_cases) * len(resolved_models)

        async def _run_limited(tc: dict, model: str) -> dict:
            async with semaphore:
                return await _run_test_case(tc, prompt_content, model, model_cache)

        tasks = [
            asyncio.create_task(_run_limited(tc, model))
            for tc in test_cases
            for model in resolved_models
        ]

        def _write_run_data(status: str, error: str | None = None) -> None:
            run_data: dict[str, Any] = {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "models": resolved_models,
                "prompt_version": prompt_version,
                "started_at": started_at,
                "status": status,
                "max_concurrency": concurrency,
                "completed": len(results),
                "total": total_count,
                "results": results,
            }
            if error is not None:
                run_data["error"] = error
            (run_dir / "raw.json").write_text(json.dumps(run_data, indent=2))

        status = "running"
        failure: str | None = None
        try:
            for completed in asyncio.as_completed(tasks):
                result = await completed
                results.append(result)
                _write_run_data("running")
            status = "complete"
        except asyncio.CancelledError:
            status = "cancelled"
            failure = "cancelled by user"
            raise
        except Exception as exc:
            status = "failed"
            failure = str(exc)
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if status in {"cancelled", "failed"}:
                with suppress(Exception):
                    _write_run_data(status, failure)

        _write_run_data(status)

        # Summary
        passed = sum(1 for r in results if result_passed(r))
        failed = len(results) - passed
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        lines = [
            f"Run complete — ID: {run_id}",
            f"Results: {passed} passed / {failed} failed / {len(results)} total",
            f"Avg latency: {avg_lat:.0f}ms",
            f"Max concurrency: {concurrency}",
            "",
        ]
        for r in results:
            status = "✓" if result_passed(r) else "✗"
            metric_summary = " | ".join(
                f"{m['type']}: {m.get('detail', '') or m.get('score', '')}"
                for m in r.get("metrics", [])
            )
            lines.append(f"  {status} [{r['test_id']}] {r['model']}  {metric_summary}")
            if r.get("error"):
                lines.append(f"      error: {r['error']}")

        lines.append(f"\nUse generate_report('{prompt_id}', '{run_id}') for full analysis.")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    def list_eval_runs(prompt_id: str) -> str:
        """List all eval runs for a prompt, most recent first.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            A list of run IDs with pass/fail counts and model info.
        """
        try:
            runs_dir = resolve_in_evals(project_dir, "results", validate_id(prompt_id, "prompt_id"))
        except ValueError as e:
            return f"Error: {e}"
        if not runs_dir.exists():
            return f"No eval runs found for '{prompt_id}'."

        runs = sorted(runs_dir.iterdir(), reverse=True)
        if not runs:
            return f"No eval runs found for '{prompt_id}'."

        lines = [f"Eval runs for '{prompt_id}':"]
        for run_dir in runs[:15]:
            raw_file = run_dir / "raw.json"
            if not raw_file.exists():
                continue
            data = json.loads(raw_file.read_text())
            passed = sum(1 for r in data["results"] if result_passed(r))
            total = len(data["results"])
            has_report = (run_dir / "report.md").exists()
            models = ", ".join(data.get("models", []))
            v = data.get("prompt_version", "?")
            run_status = data.get("status", "complete")
            lines.append(
                f"  {run_dir.name}  v{v}  {passed}/{total} passed  [{models}]  [{run_status}]"
                + (" [report]" if has_report else "")
            )
        return "\n".join(lines)

    return [run_eval, list_eval_runs]

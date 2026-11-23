import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from ..paths import resolve_in_evals, validate_id


def make_test_case_tools(project_dir: Path):
    eval_configs_dir = resolve_in_evals(project_dir, "eval_configs")

    def _test_case_dir(prompt_id: str) -> Path:
        return resolve_in_evals(project_dir, "test_cases", validate_id(prompt_id, "prompt_id"))

    def _test_case_file(prompt_id: str, test_id: str) -> Path:
        return resolve_in_evals(
            project_dir,
            "test_cases",
            validate_id(prompt_id, "prompt_id"),
            f"{validate_id(test_id, 'test_id')}.json",
        )

    def _test_case_file_for_delete(prompt_id: str, test_id: str) -> Path:
        parent = resolve_in_evals(project_dir, "test_cases", validate_id(prompt_id, "prompt_id"))
        return parent / f"{validate_id(test_id, 'test_id')}.json"

    def _eval_config_file(prompt_id: str) -> Path:
        return resolve_in_evals(project_dir, "eval_configs", f"{validate_id(prompt_id, 'prompt_id')}.json")

    def _validate_metrics(metrics: list[dict] | None, expected_output: str | None) -> str | None:
        if metrics is None:
            return None
        for index, metric in enumerate(metrics, start=1):
            if not isinstance(metric, dict):
                return f"Error: metric {index} must be an object."
            metric_type = metric.get("type")
            if metric_type not in {"latency", "json_schema", "fuzzy_match", "llm_judge"}:
                return f"Error: metric {index} has unknown type {metric_type!r}."
            if metric_type == "json_schema":
                schema = metric.get("schema")
                if not isinstance(schema, (dict, bool)):
                    return f"Error: metric {index} (json_schema) requires an object or boolean 'schema'."
            elif metric_type == "fuzzy_match":
                if expected_output is None:
                    return f"Error: metric {index} (fuzzy_match) requires expected_output."
                threshold = metric.get("threshold", 0.8)
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    return f"Error: metric {index} (fuzzy_match) requires a numeric 'threshold' between 0 and 1."
            elif metric_type == "llm_judge":
                judge_prompt = metric.get("judge_prompt")
                if not isinstance(judge_prompt, str) or not judge_prompt.strip():
                    return f"Error: metric {index} (llm_judge) requires a non-empty 'judge_prompt'."
                judge_model = metric.get("judge_model")
                if judge_model is not None and not isinstance(judge_model, str):
                    return f"Error: metric {index} (llm_judge) requires 'judge_model' to be a string."
        return None

    @tool(parse_docstring=True)
    def add_test_case(
        prompt_id: str,
        input: str,
        expected_output: str | None = None,
        metrics: list[dict] | None = None,
        notes: str | None = None,
        test_id: str | None = None,
    ) -> str:
        """Add a test case for a prompt.

        Args:
            prompt_id: The prompt identifier.
            input: The user input (or full message payload) to send to the model.
            expected_output: Optional expected output, used for fuzzy_match metric.
            metrics: List of metric config dicts. Each has a "type" key: "latency",
                "json_schema" (with "schema" key), "fuzzy_match" (with optional
                "threshold"), or "llm_judge" (with "judge_prompt" and optional
                "judge_model").
            notes: Optional description of what this test case is checking.
            test_id: Optional custom ID. Auto-generated if not provided.

        Returns:
            Confirmation with the new test case ID.
        """
        try:
            tc_dir = _test_case_dir(prompt_id)
            tid = validate_id(test_id, "test_id") if test_id is not None else f"tc_{uuid.uuid4().hex[:8]}"
            tc_file = _test_case_file(prompt_id, tid)
        except ValueError as e:
            return f"Error: {e}"

        validation_error = _validate_metrics(metrics, expected_output)
        if validation_error is not None:
            return validation_error

        tc_dir.mkdir(parents=True, exist_ok=True)
        tc = {
            "id": tid,
            "prompt_id": prompt_id,
            "input": input,
            "expected_output": expected_output,
            "metrics": metrics or [{"type": "latency"}],
            "notes": notes,
            "created_at": datetime.now().isoformat(),
        }
        tc_file.write_text(json.dumps(tc, indent=2))
        metrics_str = ", ".join(m["type"] for m in tc["metrics"])
        return f"Added test case '{tid}' for prompt '{prompt_id}' (metrics: {metrics_str})."

    @tool(parse_docstring=True)
    def list_test_cases(prompt_id: str) -> str:
        """List all test cases for a prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            A formatted list of test cases with IDs, input previews, and metrics.
        """
        try:
            tc_dir = _test_case_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not tc_dir.exists() or not list(tc_dir.glob("*.json")):
            return f"No test cases found for '{prompt_id}'."

        cases = sorted(tc_dir.glob("*.json"))
        lines = [f"Test cases for '{prompt_id}' ({len(cases)} total):"]
        for f in cases:
            tc = json.loads(f.read_text())
            metrics_str = ", ".join(m["type"] for m in tc.get("metrics", []))
            preview = tc["input"][:70] + "..." if len(tc["input"]) > 70 else tc["input"]
            lines.append(f"\n  [{tc['id']}]")
            lines.append(f"    input:   {preview}")
            if tc.get("expected_output"):
                exp_preview = tc["expected_output"][:50] + "..." if len(tc["expected_output"]) > 50 else tc["expected_output"]
                lines.append(f"    expected: {exp_preview}")
            if metrics_str:
                lines.append(f"    metrics: {metrics_str}")
            if tc.get("notes"):
                lines.append(f"    notes:   {tc['notes']}")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    def delete_test_case(prompt_id: str, test_id: str) -> str:
        """Delete a test case.

        Args:
            prompt_id: The prompt identifier.
            test_id: The test case ID to delete.

        Returns:
            Confirmation message.
        """
        try:
            tc_file = _test_case_file_for_delete(prompt_id, test_id)
        except ValueError as e:
            return f"Error: {e}"
        if not tc_file.exists() and not tc_file.is_symlink():
            return f"Test case '{test_id}' not found for prompt '{prompt_id}'."
        tc_file.unlink()
        return f"Deleted test case '{test_id}'."

    @tool(parse_docstring=True)
    def infer_json_schema(example_output: str) -> str:
        """Infer a JSON schema from an example output string.

        Use this when setting up a json_schema metric. Pass in a real example
        of what the model should output, and get back an inferred schema you
        can use directly in add_test_case.

        Args:
            example_output: A real JSON string example of expected model output.

        Returns:
            The inferred JSON schema and a ready-to-paste metric config.
        """
        try:
            parsed = json.loads(example_output)
        except json.JSONDecodeError as e:
            return f"Error: could not parse as JSON: {e}"

        from genson import SchemaBuilder
        builder = SchemaBuilder()
        builder.add_object(parsed)
        schema = builder.to_schema()

        metric_config = json.dumps({"type": "json_schema", "schema": schema}, indent=2)
        return (
            f"Inferred schema:\n\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            f"Use this as a metric in add_test_case:\n```json\n{metric_config}\n```"
        )

    @tool(parse_docstring=True)
    def save_eval_config(
        prompt_id: str,
        default_models: list[str],
        default_metrics: list[dict],
        notes: str | None = None,
    ) -> str:
        """Save the agreed evaluation methodology for a prompt.

        Call this after finalizing the eval approach with the user.
        The config is reused as defaults for future run_eval calls.

        Args:
            prompt_id: The prompt identifier.
            default_models: Models to test by default, e.g. ["anthropic:claude-sonnet-4-6", "openai:gpt-4o"].
            default_metrics: Default metric configs applied to test cases without explicit metrics.
            notes: Description of the eval approach and what we're looking for.

        Returns:
            Confirmation message.
        """
        try:
            config_file = _eval_config_file(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        eval_configs_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "prompt_id": prompt_id,
            "default_models": default_models,
            "default_metrics": default_metrics,
            "notes": notes,
            "saved_at": datetime.now().isoformat(),
        }
        config_file.write_text(json.dumps(config, indent=2))
        models_str = ", ".join(default_models)
        metrics_str = ", ".join(m["type"] for m in default_metrics)
        return f"Saved eval config for '{prompt_id}'.\n  models: {models_str}\n  metrics: {metrics_str}"

    @tool(parse_docstring=True)
    def get_eval_config(prompt_id: str) -> str:
        """Get the saved evaluation config for a prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            The eval config as formatted JSON, or a message if not yet configured.
        """
        try:
            config_file = _eval_config_file(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not config_file.exists():
            return f"No eval config for '{prompt_id}' yet. Use save_eval_config to set one up."
        return config_file.read_text()

    return [
        add_test_case,
        list_test_cases,
        delete_test_case,
        infer_json_schema,
        save_eval_config,
        get_eval_config,
    ]

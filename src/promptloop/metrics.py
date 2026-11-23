import json
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

import jsonschema


def metric_passed(metric: Mapping[str, Any]) -> bool:
    return metric.get("passed") is True


def result_passed(result: Mapping[str, Any]) -> bool:
    if result.get("error"):
        return False
    metrics = result.get("metrics", [])
    if not isinstance(metrics, list):
        return False
    return all(isinstance(m, Mapping) and metric_passed(m) for m in metrics)


def evaluate_sync_metric(
    metric: Mapping[str, Any],
    actual_output: str,
    expected_output: str | None,
) -> dict[str, Any]:
    metric_type = metric.get("type")
    result: dict[str, Any] = {"type": metric_type, "passed": None, "score": None, "detail": ""}

    if metric_type == "latency":
        result["detail"] = "measured separately"
        result["passed"] = True

    elif metric_type == "json_schema":
        try:
            parsed = json.loads(actual_output)
        except json.JSONDecodeError as e:
            result["passed"] = False
            result["detail"] = f"invalid JSON: {e}"
        else:
            schema = metric.get("schema")
            if schema is None:
                result["passed"] = True
                result["detail"] = "valid JSON"
            elif not isinstance(schema, (dict, bool)):
                result["detail"] = "failed: json_schema metric 'schema' must be an object or boolean"
            else:
                try:
                    jsonschema.validate(parsed, schema)
                except jsonschema.ValidationError as e:
                    result["passed"] = False
                    result["detail"] = f"schema mismatch: {e.message}"
                except jsonschema.SchemaError as e:
                    result["detail"] = f"invalid schema: {e.message}"
                else:
                    result["passed"] = True
                    result["detail"] = "valid JSON matching schema"

    elif metric_type == "fuzzy_match":
        if expected_output is None:
            result["detail"] = "failed: fuzzy_match requires expected_output"
        else:
            raw_threshold = metric.get("threshold", 0.8)
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError):
                result["detail"] = f"failed: invalid fuzzy_match threshold {raw_threshold!r}"
            else:
                if not 0.0 <= threshold <= 1.0:
                    result["detail"] = f"failed: fuzzy_match threshold must be between 0 and 1, got {raw_threshold!r}"
                else:
                    ratio = SequenceMatcher(None, actual_output.strip(), expected_output.strip()).ratio()
                    result["score"] = round(ratio, 3)
                    result["passed"] = ratio >= threshold
                    result["detail"] = f"similarity {ratio:.1%} (threshold {threshold:.0%})"

    elif metric_type == "llm_judge":
        result["detail"] = "failed: llm_judge metrics are evaluated asynchronously"

    else:
        result["detail"] = f"failed: unknown metric type {metric_type!r}"

    return result

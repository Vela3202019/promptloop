from promptloop.metrics import evaluate_sync_metric, metric_passed, result_passed


def test_json_schema_metric_passes_and_fails():
    schema = {"type": "object", "required": ["name"]}
    passed = evaluate_sync_metric({"type": "json_schema", "schema": schema}, '{"name": "x"}', None)
    assert passed["passed"] is True

    failed = evaluate_sync_metric({"type": "json_schema", "schema": schema}, '{"other": 1}', None)
    assert failed["passed"] is False
    assert "schema mismatch" in failed["detail"]

    invalid_json = evaluate_sync_metric({"type": "json_schema", "schema": schema}, "not json", None)
    assert invalid_json["passed"] is False
    assert "invalid JSON" in invalid_json["detail"]


def test_fuzzy_match_metric():
    result = evaluate_sync_metric({"type": "fuzzy_match", "threshold": 0.8}, "hello world", "hello world")
    assert result["passed"] is True
    assert result["score"] == 1.0

    result = evaluate_sync_metric({"type": "fuzzy_match", "threshold": 0.8}, "completely different", "hello world")
    assert result["passed"] is False
    assert result["score"] < 0.8


def test_none_metric_counts_as_failure_with_reason():
    missing_expected = evaluate_sync_metric({"type": "fuzzy_match"}, "actual", None)
    assert missing_expected["passed"] is None
    assert missing_expected["detail"] == "failed: fuzzy_match requires expected_output"

    unknown = evaluate_sync_metric({"type": "unknown"}, "actual", "expected")
    assert unknown["passed"] is None
    assert unknown["detail"] == "failed: unknown metric type 'unknown'"

    result = {"error": None, "metrics": [missing_expected]}
    assert result_passed(result) is False
    assert metric_passed(missing_expected) is False
    assert result_passed({"error": None, "metrics": [{"passed": True}]}) is True
    assert result_passed({"error": "model failed", "metrics": [{"passed": True}]}) is False

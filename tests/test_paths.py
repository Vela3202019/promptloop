import pytest

from promptloop.paths import resolve_in_evals, validate_id


@pytest.mark.parametrize("bad_id", ["", ".", "..", "...", "../x", "a/b", "a\\b", "/tmp/x"])
def test_validate_id_rejects_traversal_and_dot_only_ids(bad_id):
    with pytest.raises(ValueError):
        validate_id(bad_id, "test_id")


@pytest.mark.parametrize("good_id", ["prompt", "prompt.v2", "inline_20260101_abc123", "run_1-x"])
def test_validate_id_accepts_normal_ids(good_id):
    assert validate_id(good_id, "test_id") == good_id


def test_resolve_in_evals_rejects_symlink_escape(tmp_path):
    evals = tmp_path / ".evals"
    evals.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (evals / "results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        resolve_in_evals(tmp_path, "results", "prompt", "run")


def test_resolve_in_evals_rejects_evals_root_symlink_escape(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside-evals"
    outside.mkdir()
    (project / ".evals").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"\.evals escapes project directory"):
        resolve_in_evals(project)

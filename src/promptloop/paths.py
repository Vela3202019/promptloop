import re
from pathlib import Path

SAFE_ID = re.compile(r"^(?!\.+$)[A-Za-z0-9_.-]+$")


def validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{kind} may only contain letters, numbers, dots, underscores, and hyphens, "
            "and may not consist only of dots."
        )
    return value


def resolve_in_evals(project_dir: Path, *parts: str) -> Path:
    root = (project_dir / ".evals").resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("path escapes .evals") from None
    return candidate


def resolve_in_project(project_dir: Path, path: Path) -> Path:
    root = project_dir.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes project directory: {path}") from None
    return resolved

"""Resolve paths at invocation time; importing never creates files."""
from pathlib import Path


def resolve_artifact_root(
    override: str | Path | None = None,
    configured: str | Path | None = None,
    *,
    cwd: Path | None = None,
) -> Path:
    value = override if override is not None else configured
    root = Path(value if value is not None else "./artifacts").expanduser()
    return (root if root.is_absolute() else (cwd or Path.cwd()) / root).resolve()

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_package_modules_are_namespaced() -> None:
    root = Path(__file__).resolve().parents[1]

    assert importlib.util.find_spec("lynxai") is not None
    assert importlib.util.find_spec("lynxai.utils") is not None
    assert importlib.util.find_spec("lynxai.app") is not None
    assert importlib.util.find_spec("utils") is None
    assert importlib.util.find_spec("app") is None

    project_toml = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'lynxai = "lynxai.cli:main"' in project_toml

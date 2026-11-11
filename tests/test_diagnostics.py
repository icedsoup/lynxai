import subprocess
import sys

import pytest

from lynxai import cli
from lynxai.utils.paths import resolve_artifact_root


def test_artifact_root_precedence_and_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_artifact_root() == tmp_path / "artifacts"
    assert resolve_artifact_root(configured="yaml") == tmp_path / "yaml"
    assert resolve_artifact_root("cli", "yaml") == tmp_path / "cli"
    assert resolve_artifact_root(tmp_path / "absolute", "yaml") == tmp_path / "absolute"
    assert resolve_artifact_root(configured="yaml", cwd=tmp_path / "other") == tmp_path / "other/yaml"
    assert not list(tmp_path.iterdir())


def test_imports_are_lazy_and_create_no_directories(tmp_path):
    code = '''
import sys
from pathlib import Path
import lynxai.cli
import lynxai.environments
import lynxai.utils.smoke_config
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
assert not list(Path.cwd().iterdir())
'''
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, check=True)


def test_default_core_and_path_override(tmp_path, capsys):
    path = tmp_path / "store"
    assert cli.main(["doctor", "--artifact-root", str(path)]) == 0
    output = capsys.readouterr().out
    assert str(path) in output and path.is_dir()
    assert "CS2 readiness: unverified" in output
    assert "PyTorch" not in output
    assert list(path.iterdir()) == []


def test_doctor_custom_config_and_relative_root(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("schema_version: 1\nexperiment:\n  artifact_root: from-yaml\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["doctor", "--config", str(cfg)]) == 0
    assert str(tmp_path / "from-yaml") in capsys.readouterr().out


@pytest.mark.parametrize("content", ["invalid: value", "[broken yaml", "schema_version: 2"])
def test_bad_config_is_actionable(tmp_path, capsys, content):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(content, encoding="utf-8")
    assert cli.main(["doctor", "--config", str(cfg)]) == 1
    output = capsys.readouterr().out
    assert "FAIL  Smoke configuration" in output
    assert "SKIP  Artifact root and Smoke2D" in output


def test_missing_config(tmp_path, capsys):
    assert cli.main(["doctor", "--config", str(tmp_path / "missing")]) == 1
    assert "FAIL  Smoke configuration" in capsys.readouterr().out


def test_unwritable_storage(tmp_path, monkeypatch, capsys):
    def denied(*args, **kwargs):
        raise PermissionError("write denied by ACL")

    monkeypatch.setattr(cli, "TemporaryFile", denied)
    assert cli.main(["doctor", "--artifact-root", str(tmp_path)]) == 1
    assert "write denied by ACL" in capsys.readouterr().out


def test_file_cannot_be_artifact_directory(tmp_path, capsys):
    target = tmp_path / "file"
    target.write_text("keep", encoding="utf-8")
    assert cli.main(["doctor", "--artifact-root", str(target)]) == 1
    assert "FAIL  Artifact root" in capsys.readouterr().out
    assert target.read_text() == "keep"


def test_missing_training_dependencies(tmp_path, monkeypatch, capsys):
    def missing(module):
        raise ImportError(f"No module named {module}")

    monkeypatch.setattr(cli, "import_module", missing)
    assert cli.main(["doctor", "--profile", "training", "--artifact-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "FAIL  PyTorch" in output and "FAIL  Stable-Baselines3" in output
    assert "uv sync --locked --extra ppo" in output


def test_cpu_training_profile(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    dependency = SimpleNamespace(__version__="test", cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(cli, "import_module", lambda module: dependency)
    assert cli.main(["doctor", "--profile", "training", "--artifact-root", str(tmp_path)]) == 0
    assert "CPU is supported" in capsys.readouterr().out

"""Shared fixtures for the Ada test suite.

- Puts the src layout on sys.path so tests run without an editable install.
- `isolated_env`: points HOME / cwd / secret env vars at temp locations so
  config discovery and .env loading never touch the real machine.
- `config`: a default Config built through load_config from a tmp config file.
- `clean_registry`: snapshot/clear/restore of the global tool registry for
  tests that register their own tools. Snapshot-restore (rather than a bare
  clear) is used so tools registered by module imports (e.g. ada.tools.web)
  survive for other tests — re-importing a cached module would not re-register
  them after a plain clear().
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import ada` work straight from the source tree.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

_SECRET_VARS = ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate config discovery / env secrets from the developer's machine.

    The setenv-then-delenv dance guarantees monkeypatch records an undo entry
    for each variable even when it was initially absent, so values written to
    os.environ by load_config's .env loader are cleaned up after the test.
    """
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in (*_SECRET_VARS, "ADA_CONFIG"):
        monkeypatch.setenv(var, "__ada_test_placeholder__")
        monkeypatch.delenv(var)
    monkeypatch.chdir(cwd)
    return tmp_path


@pytest.fixture
def config(isolated_env: Path):
    """A default Config loaded via load_config from a minimal tmp config file."""
    from ada.config import load_config

    conf_dir = isolated_env / "conf"
    conf_dir.mkdir()
    conf_file = conf_dir / "config.yaml"
    conf_file.write_text("assistant:\n  name: Ada\n", encoding="utf-8")
    return load_config(str(conf_file))


@pytest.fixture
def clean_registry():
    """Empty the global tool registry for one test, restoring it afterwards."""
    from ada.tools import registry

    saved = dict(registry._REGISTRY)
    registry.clear()
    try:
        yield registry
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)

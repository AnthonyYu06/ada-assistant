"""Tests for ada.config: layering, deep merge, secrets, .env, paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from ada.config import Config, load_config


def _write_config(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_load_without_user_file(isolated_env: Path) -> None:
    cfg = load_config()
    assert cfg.config_path is None
    assert cfg.base_dir == Path.cwd()
    # Spot-check bundled defaults across sections.
    assert cfg.assistant.name == "Ada"
    assert cfg.wake.enabled is False
    assert cfg.hotkey.enabled is True
    assert cfg.wake.threshold == 0.5
    assert cfg.audio.sample_rate == 16000
    assert cfg.stt.provider == "local"
    assert cfg.stt.local.model == "small.en"
    assert cfg.llm.max_tool_iterations == 6
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.ollama.model == "gemma4:12b"
    assert cfg.llm.anthropic.model == "claude-opus-4-8"
    assert cfg.llm.active_model == "gemma4:12b"
    assert cfg.tts.provider == "say"
    assert cfg.permissions.url_schemes == ["http", "https"]
    assert "quit_app" in cfg.permissions.always_confirm
    assert cfg.logging.audit is True


def test_user_yaml_deep_merge(isolated_env: Path) -> None:
    path = _write_config(
        isolated_env / "conf",
        "wake:\n"
        "  threshold: 0.75\n"
        "tts:\n"
        "  say:\n"
        "    voice: Daniel\n"
        "permissions:\n"
        "  blocked_apps: [Terminal]\n",
    )
    cfg = load_config(str(path))
    # Overridden values took effect.
    assert cfg.wake.threshold == 0.75
    assert cfg.tts.say.voice == "Daniel"
    assert cfg.permissions.blocked_apps == ["Terminal"]
    # Sibling keys at every level kept their defaults.
    assert cfg.wake.model == "hey_mycroft_v0.1"
    assert cfg.wake.cooldown_s == 2.0
    assert cfg.tts.provider == "say"
    assert cfg.tts.say.rate_wpm == 190
    assert cfg.tts.elevenlabs.model == "eleven_flash_v2_5"
    assert cfg.permissions.url_schemes == ["http", "https"]
    assert cfg.config_path == path


def test_unknown_keys_log_a_warning(
    isolated_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_config(
        isolated_env / "conf",
        "nonsense: true\n"
        "wake:\n"
        "  bogus_key: 1\n"
        "  threshold: 0.6\n",
    )
    with caplog.at_level(logging.WARNING, logger="ada.config"):
        cfg = load_config(str(path))
    assert "nonsense" in caplog.text
    assert "wake.bogus_key" in caplog.text
    # Known keys still load despite the unknown ones.
    assert cfg.wake.threshold == 0.6


def test_secrets_come_from_env(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-from-env")
    cfg = load_config()
    assert cfg.anthropic_api_key == "sk-from-env"
    assert cfg.deepgram_api_key == "dg-from-env"
    assert cfg.elevenlabs_api_key is None


def test_secrets_never_come_from_yaml(isolated_env: Path) -> None:
    path = _write_config(
        isolated_env / "conf",
        "anthropic_api_key: sk-from-yaml\n"
        "deepgram_api_key: dg-from-yaml\n",
    )
    cfg = load_config(str(path))
    assert cfg.anthropic_api_key != "sk-from-yaml"
    assert cfg.anthropic_api_key is None
    assert cfg.deepgram_api_key is None


def test_log_dir_relative_to_config_dir(isolated_env: Path) -> None:
    conf_dir = isolated_env / "conf"
    path = _write_config(conf_dir, "assistant:\n  name: Ada\n")
    cfg = load_config(str(path))
    assert cfg.base_dir == conf_dir
    assert cfg.log_dir == conf_dir / "logs"


def test_log_dir_absolute_is_kept(isolated_env: Path) -> None:
    abs_logs = isolated_env / "abs_logs"
    path = _write_config(
        isolated_env / "conf", f"logging:\n  dir: {abs_logs}\n"
    )
    cfg = load_config(str(path))
    assert cfg.log_dir == abs_logs


def test_log_dir_without_user_file_is_cwd_relative(isolated_env: Path) -> None:
    cfg = load_config()
    assert cfg.log_dir == Path.cwd() / "logs"


def test_dotenv_next_to_config_is_loaded(isolated_env: Path) -> None:
    conf_dir = isolated_env / "conf"
    path = _write_config(conf_dir, "assistant:\n  name: Ada\n")
    (conf_dir / ".env").write_text(
        "# a comment\n"
        "\n"
        'ANTHROPIC_API_KEY = "sk-from-dotenv"\n'
        "ELEVENLABS_API_KEY='el-from-dotenv'\n"
        "not-a-valid-line\n",
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.anthropic_api_key == "sk-from-dotenv"
    assert cfg.elevenlabs_api_key == "el-from-dotenv"
    assert cfg.deepgram_api_key is None


def test_dotenv_does_not_override_existing_env(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf_dir = isolated_env / "conf"
    path = _write_config(conf_dir, "assistant:\n  name: Ada\n")
    (conf_dir / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-already-set")
    cfg = load_config(str(path))
    assert cfg.anthropic_api_key == "sk-already-set"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-already-set"


def test_explicit_missing_config_path_raises(isolated_env: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(str(isolated_env / "does-not-exist.yaml"))


def test_non_mapping_yaml_raises(isolated_env: Path) -> None:
    path = isolated_env / "conf"
    path.mkdir()
    bad = path / "config.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_config_fixture_builds_default_config(config: Config) -> None:
    assert isinstance(config, Config)
    assert config.assistant.name == "Ada"
    assert config.config_path is not None
    assert config.config_path.name == "config.yaml"

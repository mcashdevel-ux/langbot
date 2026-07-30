"""Tests for config.py — optional config file with default fallbacks.

The central guarantee: a missing, malformed, or partial config file must never
change behaviour or raise; every lookup falls back to the caller's default.
"""

import json
from pathlib import Path

import pytest

from components.config import (
    CONFIG_ENV_VAR,
    CONFIG_FILENAME,
    Config,
    candidate_paths,
    load,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path, data):
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data,
                    encoding="utf-8")
    return path


class TestLookup:
    def test_dotted_key(self):
        cfg = Config({"llm": {"model": "qwen"}})
        assert cfg.get("llm.model", "local-model") == "qwen"

    def test_missing_key_and_missing_section_use_default(self):
        cfg = Config({"llm": {"model": "qwen"}})
        assert cfg.get("llm.base_url", "http://x") == "http://x"
        assert cfg.get("memory.worker_queue_size", 50) == 50

    def test_non_dict_intermediate_node_uses_default(self):
        cfg = Config({"llm": "qwen"})
        assert cfg.get("llm.model", "local-model") == "local-model"

    def test_empty_config_returns_every_default(self):
        cfg = Config()
        assert cfg.loaded is False
        assert cfg.get("tools.read_inline_chars", 1500) == 1500
        assert "no config file" in cfg.describe()


class TestCoercion:
    def test_numeric_string_is_coerced(self):
        assert Config({"tools": {"read_inline_chars": "2500"}}).get(
            "tools.read_inline_chars", 1500) == 2500

    def test_int_default_with_float_value(self):
        assert Config({"web": {"jina_timeout": 12.7}}).get("web.jina_timeout", 25) == 12

    def test_float_default_keeps_precision(self):
        assert Config({"llm": {"temperature": 0.7}}).get("llm.temperature", 0.1) == 0.7

    def test_uncoercible_value_warns_and_falls_back(self, caplog):
        cfg = Config({"memory": {"worker_queue_size": "fifty"}})
        with caplog.at_level("WARNING"):
            assert cfg.get("memory.worker_queue_size", 50) == 50
        assert "should be int" in caplog.text

    def test_bool_is_not_silently_treated_as_a_number(self, caplog):
        cfg = Config({"tools": {"read_inline_chars": True}})
        with caplog.at_level("WARNING"):
            assert cfg.get("tools.read_inline_chars", 1500) == 1500

    def test_container_value_for_scalar_falls_back(self):
        cfg = Config({"paths": {"scratch_dir": ["a", "b"]}})
        assert cfg.get("paths.scratch_dir", "./memory/agent_scratch") == "./memory/agent_scratch"

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("1", True), ("on", True),
        ("false", False), ("0", False), ("off", False),
    ])
    def test_bool_strings(self, raw, expected):
        assert Config({"x": {"flag": raw}}).get("x.flag", False) is expected

    def test_none_default_accepts_any_value(self):
        assert Config({"x": {"y": {"z": 1}}}).get("x.y", None) == {"z": 1}


class TestEnvOverride:
    def test_env_beats_config_file(self, monkeypatch):
        cfg = Config({"paths": {"scratch_dir": "/from/file"}})
        monkeypatch.setenv("AGENT_SCRATCH_DIR", "/from/env")
        assert cfg.get("paths.scratch_dir", "./d", env="AGENT_SCRATCH_DIR") == "/from/env"

    def test_blank_env_is_ignored(self, monkeypatch):
        cfg = Config({"paths": {"scratch_dir": "/from/file"}})
        monkeypatch.setenv("AGENT_SCRATCH_DIR", "  ")
        assert cfg.get("paths.scratch_dir", "./d", env="AGENT_SCRATCH_DIR") == "/from/file"

    def test_env_is_coerced_too(self, monkeypatch):
        monkeypatch.setenv("QUEUE", "7")
        assert Config().get("memory.worker_queue_size", 50, env="QUEUE") == 7

    def test_bad_env_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setenv("QUEUE", "lots")
        with caplog.at_level("WARNING"):
            assert Config().get("memory.worker_queue_size", 50, env="QUEUE") == 50
        assert "$QUEUE" in caplog.text


class TestLoading:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        # Keep ~/.config/langbot/config.json out of the picture.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    def test_no_file_anywhere_is_not_an_error(self):
        cfg = load()
        assert cfg.loaded is False
        assert cfg.get("llm.model", "local-model") == "local-model"

    def test_loads_cwd_file(self, tmp_path):
        _write(tmp_path / CONFIG_FILENAME, {"llm": {"model": "qwen"}})
        cfg = load()
        assert cfg.loaded and cfg.source.name == CONFIG_FILENAME
        assert cfg.get("llm.model", "local-model") == "qwen"

    def test_env_path_takes_priority(self, tmp_path, monkeypatch):
        _write(tmp_path / CONFIG_FILENAME, {"llm": {"model": "from-cwd"}})
        explicit = _write(tmp_path / "custom.json", {"llm": {"model": "from-env"}})
        monkeypatch.setenv(CONFIG_ENV_VAR, str(explicit))
        assert load().get("llm.model", "x") == "from-env"

    def test_missing_env_path_warns_then_falls_back(self, tmp_path, monkeypatch, caplog):
        _write(tmp_path / CONFIG_FILENAME, {"llm": {"model": "from-cwd"}})
        monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "nope.json"))
        with caplog.at_level("WARNING"):
            cfg = load()
        assert "does not exist" in caplog.text
        assert cfg.get("llm.model", "x") == "from-cwd"

    def test_malformed_json_warns_and_uses_defaults(self, tmp_path, caplog):
        _write(tmp_path / CONFIG_FILENAME, "{not json,}")
        with caplog.at_level("WARNING"):
            cfg = load()
        assert "not valid JSON" in caplog.text
        assert cfg.loaded is False
        assert cfg.get("llm.model", "local-model") == "local-model"

    def test_non_object_json_warns_and_uses_defaults(self, tmp_path, caplog):
        _write(tmp_path / CONFIG_FILENAME, [1, 2, 3])
        with caplog.at_level("WARNING"):
            cfg = load()
        assert "must contain a JSON object" in caplog.text
        assert cfg.loaded is False

    def test_home_config_is_a_candidate(self, tmp_path):
        home_cfg = tmp_path / "home" / ".config" / "langbot"
        home_cfg.mkdir(parents=True)
        _write(home_cfg / "config.json", {"llm": {"model": "from-home"}})
        assert load().get("llm.model", "x") == "from-home"
        assert home_cfg / "config.json" in candidate_paths()


class TestExampleFile:
    def test_example_matches_the_shipped_defaults(self):
        """The example file documents defaults, so it must not change behaviour."""
        import components.code_search as code_search
        import components.file_ops as file_ops
        import components.memory_worker as memory_worker
        import components.tool_call_repair as tool_call_repair
        import components.web_tools as web_tools

        example_path = REPO_ROOT / "langbot.config.example.json"
        example = json.loads(example_path.read_text(encoding="utf-8"))
        assert example["tools"]["read_inline_chars"] == file_ops.READ_INLINE_CHARS
        assert example["tools"]["grep_inline_lines"] == code_search.GREP_INLINE_LINES
        assert example["tools"]["manyfiles_inline_chars"] == code_search.MANYFILES_INLINE_CHARS
        assert example["web"]["fetch_save_chars"] == web_tools.FETCH_SAVE_CHARS
        assert example["memory"]["worker_queue_size"] == memory_worker.MAX_QUEUE_SIZE
        assert example["memory"]["worker_batch_size"] == memory_worker.MAX_BATCH
        assert example["compat"]["repair_json_tool_calls"] == tool_call_repair.REPAIR_ENABLED
        assert example["compat"]["repair_max_candidates"] == tool_call_repair.MAX_CANDIDATES

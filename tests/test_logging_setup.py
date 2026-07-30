"""Tests for logging_setup — keeping log records out of the REPL's output.

The guarantee: after setup(), records go to the log file and the console stays
clean unless console logging is asked for explicitly.
"""

import logging

import pytest

from components import logging_setup


@pytest.fixture(autouse=True)
def clean_root_logger():
    """Restore the root logger's handlers/level around every test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)
    logging_setup._log_path = None


class TestLevels:
    @pytest.mark.parametrize("name,expected", [
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        (" warning ", logging.WARNING),
        (logging.ERROR, logging.ERROR),
    ])
    def test_resolve_level(self, name, expected):
        assert logging_setup.resolve_level(name) == expected

    def test_unknown_level_falls_back(self):
        assert logging_setup.resolve_level("chatty") == logging.WARNING
        assert logging_setup.resolve_level("chatty", logging.INFO) == logging.INFO


class TestSetup:
    def test_records_land_in_the_log_file(self, tmp_path):
        target = tmp_path / "logs" / "langbot.log"
        assert logging_setup.setup(path=target) == target
        logging.getLogger("components.memory_worker").warning("distillation skipped")
        assert "distillation skipped" in target.read_text(encoding="utf-8")

    def test_console_stays_clean_by_default(self, tmp_path, capsys):
        logging_setup.setup(path=tmp_path / "langbot.log")
        logging.getLogger("components.tool_call_repair").warning("repaired 1 call")
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_console_can_be_enabled(self, tmp_path, capsys):
        logging_setup.setup(path=tmp_path / "langbot.log", console=True)
        logging.getLogger("x").warning("also on stderr")
        assert "also on stderr" in capsys.readouterr().err

    def test_level_filters_records(self, tmp_path):
        target = tmp_path / "langbot.log"
        logging_setup.setup(path=target, level="ERROR")
        logging.getLogger("x").warning("quiet")
        logging.getLogger("x").error("loud")
        text = target.read_text(encoding="utf-8")
        assert "loud" in text and "quiet" not in text

    def test_repeated_setup_replaces_its_handlers(self, tmp_path):
        root = logging.getLogger()
        before = len(root.handlers)
        for _ in range(3):
            logging_setup.setup(path=tmp_path / "langbot.log")
        assert len(root.handlers) == before + 1

    def test_setup_keeps_foreign_handlers(self, tmp_path):
        root = logging.getLogger()
        foreign = logging.NullHandler()
        root.addHandler(foreign)
        logging_setup.setup(path=tmp_path / "langbot.log")
        logging_setup.setup(path=tmp_path / "langbot.log")
        assert foreign in root.handlers

    def test_unopenable_file_degrades_to_stderr(self, tmp_path, capsys):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        assert logging_setup.setup(path=blocker / "langbot.log") is None
        assert logging_setup.log_path() is None
        logging.getLogger("x").warning("still logged")
        err = capsys.readouterr().err
        assert "could not open" in err and "still logged" in err

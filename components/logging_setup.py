"""Where log records go.

Every component logs through the stdlib ``logging`` module, but nothing used to
configure it — so records reached logging's last-resort handler, which prints
bare text to stderr. In a REPL that means diagnostics land in the middle of the
UI: the background distiller and the tool-call repairer emit warnings whenever
they run, and the distiller runs *between* turns, so its lines appear on top of
the prompt the user is typing into (``You: memory_worker: distillation
skipped...``).

The console therefore belongs to the UI alone: handlers write to a rotating log
file under ``./memory/`` (see MEMORY_POLICY.md) and nothing else. For debugging,
``logging.console`` (or ``LANGBOT_LOG_CONSOLE=1``) mirrors records to stderr and
``logging.level`` / ``LANGBOT_LOG_LEVEL`` changes verbosity.

``setup()`` is called once at startup, before the worker threads exist.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import config

LOG_FILE = config.get("paths.log_file", "./memory/langbot.log", env="LANGBOT_LOG_FILE")
LOG_LEVEL = config.get("logging.level", "WARNING", env="LANGBOT_LOG_LEVEL")
LOG_TO_CONSOLE = config.get("logging.console", False, env="LANGBOT_LOG_CONSOLE")
LOG_MAX_BYTES = config.get("logging.max_bytes", 1_000_000)
LOG_BACKUP_COUNT = config.get("logging.backup_count", 3)

FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_MARKER = "_langbot_handler"
_log_path: "Path | None" = None


def resolve_level(name, default: int = logging.WARNING) -> int:
    """Turn a level name (``"info"``, ``"DEBUG"``) into a level number."""
    if isinstance(name, int):
        return name
    level = logging.getLevelName(str(name).strip().upper())
    return level if isinstance(level, int) else default


def log_path() -> "Path | None":
    """The active log file, or ``None`` when logging only to the console."""
    return _log_path


def _install(handler: logging.Handler, level: int) -> None:
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FORMAT))
    setattr(handler, _MARKER, True)
    logging.getLogger().addHandler(handler)


def setup(
    path=None,
    level=None,
    console: "bool | None" = None,
) -> "Path | None":
    """Point the root logger at the log file. Returns the file in use, if any.

    Safe to call more than once: handlers installed by a previous call are
    replaced rather than stacked. A log file that cannot be opened is not worth
    failing startup over, so it degrades to stderr logging plus a warning.
    """
    global _log_path

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MARKER, False)]:
        root.removeHandler(handler)
        handler.close()

    resolved = resolve_level(LOG_LEVEL if level is None else level)
    to_console = LOG_TO_CONSOLE if console is None else console
    target = Path(path if path is not None else LOG_FILE).expanduser()
    error = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _install(
            RotatingFileHandler(
                target,
                maxBytes=max(int(LOG_MAX_BYTES), 0),
                backupCount=max(int(LOG_BACKUP_COUNT), 0),
                encoding="utf-8",
            ),
            resolved,
        )
        _log_path = target
    except OSError as e:
        error, _log_path, to_console = e, None, True

    if to_console:
        _install(logging.StreamHandler(), resolved)

    root.setLevel(resolved)
    logging.captureWarnings(True)
    if error is not None:
        logging.getLogger(__name__).warning(
            "logging: could not open %s (%s) — logging to stderr instead", target, error,
        )
    return _log_path

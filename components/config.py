"""Optional configuration file support.

Every tunable in langbot has a working built-in default, so the config file is
strictly optional: if none is found (or it is unreadable, malformed, or missing
keys) the agent runs exactly as it did before this module existed. Nothing here
raises — a bad file degrades to a warning plus defaults, because an unusable
config should never stop the agent from starting.

Lookup order for the file (first hit wins):

1. ``$LANGBOT_CONFIG`` (explicit path; a missing file here *is* warned about,
   since asking for a specific file and not getting it is worth knowing)
2. ``./langbot.config.json`` in the current working directory
3. ``~/.config/langbot/config.json``

Precedence for a single value: environment variable (where one is documented)
> config file > built-in default. Env vars stay on top so a one-off
``AGENT_SCRATCH_DIR=/tmp/x python langbot.py`` still wins over a checked-in file.

Values are addressed by dotted path and validated against the *default's* type,
so a typo like ``"queue_size": "fifty"`` logs a warning and falls back instead of
blowing up somewhere deep in a worker thread:

    from .config import config
    MAX_QUEUE_SIZE = config.get("memory.worker_queue_size", 50)
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "LANGBOT_CONFIG"
CONFIG_FILENAME = "langbot.config.json"


def candidate_paths() -> "list[Path]":
    """Config file locations in priority order (the env var, if set, first)."""
    paths = []
    explicit = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.append(Path.cwd() / CONFIG_FILENAME)
    paths.append(Path.home() / ".config" / "langbot" / "config.json")
    return paths


def _load_file(path: Path) -> "dict | None":
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.warning("config: %s is not valid JSON (%s) — using defaults", path, e)
        return None
    except OSError as e:
        logger.warning("config: could not read %s (%s) — using defaults", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("config: %s must contain a JSON object — using defaults", path)
        return None
    return data


class Config:
    """A loaded config file (or an empty one) queried by dotted key path."""

    def __init__(self, data: "dict | None" = None, source: "Path | None" = None):
        self._data = data or {}
        self.source = source

    @property
    def loaded(self) -> bool:
        return self.source is not None

    def get(self, dotted_key: str, default, env: "str | None" = None):
        """Return the configured value for ``dotted_key``, else ``default``.

        ``env``, when given, is an environment variable that overrides both. The
        result is coerced to ``type(default)``; anything that cannot be coerced
        (or a bool/str mismatch) logs a warning and yields ``default``.
        """
        if env:
            raw = os.environ.get(env, "").strip()
            if raw:
                return self._coerce(raw, default, f"${env}")
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return self._coerce(node, default, dotted_key)

    def _coerce(self, value, default, where: str):
        if default is None:
            return value
        expected = type(default)
        try:
            if expected is bool:
                if isinstance(value, bool):
                    return value
                lowered = str(value).strip().lower()
                if lowered in ("1", "true", "yes", "on"):
                    return True
                if lowered in ("0", "false", "no", "off"):
                    return False
                raise ValueError(value)
            # JSON `true` for a numeric/string setting is a mistake, not a 1.
            if isinstance(value, bool):
                raise TypeError(value)
            if isinstance(value, expected):
                return value
            if isinstance(value, (dict, list)):
                raise TypeError(value)
            return expected(value)
        except (TypeError, ValueError):
            logger.warning(
                "config: %s should be %s, got %r — using default %r",
                where, expected.__name__, value, default,
            )
            return default

    def describe(self) -> str:
        return str(self.source) if self.source else "(defaults; no config file found)"


def load() -> Config:
    """Load the first config file found, or an empty (all-defaults) Config."""
    explicit = os.environ.get(CONFIG_ENV_VAR, "").strip()
    for path in candidate_paths():
        data = _load_file(path)
        if data is not None:
            logger.debug("config: loaded %s", path)
            return Config(data, path)
        if explicit and str(path) == str(Path(explicit).expanduser()) and not path.exists():
            logger.warning("config: %s=%s does not exist — using defaults",
                           CONFIG_ENV_VAR, explicit)
    return Config()


config = load()


def reload() -> Config:
    """Re-read the config file, replacing the module singleton's contents.

    Only useful for tests and a manual reload; modules read their constants at
    import time, so already-imported values are not retroactively changed.
    """
    fresh = load()
    config._data = fresh._data
    config.source = fresh.source
    return config

"""Unit tests for components/truncate.py — head+tail truncation with saved full file.

Ported from neo's maybe_truncate tests (adapted to langbot's config-driven
save dir and tool vocabulary).
"""

import os
import re
from pathlib import Path

import components.truncate as truncate_mod
from components.truncate import maybe_truncate


def _saved_path(result: str) -> str:
    """Extract the saved-file path from a truncation notice."""
    m = re.search(r"saved to (\S+)", result)
    assert m, f"no saved-to path in result: {result[:200]}"
    return m.group(1)


class TestMaybeTruncate:
    def test_short_content_returned_verbatim(self, tmp_path):
        content = "x" * 5000
        result = maybe_truncate(content, save_dir=tmp_path)
        assert result == content
        assert list(tmp_path.iterdir()) == [tmp_path / "scratch"]  # no file created (scratch dir is the autouse fixture)

    def test_large_content_truncated_with_saved_file(self, tmp_path):
        content = "A" * 40_000
        result = maybe_truncate(content, save_dir=tmp_path)
        assert len(result) < len(content)
        assert "saved to" in result
        assert content[:100] in result
        assert content[-100:] in result
        path = _saved_path(result)
        saved = Path(path).read_text(encoding="utf-8")
        assert saved == content  # round-trips byte-for-byte

    def test_identical_content_dedupes_to_same_file(self, tmp_path):
        content = "B" * 40_000
        r1 = maybe_truncate(content, save_dir=tmp_path, prefix="dup")
        r2 = maybe_truncate(content, save_dir=tmp_path, prefix="dup")
        p1 = _saved_path(r1)
        p2 = _saved_path(r2)
        assert p1 == p2
        assert len(list(tmp_path.iterdir())) == 2  # saved file + autouse scratch dir

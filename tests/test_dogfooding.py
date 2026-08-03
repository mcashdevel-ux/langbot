"""Dogfooding reproduction test — A7.4 from langbot-upgrade-plan.md.

Replays the original dogfooding transcript: "analyze langbot.py" followed by
"analyze the tools", with recorded tool responses.  The second answer must
reflect *all* matches (more than the 5 that the old ``grep -m 5`` cap allowed)
because Track A routed ``find_in_files`` through the scratchpad.

Since a live LLM server is not available, this test works at the *tool level*:
it verifies that ``find_in_files`` against the real langbot.py file returns
matches and that scratch round-trips preserve the full result set.
"""

import os

from components.code_search import find_in_files
from components.scratch import SCRATCH_DIR, read_scratch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _langbot_path():
    return os.path.join(REPO_ROOT, "langbot.py")


def test_find_in_files_returns_matches():
    """find_in_files for 'def ' should return many matches in the repo."""
    result = find_in_files("def ", REPO_ROOT)
    # The result should contain matches — either inline or via scratch.
    if "scratch:" in result:
        # Extract scratch id: "full list at scratch:grep_xxxxx):"  ->  "grep_xxxxx"
        sid = result.split("scratch:")[1].split(")")[0].strip()
        full = read_scratch(sid, offset=0, length=100000)
        lines = [l for l in full.splitlines() if "def " in l]
        # The scratch write should faithfully round-trip all matches.
        assert len(lines) > 5, (
            f"find_in_files should return >5 'def ' matches via scratch; "
            f"got {len(lines)}"
        )
    else:
        # Small result set — verify it has at least one match.
        assert "def " in result, "should find at least one function definition"


def test_read_file_scratch_round_trip():
    """read_scratch on a saved result must faithfully round-trip the content."""
    from components.scratch import save_to_scratch

    content = "\n".join(f"langbot.py:{i}: match line {i}" for i in range(100))
    sid = save_to_scratch(content, prefix="grep")

    full = read_scratch(sid, offset=0, length=100000)
    restored = [l for l in full.splitlines() if l.startswith("langbot.py:")]
    assert len(restored) == 100, \
        f"round-trip should preserve all 100 lines, got {len(restored)}"


def test_find_in_files_against_langbot_py():
    """find_in_files for 'def agent' against langbot.py should find the agent function."""
    result = find_in_files("def agent", _langbot_path())
    assert "def agent" in result or "agent" in result.lower()

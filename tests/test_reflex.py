"""Track D tests — procedural reflexes (deterministic when-X → do-Y rules).

Ported from neo's reflex store tests (adapted to langbot's config-driven
reflexes file per docs/neo-port-plan.md Track D):

  - store → match by keyword overlap → run (the run itself is tested at the
    node level in test_langbot_reflex_wiring.py)
  - disabled rules don't match
  - ``use()`` increments and persists
  - distill updates an existing rule in place (same id, keeps uses)
  - corrupt rules file degrades to an empty store, never crashes
"""

import json

import pytest

from components.reflex import ReflexStore, _words


@pytest.fixture(autouse=True)
def reflex_file(tmp_path, monkeypatch):
    """Redirect the reflex store into a temp file for every test."""
    p = tmp_path / "reflexes.json"
    monkeypatch.setattr("components.reflex.REFLEXES_FILE", str(p))
    return p


def _store(tmp_path, **kw):
    return ReflexStore(path=tmp_path / "reflexes.json", **kw)


class TestWords:
    def test_short_words_excluded(self):
        assert _words("df -h") == []          # all words ≤ 2 chars
        assert _words("check disk space") == ["check", "disk", "space"]

    def test_lowercases_and_strips_punct(self):
        assert _words("Check DISK, please!") == ["check", "disk", "please"]


class TestReflexStore:
    def test_distill_and_match_by_overlap(self, tmp_path):
        s = _store(tmp_path)
        rid = s.distill("check disk space", "df -h")
        assert rid
        hits = s.match("please check the disk space")
        assert len(hits) == 1
        assert hits[0]["id"] == rid
        assert hits[0]["command"] == "df -h"

    def test_below_min_overlap_no_match(self, tmp_path):
        s = _store(tmp_path)
        s.distill("check disk space", "df -h")
        # Only one trigger word appears — below min_overlap=2
        assert s.match("check the weather") == []

    def test_disabled_rules_do_not_match(self, tmp_path):
        s = _store(tmp_path)
        rid = s.distill("check disk space", "df -h")
        assert s.disable(rid) is True
        assert s.match("please check the disk space") == []
        assert s.enable(rid) is True
        assert len(s.match("please check the disk space")) == 1

    def test_use_increments_and_persists(self, tmp_path):
        s = _store(tmp_path)
        rid = s.distill("check disk space", "df -h")
        s.use(rid)
        s.use(rid)
        s2 = ReflexStore(path=tmp_path / "reflexes.json")
        assert s2.all()[0]["uses"] == 2

    def test_distill_updates_in_place(self, tmp_path):
        s = _store(tmp_path)
        rid = s.distill("check disk space", "df -h")
        s.use(rid)
        rid2 = s.distill("Check  DISK space", "df -hT")
        assert rid2 == rid                       # same rule, updated
        rules = s.all()
        assert len(rules) == 1
        assert rules[0]["command"] == "df -hT"
        assert rules[0]["uses"] == 1          # use count survives the update

    def test_empty_trigger_or_command_rejected(self, tmp_path):
        s = _store(tmp_path)
        assert s.distill("", "df -h") == ""
        assert s.distill("check disk", "") == ""
        assert s.all() == []

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        p = tmp_path / "reflexes.json"
        p.write_text("{not json")
        s = ReflexStore(path=p)
        assert s.all() == []
        assert s.count() == 0

    def test_persists_across_reload(self, tmp_path):
        s = _store(tmp_path)
        s.distill("check disk space", "df -h")
        s2 = ReflexStore(path=tmp_path / "reflexes.json")
        assert s2.all()[0]["trigger"] == "check disk space"

    def test_match_returns_best_first(self, tmp_path):
        s = _store(tmp_path)
        s.distill("check disk space", "df -h")
        s.distill("check disk space usage", "du -sh .")
        hits = s.match("please check the disk space usage now")
        assert hits[0]["command"] == "du -sh ."   # 4-word overlap beats 3
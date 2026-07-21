"""Unit tests for file_ops.py — hardened read/write, patching, and diff."""

import os
import subprocess

import components.file_ops as file_ops


class TestReadFile:
    def test_read_text(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hello world")
        assert "hello world" in file_ops.read_file(str(p))

    def test_missing(self, tmp_path):
        assert "does not exist" in file_ops.read_file(str(tmp_path / "nope.txt"))

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e.txt"
        p.write_text("")
        assert file_ops.read_file(str(p)) == "(empty file)"

    def test_binary_detected(self, tmp_path):
        p = tmp_path / "b.bin"
        p.write_bytes(b"\x00\x01\x02binary\x00stuff")
        out = file_ops.read_file(str(p))
        assert "Binary file" in out
        assert "b.bin" in out

    def test_directory_rejected(self, tmp_path):
        assert "Not a file" in file_ops.read_file(str(tmp_path))


class TestWriteFile:
    def test_write_and_count(self, tmp_path):
        p = tmp_path / "w.txt"
        out = file_ops.write_file(str(p), "abcd")
        assert "Wrote 4 characters" in out
        assert p.read_text() == "abcd"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "w.txt"
        file_ops.write_file(str(p), "x")
        assert p.exists()

    def test_idempotent_overwrite(self, tmp_path):
        p = tmp_path / "w.txt"
        file_ops.write_file(str(p), "same")
        out = file_ops.write_file(str(p), "same")
        assert "Already up-to-date" in out

    def test_append(self, tmp_path):
        p = tmp_path / "w.txt"
        file_ops.write_file(str(p), "a")
        file_ops.write_file(str(p), "b", append=True)
        assert p.read_text() == "ab"

    def test_coerces_non_string(self, tmp_path):
        p = tmp_path / "w.txt"
        file_ops.write_file(str(p), ["a", "b", "c"])
        assert p.read_text() == "a\nb\nc"


class TestPatchFile:
    def test_basic_replace(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("the quick brown fox")
        out = file_ops.patch_file(str(p), "quick", "slow")
        assert "Patched" in out
        assert p.read_text() == "the slow brown fox"

    def test_only_first_occurrence(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x x x")
        file_ops.patch_file(str(p), "x", "y")
        assert p.read_text() == "y x x"

    def test_old_text_missing(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("abc")
        assert "not found" in file_ops.patch_file(str(p), "zzz", "q")

    def test_idempotent(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("new value here")
        out = file_ops.patch_file(str(p), "old value", "new value")
        assert "Idempotent" in out
        assert p.read_text() == "new value here"

    def test_py_syntax_error_rolls_back(self, tmp_path):
        p = tmp_path / "m.py"
        p.write_text("x = 1\n")
        out = file_ops.patch_file(str(p), "x = 1", "def broken(:")
        assert "syntax error" in out.lower()
        assert p.read_text() == "x = 1\n"  # rolled back

    def test_py_valid_patch_applies(self, tmp_path):
        p = tmp_path / "m.py"
        p.write_text("x = 1\n")
        out = file_ops.patch_file(str(p), "x = 1", "x = 2")
        assert "Patched" in out
        assert p.read_text() == "x = 2\n"

    def test_missing_file(self, tmp_path):
        assert "not found" in file_ops.patch_file(str(tmp_path / "no.py"), "a", "b").lower()


class TestBatchPatch:
    def test_multiple(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("one")
        b.write_text("two")
        out = file_ops.batch_patch([
            {"file_path": str(a), "old_text": "one", "new_text": "1"},
            {"file_path": str(b), "old_text": "two", "new_text": "2"},
        ])
        assert "2 applied" in out
        assert a.read_text() == "1"
        assert b.read_text() == "2"

    def test_accepts_json_string(self, tmp_path):
        import json
        a = tmp_path / "a.txt"
        a.write_text("one")
        payload = json.dumps([{"file_path": str(a), "old_text": "one", "new_text": "1"}])
        out = file_ops.batch_patch(payload)
        assert "1 applied" in out

    def test_not_a_list(self):
        assert "must be a list" in file_ops.batch_patch(42)

    def test_empty(self):
        assert "no patches" in file_ops.batch_patch([]).lower()


class TestGitDiff:
    def test_not_a_repo(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("a")
        assert "not in a git repository" in file_ops.git_diff(str(p)).lower()

    def test_shows_change(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        p = tmp_path / "x.txt"
        p.write_text("line one\n")
        subprocess.run(["git", "add", "x.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        p.write_text("line two\n")
        out = file_ops.git_diff(str(p))
        assert "-line one" in out
        assert "+line two" in out

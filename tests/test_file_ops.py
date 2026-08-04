"""Unit tests for file_ops.py — hardened read/write, patching, and diff."""

import os
import subprocess

import components.file_ops as file_ops
import components.scratch as scratch


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

    def test_large_file_previewed_and_saved_to_scratch(self, tmp_path):
        content = "".join(f"line {i}\n" for i in range(5000))
        assert len(content) > 40_000
        p = tmp_path / "big.txt"
        p.write_text(content)
        out = file_ops.read_file(str(p))
        assert "scratch:file_" in out
        assert len(out) < 2 * file_ops.READ_INLINE_CHARS
        sid = out.split("scratch:")[1].split(")")[0]
        page = scratch.read_scratch(sid, offset=0, length=len(content) + 10)
        assert page.split("\n", 1)[1] == content

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

    def test_large_diff_offloaded_in_full(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_ops, "MAX_OUTPUT_CHARS", 200)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        p = tmp_path / "x.txt"
        p.write_text("".join(f"old {i}\n" for i in range(400)))
        subprocess.run(["git", "add", "x.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        p.write_text("".join(f"new {i}\n" for i in range(400)))
        out = file_ops.git_diff(str(p))
        assert "scratch:diff_" in out
        sid = out.split("scratch:")[1].split(")")[0]
        full = scratch.read_scratch(sid, offset=0, length=1_000_000)
        assert "+new 399" in full


class TestPatchFormatValidation:
    def test_json_patch_validation_success(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"name": "langbot", "status": "open"}')
        res = file_ops.patch_file(str(p), '"status": "open"', '"status": "closed"')
        assert "Patched" in res
        assert '"status": "closed"' in p.read_text()

    def test_json_patch_validation_failure_rollback(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"name": "langbot", "status": "open"}')
        res = file_ops.patch_file(str(p), '"status": "open"', '"status": "open", invalid_json')
        assert "invalid JSON after patch (rolled back)" in res
        assert '"status": "open"' in p.read_text()
        assert "invalid_json" not in p.read_text()

    def test_toml_patch_validation_success(self, tmp_path):
        p = tmp_path / "test.toml"
        p.write_text('name = "langbot"\nstatus = "open"\n')
        res = file_ops.patch_file(str(p), 'status = "open"', 'status = "closed"')
        assert "Patched" in res
        assert 'status = "closed"' in p.read_text()

    def test_toml_patch_validation_failure_rollback(self, tmp_path):
        p = tmp_path / "test.toml"
        p.write_text('name = "langbot"\nstatus = "open"\n')
        res = file_ops.patch_file(str(p), 'status = "open"', 'status = [invalid_toml')
        assert "invalid TOML after patch (rolled back)" in res
        assert 'status = "open"' in p.read_text()
        assert "invalid_toml" not in p.read_text()

    def test_yaml_patch_validation_success(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text('name: langbot\nstatus: open\n')
        res = file_ops.patch_file(str(p), 'status: open', 'status: closed')
        assert "Patched" in res
        assert 'status: closed' in p.read_text()

    def test_yaml_patch_validation_failure_rollback(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text('name: langbot\nstatus: open\n')
        res = file_ops.patch_file(str(p), 'status: open', 'status: [invalid_yaml')
        # Since PyYAML is installed, it will validate and catch this syntax error!
        assert "invalid YAML after patch (rolled back)" in res
        assert 'status: open' in p.read_text()
        assert "invalid_yaml" not in p.read_text()


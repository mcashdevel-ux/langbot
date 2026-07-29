"""Unit tests for code_search.py — search and batch read."""

import components.code_search as cs
import components.scratch as scratch


class TestFindInFiles:
    def test_finds_match(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    return 1\n")
        out = cs.find_in_files("hello", str(tmp_path))
        assert "a.py" in out
        assert "hello" in out

    def test_no_match(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        assert "No matches" in cs.find_in_files("zzznope", str(tmp_path))

    def test_empty_pattern(self):
        assert "empty pattern" in cs.find_in_files("")

    def test_python_fallback(self, tmp_path):
        (tmp_path / "a.py").write_text("needle here\n")
        out = cs._find_in_files_py("needle", str(tmp_path))
        assert "a.py" in out
        assert "needle" in out

    def test_large_result_set_paged_via_scratch(self, tmp_path):
        # 30 matching lines in one file: nothing may be silently dropped.
        (tmp_path / "a.py").write_text("".join(f"needle {i}\n" for i in range(30)))
        out = cs.find_in_files("needle", str(tmp_path))
        assert "30 matches" in out
        assert "scratch:grep_" in out
        sid = out.split("scratch:")[1].split(")")[0]
        full = scratch.read_scratch(sid, offset=0, length=100_000)
        assert full.count("needle") == 30

    def test_small_result_set_stays_inline(self, tmp_path):
        (tmp_path / "a.py").write_text("needle 1\nneedle 2\nneedle 3\n")
        out = cs.find_in_files("needle", str(tmp_path))
        assert "scratch:" not in out
        assert out.count("needle") == 3

    def test_python_fallback_pages_large_result_set(self, tmp_path):
        (tmp_path / "a.py").write_text("".join(f"needle {i}\n" for i in range(30)))
        out = cs._find_in_files_py("needle", str(tmp_path))
        assert "30 matches" in out
        assert "scratch:grep_" in out

    def test_python_fallback_skips_vcs_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "cfg.txt").write_text("needle\n")
        (tmp_path / "a.py").write_text("needle\n")
        out = cs._find_in_files_py("needle", str(tmp_path))
        assert ".git" not in out
        assert "a.py" in out


class TestReadManyFiles:
    def test_reads_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("AAA")
        (tmp_path / "b.py").write_text("BBB")
        out = cs.read_many_files(str(tmp_path / "*.py"))
        assert "AAA" in out and "BBB" in out
        assert "a.py" in out and "b.py" in out

    def test_no_match(self, tmp_path):
        assert "No files matching" in cs.read_many_files(str(tmp_path / "*.md"))

    def test_large_output_paged_via_scratch(self, tmp_path):
        body = "z" * (cs.MANYFILES_INLINE_CHARS + 1000)
        (tmp_path / "big.txt").write_text(body)
        out = cs.read_many_files(str(tmp_path / "*.txt"))
        assert "scratch:manyfiles_" in out
        assert len(out) < len(body)
        sid = out.split("scratch:")[1].split(")")[0]
        assert body in scratch.read_scratch(sid, offset=0, length=200_000)

    def test_max_files_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text(str(i))
        out = cs.read_many_files(str(tmp_path / "*.txt"), max_files=2)
        assert out.count("--- ") == 2

    def test_empty_pattern(self):
        assert "empty pattern" in cs.read_many_files("")


class TestGlobList:
    def test_lists_with_sizes(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        out = cs.glob_list(str(tmp_path / "*.txt"))
        assert "a.txt" in out
        assert "1 shown" in out

    def test_marks_directories(self, tmp_path):
        (tmp_path / "sub").mkdir()
        out = cs.glob_list(str(tmp_path / "*"))
        assert "[dir]" in out

    def test_no_match(self, tmp_path):
        assert "No files matching" in cs.glob_list(str(tmp_path / "*.zzz"))

    def test_empty_pattern(self):
        assert "empty pattern" in cs.glob_list("")

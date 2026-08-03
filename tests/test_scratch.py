"""Unit tests for scratch.py — the shared on-disk scratchpad."""

import os

import components.scratch as scratch


class TestScratch:
    def test_new_scratch_id_prefix(self):
        sid = scratch._new_scratch_id("doc")
        assert sid.startswith("doc_")
        assert len(sid.split("_")[1]) == 8

    def test_save_and_read_roundtrip(self):
        sid = scratch.save_to_scratch("hello world", prefix="t")
        out = scratch.read_scratch(sid)
        assert "hello world" in out
        assert sid in out

    def test_save_never_truncates(self):
        content = "y" * 500_000
        sid = scratch.save_to_scratch(content)
        path = os.path.join(scratch.SCRATCH_DIR, f"{sid}.txt")
        assert os.path.getsize(path) == len(content)

    def test_read_missing_id(self):
        out = scratch.read_scratch("nope_12345678")
        assert "no scratch entry found" in out

    def test_read_paging_reports_more(self):
        sid = scratch.save_to_scratch("A" * 1000)
        out = scratch.read_scratch(sid, offset=0, length=100)
        assert "more available" in out
        assert "offset=100" in out

    def test_read_offset_no_more_at_end(self):
        sid = scratch.save_to_scratch("short content")
        out = scratch.read_scratch(sid, offset=0, length=1000)
        assert "more available" not in out

    def test_read_negative_offset_clamped(self):
        sid = scratch.save_to_scratch("data here")
        out = scratch.read_scratch(sid, offset=-50)
        assert "data here" in out

    def test_read_non_ascii_byte_offsets_consistent(self):
        # Multi-byte content: offsets/total are byte-based and consistent, and
        # a chunk boundary must not corrupt output or over-report "more".
        content = "café-\u00e9\u00e9\u00e9" * 50  # 'é' is 2 bytes in UTF-8
        sid = scratch.save_to_scratch(content)
        total_bytes = len(content.encode("utf-8"))
        out = scratch.read_scratch(sid, offset=0, length=total_bytes)
        assert f"/{total_bytes}]" in out
        assert "more available" not in out
        # Reassemble via paging and confirm it round-trips exactly.
        reassembled = ""
        offset = 0
        while True:
            page = scratch.read_scratch(sid, offset=offset, length=7)
            body = page.split("\n", 1)[1]
            if body.endswith(")"):
                body = body.rsplit("\n...", 1)[0]
            reassembled += body
            marker = page.split("]", 1)[0]
            end = int(marker.split("-")[1].split("/")[0])
            if "more available" not in page:
                break
            offset = end
        assert reassembled == content


class TestOffload:
    def test_short_content_stays_inline(self):
        assert scratch.offload("abc", "t", 10, "full output") == "abc"

    def test_long_content_previews_and_saves_in_full(self):
        content = "z" * 5000
        out = scratch.offload(content, "t", 100, "full output")
        assert out.startswith("5000 chars, showing first 100 (full output at scratch:t_")
        sid = out.split("scratch:")[1].split(")")[0]
        path = os.path.join(scratch.SCRATCH_DIR, f"{sid}.txt")
        assert os.path.getsize(path) == len(content)

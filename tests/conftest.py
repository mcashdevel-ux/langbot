import pytest

import components.scratch as scratch


@pytest.fixture(autouse=True)
def scratch_dir(tmp_path, monkeypatch):
    """Redirect the shared scratchpad into a temp dir for every test.

    Several tools save large results to scratch; without this they would write
    into the repo's ./memory/agent_scratch during a test run.
    """
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_DIR", str(d))
    return d

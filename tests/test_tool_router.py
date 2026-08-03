"""Unit tests for tool_router.py — which tool schemas get bound per turn."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import components.tool_router as tool_router


class _FakeTool:
    def __init__(self, name):
        self.name = name


ALL = [_FakeTool(n) for n in [
    "execute_shell_command", "read_any_file", "write_any_file", "patch_file",
    "batch_patch", "git_diff", "find_in_files", "read_many_files", "glob_list",
    "task_start", "task_list", "task_status", "task_output", "task_kill",
    "search_web", "fetch_url", "read_scratch", "remember", "recall", "vault",
]]


def _names(messages):
    return {t.name for t in tool_router.select_tools(ALL, messages)}


class TestSelection:
    def test_core_tools_are_always_bound(self):
        names = _names([HumanMessage(content="hello")])
        assert set(tool_router.CORE_TOOLS) <= names

    def test_a_plain_greeting_binds_far_fewer_tools(self):
        assert len(_names([HumanMessage(content="hello")])) < len(ALL)

    def test_credential_talk_binds_the_vault(self):
        assert "vault" in _names([HumanMessage(content="is the api key stored?")])

    def test_git_talk_binds_the_diff_tool(self):
        assert "git_diff" in _names([HumanMessage(content="what changed in the repo?")])

    def test_background_work_binds_the_task_tools(self):
        names = _names([HumanMessage(content="start the dev server in the background")])
        assert {"task_start", "task_kill"} <= names

    def test_a_url_in_a_tool_result_binds_fetch_url(self):
        messages = [
            HumanMessage(content="who won?"),
            AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "who won"}, "id": "1"},
            ]),
            ToolMessage(content="1. Result — https://example.com/x",
                        tool_call_id="1", name="search_web"),
        ]
        assert "fetch_url" in _names(messages)

    def test_a_tool_used_this_turn_stays_bound(self):
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[
                {"name": "glob_list", "args": {"pattern": "*"}, "id": "1"},
            ]),
            ToolMessage(content="a\nb", tool_call_id="1", name="glob_list"),
        ]
        assert "glob_list" in _names(messages)

    def test_a_tool_used_in_a_previous_turn_is_released(self):
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[
                {"name": "vault", "args": {"action": "list"}, "id": "1"},
            ]),
            ToolMessage(content="none", tool_call_id="1", name="vault"),
            HumanMessage(content="thanks"),
        ]
        assert "vault" not in _names(messages)

    def test_scratch_reference_binds_the_pager(self):
        messages = [
            HumanMessage(content="read it"),
            AIMessage(content="saved at scratch:file_1234abcd"),
        ]
        assert "read_scratch" in _names(messages)

    def test_dynamic_binding_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(tool_router, "DYNAMIC_BINDING", False)
        assert len(_names([HumanMessage(content="hello")])) == len(ALL)

    def test_an_empty_core_list_falls_back_to_every_tool(self, monkeypatch):
        monkeypatch.setattr(tool_router, "CORE_TOOLS", [])
        assert len(_names([HumanMessage(content="hello")])) == len(ALL)


class TestEmbeddingRouting:
    """Track 7: embedding-based tool routing as an additive signal."""

    @staticmethod
    def _mock_model():
        """Return a fake embedding model whose ``embed_query`` / ``embed_documents``
        return fixed vectors designed so that the turn text "check what is stored
        for auth" is closer to the vault description than to any other tool."""
        import random
        from unittest.mock import MagicMock

        random.seed(7)
        base = [random.random() for _ in range(384)]

        def perturb(v, amount):
            return [x + (random.random() - 0.5) * amount for x in v]

        desc_vectors = {}
        for name in tool_router._TOOL_DESCRIPTIONS:
            v = perturb(base, 0.3)
            if name == "vault":
                v = perturb(base, 0.02)
            desc_vectors[name] = v

        query_vec = perturb(perturb(base, 0.02), 0.01)

        mock = MagicMock()
        mock.embed_query.return_value = query_vec
        mock.embed_documents.side_effect = lambda texts: [
            desc_vectors.get(
                next((n for n in tool_router._TOOL_DESCRIPTIONS
                     if tool_router._TOOL_DESCRIPTIONS[n] == t), None),
                perturb(base, 0.3),
            )
            for t in texts
        ]
        return mock

    def test_embedding_routing_adds_vault_for_fuzzy_auth_query(self, monkeypatch):
        """Fuzzy query that the regex misses should still bind vault."""
        mock = self._mock_model()
        monkeypatch.setattr(tool_router, "_embeddings_model", mock)
        monkeypatch.setattr(tool_router, "_desc_vectors", {})
        monkeypatch.setattr(tool_router, "EMBEDDING_ROUTING", True)

        msg = "check what is stored for auth"
        # With only regex, this text does NOT match vault's trigger pattern.
        assert not tool_router._COMPILED["vault"].search(msg)

        # But embedding similarity should find it.
        embedding_hits = tool_router._embedding_tool_names(msg)
        assert "vault" in embedding_hits, \
            f"Expected vault in {embedding_hits} for fuzzy auth query"

        # Full selection should include both regex hits and embedding hits.
        all_hits = tool_router.select_tool_names([HumanMessage(content=msg)])
        assert "vault" in all_hits

    def test_disabled_embedding_routing_returns_empty(self, monkeypatch):
        monkeypatch.setattr(tool_router, "EMBEDDING_ROUTING", False)
        assert tool_router._embedding_tool_names("check what is stored") == set()

    def test_empty_text_returns_empty(self):
        assert tool_router._embedding_tool_names("") == set()

    def test_embedding_routing_is_additive_not_replacement(self, monkeypatch):
        """When both regex and embedding match different tools, both appear."""
        mock = self._mock_model()
        monkeypatch.setattr(tool_router, "_embeddings_model", mock)
        monkeypatch.setattr(tool_router, "_desc_vectors", {})
        monkeypatch.setattr(tool_router, "EMBEDDING_ROUTING", True)

        msg = "check the url for auth"
        assert tool_router._COMPILED["fetch_url"].search(msg)
        assert not tool_router._COMPILED["vault"].search(msg)

        names = tool_router.select_tool_names([HumanMessage(content=msg)])
        assert "fetch_url" in names              # regex
        assert "vault" in names                   # embedding (fuzzy "auth")

    def test_embedding_routing_does_not_reduce_baseline(self, monkeypatch):
        """Embedding routing never reduces the set selected by regex + core."""
        mock = self._mock_model()
        monkeypatch.setattr(tool_router, "_embeddings_model", mock)
        monkeypatch.setattr(tool_router, "_desc_vectors", {})
        monkeypatch.setattr(tool_router, "EMBEDDING_ROUTING", True)

        names = tool_router.select_tool_names([HumanMessage(content="hello")])
        assert len(names) > 0  # baseline from core tools

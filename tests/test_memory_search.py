"""Tests for memory search: dedup on write, relevance floor, diversity, lexical leg.

These run against a real (temp) ChromaDB collection with a controllable fake
embedder, so the ranking logic is exercised end to end rather than mocked.
"""

import pytest

import components.memory_store as memory_store

DIM = 8


class _Embeddings:
    """Embeds by lookup table: registered texts get exact vectors, others a
    deterministic near-orthogonal one, so similarity is a test input."""

    def __init__(self):
        self.table: "dict[str, list[float]]" = {}
        self.fallback_index = 0

    def register(self, text, vector):
        self.table[text] = list(vector)

    def _lookup(self, text):
        if text in self.table:
            return self.table[text]
        # Unregistered text: a unit vector on its own axis (similarity 0 to all
        # other unregistered text, and to every registered vector below).
        self.fallback_index += 1
        axis = self.fallback_index % DIM
        vector = [0.0] * DIM
        vector[axis] = 1.0
        self.table[text] = vector
        return vector

    def embed_query(self, text):
        return self._lookup(text)

    def embed_documents(self, texts):
        return [self._lookup(t) for t in texts]


@pytest.fixture
def store(tmp_path, monkeypatch):
    embeddings = _Embeddings()
    monkeypatch.setattr(memory_store, "_embeddings", embeddings)
    monkeypatch.setattr(memory_store, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(memory_store, "_client", None)
    monkeypatch.setattr(memory_store, "_collection", None)
    yield memory_store, embeddings
    memory_store._client = None
    memory_store._collection = None


def _axis(index, weight=1.0):
    vector = [0.0] * DIM
    vector[index] = weight
    return vector


class TestNormalization:
    @pytest.mark.parametrize("a,b", [
        ("The repo is at ~/code", "the repo is at ~/code"),
        ("spaced   out\ttext", "spaced out text"),
        ("  padded  ", "padded"),
    ])
    def test_equivalent_texts_normalize_alike(self, a, b):
        assert memory_store.normalize(a) == memory_store.normalize(b)

    def test_identifier_tokens_survive_intact(self):
        assert "~/code/myapp" in memory_store._tokens("the project lives at ~/code/myapp")
        assert "openai_api_key" in memory_store._tokens("OPENAI_API_KEY is set")


class TestWriteDedup:
    def test_identical_fact_is_stored_once(self, store):
        ms, _ = store
        first = ms.store_memory("User greeted me with 'hi'.")
        second = ms.store_memory("User greeted me with 'hi'.")
        assert first == second
        assert ms.count() == 1

    def test_dedup_ignores_case_and_whitespace(self, store):
        ms, _ = store
        ms.store_memory("The repo is at ~/ai/repos/langbot")
        ms.store_memory("the   repo is at ~/ai/repos/langbot  ")
        assert ms.count() == 1

    def test_paraphrase_with_shared_vocabulary_is_deduped(self, store):
        ms, emb = store
        # Same vector (a paraphrase, as far as the embedder is concerned) and
        # overlapping words -> one fact.
        emb.register("the langbot repo lives at ~/code/langbot", _axis(0))
        emb.register("langbot repo lives at ~/code/langbot", _axis(0))
        ms.store_memory("the langbot repo lives at ~/code/langbot")
        ms.store_memory("langbot repo lives at ~/code/langbot")
        assert ms.count() == 1

    def test_near_identical_vectors_with_different_facts_are_both_kept(self, store):
        ms, emb = store
        # This is why dedup needs the lexical guard: these embed identically but
        # are different facts.
        emb.register("the dev server listens on port 8080", _axis(1))
        emb.register("the metrics server listens on port 9100", _axis(1))
        ms.store_memory("the dev server listens on port 8080")
        ms.store_memory("the metrics server listens on port 9100")
        assert ms.count() == 2

    def test_one_differing_identifier_blocks_dedup(self, store):
        ms, emb = store
        # Same wording, same vector, one different number: two facts, not one.
        emb.register("writer 1 fact 2-3", _axis(2))
        emb.register("writer 1 fact 2-4", _axis(2))
        ms.store_memory("writer 1 fact 2-3")
        ms.store_memory("writer 1 fact 2-4")
        assert ms.count() == 2

    def test_batch_dedups_within_itself_and_against_the_store(self, store):
        ms, _ = store
        ms.store_memory("fact one")
        ids = ms.store_memories_batch(["fact one", "fact two", "fact two", ""])
        assert ms.count() == 2
        assert ids[0] == ids[0] and ids[1] == ids[2]     # dup ids point at the original
        assert ids[3] == ""

    def test_empty_fact_is_rejected(self, store):
        ms, _ = store
        with pytest.raises(ValueError):
            ms.store_memory("   ")

    def test_source_is_recorded(self, store):
        ms, _ = store
        ms.store_memory("a manual fact")
        ms.store_memories_batch(["a distilled fact"])
        metas = ms.get_collection().get(include=["metadatas"])["metadatas"]
        by_text = {m["text"]: m["source"] for m in metas}
        assert by_text["a manual fact"] == "manual"
        assert by_text["a distilled fact"] == "distilled"


class TestRelevanceFloor:
    def test_irrelevant_memories_are_not_returned(self, store):
        ms, emb = store
        emb.register("the user prefers DuckDuckGo", _axis(0))
        emb.register("deploy with make release", _axis(1))
        ms.store_memories_batch(["the user prefers DuckDuckGo", "deploy with make release"])
        emb.register("what is the capital of France", _axis(5))
        assert ms.search_memories("what is the capital of France") == []

    def test_relevant_memory_is_returned_with_a_score(self, store):
        ms, emb = store
        emb.register("the user prefers DuckDuckGo", _axis(0))
        ms.store_memory("the user prefers DuckDuckGo")
        emb.register("which search engine?", _axis(0))
        hits = ms.search_memories("which search engine?")
        assert [h.text for h in hits] == ["the user prefers DuckDuckGo"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].source == "manual" and hits[0].timestamp
        assert "dense" in hits[0].matched

    def test_threshold_is_overridable_per_call(self, store):
        ms, emb = store
        # ~11 degrees off the query axis in a mostly-orthogonal direction.
        emb.register("mildly related fact", [0.2, 1.0] + [0.0] * (DIM - 2))
        ms.store_memory("mildly related fact")
        emb.register("query", _axis(0))
        assert ms.search_memories("query") == []
        assert len(ms.search_memories("query", min_similarity=0.05)) == 1

    def test_empty_query_and_empty_store(self, store):
        ms, _ = store
        assert ms.search_memories("") == []
        ms.store_memory("some fact")
        assert ms.search_memories("anything", n=0) == []


class TestDiversity:
    def test_duplicate_texts_do_not_fill_the_result_set(self, store):
        ms, emb = store
        # Pre-existing duplicates (written before dedup existed) must still not
        # occupy every slot.
        emb.register("langbot uses ChromaDB", _axis(0))
        collection = ms.get_collection()
        collection.add(
            ids=["a", "b", "c"],
            embeddings=[_axis(0)] * 3,
            documents=["langbot uses chromadb"] * 3,
            metadatas=[{"text": "langbot uses ChromaDB", "norm": "langbot uses chromadb",
                        "timestamp": "2024-01-01T00:00:00Z", "source": "distilled"}] * 3,
        )
        emb.register("memory store?", _axis(0))
        assert [h.text for h in ms.search_memories("memory store?", n=3)] == [
            "langbot uses ChromaDB"
        ]

    def test_mmr_prefers_a_distinct_second_hit(self):
        """Equally relevant candidates: the one that repeats the first loses."""
        def candidate(name, vector, score):
            return {"id": name, "text": name, "vector": vector, "score": score,
                    "timestamp": "", "source": "", "rrf": 0.0, "matched": []}

        first = candidate("A", _axis(0), 0.9)
        restated = candidate("A restated", _axis(0), 0.88)      # same direction as A
        other = candidate("B", _axis(1), 0.85)                  # orthogonal to A
        picked = memory_store._mmr([first, restated, other], 2)
        assert [c["text"] for c in picked] == ["A", "B"]


class TestLexicalLeg:
    def test_identifier_is_found_despite_a_distant_embedding(self, store):
        ms, emb = store
        emb.register("the project lives at ~/code/myapp", _axis(0))
        ms.store_memory("the project lives at ~/code/myapp")
        # Query embeds nowhere near the fact: only the literal token connects them.
        emb.register("~/code/myapp", _axis(4))
        hits = ms.search_memories("~/code/myapp")
        assert [h.text for h in hits] == ["the project lives at ~/code/myapp"]
        assert "lexical" in hits[0].matched

    def test_lexical_match_is_case_insensitive(self, store):
        ms, emb = store
        emb.register("OPENAI_API_KEY is stored in the vault", _axis(0))
        ms.store_memory("OPENAI_API_KEY is stored in the vault")
        emb.register("openai_api_key", _axis(4))
        assert len(ms.search_memories("openai_api_key")) == 1

    def test_lexical_leg_can_be_disabled(self, store):
        ms, emb = store
        emb.register("the project lives at ~/code/myapp", _axis(0))
        ms.store_memory("the project lives at ~/code/myapp")
        emb.register("~/code/myapp", _axis(4))
        assert ms.search_memories("~/code/myapp", lexical=False) == []

    def test_common_words_are_not_used_as_literal_filters(self, store):
        assert memory_store._query_tokens("what is the top of it") == []
        tokens = memory_store._query_tokens("where is OPENAI_API_KEY on port 8080")
        assert "openai_api_key" in tokens and "8080" in tokens


class TestTags:
    def test_clean_tags_normalizes_and_caps(self):
        assert memory_store.clean_tags(["  Preference ", "UI/UX", "ui-ux", 3, ""]) == [
            "preference", "ui-ux",
        ]
        many = [f"t{i}" for i in range(10)]
        assert len(memory_store.clean_tags(many)) == memory_store.MAX_TAGS

    @pytest.mark.parametrize("text,tag", [
        ("the user prefers DuckDuckGo", "preference"),
        ("OPENAI_API_KEY is set in the environment", "credentials"),
        ("docs live at https://docs.example.com", "web"),
        ("the project lives at ~/code/myapp", "filesystem"),
    ])
    def test_auto_tags(self, text, tag):
        assert tag in memory_store.auto_tags(text)

    def test_auto_tags_stay_out_of_plain_prose(self):
        assert memory_store.auto_tags("the meeting is on Tuesday") == []

    def test_tags_are_stored_and_returned(self, store):
        ms, emb = store
        emb.register("the user's editor is vim", _axis(0))
        ms.store_memory("the user's editor is vim", tags=["preference", "editor"])
        emb.register("which editor", _axis(0))
        hits = ms.search_memories("which editor")
        assert hits and set(hits[0].tags) >= {"preference", "editor"}

    def test_tag_query_finds_tagged_fact(self, store):
        ms, emb = store
        emb.register("the user's editor is vim", _axis(0))
        ms.store_memory("the user's editor is vim", tags=["editor"])
        # A dense-orthogonal query: only the lexical tag match can find it.
        emb.register("#editor", _axis(4))
        hits = ms.search_memories("#editor")
        assert [h.text for h in hits] == ["the user's editor is vim"]
        assert "lexical" in hits[0].matched

    def test_hash_token_is_a_query_identifier(self):
        assert "#editor" in memory_store._query_tokens("#editor")

    def test_tags_do_not_block_dedup_and_merge_instead(self, store):
        ms, emb = store
        first = ms.store_memory("the user's editor is vim", tags=["editor"])
        second = ms.store_memory("the user's editor is vim", tags=["preference"])
        assert first == second
        assert ms.count() == 1
        emb.register("which editor", emb.table["the user's editor is vim"])
        hits = ms.search_memories("which editor")
        assert set(hits[0].tags) >= {"editor", "preference"}

    def test_batch_carries_per_fact_tags(self, store):
        ms, emb = store
        ms.store_memories_batch(
            ["fact one about vim", "fact two about emacs"],
            tags_list=[["editor"], ["other"]],
        )
        emb.register("#editor", _axis(5))
        hits = ms.search_memories("#editor")
        assert [h.text for h in hits] == ["fact one about vim"]


class TestRecallCompatibility:
    def test_recall_memories_still_returns_strings(self, store):
        ms, emb = store
        emb.register("langbot stores memories in ChromaDB", _axis(0))
        ms.store_memory("langbot stores memories in ChromaDB")
        emb.register("where are memories stored", _axis(0))
        assert ms.recall_memories("where are memories stored") == [
            "langbot stores memories in ChromaDB"
        ]

import pytest

from app.rag import RagPipeline, chunk_text


def test_short_text_single_chunk():
    assert chunk_text("你好世界", chunk_size=512, overlap=64) == ["你好世界"]


def test_empty_text():
    assert chunk_text("   ", chunk_size=512, overlap=64) == []


def test_long_text_sliding_window():
    text = "甲" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=100)
    # 步长 300：起点 0/300/600/900 → 4 块
    assert len(chunks) == 4
    assert chunks[0] == text[0:400]
    assert chunks[1] == text[300:700]
    assert chunks[-1] == text[900:1000]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=64, overlap=64)


from app.store import VectorStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """确定性 fake：'猫'相关 → x 轴，其他 → y 轴。"""
    return [[1.0, 0, 0, 0] if "猫" in t else [0, 1.0, 0, 0] for t in texts]


def make_pipeline(tmp_path):
    store = VectorStore(str(tmp_path / "rag.db"), dim=4)
    return RagPipeline(store=store, embed_fn=fake_embed)


def test_ingest_chunks_and_stores(tmp_path):
    p = make_pipeline(tmp_path)
    n = p.ingest("猫" * 600, source="cat.txt", chunk_size=400, overlap=100)
    assert n == 2  # 起点 0/300 → 2 块
    assert p.store.count() == 2


def test_retrieve_returns_relevant(tmp_path):
    p = make_pipeline(tmp_path)
    p.ingest("小猫喜欢睡觉", source="cat.txt")
    p.ingest("股市今天大涨", source="stock.txt")
    results = p.retrieve("猫咪", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "cat.txt"


def test_augment_builds_prompt_with_sources(tmp_path):
    p = make_pipeline(tmp_path)
    p.ingest("小猫喜欢睡觉", source="cat.txt")
    prompt, chunks = p.augment("猫咪在干嘛？", top_k=1)
    assert "小猫喜欢睡觉" in prompt
    assert "cat.txt" in prompt
    assert "猫咪在干嘛？" in prompt
    assert chunks[0]["source"] == "cat.txt"


def test_augment_empty_store_falls_back(tmp_path):
    p = make_pipeline(tmp_path)
    prompt, chunks = p.augment("猫咪在干嘛？", top_k=3)
    assert prompt == "猫咪在干嘛？"
    assert chunks == []

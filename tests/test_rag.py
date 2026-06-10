import pytest

from app.rag import chunk_text


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

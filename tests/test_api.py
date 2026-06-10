# tests/test_api.py
import base64

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def fake_embed(texts):
    return [[1.0, 0, 0, 0] if "猫" in t else [0, 1.0, 0, 0] for t in texts]


class FakeVlm:
    def __init__(self):
        self.last_images = None

    def stream(self, messages, images=None, max_tokens=1024):
        self.last_images = images
        yield "你好"
        yield "世界"

    def reply(self, messages, images=None, max_tokens=1024):
        return "".join(self.stream(messages, images, max_tokens))


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "api.db"), dim=4, embed_fn=fake_embed, vlm=FakeVlm()
    )
    with TestClient(app) as c:
        yield c


def test_embeddings_endpoint(client):
    r = client.post("/v1/embeddings", json={"input": ["猫", "股票"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2
    assert data[0]["embedding"] == [1.0, 0, 0, 0]
    assert data[0]["object"] == "embedding"


def test_embeddings_accepts_single_string(client):
    r = client.post("/v1/embeddings", json={"input": "猫"})
    assert len(r.json()["data"]) == 1


def test_ingest_then_query(client):
    r = client.post("/ingest", json={"text": "小猫喜欢睡觉", "source": "cat.txt"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] == 1

    r = client.post("/query", json={"query": "猫咪", "top_k": 1})
    results = r.json()["results"]
    assert results[0]["source"] == "cat.txt"


def test_chat_non_stream(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "你好世界"


def test_chat_stream_sse(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
    )
    assert r.status_code == 200
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert any("你好" in l for l in lines)


def test_chat_with_rag(client):
    client.post("/ingest", json={"text": "小猫喜欢睡觉", "source": "cat.txt"})
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "猫咪在干嘛？"}],
            "use_rag": True,
        },
    )
    body = r.json()
    assert body["sources"][0]["source"] == "cat.txt"


def test_chat_with_base64_image(client, tmp_path):
    png_1px = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
            "h6FO1AAAAABJRU5ErkJggg=="
        )
    ).decode()
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是什么？"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_1px}"},
                        },
                    ],
                }
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "你好世界"


def test_chat_bad_image_returns_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,不是base64!!!"},
                        }
                    ],
                }
            ]
        },
    )
    assert r.status_code == 400

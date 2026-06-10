# 本地多模态 RAG 服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本机搭建纯本地多模态 RAG 服务：FastAPI 暴露 OpenAI 兼容接口，VLM 看图对话，embedding + sqlite-vec 做语义检索与文档问答。

**Architecture:** 两个 MLX 模型常驻（Qwen2.5-VL-7B-8bit 生成、Qwen3-Embedding-0.6B-8bit 向量化），sqlite-vec 单文件向量库。`store.py`/`embeddings.py`/`vlm.py` 各管一层，`rag.py` 编排，`main.py` 只做 HTTP 映射。依赖注入（embed_fn / vlm 对象可替换）让所有 HTTP 与 RAG 逻辑都能用 fake 做无模型单元测试。

**Tech Stack:** mlx-vlm 0.6.2 / mlx-embeddings / sqlite-vec 0.1.9 / FastAPI / uv / pytest

**已实测核实的 API 事实（实现时直接信任，不要再猜）：**

- `mlx_embeddings.load(repo) -> (model, tokenizer)`；`generate(model, tokenizer, texts: list[str])` 返回对象，**向量取 `.text_embeds`**（签名标注 `mx.array` 是错的）。Qwen3-Embedding-0.6B 输出 **1024 维、已 L2 归一化**。
- `mlx_vlm.load(repo) -> (model, processor)`；`apply_chat_template(processor, model.config, messages, num_images=N)` 生成 prompt；`stream_generate(model, processor, prompt, image=[路径或URL], max_tokens=N)` **逐块产出 `GenerationResult`，文本在 `.text`**。
- sqlite-vec：`sqlite_vec.load(conn)` 加载扩展；`create virtual table t using vec0(embedding float[1024])`；插入用 `sqlite_vec.serialize_float32(list)`；检索 `where embedding match ? and k = ?` + `order by distance`（距离越小越相似；向量已归一化，等价余弦）。
- 模型 repo：`mlx-community/Qwen3-Embedding-0.6B-8bit`（已在 HF 缓存）、`mlx-community/Qwen2.5-VL-7B-Instruct-8bit`（~8GB，首次使用会下载，给长 timeout）。
- 依赖已全部装好（`uv add mlx-vlm mlx-embeddings sqlite-vec "fastapi[standard]"` 已完成）。

**并发说明：** 本服务是单用户本地 demo。MLX 推理与 sqlite 连接都不做多线程优化；`VectorStore` 内部用 `threading.Lock` 保护，端点用 async def 串行处理即可。

---

### Task 1: 项目骨架 + pytest

**Files:**
- Create: `app/__init__.py`（空文件）
- Create: `tests/__init__.py`（空文件）
- Modify: `pyproject.toml`（uv 自动改）

- [ ] **Step 1: 加 pytest 开发依赖**

```bash
uv add --dev pytest
```

- [ ] **Step 2: 建包结构**

```bash
mkdir -p app tests
touch app/__init__.py tests/__init__.py
```

- [ ] **Step 3: 验证 pytest 可运行**

Run: `uv run pytest --collect-only -q`
Expected: `no tests ran`（无报错即可）

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: 项目骨架与测试依赖"
```

---

### Task 2: VectorStore（sqlite-vec 封装，纯单元测试）

**Files:**
- Create: `app/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_store.py
from app.store import VectorStore


def make_store(tmp_path, dim=4):
    return VectorStore(str(tmp_path / "test.db"), dim=dim)


def test_add_and_count(tmp_path):
    s = make_store(tmp_path)
    n = s.add(["猫", "狗"], [[1, 0, 0, 0], [0, 1, 0, 0]], source="a.txt")
    assert n == 2
    assert s.count() == 2


def test_search_returns_nearest_first(tmp_path):
    s = make_store(tmp_path)
    s.add(
        ["猫", "狗", "小猫"],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
        source="a.txt",
    )
    results = s.search([1, 0, 0, 0], top_k=2)
    assert [r["text"] for r in results] == ["猫", "小猫"]
    assert results[0]["distance"] <= results[1]["distance"]
    assert results[0]["source"] == "a.txt"


def test_search_empty_store(tmp_path):
    s = make_store(tmp_path)
    assert s.search([1, 0, 0, 0], top_k=3) == []


def test_persistence(tmp_path):
    path = str(tmp_path / "p.db")
    VectorStore(path, dim=4).add(["猫"], [[1, 0, 0, 0]], source="a")
    s2 = VectorStore(path, dim=4)
    assert s2.count() == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: 实现 VectorStore**

```python
# app/store.py
import sqlite3
import threading

import sqlite_vec


class VectorStore:
    """sqlite-vec 向量库：存 (向量, 原文, 来源)，按余弦距离检索。"""

    def __init__(self, path: str, dim: int = 1024):
        self.dim = dim
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute(
            "create table if not exists chunks("
            "id integer primary key autoincrement, "
            "text text not null, source text not null)"
        )
        self.db.execute(
            f"create virtual table if not exists chunks_vec "
            f"using vec0(embedding float[{dim}])"
        )

    def add(self, texts: list[str], vectors: list[list[float]], source: str) -> int:
        with self._lock:
            for text, vec in zip(texts, vectors, strict=True):
                cur = self.db.execute(
                    "insert into chunks(text, source) values (?, ?)", (text, source)
                )
                self.db.execute(
                    "insert into chunks_vec(rowid, embedding) values (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(list(vec))),
                )
            self.db.commit()
        return len(texts)

    def search(self, vector: list[float], top_k: int = 4) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "select c.text, c.source, v.distance "
                "from chunks_vec v join chunks c on c.id = v.rowid "
                "where v.embedding match ? and k = ? order by v.distance",
                (sqlite_vec.serialize_float32(list(vector)), top_k),
            ).fetchall()
        return [
            {"text": t, "source": s, "distance": d} for t, s, d in rows
        ]

    def count(self) -> int:
        with self._lock:
            return self.db.execute("select count(*) from chunks").fetchone()[0]
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/store.py tests/test_store.py
git commit -m "feat: sqlite-vec 向量库封装"
```

---

### Task 3: 文本分块（rag.py 第一部分，纯函数）

**Files:**
- Create: `app/rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rag.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL，`No module named 'app.rag'`

- [ ] **Step 3: 实现 chunk_text**

```python
# app/rag.py
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """按字符滑动窗口分块，相邻块重叠 overlap 字符。"""
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: 文本滑动窗口分块"
```

---

### Task 4: 模型单例（models.py）+ embedding 封装（embeddings.py）

**Files:**
- Create: `app/models.py`
- Create: `app/embeddings.py`
- Test: `tests/test_embeddings.py`（真模型冒烟，模型仅 0.6GB 且已在缓存）

- [ ] **Step 1: 实现 models.py（懒加载单例；无逻辑可测，直接写）**

```python
# app/models.py
"""模型单例：懒加载，进程内只加载一次。

/ingest、/query、/v1/embeddings 只触发 embedding 模型（~0.6GB）；
VLM（~8GB）只有对话接口用到时才加载。
"""
from functools import lru_cache

EMBED_MODEL = "mlx-community/Qwen3-Embedding-0.6B-8bit"
VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
EMBED_DIM = 1024


@lru_cache(maxsize=1)
def get_embedder():
    from mlx_embeddings import load

    return load(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vlm():
    from mlx_vlm import load

    return load(VLM_MODEL)
```

- [ ] **Step 2: 写 embedding 冒烟测试（真模型）**

```python
# tests/test_embeddings.py
import numpy as np

from app.embeddings import embed_texts


def test_embed_shape_and_norm():
    vecs = embed_texts(["猫是哺乳动物", "今天天气不错"])
    arr = np.array(vecs)
    assert arr.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(arr, axis=-1), 1.0, atol=1e-3)


def test_semantic_similarity_sanity():
    vecs = np.array(embed_texts(["小猫在睡觉", "猫科动物", "股市大涨"]))
    sim_cat = vecs[0] @ vecs[1]
    sim_unrelated = vecs[0] @ vecs[2]
    assert sim_cat > sim_unrelated
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL，`No module named 'app.embeddings'`

- [ ] **Step 4: 实现 embed_texts**

```python
# app/embeddings.py
import numpy as np

from app.models import get_embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """文本批量转 1024 维归一化向量。注意：向量在返回对象的 .text_embeds 上。"""
    model, tokenizer = get_embedder()
    from mlx_embeddings import generate

    out = generate(model, tokenizer, texts)
    return np.array(out.text_embeds, dtype=np.float32).tolist()
```

- [ ] **Step 5: 运行确认通过（首次加载模型需几秒）**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add app/models.py app/embeddings.py tests/test_embeddings.py
git commit -m "feat: 模型单例与 embedding 封装"
```

---

### Task 5: RAG 编排（ingest / retrieve / 拼 prompt，依赖注入可测）

**Files:**
- Modify: `app/rag.py`（追加 RagPipeline）
- Test: `tests/test_rag.py`（追加）

- [ ] **Step 1: 写失败测试（fake embed_fn，无模型）**

在 `tests/test_rag.py` 追加：

```python
from app.rag import RagPipeline, chunk_text  # noqa: F811 (更新原有 import 行)
from app.store import VectorStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """确定性 fake：'猫'相关 → x 轴，其他 → y 轴。"""
    return [[1.0, 0, 0, 0] if "猫" in t else [0, 1.0, 0, 0] for t in texts]


def make_pipeline(tmp_path):
    store = VectorStore(str(tmp_path / "rag.db"), dim=4)
    return RagPipeline(store=store, embed_fn=fake_embed)


def test_ingest_chunks_and_stores(tmp_path):
    p = make_pipeline(tmp_path)
    n = p.ingest("猫" * 700, source="cat.txt", chunk_size=400, overlap=100)
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 新增测试 FAIL，`cannot import name 'RagPipeline'`

- [ ] **Step 3: 实现 RagPipeline**

在 `app/rag.py` 追加：

```python
from typing import Callable

from app.store import VectorStore

RAG_TEMPLATE = """请仅根据以下资料回答问题。如果资料中没有相关信息，请明确说明没有找到相关资料。

{context}

问题：{question}"""


class RagPipeline:
    """编排层：分块→向量化→入库；检索→拼 prompt。生成交给调用方。"""

    def __init__(self, store: VectorStore, embed_fn: Callable[[list[str]], list[list[float]]]):
        self.store = store
        self.embed_fn = embed_fn

    def ingest(self, text: str, source: str, chunk_size: int = 512, overlap: int = 64) -> int:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        return self.store.add(chunks, self.embed_fn(chunks), source=source)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        [qvec] = self.embed_fn([question])
        return self.store.search(qvec, top_k=top_k)

    def augment(self, question: str, top_k: int = 4) -> tuple[str, list[dict]]:
        """检索并拼 RAG prompt；无结果时原样返回问题（退化为普通对话）。"""
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return question, []
        context = "\n\n".join(
            f"[{i + 1}] 来源：{c['source']}\n{c['text']}" for i, c in enumerate(chunks)
        )
        return RAG_TEMPLATE.format(context=context, question=question), chunks
```

- [ ] **Step 4: 运行全部测试确认通过**

Run: `uv run pytest -v`
Expected: 全部 passed（store 4 + rag 9 + embeddings 2）

- [ ] **Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: RAG 编排（入库/检索/拼 prompt）"
```

---

### Task 6: VLM 封装 + 看图冒烟

**Files:**
- Create: `app/vlm.py`
- Create: `scripts/smoke_vlm.py`（手动冒烟，不进 pytest——8GB 模型太重）

- [ ] **Step 1: 实现 vlm.py（薄封装，逻辑极少，靠冒烟验证）**

```python
# app/vlm.py
from typing import Iterator

from mlx_vlm import apply_chat_template, stream_generate

from app.models import get_vlm


class Vlm:
    """VLM 推理封装。messages 为 [{'role','content':str}]，图片以本地路径/URL 列表单独传。"""

    def __init__(self, model=None, processor=None):
        if model is None or processor is None:
            model, processor = get_vlm()
        self.model = model
        self.processor = processor

    def stream(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        images = images or []
        prompt = apply_chat_template(
            self.processor, self.model.config, messages, num_images=len(images)
        )
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            max_tokens=max_tokens,
        ):
            yield chunk.text

    def reply(self, messages, images=None, max_tokens: int = 1024) -> str:
        return "".join(self.stream(messages, images, max_tokens))
```

- [ ] **Step 2: 写冒烟脚本**

```python
# scripts/smoke_vlm.py
"""手动冒烟：生成一张纯红色图，问 VLM 是什么颜色。首次运行会下载 ~8GB 模型。"""
import tempfile

from PIL import Image

from app.vlm import Vlm


def main():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGB", (256, 256), (255, 0, 0)).save(f.name)
        path = f.name

    vlm = Vlm()
    print("--- 看图 ---")
    answer = ""
    for piece in vlm.stream(
        [{"role": "user", "content": "这张图片是什么颜色？用一句话回答。"}],
        images=[path],
        max_tokens=64,
    ):
        answer += piece
        print(piece, end="", flush=True)
    print()
    assert "红" in answer or "red" in answer.lower(), f"看图回答异常: {answer}"

    print("--- 纯文字 ---")
    text = vlm.reply([{"role": "user", "content": "用一句话介绍猫。"}], max_tokens=64)
    print(text)
    assert len(text.strip()) > 0
    print("SMOKE OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行冒烟（首次下载 8GB，需要耐心；命令给长超时）**

Run: `uv run python scripts/smoke_vlm.py`（Bash timeout 设 600000ms；若下载超时就重跑，HF 支持断点续传）
Expected: 看图回答含「红」，纯文字正常生成，最后打印 `SMOKE OK`

- [ ] **Step 4: Commit**

```bash
git add app/vlm.py scripts/smoke_vlm.py
git commit -m "feat: VLM 封装与看图冒烟脚本"
```

---

### Task 7: FastAPI 服务（全部四个接口 + TestClient 测试）

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: 写失败测试（fake embed + fake vlm，无模型、秒级跑完）**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL，`No module named 'app.main'`

- [ ] **Step 3: 实现 main.py**

```python
# app/main.py
import base64
import binascii
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models import EMBED_DIM
from app.rag import RagPipeline
from app.store import VectorStore


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class IngestRequest(BaseModel):
    text: str
    source: str = "unknown"
    chunk_size: int = 512
    overlap: int = 64


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None
    stream: bool = False
    max_tokens: int = 1024
    use_rag: bool = False
    rag_top_k: int = 4


def _decode_data_url(url: str) -> str:
    """data:image/...;base64,xxx → 临时文件路径。http(s) URL 原样返回。"""
    if url.startswith(("http://", "https://")):
        return url
    if not url.startswith("data:"):
        raise HTTPException(400, f"不支持的图片 URL 形式: {url[:30]}")
    try:
        _, b64 = url.split(",", 1)
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error) as e:
        raise HTTPException(400, f"图片 base64 解析失败: {e}")
    suffix = ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        return f.name


def _parse_messages(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """把 OpenAI 多模态 content 拆成 (纯文本 messages, 图片路径列表)。"""
    parsed, images = [], []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part["text"])
                elif part.get("type") == "image_url":
                    images.append(_decode_data_url(part["image_url"]["url"]))
            content = "\n".join(texts)
        parsed.append({"role": m["role"], "content": content})
    return parsed, images


def _completion_body(text: str, model: str, sources: list[dict] | None = None) -> dict:
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    if sources is not None:
        body["sources"] = sources
    return body


def _sse_chunk(piece: str, model: str) -> str:
    import json

    return "data: " + json.dumps(
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        },
        ensure_ascii=False,
    ) + "\n\n"


def create_app(
    db_path: str = "data/vectors.db",
    dim: int = EMBED_DIM,
    embed_fn=None,
    vlm=None,
) -> FastAPI:
    """embed_fn / vlm 传 None 时使用真模型（懒加载）；测试时注入 fake。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal embed_fn, vlm
        if embed_fn is None:
            from app.embeddings import embed_texts

            embed_fn = embed_texts
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        app.state.pipeline = RagPipeline(
            store=VectorStore(db_path, dim=dim), embed_fn=embed_fn
        )
        app.state.get_vlm = (lambda: vlm) if vlm is not None else _real_vlm
        yield

    def _real_vlm():
        from app.vlm import Vlm

        return Vlm()

    app = FastAPI(title="mlx-lm-local-demo", lifespan=lifespan)

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingsRequest):
        texts = [req.input] if isinstance(req.input, str) else req.input
        vecs = app.state.pipeline.embed_fn(texts)
        return {
            "object": "list",
            "model": req.model or "qwen3-embedding-0.6b",
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vecs)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        n = app.state.pipeline.ingest(
            req.text, source=req.source, chunk_size=req.chunk_size, overlap=req.overlap
        )
        return {"chunks_added": n, "total_chunks": app.state.pipeline.store.count()}

    @app.post("/query")
    async def query(req: QueryRequest):
        return {"results": app.state.pipeline.retrieve(req.query, top_k=req.top_k)}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        messages, images = _parse_messages(req.messages)
        model_name = req.model or "qwen2.5-vl-7b"
        sources = None
        if req.use_rag and messages:
            question = messages[-1]["content"]
            prompt, sources = app.state.pipeline.augment(question, top_k=req.rag_top_k)
            messages = messages[:-1] + [{"role": "user", "content": prompt}]

        vlm_instance = app.state.get_vlm()
        if not req.stream:
            text = vlm_instance.reply(messages, images=images, max_tokens=req.max_tokens)
            return _completion_body(text, model_name, sources)

        def event_stream():
            for piece in vlm_instance.stream(
                messages, images=images, max_tokens=req.max_tokens
            ):
                yield _sse_chunk(piece, model_name)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
```

- [ ] **Step 4: 运行 API 测试确认通过**

Run: `uv run pytest tests/test_api.py -v`
Expected: 8 passed

- [ ] **Step 5: 跑全量测试**

Run: `uv run pytest -v`
Expected: 全部 passed

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: FastAPI 服务（OpenAI 兼容 + RAG 接口）"
```

---

### Task 8: 端到端冒烟 + README

**Files:**
- Modify: `README.md`
- Delete: `main.py`（根目录旧 hello demo，已被 app/ 取代）

- [ ] **Step 1: 起真实服务（后台，VLM 首次加载较慢）**

```bash
uv run uvicorn app.main:app --port 8000
```
（后台运行；等日志出现 `Application startup complete`）

- [ ] **Step 2: 依次 curl 验证四个接口**

```bash
# 1. embedding
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "你好"}' | head -c 200

# 2. 入库
curl -s localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text": "MLX 是 Apple 为 Apple Silicon 设计的机器学习框架，利用统一内存在 GPU 上高效运行。", "source": "mlx-intro"}'

# 3. 检索
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "什么是 MLX？", "top_k": 2}'

# 4a. RAG 对话（非流式）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "什么是 MLX？"}], "use_rag": true}'

# 4b. 流式
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "用一句话介绍猫"}], "stream": true}' | head -20
```

Expected: 1 返回 1024 维向量；2 返回 chunks_added≥1；3 命中 mlx-intro；4a 回答提到 Apple/机器学习且 sources 含 mlx-intro；4b 逐行 SSE 输出以 `data: [DONE]` 结束。

- [ ] **Step 3: 看图冒烟（真实图片走 HTTP）**

```bash
uv run python - <<'EOF'
import base64, json, urllib.request
from PIL import Image
import io
buf = io.BytesIO()
Image.new("RGB", (256, 256), (0, 0, 255)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "这张图是什么颜色？一句话回答。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "max_tokens": 64,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
EOF
```

Expected: 回答提到「蓝」。

- [ ] **Step 4: 停服务，删除根目录旧 main.py，更新 README**

README 重写为（保留原有环境/运行小节风格）：

```markdown
# mlx-lm-local-demo

纯本地多模态 RAG 服务（Apple Silicon + MLX）：看图对话、语义搜索、文档问答，OpenAI 兼容接口。

## 模型

- 对话/看图：`mlx-community/Qwen2.5-VL-7B-Instruct-8bit`（~8GB，首次使用自动下载）
- 向量化：`mlx-community/Qwen3-Embedding-0.6B-8bit`（~0.6GB）
- 向量库：sqlite-vec（`data/vectors.db`，单文件）

## 启动

```bash
uv run uvicorn app.main:app --port 8000
```

## 接口

| 路径 | 用途 |
|------|------|
| `POST /v1/chat/completions` | 对话；支持图片（base64/URL）、`stream`、`use_rag` |
| `POST /v1/embeddings` | 文本转向量（OpenAI 格式） |
| `POST /ingest` | 文本入库：分块 → 向量化 → 写入 sqlite-vec |
| `POST /query` | 纯语义检索，返回 top-k 片段 |

示例：

```bash
# 入库
curl localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text": "要入库的长文本……", "source": "notes.md"}'

# RAG 问答
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "我的笔记里说了什么？"}], "use_rag": true}'
```

## 测试

```bash
uv run pytest           # 单元测试（不依赖 8GB VLM）
uv run python scripts/smoke_vlm.py   # VLM 看图冒烟（首次下载 ~8GB）
```

注：`/ingest`、`/query`、`/v1/embeddings` 仅依赖 0.6GB 的 embedding 模型；VLM 只在对话接口首次调用时加载。
```

- [ ] **Step 5: 最终全量测试 + Commit**

```bash
uv run pytest
git add -A
git commit -m "feat: 端到端冒烟通过，更新 README"
```

---

## Self-Review 记录

- **Spec 覆盖**：四个接口（chat/embeddings/ingest/query）✅；流式 SSE ✅；图片 base64/URL ✅；RAG 检索为空退化 ✅（Task 5 augment + Task 7）；图片解析失败 400 ✅；分块参数可配 ✅；测试策略（store/rag 纯单元、API fake 注入、VLM 冒烟）✅；模型启动失败即退出——FastAPI lifespan 中加载失败会让启动直接抛错 ✅。
- **占位符扫描**：无 TBD/TODO；所有步骤含完整代码与期望输出。
- **类型一致性**：`VectorStore.add/search/count`、`chunk_text`、`RagPipeline.ingest/retrieve/augment`、`Vlm.stream/reply`、`create_app(db_path, dim, embed_fn, vlm)` 在各 Task 间签名一致；fake 与真实现接口一致（`stream/reply`）。
- **偏差说明**：spec 中「启动时加载两个模型」细化为：embedding 在 lifespan 即就位，VLM 懒加载（首次对话时）——保留了用户明确想要的「ingest/query 不碰 VLM」性质，且首次启动不强制等 8GB 下载。

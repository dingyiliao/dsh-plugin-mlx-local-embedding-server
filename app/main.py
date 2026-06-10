# app/main.py
import base64
import binascii
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

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
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw)
        return f.name


def _parse_messages(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """把 OpenAI 多模态 content 拆成 (纯文本 messages, 图片路径列表)。"""
    parsed, images = [], []
    try:
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
    except HTTPException:
        # Clean up any temp files already written before re-raising
        temp_files = [p for p in images if not p.startswith(("http://", "https://"))]
        for p in temp_files:
            if os.path.exists(p):
                os.unlink(p)
        raise
    return parsed, images


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


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


def _sse_chunk(piece: str, model: str, stream_id: str, first: bool = False) -> str:
    delta: dict = {"content": piece}
    if first:
        delta["role"] = "assistant"
    return "data: " + json.dumps(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
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

    def _real_vlm():
        from app.vlm import Vlm

        return Vlm()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal embed_fn
        if embed_fn is None:
            from app.embeddings import embed_texts

            embed_fn = embed_texts
            embed_fn(["warmup"])  # 启动时即加载 embedding 模型，加载失败直接拒绝启动
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        app.state.pipeline = RagPipeline(
            store=VectorStore(db_path, dim=dim), embed_fn=embed_fn
        )
        app.state.get_vlm = (lambda: vlm) if vlm is not None else _real_vlm
        yield

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
        try:
            n = app.state.pipeline.ingest(
                req.text, source=req.source, chunk_size=req.chunk_size, overlap=req.overlap
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"chunks_added": n, "total_chunks": app.state.pipeline.store.count()}

    @app.post("/query")
    async def query(req: QueryRequest):
        return {"results": app.state.pipeline.retrieve(req.query, top_k=req.top_k)}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        messages, images = _parse_messages(req.messages)
        temp_files = [p for p in images if not p.startswith(("http://", "https://"))]
        model_name = req.model or "qwen2.5-vl-7b"
        sources = None
        if req.use_rag and messages:
            question = messages[-1]["content"]
            prompt, sources = app.state.pipeline.augment(question, top_k=req.rag_top_k)
            messages = messages[:-1] + [{"role": messages[-1]["role"], "content": prompt}]

        vlm_instance = app.state.get_vlm()
        if not req.stream:
            text = vlm_instance.reply(messages, images=images, max_tokens=req.max_tokens)
            return JSONResponse(
                content=_completion_body(text, model_name, sources),
                background=BackgroundTask(_cleanup, temp_files),
            )

        stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def event_stream():
            first = True
            for piece in vlm_instance.stream(
                messages, images=images, max_tokens=req.max_tokens
            ):
                yield _sse_chunk(piece, model_name, stream_id, first=first)
                first = False
            # Final chunk: finish_reason + sources
            final: dict = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if sources is not None:
                final["sources"] = sources
            yield "data: " + json.dumps(final, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            background=BackgroundTask(_cleanup, temp_files),
        )

    return app


app = create_app()

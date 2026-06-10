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

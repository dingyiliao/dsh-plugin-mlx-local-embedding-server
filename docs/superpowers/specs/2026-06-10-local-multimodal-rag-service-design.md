# 本地多模态 RAG 服务 — 设计文档

日期：2026-06-10
状态：已与用户确认架构，待审阅

## 目标

在 Apple Silicon（M5 Pro / 24GB）上搭建一个**纯本地**的多模态助手服务，提供三类能力：

1. **看图对话** —— 上传图片，模型描述/分析/回答关于图片的问题
2. **语义搜索** —— 对本地文本资料建索引，用自然语言检索最相关片段
3. **文档问答（RAG）** —— 基于检索到的资料生成回答

对外以 **HTTP API** 形式提供，接口尽量**兼容 OpenAI**，方便任意 OpenAI 客户端直接接入。

## 关键决策（已确认）

| 决策点 | 选择 | 备注 |
|--------|------|------|
| 技术栈 | 纯 MLX 轻量自研 | `mlx-vlm` + `mlx-embeddings` + `sqlite-vec` + 自写 FastAPI 编排 |
| 检索对象 | 仅文本 | 图片交给 VLM 看；不做按图搜图（不引入多模态 embedding） |
| 交互形态 | 本地 API 服务 | OpenAI 兼容接口 + RAG 扩展接口 |
| VLM 模型 | `mlx-community/Qwen2.5-VL-7B-Instruct-8bit`（~8GB） | 看图质量优先，24GB 内存吃得下 |
| Embedding 模型 | `mlx-community/Qwen3-Embedding-0.6B-8bit`（~0.6GB） | 轻量、专做检索 |
| 向量库 | `sqlite-vec` | 单文件、零额外服务，最适合本地 |
| 对话输出 | 流式（SSE） | 打字机效果，OpenAI 客户端通用 |

**内存预算**：VLM(~8GB) + Embedding(~0.6GB) ≈ 9GB 常驻，24GB 余量充足，可长上下文。

## 架构

```
                  ┌─────────────────────────────┐
   HTTP 请求 ───▶ │   本地 API 服务 (FastAPI)     │
                  │                              │
  /v1/chat...  ──▶│  ① VLM：看图 + 对话 + 生成    │◀── Qwen2.5-VL-7B-8bit
  /v1/embeddings─▶│  ② Embedding：文本转向量      │◀── Qwen3-Embedding-0.6B-8bit
  /ingest /query ▶│  ③ RAG 编排 + 向量库          │◀── sqlite-vec (data/vectors.db)
                  └─────────────────────────────┘
```

两个模型在服务**启动时各加载一次**并常驻内存，请求间复用。

## 项目结构

```
mlx-lm-local-demo/
├── app/
│   ├── main.py          # FastAPI 入口，路由定义、启动加载模型
│   ├── models.py        # 持有 VLM + embedding 两个模型实例（单例）
│   ├── vlm.py           # 看图对话生成，封装 mlx-vlm，含流式
│   ├── embeddings.py    # 文本转向量，封装 mlx-embeddings
│   ├── store.py         # sqlite-vec：建表、写入、相似检索
│   └── rag.py           # 编排：分块→embed→入库；检索→拼prompt→调VLM
├── data/
│   └── vectors.db       # sqlite-vec 数据文件（gitignore）
├── tests/
│   ├── test_store.py
│   └── test_rag.py
└── pyproject.toml
```

每个模块单一职责，可独立理解与测试。

## 模块职责与接口

- **`store.py`**：`add(chunks, vectors, metadata)` / `search(query_vector, top_k) -> [(chunk, score, meta)]`。只管向量库，不依赖模型。
- **`embeddings.py`**：`embed_texts(list[str]) -> list[vector]`。只管文本转向量。
- **`vlm.py`**：`generate(messages, images=None, stream=False)`。只管 VLM 推理。
- **`rag.py`**：`ingest(text, source)`（分块→embed→store.add）；`answer(question, images, top_k)`（embed问题→store.search→拼prompt→vlm.generate）。编排层，依赖上述三者。
- **`main.py`**：HTTP 路由，把请求映射到 rag/vlm/embeddings，处理流式响应。

## HTTP 接口

| 方法/路径 | 用途 | 说明 |
|-----------|------|------|
| `POST /v1/chat/completions` | 对话 | OpenAI 格式；消息可含图片（base64/URL）；可选 `use_rag: true` 先检索再答；支持 `stream: true` |
| `POST /v1/embeddings` | 文本转向量 | OpenAI 格式 |
| `POST /ingest` | 入库 | 上传文档/文本 → 分块 → embed → 写库 |
| `POST /query` | 纯检索 | 返回 top-k 相关片段，用于调试 RAG |

## 数据流（一次 RAG 问答）

```
用户问题 ─▶ embed 问题 ─▶ sqlite-vec 取 top-k 片段 ─▶
拼进 prompt（检索内容 + 问题 [+ 图片]）─▶ VLM 流式生成 ─▶ 返回答案 + 引用来源
```

## 分块策略

- 默认按字符数滑动窗口分块（如 chunk_size ≈ 512、overlap ≈ 64），参数可配置。
- 每个 chunk 记录来源 `source` 与序号，便于回答时标注引用。

## 错误处理

- **模型加载失败**：服务启动即报错退出，不静默继续。
- **检索为空**：退化为普通对话，并在回答中提示「未找到相关资料」。
- **图片解析失败**：返回 4xx，明确错误信息。

## 测试策略

- `test_store.py`：写入若干向量后检索，验证 top-k 顺序正确（纯单元测试，不需模型）。
- `test_rag.py`：分块逻辑、prompt 拼接逻辑（可对 embedding/VLM 做 mock）。
- 冒烟测试：用真实小模型跑一次 ingest + answer，确认链路通。

## 非目标（YAGNI）

- 不做按图搜图（多模态 embedding）。
- 不做用户认证、多租户、持久会话。
- 不做前端界面（纯 API；以后需要可另起）。
- 不做重排（rerank）模型；top-k 直接喂 VLM，先验证效果。

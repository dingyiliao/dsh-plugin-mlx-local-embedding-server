# DSH MLX 本地插件

仅在 macOS 上把 Apple Silicon 本地 MLX embedding/RAG FastAPI 服务作为独立进程接入 DeepSeek Harness。MLX 依赖 Apple Silicon/macOS；在其他平台插件会记录警告并不启动服务。

插件应用时启动服务进程，插件进入 pending 或卸载时由 DSH shell/subprocess 生命周期关闭该进程。本插件不向模型注册 MLX 工具，也不提供内部 HTTP 代理。

默认配置已经内置，不需要在 `cordis.patch.yml` 中重复填写：

- `baseURL`：`http://127.0.0.1:8000`
- `command`：`uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`
- `workdir`：`/Users/dylan/Workspace/AI/mlx-lm-local-demo`

如需覆盖默认值，可在 profile patch 中写：

```yaml
- id: mlx-local
  config:
    command: uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
    workdir: /Users/dylan/Workspace/AI/mlx-lm-local-demo
```

Embedding 服务配置使用：

- endpoint：`http://127.0.0.1:8000/v1/embeddings`
- API key reference：留空
- model：`mlx-community/Qwen3-Embedding-0.6B-8bit`
- vector dimensions：`1024`

# DSH MLX local

Run the local MLX embedding/RAG FastAPI server as a managed process of DeepSeek Harness on macOS only. MLX requires Apple Silicon/macOS; on other platforms the plugin logs a warning and does not start the server.

The plugin starts the server when applied. When the plugin enters pending or unloads, the DSH shell/subprocess lifecycle closes the process. This plugin does not register model-facing MLX tools or provide an internal HTTP proxy.

The defaults are built in, so `cordis.patch.yml` does not need to repeat them:

- `baseURL`: `http://127.0.0.1:8000`
- `command`: `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`
- `workdir`: `/Users/dylan/Workspace/AI/mlx-lm-local-demo`

Override them with a profile patch when needed:

```yaml
- id: mlx-local
  config:
    command: uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
    workdir: /Users/dylan/Workspace/AI/mlx-lm-local-demo
```

Embedding configuration: endpoint `http://127.0.0.1:8000/v1/embeddings`, no API key reference, model `mlx-community/Qwen3-Embedding-0.6B-8bit`, and vector dimensions `1024`.

# mlx-lm-local-demo

在 Apple Silicon 上用 [mlx-lm](https://github.com/ml-explore/mlx-lm) 跑本地大模型推理的 demo。

## 环境

- macOS + Apple Silicon (arm64)
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖，Python 3.13

## 运行

```bash
uv run main.py
```

首次运行会自动从 Hugging Face 下载模型（`Qwen2.5-1.5B-Instruct-4bit`，约 1GB）到本地缓存。

也可以直接用 mlx-lm 自带的命令行：

```bash
uv run mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "你好"
```

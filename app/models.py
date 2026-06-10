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

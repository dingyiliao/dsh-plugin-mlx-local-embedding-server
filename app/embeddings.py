# app/embeddings.py
import numpy as np

from app.models import get_embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """文本批量转 1024 维归一化向量。注意：向量在返回对象的 .text_embeds 上。"""
    model, tokenizer = get_embedder()
    from mlx_embeddings import generate

    out = generate(model, tokenizer, texts)
    return np.array(out.text_embeds, dtype=np.float32).tolist()

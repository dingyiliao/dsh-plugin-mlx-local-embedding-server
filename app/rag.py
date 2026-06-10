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

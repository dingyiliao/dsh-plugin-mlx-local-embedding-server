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

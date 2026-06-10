# app/vlm.py
from typing import Iterator

from mlx_vlm import apply_chat_template, stream_generate

from app.models import get_vlm


class Vlm:
    """VLM 推理封装。messages 为 [{'role','content':str}]，图片以本地路径/URL 列表单独传。"""

    def __init__(self, model=None, processor=None):
        if model is None or processor is None:
            model, processor = get_vlm()
        self.model = model
        self.processor = processor

    def stream(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        images = images or []
        prompt = apply_chat_template(
            self.processor, self.model.config, messages, num_images=len(images)
        )
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            max_tokens=max_tokens,
        ):
            yield chunk.text

    def reply(self, messages, images=None, max_tokens: int = 1024) -> str:
        return "".join(self.stream(messages, images, max_tokens))

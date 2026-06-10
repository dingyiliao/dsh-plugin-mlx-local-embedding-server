# scripts/smoke_vlm.py
"""手动冒烟：生成黑底红圆图，问 VLM 看到什么。首次运行会下载 ~8GB 模型。

注意：不要用纯色图做探针——无特征的均匀图像会让视觉编码器输出退化
（实测纯红图被回答成"白色"），有形状的图才能真正验证看图能力。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from app.vlm import Vlm


def main():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (448, 448), (0, 0, 0))
        ImageDraw.Draw(img).ellipse([100, 100, 348, 348], fill=(255, 0, 0))
        img.save(f.name)
        path = f.name

    vlm = Vlm()
    print("--- 看图 ---")
    answer = ""
    for piece in vlm.stream(
        [{"role": "user", "content": "这张图里有什么形状？是什么颜色？用一句话回答。"}],
        images=[path],
        max_tokens=64,
    ):
        answer += piece
        print(piece, end="", flush=True)
    print()
    assert "红" in answer or "red" in answer.lower(), f"看图回答异常: {answer}"

    print("--- 纯文字 ---")
    text = vlm.reply([{"role": "user", "content": "用一句话介绍猫。"}], max_tokens=64)
    print(text)
    assert len(text.strip()) > 0
    print("SMOKE OK")


if __name__ == "__main__":
    main()

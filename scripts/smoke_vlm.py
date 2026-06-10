# scripts/smoke_vlm.py
"""手动冒烟：生成一张纯红色图，问 VLM 是什么颜色。首次运行会下载 ~8GB 模型。"""
import tempfile

from PIL import Image

from app.vlm import Vlm


def main():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGB", (256, 256), (255, 0, 0)).save(f.name)
        path = f.name

    vlm = Vlm()
    print("--- 看图 ---")
    answer = ""
    for piece in vlm.stream(
        [{"role": "user", "content": "这张图片是什么颜色？用一句话回答。"}],
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

from mlx_lm import generate, load

# 4-bit 量化的小模型，首次运行会自动从 Hugging Face 下载到本地缓存
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def main():
    model, tokenizer = load(MODEL)

    messages = [{"role": "user", "content": "用一句话介绍一下 MLX 框架。"}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=True,
    )
    print(text)


if __name__ == "__main__":
    main()

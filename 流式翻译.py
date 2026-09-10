from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
import time

client = OpenAI(
    api_key="sk-5d1450437ddd4cd5a3d37ba2b9c2b947",
    base_url="https://api.deepseek.com"
)

system_prompt = "你是一个翻译助手。如果用户输入中文，请翻译成英文；如果用户输入英文，请翻译成中文。只输出翻译结果，不要有任何前缀、解释或引号。"

print("=== 命令行翻译助手 ===")
print("输入要翻译的文字，输入 exit 退出程序")

while True:
    try:
        user_input = input(">>> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n程序退出")
        break

    if user_input.lower() in ["exit", "quit"]:
        print("程序退出")
        break

    if not user_input:
        continue

    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2,
            stream=True
        )

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta

        with open("output.txt", "a", encoding="utf-8") as f:
            f.write("原文: " + user_input + "\n")
            f.write("译文: " + full_text + "\n")
            f.write("-" * 40 + "\n")

        print("已完成翻译，结果已保存到 output.txt")

    except AuthenticationError:
        print("API Key 错误，请检查 key 是否正确。")
    except RateLimitError:
        print("请求过于频繁，请稍后再试。")
        time.sleep(2)
    except APIConnectionError:
        print("网络连接失败，请检查网络。")
    except APIStatusError as e:
        print("服务端返回错误：", e.status_code)
    except Exception as e:
        print("发生未知错误：", type(e).__name__, ":", e)

    print("-" * 40)
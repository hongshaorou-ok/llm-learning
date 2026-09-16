from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
import time

client = OpenAI(
    api_key="sk-5d1450437ddd4cd5a3d37ba2b9c2b947",
    base_url="https://api.deepseek.com"
)

# 默认系统提示
system_prompt = "你是一个乐于助人的助手。"

# 保存对话历史
messages = [{"role": "system", "content": system_prompt}]

print("=== 命令行聊天机器人 ===")
print("输入 /system 角色描述  来设定角色，例如 /system 你是一个数学老师")
print("输入 exit 退出程序")

while True:
    try:
        user_input = input("你：").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n程序退出")
        break

    if user_input.lower() in ["exit", "quit"]:
        print("程序退出")
        break

    if not user_input:
        continue

    # TODO 1: 处理 /system 命令
    if user_input.startswith("/system"):
        system_prompt = user_input[len("/system"):].strip()
        # 更新 messages 列表里的 system 消息
        messages = [{"role": "system", "content": system_prompt}]
        print("已切换角色为：", system_prompt)
        continue

    # 把用户消息加入历史
    messages.append({"role": "user", "content": user_input})

    try:
        # TODO 2: 调用流式 API，记得带上完整的 messages 历史
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            stream=True
        )

        print("AI：", end="", flush=True)
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_text += delta
        print()

        # 把 AI 回复也加入历史
        messages.append({"role": "assistant", "content": full_text})

        # TODO 3: 把本轮对话写入 chat_history.txt
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            f.write("你：" + user_input + "\n")
            f.write("AI：" + full_text + "\n")
            f.write("-" * 40 + "\n")

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
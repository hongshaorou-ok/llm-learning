from openai import OpenAI
import chromadb

# 初始化客户端
client = OpenAI(
    api_key="your API key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 准备长文档切分
document = (
               "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
               "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，"
               "应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。人工智能可以对人的意识、思维的信息过程进行模拟。"
               "人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"
           ) * 10

chunk_size = 200
overlap = 50
stride = chunk_size - overlap

chunks = []
for start in range(0, len(document), stride):
    end = start + chunk_size
    chunk = document[start:end]
    chunks.append(chunk)
    if end >= len(document):
        break

print(f"切分出了{len(chunks)}个文本块")


# 定义embedding函数
def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return response.data[0].embedding


# 初始化chroma并且存入数据
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_knowledge_base")

embeddings_list = []
ids_list = []
for i, chunk in enumerate(chunks):
    vec = get_embedding(chunk)
    embeddings_list.append(vec)
    ids_list.append(f"id_{i}")

collection.add(
    documents=chunks,
    embeddings=embeddings_list,
    ids=ids_list
)
print("已存入向量数据库")

question = "人工智能有什么优点"

question_vec = get_embedding(question)

# 检索相关的三个文本块
results = collection.query(
    query_embeddings=[question_vec],
    n_results=3
)

# 拼接检索到的文本块
retrieved_docs = results["documents"][0]
context = "\n\n".join(retrieved_docs)

print("\n检索到的的资料片段为:")
print(context[:200], "...")
# 构造prompt
prompt = f"""你是一个知识助手,请根据下面的资料回答问题。如果资料中没有答案,请回答“资料未提及”
【资料】
{context}

【问题】
{question}

【回答】
"""

# 调用大模型生成回答(流式)
stream = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    stream=True
)
print("\nAI回答:", end="", flush=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()

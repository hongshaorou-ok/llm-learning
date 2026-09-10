from openai import OpenAI
import chromadb
import numpy as np

# ---------- 1. 初始化智谱客户端 ----------
client = OpenAI(
    api_key="25e80698c62a40c7b2d4109b48411b00.5f9ReM35vOjkmhyp",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ---------- 2. 准备一个长文档并切分 ----------
document = (
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，"
    "应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。人工智能可以对人的意识、思维的信息过程进行模拟。"
    "人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"
) * 10   # 复制10遍，让文本足够长

chunk_size = 100
overlap = 20
stride = chunk_size - overlap   # 80

chunks = []
for start in range(0, len(document), stride):
    end = start + chunk_size
    chunk = document[start:end]
    chunks.append(chunk)
    if end >= len(document):
        break

print(f"切分出 {len(chunks)} 个文本块")

# ---------- 3. 定义 embedding 函数 ----------
def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return response.data[0].embedding

# ---------- 4. 初始化 Chroma ----------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_knowledge_base")

# ---------- 5. 将 chunks 向量化并存入 Chroma ----------
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

print("已将所有文本块存入向量数据库")

# ---------- 6. 查询测试 ----------
question = "人工智能可以超越人类吗"
question_vec = get_embedding(question)

results = collection.query(
    query_embeddings=[question_vec],
    n_results=3
)

print("\n查询问题：", question)
print("最相似的3个文本块：\n")

# 注意：results 是一个字典，里面有 documents、ids、distances 等
for i, doc in enumerate(results["documents"][0]):
    print(f"第{i+1}个匹配：")
    print(doc[:200])   # 只打印前200字符，避免太长
    print("-" * 40)
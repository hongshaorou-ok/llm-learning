from openai import OpenAI
import numpy as np

# 初始化客户端
client = OpenAI(
    api_key="25e80698c62a40c7b2d4109b48411b00.5f9ReM35vOjkmhyp",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 三个待比较的文本
texts = [
    "苹果是一种水果",
    "香蕉也是一种水果",
    "今天天气很好"
]
names = ["A", "B", "C"]

# 调用 Embedding API，一次传入三个文本
response = client.embeddings.create(
    model="embedding-2",   # 如果报错，请查 DeepSeek 文档确认模型名
    input=texts
)

# 取出三个向量（每个向量是一个很长的数字列表）
vectors = []
for data in response.data:
    vectors.append(data.embedding)

# 定义余弦相似度函数
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 两两比较
pairs = [(0, 1), (0, 2), (1, 2)]   # 代表 (A,B), (A,C), (B,C)
best_pair = None
best_score = -1.0

for i, j in pairs:
    score = cosine_similarity(vectors[i], vectors[j])
    print(f"{names[i]} 和 {names[j]} 的相似度：{score:.4f}")
    if score > best_score:
        best_score = score
        best_pair = (names[i], names[j])

print(f"\n最相似的一对是：{best_pair[0]} 和 {best_pair[1]}，相似度：{best_score:.4f}")
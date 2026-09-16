from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
import uvicorn

# 初始化智谱客户端
client = OpenAI(
    api_key="f5b46cf5fd9e4bc592f342f98bff9621.mcFRrVHS2NyfXXBM",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 准备文档并且创建chroma(只在启动时执行一次)
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


def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return response.data[0].embedding


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


# 封装RAG函数

def get_answer(question: str) -> str:
    question_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=(question_vec),
        n_results=3
    )
    retrieved_docs = results["documents"][0]
    context = "/n/n".join(retrieved_docs)

    prompt = f"""请只根据下面资料回答问题。如果资料中没有答案,就回答"资料中未提及"。
    
【资料】
{context}

【问题】
{question}

请直接给出答案:"""

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是严谨的知识助手,只根据给定的资料回答问题"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# 请求体模型
class QuestionRequest(BaseModel):
    question: str


#     创建FastAPI应用
app = FastAPI()


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = get_answer(request.question)
    return {"answer": answer}


# 启动
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

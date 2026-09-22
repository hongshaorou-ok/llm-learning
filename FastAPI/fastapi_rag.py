from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from docx import Document
import chromadb
import uvicorn
import pymupdf
import io


# ==========  解析函数 ============
def extract_pdf_text(content: bytes) -> str:
    """从 PDF 二进制内容中提取文本"""
    doc = pymupdf.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_docx_text(content: bytes) -> str:
    """从 DOCX 二进制内容中提取文本"""
    doc = Document(io.BytesIO(content))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text


# ========== 1. 初始化客户端 ==========
client = OpenAI(
    api_key="你的 APIkey(根据你的大模型更改网站和思考模型)",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "my_knowledge_base"


# ========== 2. 公共函数（顶格，所有地方都能用） ==========
def split_text(text, chunk_size=200, overlap=50):
    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), stride):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response.data[0].embedding


# ========== 3. 判断集合是否存在 ==========
existing = [c.name for c in chroma_client.list_collections()]

if COLLECTION_NAME in existing:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print("已加载已有集合，跳过向量化")
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    document = (
                   "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
                   "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，"
                   "应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。人工智能可以对人的意识、思维的信息过程进行模拟。"
                   "人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"
               ) * 10

    chunks = split_text(document)

    embeddings_list = []
    ids_list = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        embeddings_list.append(vec)
        ids_list.append(f"id_{i}")

    collection.add(
        documents=chunks,
        embeddings=embeddings_list,
        ids=ids_list,
        metadatas=[{"source": "default"} for _ in chunks]
    )
    print("首次构建完成，已存入磁盘")


# ========== 4. RAG 函数 ==========
def get_answer(question: str) -> str:
    question_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=[question_vec],
        n_results=5
    )
    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)

    prompt = f"""请只根据下面资料回答问题。如果资料中没有答案，就回答"资料中未提及"。

【资料】
{context}

【问题】
{question}

请直接给出答案："""

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是严谨的知识助手，只根据给定的资料回答问题"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# ========== 5. FastAPI ==========
class QuestionRequest(BaseModel):
    question: str


app = FastAPI()


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = get_answer(request.question)
    return {"answer": answer}


SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"]


@app.get("/list")
def list_files():
    all_data = collection.get()
    sources = set()
    for meta in all_data["metadatas"]:
        sources.add(meta["source"])
    return {"files": list(sources)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1检查扩展名
    filename = file.filename.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"不支持的文件类型：{ext}，仅支持 {SUPPORTED_EXTENSIONS}"}

    content = await file.read()
    try:
        if ext == ".txt":
            text = content.decode("utf-8")
        elif ext == ".pdf":
            text = extract_docx_text(content)
        elif ext == ".docx":
            text = extract_docx_text(content)
    except Exception as e:
        return {"error": f"文件解析失败:{str(e)}"}

    if not text.strip():
        return {"error": "文件内容为空/无法提取文本"}
    chunks = split_text(text)

    # 先检查是否重复（用文件名 + 序号作为 id）
    ids_list = [f"{file.filename}_{i}" for i in range(len(chunks))]
    old = collection.get(ids=ids_list)
    if old["ids"]:
        return {"error": f"文件 {file.filename} 已存在，请先删除或改名"}

    # 再向量化
    embeddings_list = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        embeddings_list.append(vec)

    # 入库
    collection.add(
        documents=chunks,
        embeddings=embeddings_list,
        ids=ids_list,
        metadatas=[{"source": filename} for _ in chunks]
    )
    return {"message": "上传成功", "filename": file.filename, "type": ext, "chunks": len(chunks)}


@app.delete("/delete")
def delete_file(filename: str):
    existing = collection.get(where={"source": filename})
    if not existing["ids"]:
        return {"error": f"文件 {filename} 不存在"}
    collection.delete(where={"source": filename})
    return {"message": f"文件 {filename} 已删除"}


# ========== 6. 启动 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from docx import Document
from datetime import datetime
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
    api_key="你的智谱APIkey",
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
    print("已创建空集合,请通过/upload上传文档")


# ========= 工具函数  ============
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =======字典映射========
func_map = {
    "get_current_time": get_current_time
}

# =======工具描述========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户询问现在几点、今天几号等问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ========== 4.判断是否闲聊进行直接回答或者走RAG流程 ==========
def get_answer(question: str) -> str:
    # ========== 第一步：带 tools 调用大模型 ==========
    messages = [{"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        tools=tools,
        temperature=0
    )
    msg = response.choices[0].message

    # ========== 第二步：判断是否调工具 ==========
    if msg.tool_calls:
        # 调工具
        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            if func_name in func_map:
                result = func_map[func_name]()
            else:
                result = f"未知函数：{func_name}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
            # 再次调用大模型，生成最终回答
        final = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            temperature=0.3
        )
        return final.choices[0].message.content

    # ========== 第三步：不调工具，走意图判断 ==========
    intent_prompt = f"""请判断下面这句话属于哪类，只回答一个词：闲聊 或 知识。
    
用户输入: {question}
"""
    intent_resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0
    )
    intent = intent_resp.choices[0].message.content.strip()
    # 第二步:闲聊分支,直接回答
    if "闲聊" in intent:
        chat_response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一个友好的客服助手，请用自然、亲切的语气回答用户。"},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return chat_response.choices[0].message.content

    # 第三步：知识分支，走 RAG

    question_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=[question_vec],
        n_results=3
    )
    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)

    prompt = f"""请只根据下面资料回答问题。如果资料中没有答案，就回答"资料中未提及"。

【资料】
{context}

【问题】
{question}

请直接给出答案："""

    rag_resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是严谨的知识助手，只根据给定的资料回答问题"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return rag_resp.choices[0].message.content


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
            text = extract_pdf_text(content)
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

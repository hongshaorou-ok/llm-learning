# RAG 知识库问答 Web 服务

## 项目简介

基于 FastAPI + 智谱 AI + ChromaDB 搭建的检索增强生成（RAG）知识库问答服务。
用户通过 HTTP 接口提交问题，服务先从知识库中检索相关片段，再由大模型基于片段生成回答。

## 技术栈

- Python 3.12
- FastAPI（Web 框架）
- Uvicorn（ASGI 服务器）
- 智谱 AI（Embedding + Chat）
- ChromaDB（向量数据库）

## 核心流程

1. 文档切分（chunking）
2. 文本向量化（Embedding）
3. 存入 Chroma 向量数据库
4. 用户提问 → 问题向量化 → 相似度检索
5. 检索结果拼接成 context
6. 大模型基于 context 生成回答

## 接口说明

### POST /ask

请求体：

```json
{
  "question": "人工智能是什么？"
}
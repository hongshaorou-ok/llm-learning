# RAG 知识库问答 Web 服务

## 项目简介

基于 FastAPI + 智谱 AI + ChromaDB 搭建的检索增强生成（RAG）知识库问答服务。
支持上传TXT/PDF/DOCX文档,自动切分,向量化,存入向量数据库并通过 HTTP 接口提供问答、文件列表和删除功能。

## 技术栈

- Python 3.12
- FastAPI（Web 框架）
- Uvicorn（ASGI 服务器）
- 智谱 AI（Embedding + Chat）
- ChromaDB（向量数据库）
- - pymupdf（PDF 解析）
- python-docx（DOCX 解析）

## 核心流程

1.文档切分（chunking）
2.文本向量化（Embedding）
3.用户上传资料文档
4.存入 Chroma 向量数据库
5.用户提问 → 问题向量化 → 相似度检索
6.检索结果拼接成 context
7.大模型基于 context 生成回答

## 接口说明

### POST /ask

请求体：

```json
{
  "question": "人工智能是什么？"
}
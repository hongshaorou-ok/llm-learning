# 本地知识库问答机器人（RAG）

## 项目简介
基于大模型和向量数据库实现的知识库问答系统。将长文档切分、向量化后存入 Chroma，用户提问时检索最相关片段，再由大模型生成回答。

## 技术栈
- Python
- 智谱 AI Embedding API
- 智谱 AI Chat API
- ChromaDB

## 功能
- 文档切分（chunking）
- 文本向量化
- 向量检索
- 基于检索结果生成回答

## 运行方法
1. 安装依赖：pip install openai chromadb
2. 替换代码中的 API Key
3. 运行：python rag_complete.py

## 效果展示
（可粘贴运行截图或回答示例）
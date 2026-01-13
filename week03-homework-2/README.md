# Week 03 Homework 2: RAG System with GraphRAG and Milvus

本项目包含两个部分：
1. **GraphRAG**: 基于图数据库 (Neo4j) 的多跳问答系统。
2. **Milvus FAQ**: 基于向量数据库 (Milvus) 的 FAQ 检索系统。

## 环境准备

### 1. 依赖安装

确保已安装 Python 3.11+，并在项目根目录下运行：

```bash
pip install -e .
# 如果使用的是本地文件版 Milvus (Milvus Lite)，还需要安装：
pip install "pymilvus[milvus_lite]"
```

### 2. 环境变量配置

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

`.env` 文件内容：
```env
DASHSCOPE_API_KEY=sk-your-api-key-here
```

---

## 项目一：Milvus FAQ 检索系统

位于 `milvus_faq/` 目录。使用 FastAPI 提供 RESTful 接口。

### 启动服务

```bash
python milvus_faq/main.py
```
服务默认运行在 `http://0.0.0.0:8000`。

### API 使用示例

#### 1. 查询 (Query)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "如何退货？"}'
```

#### 2. 重建索引 (Reindex)
如果更新了 `data/faq.txt` 数据，可以调用此接口重新构建向量索引：
```bash
curl -X POST "http://localhost:8000/reindex" \
     -H "Content-Type: application/json" \
     -d '{}'
```

---

## 项目二：GraphRAG 多跳问答系统

位于 `graph_rag/` 目录。演示了如何结合 Knowledge Graph 和 RAG 进行复杂推理。

### 运行 Demo

确保你已配置好 Neo4j（如果使用的是远程 Neo4j，请在代码中修改连接配置，或者使用 Docker 启动本地 Neo4j）。

```bash
python graph_rag/main.py
```

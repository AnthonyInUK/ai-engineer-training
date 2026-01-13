# Milvus FAQ Retrieval System

基于 FastAPI + Milvus + LlamaIndex 构建的 FAQ 问答系统，支持语义切片（Semantic Splitting）和向量检索。

## 功能特性

- **语义检索**：使用 `SemanticSplitterNodeParser` 基于语义相似度进行文档切片，而非简单的固定字符长度。
- **向量存储**：使用 Milvus (Lite) 存储向量索引。
- **REST API**：提供标准的 FastAPI 接口进行查询和索引更新。

## 快速开始

### 1. 环境准备

确保已安装 Python 3.11+。

安装依赖：
```bash
pip install -r requirements.txt
# 或者如果使用 uv/poetry
uv pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp ../.env.example .env
```

编辑 `.env` 文件：
```ini
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 启动服务

在 `week03-homework-2` 目录下运行：

```bash
python milvus_faq/main.py
```
或者直接使用 uvicorn：
```bash
uvicorn milvus_faq.main:app --reload --port 8000
```

服务启动后，Milvus Lite 会在当前目录下生成 `milvus_demo.db`。

### 4. API 使用指南

访问 Swagger UI 文档：http://localhost:8000/docs

#### 查询 FAQ
**POST** `/query`

```json
{
  "question": "运费谁来出？"
}
```

#### 重建索引
**POST** `/reindex`

```json
{
  "data_dir": "./data"
}
```

## 项目结构

- `main.py`: FastAPI 应用主入口。
- `data/`: 存放 FAQ 原始文本文件。
- `milvus_demo.db`: Milvus Lite 本地数据库文件。



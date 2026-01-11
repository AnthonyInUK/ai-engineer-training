# Milvus FAQ System Report

## 1. 系统架构
本系统实现了一个基于 Milvus 向量数据库和 LlamaIndex 框架的 FAQ 检索系统。

### 核心组件
- **向量数据库**: Milvus (Standalone)
- **RAG 框架**: LlamaIndex
- **Web 框架**: FastAPI
- **LLM**: Qwen-Plus (via DashScope)
- **Embedding**: text-embedding-v3 (via DashScope)

## 2. 关键技术实现

### 2.1 语义切片
使用 `SemanticSplitterNodeParser` 对文档进行切片。与传统的固定长度切片不同，语义切片通过比较相邻句子的语义相似度（embedding 距离）来决定切分点，能够更好地保持文本的语义完整性。

```python
splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=95, 
    embed_model=embed_model
)
```

### 2.2 向量检索与存储
利用 `MilvusVectorStore` 将切分后的 Document 节点存储到 Milvus 中。设置 `overwrite=True` 实现了简化的全量热更新逻辑。

### 2.3 接口设计
- `POST /query`: 接收自然语言问题，返回最相关的回答。
- `POST /reindex`: 触发知识库的重新索引，支持指定数据目录，实现了知识库的热更新。

## 3. 部署说明

### 前置条件
1. 启动 Milvus 服务 (Docker):
   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
   docker-compose up -d
   ```
2. 配置环境变量 `.env`:
   ```env
   DASHSCOPE_API_KEY=your_key_here
   ```

### 启动服务
```bash
python main.py
```

### API 测试
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "如何退货"}'
```

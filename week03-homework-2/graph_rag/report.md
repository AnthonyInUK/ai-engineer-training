# GraphRAG Multi-hop QA System Report

## 1. 系统架构
本系统融合了 **Document RAG (LlamaIndex)** 和 **Knowledge Graph (Neo4j)**，旨在解决复杂的企业股权多跳问答问题。

### 核心组件
1.  **Document RAG**: 负责检索非结构化文本（如公司业务介绍、历史背景）。
    *   Stack: LlamaIndex + DashScope Embedding + Qwen-Plus
2.  **Knowledge Graph**: 负责存储和推理结构化的股权关系。
    *   Stack: Neo4j + Cypher Query
3.  **Hybrid Orchestrator**: 负责意图识别、分发查询、上下文融合和最终答案生成。

## 2. 关键技术实现

### 2.1 图谱构建 (Graph Construction)
使用 Neo4j 存储企业实体（Company, Person）及其关系（CONTROLS, SHARES）。
```cypher
CREATE (a:Company {name: 'A公司'})-[:CONTROLS {share: 0.60}]->(b:Company {name: 'B公司'})
```

### 2.2 多跳查询逻辑 (Multi-hop Logic)
针对 "谁是实际控制人" 这类问题，单纯的向量检索难以回答，必须依赖图谱的多跳路径查询。
我们实现了 `find_ultimate_beneficiary` 方法，利用 Cypher 的变长路径查询：
```cypher
MATCH path = (root)-[:CONTROLS|SHARES*1..]->(c:Company {name: $name})
RETURN root, length(path)
ORDER BY length(path) DESC
```

### 2.3 RAG 与 KG 的融合
系统采用了 **Late Fusion (后融合)** 策略：
1.  **并行检索**: 同时向 Document Store 发起语义查询，向 Graph Store 发起结构化查询。
2.  **上下文拼接**: 将文档检索到的“公司背景”与图谱检索到的“股权路径”合并到 Prompt 中。
3.  **LLM 综合**: 让 LLM 阅读组合后的上下文，生成包含推理过程的自然语言回答。

## 3. 运行效果示例

**用户问题**: "C公司的实际控制人是谁？"

**推理路径**:
1.  **RAG**: 检索到 C 公司是 B 公司的全资子公司，负责硬件制造。
2.  **KG**: 发现路径 `A公司 -> [CONTROLS] -> B公司 -> [CONTROLS] -> C公司`。
3.  **LLM**: 结合信息，判定 A 公司通过控股 B 公司间接控制 C 公司。

**最终回答**:
"C公司的实际控制人是A公司。
推理依据：
1. 图谱显示 B 公司持有 C 公司 100% 股份。
2. A 公司持有 B 公司 60% 股份，拥有控股权。
3. 因此 A 公司通过 B 公司间接控制 C 公司。"

## 4. 部署说明
1. 启动 Neo4j (Docker)
2. 配置 `.env` (DASHSCOPE_API_KEY)
3. 运行 `python main.py`

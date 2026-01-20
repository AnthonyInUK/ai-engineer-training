# 基于MCP协议的多代理文章自动编写系统

## 项目简介

一个使用MCP协议和多代理协作完成文章写作的系统。四个专业化代理按顺序协作：

```
📚 ResearchAgent → ✍️ WritingAgent → 🔍 ReviewAgent → ✨ PolishingAgent
   (研究资料)        (撰写初稿)       (审核建议)       (润色优化)
```

## 快速开始

### 1. 安装依赖

```bash
cd week05-homework
pip install langchain langchain-openai langchain-mcp-adapters mcp fastmcp rich python-dotenv
```

### 2. 配置API密钥

在 `week05-homework/.env` 文件中设置：

```bash
OPENAI_API_KEY=你的API密钥
```

### 3. 运行系统

```bash
python -m multi-agent.main
```

选择模式2（独立模式），输入主题如"AI Agent"，等待执行完成。

系统会生成 `report.md` 包含完整的文章和执行过程。

## 核心概念

### MCP (Model Context Protocol)

让AI能够调用外部工具的标准化协议。

```python
# MCP服务器（提供工具）
@mcp.tool()
def search_web(query: str) -> str:
    """搜索工具"""
    return results

# AI代理调用工具
result = search_tool("AI Agent")
```

### 多代理系统

多个专业化Agent协作完成任务，每个Agent负责特定步骤：

1. **ResearchAgent**: 使用搜索工具收集资料
2. **WritingAgent**: 基于研究生成文章初稿
3. **ReviewAgent**: 检查质量提供建议
4. **PolishingAgent**: 优化语言和结构

## 文件结构

```
multi-agent/
├── mcp_server.py       # MCP工具服务器
├── agents.py           # 四个代理实现
├── workflow.py         # 协作流程
├── main.py            # 主程序入口
├── retry_handler.py   # 错误重试机制（可选）
└── report.md          # 示例输出
```

## 运行模式

### 模式1：独立模式（推荐）

使用模拟数据，无需启动MCP服务器：

```bash
python -m multi-agent.main
# 选择: 2
```

### 模式2：MCP模式

使用真实MCP协议通信：

**终端1：**
```bash
python3 -m multi-agent.mcp_server
```

**终端2：**
```bash
python3 -m multi-agent.main
# 选择: 1
```

## 扩展功能

### 启用重试机制

在 `main.py` 中使用 `RobustWorkflow`：

```python
from .retry_handler import RobustWorkflow

robust_workflow = RobustWorkflow(workflow)
result = await robust_workflow.execute_with_retry(topic)
```

实现三级重试：
- 一级：同代理重试2次
- 二级：切换备用代理
- 三级：请求用户输入

## 常见问题

**Q: ModuleNotFoundError**
```bash
pip install langchain-mcp-adapters mcp fastmcp
```

**Q: OpenAI API错误**  
检查 `.env` 中的API密钥是否正确

**Q: MCP连接失败**  
使用独立模式（模式2）或先启动 `mcp_server.py`

## 输出示例

查看 [report.md](./report.md) 了解完整输出格式，包括：
- 最终文章（Markdown格式）
- 每个阶段的详细输出
- 完整执行日志

## 技术栈

- **LangChain**: AI应用框架
- **LangGraph**: 多代理编排
- **FastMCP**: MCP服务器实现
- **OpenAI GPT**: 大语言模型
- **Rich**: 终端美化

## 作业要求

- ✅ 四个代理协作完成文章写作
- ✅ 使用MCP协议提供工具
- ✅ 终端实时显示执行过程
- ✅ 生成完整的report.md
- ✅ (可选) 实现三级重试机制

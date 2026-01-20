# 🎓 多代理系统学习教程

## 📚 核心概念详解

### 1. MCP (Model Context Protocol) 是什么？

**简单理解：MCP就像是AI的"工具箱协议"**

想象你要盖房子：
- 你（AI模型）有想法和计划
- 但你需要**工具**：锤子、电钻、锯子
- MCP就是**标准化的工具接口规范**

```
传统方式：每个工具都有不同的使用方法 ❌
MCP方式：所有工具遵循统一的标准 ✅
```

**具体例子：**

```python
# MCP服务器（工具提供方）
@mcp.tool()
def search_web(query: str) -> str:
    """搜索工具"""
    return search_results

# MCP客户端（AI使用方）
client = MultiServerMCPClient({
    "search": {"url": "http://localhost:8000/mcp"}
})
# AI就可以调用这个搜索工具了！
```

**MCP的核心价值：**
1. **标准化**：不同的AI框架可以使用同一个工具服务器
2. **安全性**：工具在服务器端执行，可以控制权限
3. **扩展性**：轻松添加新工具，无需修改AI代码

---

### 2. 多代理系统（Multi-Agent System）

**简单理解：多代理系统就像一个专业团队**

类比编辑部写文章：
```
记者（Research Agent）  → 采访收集素材
编辑（Writing Agent）   → 撰写初稿
审核（Review Agent）    → 检查质量
美编（Polishing Agent） → 美化排版
```

**为什么需要多代理？**

❌ **单一Agent（一人包办）**
```python
agent = ChatAgent()
result = agent.process("写一篇文章")
# 问题：什么都做，什么都不精
```

✅ **多代理系统（专业分工）**
```python
# 每个代理专注一件事
research_result = research_agent.research(topic)
draft = writing_agent.write(research_result)
feedback = review_agent.review(draft)
final = polishing_agent.polish(feedback)
# 优势：专业、高效、质量好
```

---

## 🔍 代码逐行解析

### 示例1：创建MCP工具服务器

```python
from mcp.server.fastmcp import FastMCP

# 1. 创建MCP服务器实例
mcp = FastMCP("ArticleWritingTools")

# 2. 用装饰器注册工具
@mcp.tool()
def search_web(query: str, num_results: int = 5) -> str:
    """
    搜索工具 - AI可以调用这个函数
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
    """
    # 这里是你的搜索逻辑
    # 可以调用真实的搜索API，或返回模拟数据
    return search_results

# 3. 启动服务器
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # 服务器会在 http://localhost:8000/mcp 监听
```

**关键点：**
- `@mcp.tool()` 装饰器让函数变成"AI可调用的工具"
- 函数的docstring会帮助AI理解工具的用途
- 参数类型注解很重要，帮助AI正确传参

---

### 示例2：创建一个专业化代理

```python
class ResearchAgent(BaseAgent):
    """研究代理 - 专门负责收集资料"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            name="ResearchAgent",
            role="研究员",
            llm=llm
        )
        
        # 定义这个代理的专属提示词
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的研究员。
你的任务是收集和整理资料。
请系统性地研究主题的各个方面。"""),
            ("user", "{input}")
        ])
    
    async def research(self, topic: str, search_tool) -> dict:
        """执行研究任务"""
        # 1. 使用工具收集数据
        search_results = search_tool(topic)
        
        # 2. 让LLM整理分析
        prompt = self.prompt_template.format_messages(
            input=f"主题: {topic}\n数据: {search_results}"
        )
        response = await self.llm.ainvoke(prompt)
        
        # 3. 返回结构化结果
        return {
            "topic": topic,
            "report": response.content,
            "status": "success"
        }
```

**关键点：**
- 每个代理有**专门的提示词**（system message）
- 代理封装了**特定的业务逻辑**
- 返回**结构化数据**便于下一个代理使用

---

### 示例3：代理协作工作流

```python
class ArticleWritingWorkflow:
    """工作流 - 协调多个代理按顺序工作"""
    
    async def execute(self, topic: str):
        # 阶段1: 研究
        research_result = await self.research_agent.research(topic)
        # research_result = {
        #     "topic": "AI Agent",
        #     "report": "研究报告内容...",
        #     "status": "success"
        # }
        
        # 阶段2: 写作（使用研究结果）
        writing_result = await self.writing_agent.write(research_result)
        # writing_result = {
        #     "topic": "AI Agent",
        #     "draft": "文章初稿内容...",
        #     "status": "success"
        # }
        
        # 阶段3: 审核（检查初稿）
        review_result = await self.review_agent.review(writing_result)
        # review_result = {
        #     "feedback": "审核意见...",
        #     "original_draft": "初稿...",
        #     "status": "success"
        # }
        
        # 阶段4: 润色（基于审核意见）
        final_result = await self.polishing_agent.polish(review_result)
        
        return final_result
```

**数据流转示意：**
```
topic (str)
    ↓
[ResearchAgent] → {"report": "...", "topic": "..."}
    ↓
[WritingAgent]  → {"draft": "...", "topic": "..."}
    ↓
[ReviewAgent]   → {"feedback": "...", "original_draft": "..."}
    ↓
[PolishingAgent] → {"final_article": "...", "word_count": 1200}
```

---

## 🎯 实战练习

### 练习1：理解MCP

**任务：运行MCP服务器**

```bash
# 终端1：启动服务器
cd week05-homework
python -m multi-agent.mcp_server

# 你会看到：
# 🚀 启动 MCP 服务器...
# 📡 监听地址: http://localhost:8000/mcp
```

**思考：**
1. 服务器提供了哪些工具？（查看 `mcp_server.py`）
2. 如果要添加一个"翻译工具"，怎么做？

<details>
<summary>点击查看答案</summary>

```python
@mcp.tool()
def translate_text(text: str, target_lang: str = "zh") -> str:
    """
    翻译工具
    
    Args:
        text: 要翻译的文本
        target_lang: 目标语言（zh/en/ja等）
    """
    # 这里可以调用翻译API
    return translated_text
```
</details>

---

### 练习2：理解代理

**任务：创建一个新代理**

假设你要添加一个"标题生成代理"（TitleAgent），它的任务是为文章生成吸引人的标题。

**步骤：**

1. **定义代理职责**
   - 输入：文章内容
   - 输出：3个标题候选

2. **编写代码**

```python
class TitleAgent(BaseAgent):
    """标题生成代理"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            name="TitleAgent",
            role="标题创作专家",
            llm=llm
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是标题创作专家。
请为文章生成3个吸引人的标题候选。
标题要简洁有力，吸引读者点击。"""),
            ("user", "{input}")
        ])
    
    async def generate_titles(self, article: str) -> dict:
        prompt = self.prompt_template.format_messages(
            input=f"请为以下文章生成3个标题：\n\n{article}"
        )
        response = await self.llm.ainvoke(prompt)
        
        return {
            "titles": response.content,
            "status": "success"
        }
```

3. **集成到工作流**

在 `workflow.py` 中添加：
```python
# 在 __init__ 中
self.title_agent = TitleAgent(self.llm)

# 在 execute 中（在润色之后）
title_result = await self.title_agent.generate_titles(
    final_result['final_article']
)
```

---

### 练习3：理解工作流

**任务：改变代理执行顺序**

当前顺序：Research → Write → Review → Polish

如果我们想要：Research → Review研究质量 → Write → Polish

该怎么修改？

<details>
<summary>点击查看答案</summary>

在 `workflow.py` 的 `execute` 方法中：

```python
async def execute(self, topic):
    # 1. 研究
    research_result = await self.research_agent.research(topic)
    
    # 2. 审核研究质量（新增）
    research_review = await self.review_agent.review({
        "draft": research_result['report'],
        "topic": topic
    })
    
    # 3. 基于审核后的研究写作
    writing_result = await self.writing_agent.write({
        "report": research_review['feedback'],  # 使用审核后的内容
        "topic": topic
    })
    
    # 4. 润色
    final_result = await self.polishing_agent.polish(writing_result)
    
    return final_result
```
</details>

---

## 🚀 运行完整系统

### 方式一：独立模式（推荐学习）

```bash
cd week05-homework
python -m multi-agent.main

# 选择：2（独立模式）
# 输入主题：AI Agent
# 等待系统执行...
# 查看生成的 report.md
```

### 方式二：完整MCP模式

**终端1：**
```bash
python -m multi-agent.mcp_server
```

**终端2：**
```bash
python -m multi-agent.main
# 选择：1（MCP模式）
```

---

## 📖 深入理解

### Q1: 为什么要用 async/await？

```python
# ❌ 同步方式（慢）
result1 = agent1.work()  # 等10秒
result2 = agent2.work()  # 等10秒
# 总耗时：20秒

# ✅ 异步方式（快）
result1 = await agent1.work()  # 等10秒
result2 = await agent2.work()  # 等10秒
# 总耗时：10秒（并行执行）
```

### Q2: 代理之间如何通信？

通过**结构化的字典数据**：

```python
# ResearchAgent 输出
{
    "topic": "AI Agent",
    "report": "研究内容...",
    "status": "success"
}

# WritingAgent 接收并使用
def write(self, research_data: dict):
    report = research_data['report']  # 读取研究结果
    topic = research_data['topic']    # 读取主题
    # 基于这些数据生成文章
```

**关键：** 定义清晰的**数据契约**（data contract）

### Q3: 如何让代理更聪明？

**1. 优化提示词（Prompt Engineering）**

```python
# ❌ 模糊的提示词
"system": "你是一个写作助手"

# ✅ 清晰的提示词
"system": """你是一位专业的技术文章撰稿人。
你的任务是：
1. 基于研究资料撰写文章
2. 使用Markdown格式
3. 包含引言、主体、总结
4. 长度800-1200字
5. 语言专业但易懂"""
```

**2. 提供示例（Few-shot Learning）**

```python
"system": """你是研究员。

示例输入：
主题: 人工智能

示例输出：
# 研究报告
## 核心概念
人工智能（AI）是...
## 关键技术
...

现在请处理用户的输入："""
```

**3. 使用工具（Tool Use）**

```python
# 让代理可以调用工具
tools = [search_tool, calculator_tool, database_tool]
agent = create_agent(llm, tools=tools)
```

---

## 🎨 扩展思路

### 1. 添加并行执行

```python
# 多个研究员同时工作
results = await asyncio.gather(
    research_agent1.research("技术方面"),
    research_agent2.research("应用方面"),
    research_agent3.research("市场方面")
)
```

### 2. 添加人工反馈

```python
# 在审核后，让用户决定是否继续
review_result = await review_agent.review(draft)
print(review_result['feedback'])
user_choice = input("是否继续？(y/n)")
if user_choice == 'y':
    final = await polishing_agent.polish(review_result)
```

### 3. 添加重试机制

```python
async def execute_with_retry(self, topic, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await self.execute(topic)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"失败，重试 {attempt+1}/{max_retries}")
                continue
            raise
```

---

## 📚 学习资源

### 官方文档
- [LangChain文档](https://python.langchain.com/)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [MCP协议规范](https://modelcontextprotocol.io/)

### 推荐阅读
- ReAct: Synergizing Reasoning and Acting in Language Models
- Communicative Agents for Software Development

### 实践项目
1. 智能客服系统（多Agent协作）
2. 代码审查助手（代码分析Agent）
3. 数据分析报告生成器（数据Agent + 可视化Agent + 写作Agent）

---

## 🎯 总结

通过本项目，你应该掌握：

✅ **MCP协议**
- MCP是AI工具调用的标准化协议
- 服务器端注册工具，客户端调用工具
- 类似于API，但专为AI设计

✅ **多代理系统**
- 每个代理专注一个任务
- 通过结构化数据通信
- 工作流协调代理执行顺序

✅ **实战技能**
- 创建MCP工具服务器
- 设计专业化代理
- 构建代理协作流程
- 优化提示词提升效果

**下一步：**
1. 运行完整系统，理解整个流程
2. 尝试添加新代理或新工具
3. 思考如何应用到实际项目中

祝学习愉快！🎉


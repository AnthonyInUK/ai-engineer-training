"""
多代理系统 - 四个专业化代理的实现

每个代理都是一个专门的AI，负责文章创作流程中的特定任务：
- ResearchAgent: 研究收集资料
- WritingAgent: 撰写文章初稿  
- ReviewAgent: 审核内容质量
- PolishingAgent: 润色优化文章
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class BaseAgent:
    """代理基类 - 所有代理的共同功能"""

    def __init__(self, name: str, role: str, llm: ChatOpenAI):
        self.name = name
        self.role = role
        self.llm = llm
        self.execution_log = []

    def log(self, message: str, level: str = "info"):
        """记录执行日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "level": level,
            "message": message
        }
        self.execution_log.append(log_entry)

        # 终端显示
        if level == "info":
            console.print(
                f"[cyan]ℹ️  [{timestamp}] {self.name}:[/cyan] {message}")
        elif level == "success":
            console.print(
                f"[green]✅ [{timestamp}] {self.name}:[/green] {message}")
        elif level == "error":
            console.print(f"[red]❌ [{timestamp}] {self.name}:[/red] {message}")
        elif level == "warning":
            console.print(
                f"[yellow]⚠️  [{timestamp}] {self.name}:[/yellow] {message}")

    def get_logs(self) -> List[Dict[str, Any]]:
        """获取执行日志"""
        return self.execution_log


class ResearchAgent(BaseAgent):
    """研究代理 - 负责收集和整理资料"""

    def __init__(self, llm: ChatOpenAI, tools: list = None):
        super().__init__(
            name="ResearchAgent",
            role="研究员",
            llm=llm
        )
        self.tools = tools or []
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的研究员。你的任务是：
1. 根据用户的主题，确定需要研究的关键问题
2. 使用搜索工具收集相关资料
3. 整理和总结研究结果，形成结构化的资料

请系统性地收集信息，确保覆盖主题的各个方面。
输出格式应该是清晰的研究报告，包含：
- 研究主题
- 关键发现（分点列出）
- 重要资料来源
"""),
            ("user", "{input}")
        ])

    async def research(self, topic: str, search_tool=None) -> Dict[str, Any]:
        """
        执行研究任务

        Args:
            topic: 研究主题
            search_tool: 搜索工具函数

        Returns:
            研究结果字典
        """
        self.log(f"开始研究主题: {topic}")

        try:
            # 使用搜索工具收集资料
            search_results = ""
            if search_tool:
                self.log("正在搜索相关资料...")
                # 判断是工具对象还是普通函数
                if hasattr(search_tool, 'ainvoke'):
                    # LangChain 工具对象，使用异步 ainvoke
                    search_results = await search_tool.ainvoke({"query": topic})
                elif hasattr(search_tool, 'invoke'):
                    # 同步 invoke
                    search_results = search_tool.invoke({"query": topic})
                elif callable(search_tool):
                    # 普通函数，直接调用
                    search_results = search_tool(topic)
                else:
                    search_results = str(search_tool)
                self.log(f"搜索完成，找到相关资料")

            # 使用 LLM 整理研究结果
            self.log("正在整理和分析研究结果...")
            prompt = self.prompt_template.format_messages(
                input=f"""主题: {topic}

搜索结果:
{search_results}

请基于以上搜索结果，整理出一份结构化的研究报告。包括：
1. 主题概述
2. 核心概念和定义
3. 关键技术要点
4. 应用场景
5. 最新发展趋势
"""
            )

            response = await self.llm.ainvoke(prompt)
            research_report = response.content

            # 显示研究结果
            console.print(Panel(
                Markdown(research_report),
                title=f"[bold cyan]📚 {self.name} - 研究报告[/bold cyan]",
                border_style="cyan"
            ))

            self.log("研究任务完成", "success")

            return {
                "topic": topic,
                "report": research_report,
                "sources": search_results,
                "status": "success"
            }

        except Exception as e:
            self.log(f"研究过程中出错: {str(e)}", "error")
            return {
                "topic": topic,
                "status": "failed",
                "error": str(e)
            }


class WritingAgent(BaseAgent):
    """撰写代理 - 负责生成文章初稿"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            name="WritingAgent",
            role="撰稿人",
            llm=llm
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的技术文章撰稿人。你的任务是：
1. 基于研究资料，撰写一篇结构完整、内容丰富的文章
2. 文章应该包含清晰的标题、引言、主体内容和总结
3. 使用Markdown格式，确保排版美观
4. 语言流畅，逻辑清晰，适合技术读者阅读

文章结构建议：
- # 主标题
- ## 引言
- ## 核心内容（可分多个小节）
- ## 应用场景
- ## 总结
"""),
            ("user", "{input}")
        ])

    async def write(self, research_data: Dict[str, Any], style: str = "technical") -> Dict[str, Any]:
        """
        撰写文章初稿

        Args:
            research_data: 研究资料
            style: 写作风格

        Returns:
            文章初稿
        """
        self.log(f"开始撰写文章: {research_data.get('topic', '未知主题')}")

        try:
            research_report = research_data.get('report', '')
            topic = research_data.get('topic', '')

            self.log("正在生成文章初稿...")
            prompt = self.prompt_template.format_messages(
                input=f"""请基于以下研究资料，撰写一篇关于"{topic}"的技术文章。

研究资料:
{research_report}

要求：
1. 文章长度严格控制在600-800字（重要！）
2. 使用Markdown格式
3. 包含完整的结构（标题、引言、主体、总结）
4. 语言简洁精炼，直击要点
5. 内容聚焦核心概念，避免冗长描述
"""
            )

            response = await self.llm.ainvoke(prompt)
            article_draft = response.content

            # 显示初稿
            console.print(Panel(
                Markdown(article_draft),
                title=f"[bold green]✍️  {self.name} - 文章初稿[/bold green]",
                border_style="green"
            ))

            self.log("文章初稿完成", "success")

            return {
                "topic": topic,
                "draft": article_draft,
                "word_count": len(article_draft),
                "status": "success"
            }

        except Exception as e:
            self.log(f"撰写过程中出错: {str(e)}", "error")
            return {
                "status": "failed",
                "error": str(e)
            }


class ReviewAgent(BaseAgent):
    """审核代理 - 负责检查内容质量"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            name="ReviewAgent",
            role="审核编辑",
            llm=llm
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位严谨的审核编辑。你的任务是：
1. 检查文章的逻辑性和连贯性
2. 发现事实错误、表述不清或逻辑漏洞
3. 评估文章结构是否合理
4. 提供具体的修改建议

请给出客观、建设性的反馈，包括：
- 优点（做得好的地方）
- 问题（需要改进的地方）
- 具体修改建议
"""),
            ("user", "{input}")
        ])

    async def review(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        审核文章

        Args:
            article_data: 文章数据

        Returns:
            审核意见
        """
        self.log("开始审核文章")

        try:
            draft = article_data.get('draft', '')
            topic = article_data.get('topic', '')

            self.log("正在检查文章质量...")
            prompt = self.prompt_template.format_messages(
                input=f"""请审核以下关于"{topic}"的文章草稿：

{draft}

请从以下几个方面进行评估：
1. 内容准确性和完整性
2. 逻辑结构和连贯性
3. 语言表达和可读性
4. 标题和章节划分
5. 是否需要补充或删减内容

请给出详细的审核意见和具体的修改建议。
"""
            )

            response = await self.llm.ainvoke(prompt)
            review_feedback = response.content

            # 显示审核意见
            console.print(Panel(
                Markdown(review_feedback),
                title=f"[bold yellow]🔍 {self.name} - 审核意见[/bold yellow]",
                border_style="yellow"
            ))

            self.log("审核完成", "success")

            return {
                "topic": topic,
                "feedback": review_feedback,
                "original_draft": draft,
                "status": "success"
            }

        except Exception as e:
            self.log(f"审核过程中出错: {str(e)}", "error")
            return {
                "status": "failed",
                "error": str(e)
            }


class PolishingAgent(BaseAgent):
    """润色代理 - 负责优化文章"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            name="PolishingAgent",
            role="文字润色师",
            llm=llm
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的文字润色师。你的任务是：
1. 根据审核意见优化文章
2. 提升语言表达的流畅性和专业性
3. 确保风格一致性
4. 优化文章结构和排版

请生成最终版本的文章，确保：
- 语言精炼优美
- 逻辑严密
- 格式规范
- 易于阅读
"""),
            ("user", "{input}")
        ])

    async def polish(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        润色文章

        Args:
            review_data: 审核数据

        Returns:
            最终文章
        """
        self.log("开始润色文章")

        try:
            draft = review_data.get('original_draft', '')
            feedback = review_data.get('feedback', '')
            topic = review_data.get('topic', '')

            self.log("正在优化文章内容和表达...")
            prompt = self.prompt_template.format_messages(
                input=f"""请基于以下审核意见，对文章进行润色和优化：

原始草稿：
{draft}

审核意见：
{feedback}

请生成最终版本的文章，要求：
1. 采纳审核意见中的合理建议
2. 优化语言表达，使其更加流畅专业
3. 确保Markdown格式规范
4. 保持文章的完整性和连贯性
"""
            )

            response = await self.llm.ainvoke(prompt)
            final_article = response.content

            # 显示最终文章
            console.print(Panel(
                Markdown(final_article),
                title=f"[bold magenta]✨ {self.name} - 最终文章[/bold magenta]",
                border_style="magenta"
            ))

            self.log("润色完成，文章已完成", "success")

            return {
                "topic": topic,
                "final_article": final_article,
                "word_count": len(final_article),
                "status": "success"
            }

        except Exception as e:
            self.log(f"润色过程中出错: {str(e)}", "error")
            return {
                "status": "failed",
                "error": str(e)
            }

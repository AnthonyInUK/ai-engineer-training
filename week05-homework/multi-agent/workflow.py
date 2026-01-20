"""
多代理协作工作流

这个模块定义了四个代理如何协作完成文章写作任务：
Research → Writing → Review → Polishing
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from .agents import ResearchAgent, WritingAgent, ReviewAgent, PolishingAgent
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
import os

console = Console()


class ArticleWritingWorkflow:
    """文章写作工作流 - 协调四个代理的协作"""

    def __init__(self, openai_api_key: str = None):
        """
        初始化工作流

        Args:
            openai_api_key: OpenAI API密钥
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要提供 OPENAI_API_KEY")

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # 使用 GPT-4o-mini 性价比高
            temperature=0.7,
            api_key=api_key
        )

        # 初始化四个代理
        self.research_agent = None
        self.writing_agent = WritingAgent(self.llm)
        self.review_agent = ReviewAgent(self.llm)
        self.polishing_agent = PolishingAgent(self.llm)

        # 工作流状态
        self.workflow_data = {}
        self.execution_log = []

    def set_research_agent(self, search_tool=None):
        """设置研究代理（需要在MCP客户端连接后调用）"""
        self.research_agent = ResearchAgent(
            self.llm, tools=[search_tool] if search_tool else [])

    async def execute(self, topic: str, search_tool=None) -> Dict[str, Any]:
        """
        执行完整的文章写作流程

        Args:
            topic: 文章主题
            search_tool: 搜索工具（来自MCP服务器）

        Returns:
            完整的执行结果
        """
        console.print("\n")
        console.print(Panel.fit(
            f"[bold cyan]🚀 开始多代理协作写作任务[/bold cyan]\n"
            f"主题: {topic}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="cyan"
        ))

        start_time = datetime.now()

        try:
            # 确保研究代理已初始化
            if not self.research_agent:
                self.set_research_agent(search_tool)

            # 阶段1: 研究
            console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
            console.print("[bold cyan]阶段 1/4: 资料研究[/bold cyan]")
            console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")

            research_result = await self.research_agent.research(topic, search_tool)
            if research_result['status'] != 'success':
                raise Exception(
                    f"研究阶段失败: {research_result.get('error', '未知错误')}")

            self.workflow_data['research'] = research_result

            # 阶段2: 撰写
            console.print("\n[bold green]" + "="*60 + "[/bold green]")
            console.print("[bold green]阶段 2/4: 撰写初稿[/bold green]")
            console.print("[bold green]" + "="*60 + "[/bold green]\n")

            writing_result = await self.writing_agent.write(research_result)
            if writing_result['status'] != 'success':
                raise Exception(
                    f"撰写阶段失败: {writing_result.get('error', '未知错误')}")

            self.workflow_data['writing'] = writing_result

            # 阶段3: 审核
            console.print("\n[bold yellow]" + "="*60 + "[/bold yellow]")
            console.print("[bold yellow]阶段 3/4: 内容审核[/bold yellow]")
            console.print("[bold yellow]" + "="*60 + "[/bold yellow]\n")

            review_result = await self.review_agent.review(writing_result)
            if review_result['status'] != 'success':
                raise Exception(
                    f"审核阶段失败: {review_result.get('error', '未知错误')}")

            self.workflow_data['review'] = review_result

            # 阶段4: 润色
            console.print("\n[bold magenta]" + "="*60 + "[/bold magenta]")
            console.print("[bold magenta]阶段 4/4: 文章润色[/bold magenta]")
            console.print("[bold magenta]" + "="*60 + "[/bold magenta]\n")

            polishing_result = await self.polishing_agent.polish(review_result)
            if polishing_result['status'] != 'success':
                raise Exception(
                    f"润色阶段失败: {polishing_result.get('error', '未知错误')}")

            self.workflow_data['polishing'] = polishing_result

            # 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            console.print("\n")
            console.print(Panel.fit(
                f"[bold green]✅ 文章创作完成！[/bold green]\n"
                f"总耗时: {duration:.2f} 秒\n"
                f"最终字数: {polishing_result['word_count']} 字",
                border_style="green"
            ))

            # 收集所有日志
            all_logs = []
            all_logs.extend(self.research_agent.get_logs())
            all_logs.extend(self.writing_agent.get_logs())
            all_logs.extend(self.review_agent.get_logs())
            all_logs.extend(self.polishing_agent.get_logs())

            return {
                "status": "success",
                "topic": topic,
                "final_article": polishing_result['final_article'],
                "workflow_data": self.workflow_data,
                "execution_log": all_logs,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

        except Exception as e:
            console.print(f"\n[bold red]❌ 工作流执行失败: {str(e)}[/bold red]\n")
            return {
                "status": "failed",
                "error": str(e),
                "workflow_data": self.workflow_data
            }

    def generate_report(self, result: Dict[str, Any]) -> str:
        """
        生成执行报告（Markdown格式）

        Args:
            result: 执行结果

        Returns:
            Markdown格式的报告
        """
        if result['status'] != 'success':
            return f"# 执行失败\n\n错误信息: {result.get('error', '未知错误')}"

        report = f"""# 多代理文章写作系统 - 执行报告

## 基本信息

- **主题**: {result['topic']}
- **开始时间**: {result['start_time']}
- **结束时间**: {result['end_time']}
- **总耗时**: {result['duration']:.2f} 秒
- **执行状态**: ✅ 成功

---

## 最终文章

{result['final_article']}

---

## 执行过程详情

### 阶段 1: 资料研究 (ResearchAgent)

**研究报告:**

{result['workflow_data']['research']['report']}

**搜索来源:**

{result['workflow_data']['research']['sources']}

---

### 阶段 2: 撰写初稿 (WritingAgent)

**初稿内容:**

{result['workflow_data']['writing']['draft']}

**字数统计:** {result['workflow_data']['writing']['word_count']} 字

---

### 阶段 3: 内容审核 (ReviewAgent)

**审核反馈:**

{result['workflow_data']['review']['feedback']}

---

### 阶段 4: 文章润色 (PolishingAgent)

**最终字数:** {result['workflow_data']['polishing']['word_count']} 字

---

## 执行日志

| 时间 | 代理 | 级别 | 消息 |
|------|------|------|------|
"""

        # 添加日志表格
        for log in result['execution_log']:
            report += f"| {log['timestamp']} | {log['agent']} | {log['level']} | {log['message']} |\n"

        report += """
---

## 系统说明

本报告由多代理文章写作系统自动生成。系统包含四个专业化代理：

1. **ResearchAgent (研究代理)**: 使用MCP工具搜索和收集相关资料
2. **WritingAgent (撰写代理)**: 基于研究结果生成文章初稿
3. **ReviewAgent (审核代理)**: 检查内容质量并提供改进建议
4. **PolishingAgent (润色代理)**: 优化语言表达和文章结构

代理间通过结构化数据进行通信，确保协作流程的顺畅执行。
"""

        return report

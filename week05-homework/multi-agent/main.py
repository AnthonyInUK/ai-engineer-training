"""
基于MCP协议的多代理文章自动编写系统 - 主程序

这个系统演示了如何使用MCP协议和多代理协作来完成复杂任务。

运行方式:
1. 启动MCP服务器: python -m multi-agent.mcp_server
2. 运行主程序: python -m multi-agent.main
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# 加载环境变量 - 明确指定.env文件路径
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

console = Console()


async def main():
    """主程序入口"""

    # 显示欢迎界面
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 基于MCP协议的多代理文章自动编写系统[/bold cyan]\n\n"
        "本系统包含四个专业化代理:\n"
        "  📚 ResearchAgent   - 研究收集资料\n"
        "  ✍️  WritingAgent    - 撰写文章初稿\n"
        "  🔍 ReviewAgent     - 审核内容质量\n"
        "  ✨ PolishingAgent  - 润色优化文章\n\n"
        "[dim]提示: 输入 'quit' 退出系统[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]❌ 错误: 未找到 OPENAI_API_KEY[/bold red]")
        console.print("请在 .env 文件中设置 OPENAI_API_KEY")
        return

    # 询问运行模式
    console.print("\n[bold]请选择运行模式:[/bold]")
    console.print("1. 使用MCP服务器（需要先启动 mcp_server.py）")
    console.print("2. 独立模式（不使用MCP，使用模拟数据）")

    mode = Prompt.ask("请选择", choices=["1", "2"], default="2")

    if mode == "1":
        await run_with_mcp()
    else:
        await run_standalone()


async def run_with_mcp():
    """使用MCP服务器运行"""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from .workflow import ArticleWritingWorkflow

    console.print("\n[cyan]正在连接MCP服务器...[/cyan]")

    # 初始化MCP客户端
    client = MultiServerMCPClient({
        "article_tools": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })

    try:
        async with client.session("article_tools") as session:
            console.print("[green]✅ MCP服务器连接成功[/green]\n")

            # 加载MCP工具
            tools = await load_mcp_tools(session)
            search_tool = None
            for tool in tools:
                if "search" in tool.name.lower():
                    search_tool = tool
                    break

            # 初始化工作流
            workflow = ArticleWritingWorkflow()

            # 交互式循环
            while True:
                topic = Prompt.ask("\n[bold cyan]请输入文章主题[/bold cyan]")

                if topic.lower() in ['quit', 'exit', '退出', 'q']:
                    console.print("\n[yellow]👋 再见！[/yellow]\n")
                    break

                # 执行工作流
                result = await workflow.execute(topic, search_tool)

                if result['status'] == 'success':
                    # 生成报告
                    report = workflow.generate_report(result)

                    # 保存报告
                    report_path = Path(__file__).parent / "report.md"
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)

                    console.print(f"\n[green]✅ 报告已保存到: {report_path}[/green]")
                else:
                    console.print(
                        f"\n[red]❌ 执行失败: {result.get('error')}[/red]")

    except Exception as e:
        console.print(f"[red]❌ MCP连接失败: {str(e)}[/red]")
        console.print("[yellow]提示: 请先启动MCP服务器[/yellow]")
        console.print("[dim]命令: python -m multi-agent.mcp_server[/dim]")


async def run_standalone():
    """独立模式运行（不使用MCP）"""
    from .workflow import ArticleWritingWorkflow

    console.print("\n[cyan]独立模式启动...[/cyan]\n")

    # 模拟搜索工具
    def mock_search_tool(query: str) -> str:
        """模拟搜索工具"""
        mock_data = {
            "AI Agent": """
搜索结果:
1. AI Agent是一种智能体，能够感知环境、做出决策并采取行动
2. 典型的AI Agent包含感知、推理、规划和执行模块
3. ReAct框架是AI Agent的重要实现方式
4. 多Agent系统可以通过协作完成复杂任务
5. LangGraph、LangChain等框架支持构建Agent应用
            """,
            "LangGraph": """
搜索结果:
1. LangGraph是用于构建有状态多参与者应用的框架
2. 支持循环、持久化和人工介入等高级功能
3. 基于图的结构定义Agent工作流
4. 提供检查点机制实现状态管理
5. 与LangChain生态系统无缝集成
            """
        }

        # 返回相关搜索结果
        for key in mock_data:
            if key.lower() in query.lower():
                return mock_data[key]

        return f"关于 {query} 的搜索结果：这是一个技术主题，包含相关的概念、应用和最新发展。"

    # 初始化工作流
    workflow = ArticleWritingWorkflow()
    workflow.set_research_agent(mock_search_tool)

    # 交互式循环
    while True:
        topic = Prompt.ask("\n[bold cyan]请输入文章主题[/bold cyan]")

        if topic.lower() in ['quit', 'exit', '退出', 'q']:
            console.print("\n[yellow]👋 再见！[/yellow]\n")
            break

        # 执行工作流
        result = await workflow.execute(topic, mock_search_tool)

        if result['status'] == 'success':
            # 生成报告
            report = workflow.generate_report(result)

            # 保存报告
            report_path = Path(__file__).parent / "report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            console.print(f"\n[green]✅ 报告已保存到: {report_path}[/green]")

            # 询问是否继续
            continue_choice = Prompt.ask(
                "\n是否继续写作另一篇文章？",
                choices=["y", "n"],
                default="n"
            )
            if continue_choice.lower() == 'n':
                console.print("\n[yellow]👋 再见！[/yellow]\n")
                break
        else:
            console.print(f"\n[red]❌ 执行失败: {result.get('error')}[/red]")


if __name__ == "__main__":
    asyncio.run(main())

"""
错误处理和重试机制

实现三级重试策略：
1. 一级：相同代理重新执行（最多2次）
2. 二级：切换至备用代理执行
3. 三级：向用户请求补充信息
"""

from typing import Dict, Any, Callable, Optional
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()


class RetryHandler:
    """重试处理器 - 管理代理执行失败时的重试逻辑"""

    def __init__(self):
        self.retry_log = []

    async def execute_with_retry(
        self,
        agent_func: Callable,
        agent_name: str,
        *args,
        max_retries: int = 2,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带重试机制的代理执行

        Args:
            agent_func: 代理的执行函数
            agent_name: 代理名称
            max_retries: 最大重试次数
            fallback_func: 备用代理函数（二级重试）
            *args, **kwargs: 传递给代理函数的参数

        Returns:
            执行结果
        """

        # 一级重试：相同代理重新执行
        for attempt in range(max_retries + 1):
            try:
                console.print(
                    f"[cyan]执行 {agent_name}（尝试 {attempt + 1}/{max_retries + 1}）[/cyan]")

                result = await agent_func(*args, **kwargs)

                # 检查结果状态
                if result.get('status') == 'success':
                    if attempt > 0:
                        self._log_retry(
                            agent_name,
                            level="success",
                            message=f"重试成功（第 {attempt + 1} 次尝试）"
                        )
                    return result
                else:
                    raise Exception(f"代理返回失败状态: {result.get('error', '未知错误')}")

            except Exception as e:
                self._log_retry(
                    agent_name,
                    level="error",
                    message=f"执行失败: {str(e)}",
                    attempt=attempt + 1
                )

                # 如果还有重试次数，继续重试
                if attempt < max_retries:
                    console.print(f"[yellow]⚠️  执行失败，准备重试...[/yellow]")
                    continue

                # 一级重试用尽，尝试二级重试
                console.print(
                    f"[red]❌ {agent_name} 一级重试失败（{max_retries + 1}次）[/red]")
                break

        # 二级重试：切换至备用代理
        if fallback_func:
            console.print(f"\n[yellow]🔄 启动二级重试：切换至备用代理[/yellow]")
            try:
                result = await fallback_func(*args, **kwargs)

                if result.get('status') == 'success':
                    self._log_retry(
                        f"{agent_name}_fallback",
                        level="success",
                        message="备用代理执行成功"
                    )
                    return result

            except Exception as e:
                self._log_retry(
                    f"{agent_name}_fallback",
                    level="error",
                    message=f"备用代理也失败: {str(e)}"
                )
                console.print(f"[red]❌ 备用代理执行失败[/red]")

        # 三级重试：请求用户输入
        console.print(f"\n[bold yellow]🙋 启动三级重试：需要用户帮助[/bold yellow]")
        return await self._request_user_help(agent_name, *args, **kwargs)

    async def _request_user_help(
        self,
        agent_name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        三级重试：向用户请求补充信息

        Args:
            agent_name: 代理名称

        Returns:
            用户提供的信息或跳过指令
        """
        console.print(Panel(
            f"[bold yellow]{agent_name} 执行遇到困难[/bold yellow]\n\n"
            f"可能的原因：\n"
            f"1. 任务过于复杂或模糊\n"
            f"2. 缺少必要信息\n"
            f"3. 网络或API问题\n\n"
            f"请选择操作：",
            border_style="yellow"
        ))

        choice = Prompt.ask(
            "请选择",
            choices=["1", "2", "3"],
            default="1"
        )

        console.print("1. 提供补充信息")
        console.print("2. 跳过此步骤")
        console.print("3. 终止整个流程")

        if choice == "1":
            # 用户提供补充信息
            user_input = Prompt.ask("请提供补充信息")

            self._log_retry(
                agent_name,
                level="info",
                message=f"用户提供补充信息: {user_input[:50]}..."
            )

            return {
                "status": "success",
                "user_provided": True,
                "content": user_input,
                "message": "使用用户提供的信息"
            }

        elif choice == "2":
            # 跳过此步骤
            self._log_retry(
                agent_name,
                level="warning",
                message="用户选择跳过此步骤"
            )

            return {
                "status": "skipped",
                "message": f"{agent_name} 步骤已跳过"
            }

        else:
            # 终止流程
            self._log_retry(
                agent_name,
                level="error",
                message="用户选择终止流程"
            )

            return {
                "status": "terminated",
                "message": "用户终止了执行流程"
            }

    def _log_retry(
        self,
        agent_name: str,
        level: str,
        message: str,
        attempt: Optional[int] = None
    ):
        """记录重试日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "level": level,
            "message": message,
            "attempt": attempt
        }
        self.retry_log.append(log_entry)

        # 终端显示
        emoji_map = {
            "info": "ℹ️ ",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️ "
        }
        emoji = emoji_map.get(level, "📝")

        color_map = {
            "info": "cyan",
            "success": "green",
            "error": "red",
            "warning": "yellow"
        }
        color = color_map.get(level, "white")

        console.print(
            f"[{color}]{emoji} [重试日志] {agent_name}: {message}[/{color}]")

    def get_retry_log(self) -> list:
        """获取重试日志"""
        return self.retry_log

    def generate_retry_report(self) -> str:
        """生成重试报告"""
        if not self.retry_log:
            return "## 异常处理日志\n\n未发生异常，所有代理执行顺利。\n"

        report = "## 异常处理日志\n\n"
        report += "本次执行过程中发生了以下异常和重试：\n\n"
        report += "| 时间 | 代理 | 级别 | 尝试次数 | 消息 |\n"
        report += "|------|------|------|----------|------|\n"

        for log in self.retry_log:
            attempt_str = f"第{log.get('attempt', '-')}次" if log.get(
                'attempt') else "-"
            report += (
                f"| {log['timestamp']} "
                f"| {log['agent']} "
                f"| {log['level']} "
                f"| {attempt_str} "
                f"| {log['message']} |\n"
            )

        return report


class RobustWorkflow:
    """
    增强版工作流 - 集成重试机制

    使用示例：
        workflow = RobustWorkflow(api_key)
        result = await workflow.execute_with_retry(topic)
    """

    def __init__(self, workflow):
        """
        Args:
            workflow: 原始工作流对象
        """
        self.workflow = workflow
        self.retry_handler = RetryHandler()

    async def execute_with_retry(self, topic: str, search_tool=None) -> Dict[str, Any]:
        """
        带重试机制的工作流执行

        Args:
            topic: 文章主题
            search_tool: 搜索工具

        Returns:
            执行结果（包含重试日志）
        """
        console.print("\n[bold cyan]🛡️  启动增强版工作流（带重试机制）[/bold cyan]\n")

        workflow_data = {}

        try:
            # 确保研究代理已初始化
            if not self.workflow.research_agent:
                self.workflow.set_research_agent(search_tool)

            # 阶段1: 研究（带重试）
            research_result = await self.retry_handler.execute_with_retry(
                self.workflow.research_agent.research,
                "ResearchAgent",
                topic,
                search_tool,
                max_retries=2
            )

            if research_result.get('status') == 'terminated':
                return self._build_result('terminated', workflow_data)

            workflow_data['research'] = research_result

            # 阶段2: 撰写（带重试）
            writing_result = await self.retry_handler.execute_with_retry(
                self.workflow.writing_agent.write,
                "WritingAgent",
                research_result,
                max_retries=2
            )

            if writing_result.get('status') == 'terminated':
                return self._build_result('terminated', workflow_data)

            workflow_data['writing'] = writing_result

            # 阶段3: 审核（带重试）
            review_result = await self.retry_handler.execute_with_retry(
                self.workflow.review_agent.review,
                "ReviewAgent",
                writing_result,
                max_retries=2
            )

            if review_result.get('status') == 'terminated':
                return self._build_result('terminated', workflow_data)

            workflow_data['review'] = review_result

            # 阶段4: 润色（带重试）
            polishing_result = await self.retry_handler.execute_with_retry(
                self.workflow.polishing_agent.polish,
                "PolishingAgent",
                review_result,
                max_retries=2
            )

            if polishing_result.get('status') == 'terminated':
                return self._build_result('terminated', workflow_data)

            workflow_data['polishing'] = polishing_result

            # 生成完整结果
            result = self._build_result('success', workflow_data)
            result['final_article'] = polishing_result.get('final_article', '')

            return result

        except Exception as e:
            console.print(f"\n[bold red]❌ 工作流执行失败: {str(e)}[/bold red]\n")
            return self._build_result('failed', workflow_data, error=str(e))

    def _build_result(
        self,
        status: str,
        workflow_data: dict,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建结果字典"""
        result = {
            "status": status,
            "workflow_data": workflow_data,
            "retry_log": self.retry_handler.get_retry_log(),
            "retry_report": self.retry_handler.generate_retry_report()
        }

        if error:
            result['error'] = error

        return result

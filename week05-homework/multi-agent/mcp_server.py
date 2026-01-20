"""
MCP 服务器 - 为多代理系统提供工具
这个服务器暴露了搜索、数据检索等工具供代理使用
"""
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# 创建 MCP 服务器实例
mcp = FastMCP("ArticleWritingTools")


@mcp.tool()
def search_web(query: str, num_results: int = 5) -> str:
    """
    模拟网络搜索功能（实际项目中可以接入真实的搜索API）

    Args:
        query: 搜索关键词
        num_results: 返回结果数量

    Returns:
        搜索结果的JSON字符串
    """
    # 这里是模拟数据，实际项目中可以接入 Tavily、Google Search API 等
    mock_results = {
        "AI Agent": [
            {
                "title": "什么是AI Agent？",
                "snippet": "AI Agent是一种能够感知环境、做出决策并采取行动的智能体。它可以自主完成任务，无需人工干预。",
                "url": "https://example.com/ai-agent-intro"
            },
            {
                "title": "AI Agent的应用场景",
                "snippet": "AI Agent广泛应用于客户服务、智能助手、自动化流程等领域，能够显著提升工作效率。",
                "url": "https://example.com/ai-agent-applications"
            },
            {
                "title": "AI Agent的技术架构",
                "snippet": "典型的AI Agent包含感知模块、决策模块和执行模块，通过大语言模型进行推理和规划。",
                "url": "https://example.com/ai-agent-architecture"
            },
            {
                "title": "ReAct框架：AI Agent的核心思想",
                "snippet": "ReAct框架结合了推理(Reasoning)和行动(Acting)，让AI Agent能够更好地解决复杂问题。",
                "url": "https://example.com/react-framework"
            },
            {
                "title": "多Agent协作系统",
                "snippet": "多个专业化的Agent可以协作完成复杂任务，每个Agent负责特定的子任务，提高整体效率。",
                "url": "https://example.com/multi-agent-systems"
            }
        ],
        "LangGraph": [
            {
                "title": "LangGraph简介",
                "snippet": "LangGraph是用于构建有状态、多参与者应用的框架，特别适合构建Agent和多Agent工作流。",
                "url": "https://example.com/langgraph-intro"
            },
            {
                "title": "LangGraph的状态管理",
                "snippet": "LangGraph通过StateGraph管理应用状态，支持检查点、时间旅行等高级功能。",
                "url": "https://example.com/langgraph-state"
            },
            {
                "title": "使用LangGraph构建Agent",
                "snippet": "LangGraph提供了预构建的Agent节点和工具调用功能，简化Agent开发流程。",
                "url": "https://example.com/langgraph-agents"
            }
        ]
    }

    # 返回相关结果或通用搜索提示
    results = mock_results.get(query, [
        {
            "title": f"关于{query}的搜索结果",
            "snippet": f"这是关于{query}的相关信息。在实际应用中，这里会返回真实的搜索结果。",
            "url": "https://example.com/search"
        }
    ])

    # 限制返回数量
    results = results[:num_results]

    # 格式化输出
    output = f"搜索关键词: {query}\n找到 {len(results)} 条结果:\n\n"
    for i, result in enumerate(results, 1):
        output += f"{i}. {result['title']}\n"
        output += f"   {result['snippet']}\n"
        output += f"   来源: {result['url']}\n\n"

    return output


@mcp.tool()
def get_writing_guidelines(article_type: str = "technical") -> str:
    """
    获取写作指南和风格建议

    Args:
        article_type: 文章类型 (technical, blog, academic, etc.)

    Returns:
        写作指南
    """
    guidelines = {
        "technical": """
技术文章写作指南：
1. 结构清晰：引言 -> 背景 -> 核心内容 -> 应用场景 -> 总结
2. 语言准确：使用专业术语，解释清楚技术概念
3. 示例丰富：提供代码示例或实际案例
4. 逻辑严密：确保技术细节准确，逻辑连贯
5. 读者友好：考虑不同技术水平的读者
        """,
        "blog": """
博客文章写作指南：
1. 标题吸引人：使用引人注目的标题
2. 开头引入：用故事或问题引起读者兴趣
3. 段落简短：每段3-5句话
4. 语言生动：使用类比、比喻等修辞手法
5. 互动性：鼓励读者评论和分享
        """,
        "academic": """
学术论文写作指南：
1. 摘要精炼：200-300字概括核心内容
2. 文献综述：充分引用相关研究
3. 方法严谨：详细描述研究方法
4. 数据支撑：用数据和实验支持观点
5. 结论客观：总结发现并指出局限性
        """
    }

    return guidelines.get(article_type, guidelines["technical"])


@mcp.tool()
def check_article_quality(article_text: str) -> str:
    """
    检查文章质量，提供改进建议

    Args:
        article_text: 文章内容

    Returns:
        质量评估和建议
    """
    issues = []
    suggestions = []

    # 基础检查
    word_count = len(article_text)
    if word_count < 500:
        issues.append("文章过短，建议至少500字")

    # 检查结构
    if "##" not in article_text and "#" not in article_text:
        issues.append("缺少明确的标题和章节划分")
        suggestions.append("添加清晰的标题层级结构")

    # 检查段落
    paragraphs = [p for p in article_text.split('\n\n') if p.strip()]
    if len(paragraphs) < 3:
        issues.append("段落数量较少，可能结构不够丰富")
        suggestions.append("将内容分成更多段落，提高可读性")

    # 检查关键元素
    if "引言" not in article_text and "简介" not in article_text:
        suggestions.append("考虑添加引言部分")

    if "总结" not in article_text and "结论" not in article_text:
        suggestions.append("考虑添加总结部分")

    # 生成报告
    report = f"文章质量评估报告\n"
    report += f"{'='*50}\n"
    report += f"字数统计: {word_count} 字\n"
    report += f"段落数量: {len(paragraphs)} 段\n\n"

    if issues:
        report += "发现的问题:\n"
        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue}\n"
        report += "\n"

    if suggestions:
        report += "改进建议:\n"
        for i, suggestion in enumerate(suggestions, 1):
            report += f"{i}. {suggestion}\n"
    else:
        report += "文章质量良好，未发现明显问题。\n"

    return report


@mcp.prompt()
def system_prompt() -> str:
    """返回系统提示词"""
    return """你是一个专业的文章写作助手，可以调用各种工具来帮助完成文章写作任务。
你可以搜索资料、获取写作指南、检查文章质量等。请根据用户需求合理使用工具。"""


if __name__ == "__main__":
    # 启动 MCP 服务器，使用 HTTP 传输
    print("🚀 启动 MCP 服务器...")
    print("📡 监听地址: http://localhost:8000/mcp")
    mcp.run(transport="streamable-http")

from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
import os
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1.加载 .env 文件
_ = load_dotenv(find_dotenv())


def basicChat():
    # 2. 初始化 LLM
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0
    )
    # 3. 创建 Prompt 模板
    template = """
    当前日期是：{current_date}

    用户的输入是："{user_input}"

    请根据用户的输入，推断用户提到的具体日期（格式：YYYY-MM-DD）。
    只返回日期，不要废话。
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["current_date", "user_input"]
    )
    # 4. 创建输出解析器
    parser = StrOutputParser()

    # 5. 创建完整的链
    chain = prompt | llm | parser

    # 6.获取今天的时间
    today = datetime.date.today()
    user_text = "我昨天下的单"

    print(f"基准日期: {today}")
    print(f"用户输入: {user_text}")

    # 传入当前日期
    result = chain.invoke({
        "current_date": today,
        "user_input": user_text
    })

    print(f"推断日期: {result}")


def advancedChat():
    # LLM 初始化 (统一使用 ChatOpenAI + Qwen)
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0
    )

    # 专用 JSON 模式 LLM
    # llm_json_mode = ChatOpenAI(
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     model="qwen-turbo",
    #     temperature=0,
    #     model_kwargs={"response_format": {"type": "json_object"}}
    # )

    @tool
    def query_order(order_id: str):
        """查询订单状态和物流信息。输入必须是订单号。"""
        # 这里模拟数据库
        mock_db = {
            "1234567890": {"status": "已发货", "location": "上海转运中心", "eta": "2024-02-01"},
            "ORD-999": {"status": "待发货", "location": "仓库", "eta": "未知"},
        }

        if not order_id:
            return "错误：订单号不能为空。"

        result = mock_db.get(order_id)
        if result:
            return f"订单 {order_id} 状态：{result['status']}，当前位置：{result['location']}，预计到达：{result['eta']}"
        else:
            return "未找到该订单号，请核对。"

    tools = [query_order]
    llm_with_tools = llm.bind_tools(tools)

    # 3. 定义状态

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 4. 定义 Chatbot 节点
    def chatbot(state: State): return {"messages": [
        llm_with_tools.invoke(state["messages"])]}

    # 5. 构建图
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # 工具执行完 -> 回到 chatbot 继续对话
    workflow.add_edge("tools", "chatbot")
    app = workflow.compile()

    # 6. 运行对话循环
    print("客服助手已启动（输入 q 退出）...")
    print("提示：试试输入 '查一下订单1234567890'")

    while True:
        try:
            user_input = input("\n用户: ")
            if user_input.lower() in ['q', 'exit']:
                break

            # 运行图
            # stream_mode="values" 会返回每一步的消息列表更新
            events = app.stream(
                {"messages": [("user", user_input)]},
                stream_mode="values"
            )

            for event in events:
                # 获取最后一条消息
                last_msg = event["messages"][-1]
                # 只打印 AI 的回复（过滤掉用户输入和工具调用的中间状态）
                if last_msg.type == "ai" and last_msg.content:
                    print(f"AI: {last_msg.content}")

        except Exception as e:
            print(f"发生错误: {e}")


def main():
    # 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
    # 在根目录可以通过python -m base_chat_system.main 运行
    # basicChat()
    advancedChat()


if __name__ == "__main__":
    main()

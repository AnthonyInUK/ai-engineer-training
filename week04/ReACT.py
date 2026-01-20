from langchain_community.llms import Tongyi
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 加载环境变量


def query_order(order_id):
    """包裹追踪工具"""
    order_id = str(order_id).strip()

    mock_db = {
        "1234567890": {
            "status": "运输中",
            "location": "北京分拣中心",
            "estimated_delivery": "2024-01-15"
        },
        "YT789012": {
            "status": "已签收",
            "location": "上海浦东",
            "delivery_date": "2024-01-10"
        }
    }

    if order_id in mock_db:
        info = mock_db[order_id]
        return f"订单 {order_id} 状态：{info['status']}，当前位置：{info['location']}，{info['eta']}"
    elif len(order_id) < 5:
        return "订单号似乎太短了，请核对后重新输入。"
    else:
        # 2. 体验优化：对于不在库里的长单号，模拟一个状态，而不是直接报错
        return f"订单 {order_id} 正在出库处理中，暂时没有物流详情。"

    if tracking_number in package_status:
        info = package_status[tracking_number]
        if 'estimated_delivery' in info:
            return f"快递单号: {tracking_number}, 状态: {info['status']}, 位置: {info['location']}, 预计送达: {info['estimated_delivery']}"
        else:
            return f"快递单号: {tracking_number}, 状态: {info['status']}, 位置: {info['location']}, 送达日期: {info['delivery_date']}"
    else:
        return f"未找到快递单号 {tracking_number} 的信息"


# 创建工具
tools = [
    Tool(
        name="query_order",
        description="追踪包裹状态，输入快递单号",
        func=query_order
    )
]

# 简化的 ReACT 提示模板
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True  # 如果用的是 ChatModel (ChatOpenAI)，建议设为 True
)

# 初始化通义千问模型
try:
    model = Tongyi(temperature=0)
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True
    )

    # 测试案例
    if __name__ == "__main__":
        print("=== 物流 ReACT Agent===\n")

        test_cases = [
            "查询快递单号 SF123456",
            "它现在到哪里了？",
            "预计什么时候能送到？"
            # ,
            # "从北京到上海寄2公斤包裹多少钱",
            # "查询商品A001的库存"
        ]

        for i, question in enumerate(test_cases, 1):
            print(f"--- 测试案例 {i} ---")
            print(f"问题: {question}")
            print("-" * 50)

            try:
                result = agent_executor.invoke({"input": question})
                print(f"\n最终答案: {result.get('output', '无结果')}")

                # 显示中间步骤
                if 'intermediate_steps' in result:
                    print(f"执行步骤数: {len(result['intermediate_steps'])}")

            except Exception as e:
                print(f"Agent执行错误: {e}")

                # 提供备选答案，确保用户能看到正确结果
                print("\n使用直接工具调用作为备选:")
                if i == 1:  # 查询快递单号
                    backup_result = query_order("SF123456")
                    print(f"备选答案: {backup_result}")

            print("\n" + "="*60 + "\n")


except Exception as e:
    print(f"初始化错误: {e}")
    print("请确保已设置 DASHSCOPE_API_KEY 环境变量")

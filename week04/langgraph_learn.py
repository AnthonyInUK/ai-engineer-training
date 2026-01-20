import os
import json
import operator
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. 配置与初始化
# ==========================================
_ = load_dotenv(find_dotenv())

# LLM 初始化 (统一使用 ChatOpenAI + Qwen)
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0
)

# 专用 JSON 模式 LLM
llm_json_mode = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 搜索工具
try:
    if os.getenv("TAVILY_API_KEY"):
        web_search_tool = TavilySearchResults(k=3)
    else:
        print("未检测到 Tavily Key，降级使用 DuckDuckGo。")
        from langchain_community.tools import DuckDuckGoSearchResults
        web_search_tool = DuckDuckGoSearchResults(max_results=3)
except:
    print("搜索工具初始化异常，使用模拟数据。")

    class MockSearch:
        def invoke(self, q): return [
            {"content": "模拟搜索结果：LangChain 是一个开发框架..."}]
    web_search_tool = MockSearch()

# 向量库初始化
print("正在初始化知识库...")
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]
try:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local"),
    )
    retriever = vectorstore.as_retriever(k=3)
    print("知识库初始化完成。")
except Exception as e:
    print(f"知识库初始化受限 (Mock): {e}")

    class MockRetriever:
        def invoke(self, q): return [Document(
            page_content="Mock doc content.")]
    retriever = MockRetriever()

# ==========================================
# 2. 状态与 Prompt 定义
# ==========================================


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    max_retries: int
    loop_step: int


# Prompts
rag_prompt = """You are an assistant for question-answering tasks. 
Here is the context to use to answer the question:
{context} 
Question: {question}
Answer:"""

router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering.
Use the vectorstore for questions on these topics. For all else, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore'."""

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
Return JSON with single key, binary_score, that is 'yes' or 'no'."""
doc_grader_prompt = "Document: {document}\nQuestion: {question}"

hallucination_grader_instructions = """You are a teacher grading a quiz. 
Ensure the STUDENT ANSWER is grounded in the FACTS.
Return JSON with two keys: binary_score ('yes'/'no') and explanation."""
hallucination_grader_prompt = "FACTS: {documents}\nSTUDENT ANSWER: {generation}"

answer_grader_instructions = """You are a teacher grading a quiz. 
Ensure the STUDENT ANSWER helps to answer the QUESTION.
Return JSON with two keys: binary_score ('yes'/'no') and explanation."""
answer_grader_prompt = "QUESTION: {question}\nSTUDENT ANSWER: {generation}"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 3. 节点函数 (Nodes)
# ==========================================


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    try:
        try:
            docs = web_search_tool.invoke({"query": question})
        except:
            docs = web_search_tool.invoke(question)

        if isinstance(docs, str):
            content = docs
        else:
            content = "\n".join([d.get("content", "") if isinstance(
                d, dict) else d.page_content for d in docs])

        documents.append(Document(page_content=content))
    except Exception as e:
        print(f"搜索出错: {e}")
    return {"documents": documents}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    context = format_docs(documents)
    prompt = rag_prompt.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"generation": response, "loop_step": loop_step + 1}


def grade_documents(state):
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    for d in documents:
        prompt = doc_grader_prompt.format(
            document=d.page_content, question=question)
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions), HumanMessage(content=prompt)])
        try:
            grade = json.loads(result.content).get("binary_score", "yes")
        except:
            grade = "yes"

        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"

    if not filtered_docs:
        web_search = "Yes"

    return {"documents": filtered_docs, "web_search": web_search}

# ==========================================
# 4. 路由逻辑 (Edges)
# ==========================================


def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    result = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=question)])
    try:
        source = json.loads(result.content).get("datasource", "websearch")
    except:
        source = "websearch"

    if source == "websearch":
        print("-> 路由到: 联网搜索")
        return "websearch"
    else:
        print("-> 路由到: 本地知识库")
        return "vectorstore"


def decide_to_generate(state):
    print("---DECIDE TO GENERATE---")
    if state["web_search"] == "Yes":
        return "websearch"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state):
    print("---CHECK GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    loop_step = state.get("loop_step", 0)

    if loop_step > max_retries:
        return "max retries"

    # 1. 幻觉检查
    prompt1 = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content)
    res1 = llm_json_mode.invoke([SystemMessage(
        content=hallucination_grader_instructions), HumanMessage(content=prompt1)])
    try:
        grade1 = json.loads(res1.content).get("binary_score", "yes")
    except:
        grade1 = "yes"

    if grade1 == "yes":
        print("-> 无幻觉")
        # 2. 答案质量检查
        prompt2 = answer_grader_prompt.format(
            question=question, generation=generation.content)
        res2 = llm_json_mode.invoke([SystemMessage(
            content=answer_grader_instructions), HumanMessage(content=prompt2)])
        try:
            grade2 = json.loads(res2.content).get("binary_score", "yes")
        except:
            grade2 = "yes"

        if grade2 == "yes":
            print("-> 回答有效")
            return "useful"
        else:
            print("-> 回答无效，重新搜索")
            return "not useful"
    else:
        print("-> 存在幻觉，重试")
        return "not supported"


# ==========================================
# 5. 构建工作流 (Workflow)
# ==========================================
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# 入口路由
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

# 常规边
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")

# 条件边：评分 -> 决定
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

# 条件边：生成 -> 检查
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

graph = workflow.compile()

# ==========================================
# 6. 运行
# ==========================================
if __name__ == "__main__":
    # 生成图片
    try:
        print("正在生成流程图...")
        png_data = graph.get_graph().draw_mermaid_png()
        with open("graph_workflow.png", "wb") as f:
            f.write(png_data)
        print("✅ 流程图已保存为 graph_workflow.png")
    except Exception as e:
        print(f"❌ 生成流程图失败: {e}")

    print("\n=== 开始测试 ===")

    # 测试 1: RAG
    q1 = "agent memory"
    print(f"\n用户: {q1}")
    result = graph.invoke({"question": q1, "max_retries": 3, "loop_step": 0})
    print(f"AI: {result['generation'].content}")

    print("-" * 30)

    # 测试 2: Search
    q2 = "今天的大模型新闻"
    print(f"\n用户: {q2}")
    result = graph.invoke({"question": q2, "max_retries": 3, "loop_step": 0})
    print(f"AI: {result['generation'].content}")

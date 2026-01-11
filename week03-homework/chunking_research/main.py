import os
import time
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("Warning: DASHSCOPE_API_KEY not found in environment variables. Please set it in .env or your shell.")
    # You can uncomment the line below to set it directly for testing if needed
    # os.environ["DASHSCOPE_API_KEY"] = "your_key_here"

# 2. 配置 LlamaIndex
# 使用 Qwen-Plus 作为 LLM
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

# 使用 DashScope Embedding
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)


def run_experiment(name, splitter, documents, question):
    """
    运行切片实验
    :param name: 实验名称
    :param splitter: 切片器 (NodeParser)
    :param documents: 文档列表
    :param question: 测试问题
    """
    print(f"\n{'='*20} Experiment: {name} {'='*20}")

    # 全局设置当前的 splitter
    Settings.text_splitter = splitter

    # 特殊处理：如果是 SentenceWindowNodeParser，需要在构建索引时传入 transformations
    # 或者直接从 nodes 构建索引
    if isinstance(splitter, SentenceWindowNodeParser):
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)

        # 句子窗口检索需要 MetadataReplacementPostProcessor
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ]
        )
    else:
        # 普通切片器
        # 注意：VectorStoreIndex.from_documents 默认使用 Settings.text_splitter
        # 为了确保隔离，我们这里显式传递 transformations
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter]
        )
        query_engine = index.as_query_engine(similarity_top_k=3)

    # 执行检索和生成
    print(f"Question: {question}")
    response = query_engine.query(question)

    print("\n[Retrieved Context (Top 1)]:")
    # 打印最相关的上下文片段，方便观察切片效果
    if len(response.source_nodes) > 0:
        node = response.source_nodes[0]
        content = node.node.get_content(metadata_mode="all")
        print(f"{content[:500]}..." if len(content) > 500 else content)
    else:
        print("No context retrieved.")

    print("\n[Generated Answer]:")
    print(response.response)
    print("-" * 60)


def main():
    # 加载数据
    # 注意：请确保运行目录正确，或者使用绝对路径
    data_dir = "../data"  # 假设在 chunking_research 目录下运行
    if not os.path.exists(data_dir):
        # 尝试使用相对于 workspace 的路径
        data_dir = "week03-homework/data"

    print(f"Loading data from {data_dir}...")
    try:
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    question = "人工智能历史上的低谷期（AI Winter）是指什么？"

    # --- 实验 1: Sentence Splitter (默认) ---
    sentence_splitter = SentenceSplitter(
        # chunk_size=256,
        chunk_size=512,
        # chunk_overlap=50,
        chunk_overlap=1
    )
    run_experiment("Sentence Splitter", sentence_splitter, documents, question)

    # --- 实验 2: Token Text Splitter ---
    token_splitter = TokenTextSplitter(
        # chunk_size=128,
        chunk_size=256,
        # chunk_overlap=10,
        chunk_overlap=1,
        separator="\n"
    )
    run_experiment("Token Splitter", token_splitter, documents, question)

    # --- 实验 3: Sentence Window Retrieval ---
    # 窗口大小为3，意味着取当前句子 + 前3句 + 后3句作为上下文
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        # window_size=3,
        window_size=10,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    run_experiment("Sentence Window Splitter",
                   sentence_window_splitter, documents, question)


if __name__ == "__main__":
    main()

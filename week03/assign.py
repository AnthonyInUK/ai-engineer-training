import os
import re
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "DASHSCOPE_API_KEY not found. Set it in environment or a .env file.")

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=_API_KEY,
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)


def evaluate_splitter(splitter, documents, question, ground_truth, label="Sentence", top_k=8):
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    retrieved = index.as_retriever(similarity_top_k=top_k).retrieve(question)
    contexts = [r.node.get_content() for r in retrieved]
    answer = str(index.as_query_engine(similarity_top_k=top_k).query(question))
    avg_len = (sum(len(n.get_content())
               for n in nodes) // len(nodes)) if nodes else 0
    norm_gt = _normalize(ground_truth or "")
    hit = any(norm_gt in _normalize(c)
              for c in contexts) if ground_truth else None
    print(f"[{label}] nodes={len(nodes)} avg_chars={avg_len} retrieved={len(contexts)} hit={hit}")
    # Show top-2 context snippets with source metadata (if available)
    for i, r in enumerate(retrieved[:2], start=1):
        meta = r.node.metadata or {}
        src = meta.get("file_path") or meta.get(
            "file_name") or meta.get("source") or "unknown"
        page = meta.get("page_label") or meta.get(
            "page") or meta.get("page_number") or "?"
        snippet = r.node.get_content().replace("\n", " ")[:400]
        print(f"[{label}] ctx{i} src={src} page={page} :: {snippet}")
    print(f"[{label}] answer: {answer}")
    return {"nodes": len(nodes), "avg_len": avg_len, "hit": hit, "answer": answer, "contexts": contexts}


documents = SimpleDirectoryReader(
    input_files=[
        # "/Users/anthony/Desktop/llm/ai-engineer-training/week03/data/Extreme weather.pdf",
        # "/Users/anthony/Desktop/llm/ai-engineer-training/week03/data/Large language model.pdf",
        "/Users/anthony/Desktop/llm/ai-engineer-training/week03/data/Quantum computing.pdf",
    ]
).load_data()


# 句子切片
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

question = "What is a quantum computer?"
ground_truth = "A quantum computer is a (real or theoretical) computer that exploits superposed and entangled states, and the intrinsically non-deterministic outcomes of quantum measurements, as features of its computation."


evaluate_splitter(sentence_splitter, documents,
                  question, ground_truth, "Sentence")

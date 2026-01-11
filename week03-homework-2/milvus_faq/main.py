import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
# 使用 OpenAIEmbedding 代替 DashScopeEmbedding 以解决 _api_key 问题
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

# Load environment variables
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# --- Configuration ---
DIMENSION = 1024  # text-embedding-v3 dimension
MILVUS_URI = "./milvus_demo.db"  # Use Milvus Lite for simplicity

# --- Setup Models ---
# LLM
llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True
)

# Embedding Model (Use OpenAIEmbedding interface for DashScope)
embed_model = OpenAIEmbedding(
    model_name="text-embedding-v3",
    api_key=DASHSCOPE_API_KEY,
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Set Global Settings
Settings.llm = llm
Settings.embed_model = embed_model

# --- App Initialization ---
app = FastAPI(title="Milvus FAQ Retrieval System")

# Global variables for index and query engine
query_engine = None


class FAQRequest(BaseModel):
    question: str


class FAQResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]


class ReindexRequest(BaseModel):
    data_dir: str = "./data"


# --- Helper Functions ---

def init_milvus_index(data_dir: str = "./data"):
    global query_engine

    print(f"Initializing index from {data_dir}...")

    # 1. Setup Milvus Vector Store
    # overwrite=True ensures we rebuild the index (simulating re-indexing)
    vector_store = MilvusVectorStore(
        uri=MILVUS_URI,
        dim=DIMENSION,
        overwrite=True
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. Load Documents
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        # Create dummy data if empty
        faq_file = os.path.join(data_dir, "faq.txt")
        if not os.path.exists(faq_file):
            dummy_faq = """
            Q: 如何退货？
            A: 您可以在订单详情页点击"申请售后"，选择"退货退款"，填写原因并提交。我们会在24小时内审核。
            
            Q: 退款多久到账？
            A: 商家同意退款后，款项通常会在1-3个工作日原路退回您的支付账户。
            
            Q: 运费谁承担？
            A: 如果是商品质量问题，运费由商家承担；如果是个人原因不喜欢，运费由买家承担。
            """
            with open(faq_file, "w") as f:
                f.write(dummy_faq)

    # 修改：只加载 faq.txt，避免加载该目录下的其他无关文件（如 company_info.txt）
    documents = SimpleDirectoryReader(
        input_files=[os.path.join(data_dir, "faq.txt")]
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    # 3. Document Splitting (Semantic Splitting)
    # Using SemanticSplitterNodeParser for semantic chunking
    # This ensures chunks are broken by meaning, not just character count
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )

    # 4. Create Index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True
    )

    # 5. Create Query Engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    print("Index initialized successfully.")


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    try:
        init_milvus_index()
    except Exception as e:
        print(f"Error initializing index on startup: {e}")
        print("System will start, but query endpoints may fail until /reindex is called.")


@app.get("/")
async def root():
    return {"message": "Welcome to Milvus FAQ Retrieval System. Visit /docs for API documentation."}


@app.post("/query", response_model=FAQResponse)
async def query_faq(request: FAQRequest):
    global query_engine
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index not initialized")

    response = query_engine.query(request.question)

    # Extract source content for reference
    sources = [node.node.get_content()[:100] +
               "..." for node in response.source_nodes]

    return FAQResponse(
        question=request.question,
        answer=response.response,
        sources=sources
    )


@app.post("/reindex")
async def reindex(request: ReindexRequest):
    try:
        # 再次调用初始化函数，重新读取文件 -> 切片 -> 存入 Milvus
        init_milvus_index(request.data_dir)
        return {"status": "success", "message": "Knowledge base re-indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

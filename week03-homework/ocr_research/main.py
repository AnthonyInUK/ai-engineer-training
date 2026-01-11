import os
import sys
import types
from dotenv import load_dotenv
from typing import List, Union
from pathlib import Path

# --- Monkey Patch Start ---
# 【必须保留】修复 paddlex 依赖 langchain.docstore 的问题
try:
    import langchain.docstore
except ImportError:
    class MockDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    docstore_module = types.ModuleType("langchain.docstore")
    document_module = types.ModuleType("langchain.docstore.document")
    document_module.Document = MockDocument
    docstore_module.document = document_module

    sys.modules["langchain.docstore"] = docstore_module
    sys.modules["langchain.docstore.document"] = document_module

try:
    import langchain.text_splitter
except ImportError:
    class MockRecursiveCharacterTextSplitter:
        def __init__(self, **kwargs):
            pass

        def split_text(self, text):
            return [text]

    text_splitter_module = types.ModuleType("langchain.text_splitter")
    text_splitter_module.RecursiveCharacterTextSplitter = MockRecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = text_splitter_module
# --- Monkey Patch End ---

# 引入 LlamaIndex 核心组件
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 加载环境变量
load_dotenv()

# 配置 LlamaIndex
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3
)


class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""

    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        # 延迟导入 PaddleOCR
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        from paddleocr import PaddleOCR

        # 按照老师的模版配置 PaddleOCR
        self.ocr = PaddleOCR(
            lang=lang,
            # use_gpu=use_gpu, # 已移除
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **kwargs
        )

    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        """
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file

        documents = []

        for file_path in files:
            file_path_str = str(file_path)
            if not os.path.exists(file_path_str):
                print(f"Warning: File not found {file_path_str}")
                continue

            print(f"Processing {file_path_str}...")

            try:
                # 使用 ocr.predict 接口
                result = self.ocr.predict(file_path_str)
            except Exception as e:
                print(f"Error processing {file_path_str}: {e}")
                continue

            # 解析结果
            text_content = []

            for res in result:
                # PaddleX OCRResult 对象
                # 优先尝试从 rec_texts 字段提取
                try:
                    if 'rec_texts' in res:
                        texts = res['rec_texts']
                        if isinstance(texts, list):
                            text_content.extend(texts)
                            continue
                except:
                    pass

                # 尝试属性访问
                if hasattr(res, 'rec_texts') and res.rec_texts:
                    text_content.extend(res.rec_texts)
                    continue

                # 兜底：尝试 rec_text (单数)
                if hasattr(res, 'rec_text') and res.rec_text:
                    text_content.append(res.rec_text)
                    continue

            full_text = "\n".join(text_content)

            # 构造 Document 对象
            metadata = {
                "image_path": file_path_str,
                "file_name": os.path.basename(file_path_str),
            }

            doc = Document(text=full_text, metadata=metadata)
            documents.append(doc)

        return documents


def main():
    image_dir = "test_images"
    if not os.path.exists(image_dir) and os.path.exists(f"week03-homework/ocr_research/{image_dir}"):
        image_dir = f"week03-homework/ocr_research/{image_dir}"

    image_files = [os.path.join(image_dir, f) for f in os.listdir(
        image_dir) if f.endswith(('.png', '.jpg'))]

    if not image_files:
        print("No images found. Please run generate_images.py first.")
        return

    print(f"Testing with images: {image_files}")

    try:
        # 实例化 Reader (使用英文模型)
        reader = ImageOCRReader(lang='en')
    except Exception as e:
        print(f"Failed to initialize OCR: {e}")
        import traceback
        traceback.print_exc()
        return

    # 加载数据
    documents = reader.load_data(image_files)

    print("\n--- Extracted Text Preview ---")
    for doc in documents:
        print(f"\n[File: {doc.metadata['file_name']}]")
        print(doc.text[:200])
    print("------------------------------\n")

    # 构建索引并查询
    if documents:
        print("Building index and querying...")
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        response = query_engine.query("What is the price of Banana?")
        print(f"\nQ: What is the price of Banana?\nA: {response}")


if __name__ == "__main__":
    main()

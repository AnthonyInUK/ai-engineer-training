import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")

# 加载环境变量
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 1. 定义 LLM (使用 OpenAILike)
llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True
)

# 2. 定义 Embedding (Switch to OpenAIEmbedding compatible with DashScope)
embed_model = OpenAIEmbedding(
    model_name="text-embedding-v3",
    api_key=DASHSCOPE_API_KEY,
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
Settings.llm = llm
Settings.embed_model = embed_model


class DocumentRAG:
    """负责文档检索的 RAG 模块"""

    def __init__(self):
        self.index = None
        self.query_engine = None

    def build_index(self, data_dir: str = "./data"):
        # 如果目录不存在，我们需要创建一些模拟数据
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "company_info.txt"), "w") as f:
                f.write("""
                A公司成立于2010年，主要从事人工智能技术研发。
                B公司是A公司的子公司，A公司持有B公司60%的股份，B公司专注于自动驾驶领域。
                C公司是B公司的全资子公司，负责硬件制造，B公司持有C公司100%的股份。
                D公司是A公司的投资部门，A公司持有D公司55%的股份，D公司持有多家初创公司股份。
                H集团持有A公司25%的股份，是其重要的战略投资者。
                K资本持有A公司15%的股份。
                张三作为创始人，目前持有A公司20%的股份。
                另外，A公司最近刚举办了十周年庆典，邀请了多位行业专家出席。
                公司食堂的红烧肉非常有名，员工对此赞不绝口。
                去年A公司在环保公益活动中捐款了500万元。
                """)
        # --- TODO 2: 加载文档并构建索引 ---
        documents = SimpleDirectoryReader(
            input_files=[os.path.join(data_dir, "company_info.txt")]).load_data()
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        # 构建向量索引
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter])
        # 创建查询引擎
        self.query_engine = self.index.as_query_engine(similarity_top_k=2)
        print("RAG Index built successfully.")

    def query(self, question: str):
        if not self.query_engine:
            return "RAG Engine not initialized."
        return self.query_engine.query(question)


# --- 2. Knowledge Graph Component (Neo4j) ---
class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def init_data(self):
        """初始化图谱数据 (清空旧数据，不再硬编码插入)"""
        with self.driver.session() as session:
            # 清空旧数据
            session.run("MATCH (n) DETACH DELETE n")
            print("Knowledge Graph cleared.")

    def find_shareholders(self, company_name: str):
        """查询直接股东"""
        query = """
        MATCH (holder)-[r:SHARES|CONTROLS]->(c:Company {name: $name})
        RETURN holder.name as holder, r.share as share
        ORDER BY r.share DESC
        """
        with self.driver.session() as session:
            result = session.run(query, name=company_name)
            data = [record.data() for record in result]
            return data

    def extract_and_store(self, text: str):
        """从文本提取实体关系并存入图谱"""
        triplets = self._extract_triplets(text)
        print(f"Extracted Triplets: {triplets}")

        if not triplets:
            print("Warning: No triplets extracted from text.")
            return

        import re

        with self.driver.session() as session:
            for source, rel, target, props in triplets:
                # 更加鲁棒的数值提取
                share_val = 0.0
                # 匹配 25%, 25.5%, 0.25
                match = re.search(r"(\d+(\.\d+)?)%?", props)
                if match:
                    try:
                        val = float(match.group(1))
                        # 简单的逻辑：如果含有 % 或者值大于1，通常是百分比；小于1通常是小数
                        if "%" in props or val > 1.0:
                            share_val = val / 100.0
                        else:
                            share_val = val
                    except:
                        pass

                # 简单的关系映射
                rel_type = "RELATION"
                if any(x in rel for x in ["股东", "持股", "投资", "持有"]):
                    rel_type = "SHARES"
                elif "控制" in rel:
                    rel_type = "CONTROLS"

                # 区分 Company 和 Person
                # 简单规则：名字里没这些词的当作人
                source_label = "Company"
                if not any(x in source for x in ["公司", "集团", "资本", "部", "局"]):
                    source_label = "Person"

                target_label = "Company"
                if not any(x in target for x in ["公司", "集团", "资本", "部", "局"]):
                    target_label = "Person"

                query = f"""
                MERGE (a:{source_label} {{name: $source}})
                MERGE (b:{target_label} {{name: $target}})
                MERGE (a)-[:{rel_type} {{share: $share}}]->(b)
                """
                session.run(query, source=source,
                            share=share_val, target=target)

        print(f"Stored {len(triplets)} relationships in Graph.")

    def _extract_triplets(self, text: str):
        """让 LLM 提取三元组"""
        prompt = f"""
        你是一个知识图谱专家。请从以下文本中提取公司股权结构和投资关系。
        忽略无关信息（如食堂、活动等）。
        
        文本：
        {text}
        
        请严格按以下 CSV 格式输出（不要表头，每行一个）：
        主体, 关系, 客体, 属性
        
        规则：
        1. 关系必须是以下之一：股东, 持股, 子公司
        2. 如果提到持股比例，请在属性中写明（如 "持股25%"）
        3. 实体名称要完整
        
        示例：
        张三, 股东, A公司, 持股35%
        A公司, 持股, B公司, 持股60%
        
        输出：
        """
        response = llm.complete(prompt)
        content = response.text.strip()

        triplets = []
        for line in content.split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                s = parts[0].strip()
                r = parts[1].strip()
                t = parts[2].strip()
                p = parts[3].strip() if len(parts) > 3 else ""
                triplets.append((s, r, t, p))
        return triplets

    def close(self):
        self.driver.close()


class HybridQA:
    def __init__(self):
        self.rag = DocumentRAG()
        self.kg = KnowledgeGraph()

    def initialize(self):
        self.rag.build_index()
        self.kg.init_data()

        with open("./data/company_info.txt", "r") as f:
            text = f.read()
        self.kg.extract_and_store(text)

    def answer(self, question: str):
        print(f"\n--- Processing Question: {question} ---")

        # 1. 简单的意图分析（这里硬编码简化，实际可以用 LLM 提取实体）
        target_company = "A公司"
        if "B公司" in question:
            target_company = "B公司"
        if "C公司" in question:
            target_company = "C公司"

        print(f"[Plan] 1. RAG: Retrieve info about {target_company}")
        print(f"[Plan] 2. KG: Query shareholders of {target_company}")

        # 2. 并行检索
        rag_context = self.rag.query(f"{target_company}的基本情况")
        kg_data = self.kg.find_shareholders(target_company)
        kg_context = str(kg_data) if kg_data else "未找到股东信息"

        print(f"[Context] RAG: {rag_context}")
        print(f"[Context] KG: {kg_context}")

        # 3. LLM 综合回答（包含简单的联合评分与冲突处理机制）
        # 这里的机制是通过 Prompt Engineering 实现的，指导 LLM 如何权衡不同来源的信息。
        final_prompt = f"""
        基于以下信息回答问题："{question}"
        
        [文档信息] (权重: 参考细节描述):
        {rag_context}
        
        [股权图谱信息] (权重: 高 - 精确数值以图谱为准):
        {kg_context}
        
        任务：
        1. 综合两部分信息。
        2. 联合评分逻辑：如果图谱中有明确的股权比例数据，优先采用图谱数据（防止文档过时或非结构化提取错误）。
           - 原理：图谱数据通常来自结构化数据库，精度高于非结构化文本检索。
        3. 错误传播防止：如果两者信息严重冲突（如数字完全对不上），请明确指出冲突点，并建议人工核实。
           - 目的：避免单一来源的幻觉或过时信息误导用户。
        
        请给出最终回答：
        """

        response = llm.complete(final_prompt)
        print(f"\n[Final Answer]:\n{response.text}")
        return response.text

    def close(self):
        self.kg.close()


def main():
    system = HybridQA()
    try:
        system.initialize()

        # Scenario 1
        system.answer("A公司的最大股东是谁？")

        # Scenario 2 (Multi-hop)
        # system.answer("C公司的实际控制人是谁？")

    finally:
        system.close()


if __name__ == "__main__":
    main()

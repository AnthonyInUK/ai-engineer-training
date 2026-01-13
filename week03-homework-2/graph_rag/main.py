import os
import json
import re
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

# 1. 定义 LLM
llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True
)

# 2. 定义 Embedding
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

        documents = SimpleDirectoryReader(
            input_files=[os.path.join(data_dir, "company_info.txt")]).load_data()
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter])
        self.query_engine = self.index.as_query_engine(similarity_top_k=2)
        print("RAG Index built successfully.")

    def query(self, question: str):
        if not self.query_engine:
            return "RAG Engine not initialized."
        return self.query_engine.query(question)


class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def init_data(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Knowledge Graph cleared.")

    def find_shareholders(self, company_name: str):
        """查询直接股东 (单跳)"""
        query = """
        MATCH (holder)-[r:SHARES|CONTROLS]->(c:Company {name: $name})
        RETURN holder.name as holder, r.share as share
        ORDER BY r.share DESC
        """
        with self.driver.session() as session:
            result = session.run(query, name=company_name)
            data = [record.data() for record in result]
            return data

    def find_ultimate_controller(self, company_name: str):
        """
        [改进点1] 多跳路径查询
        查询到达目标公司的所有控制路径，层级深度设为1到5层
        """
        query = """
        MATCH path = (root)-[:SHARES|CONTROLS*1..5]->(c:Company {name: $name})
        RETURN [node in nodes(path) | node.name] as path_nodes,
               [rel in relationships(path) | rel.share] as path_shares
        """
        with self.driver.session() as session:
            result = session.run(query, name=company_name)
            data = [record.data() for record in result]
            return data

    def extract_and_store(self, text: str):
        triplets = self._extract_triplets(text)
        print(f"Extracted Triplets: {triplets}")

        if not triplets:
            print("Warning: No triplets extracted!")
            return

        with self.driver.session() as session:
            for source, rel, target, props in triplets:
                share_val = 0.0
                match = re.search(r"(\d+(\.\d+)?)%?", props)
                if match:
                    try:
                        val = float(match.group(1))
                        if "%" in props or val > 1.0:
                            share_val = val / 100.0
                        else:
                            share_val = val
                    except:
                        pass

                rel_type = "SHARES"
                final_source = source
                final_target = target

                # 修正：处理 "子公司" 关系的方向
                # 如果 source=A, rel=子公司, target=B -> 意味着 A是B的子公司 -> B持股A
                # 我们希望图谱里是: 投资方 -> SHARES -> 被投资方
                if "子公司" in rel:
                    final_source = target
                    final_target = source
                    rel_type = "SHARES"
                elif "控制" in rel:
                    rel_type = "CONTROLS"

                source_label = "Company"
                if not any(x in final_source for x in ["公司", "集团", "资本", "部", "局"]):
                    source_label = "Person"

                target_label = "Company"
                if not any(x in final_target for x in ["公司", "集团", "资本", "部", "局"]):
                    target_label = "Person"

                query = f"""
                MERGE (a:{source_label} {{name: $source}})
                MERGE (b:{target_label} {{name: $target}})
                MERGE (a)-[:{rel_type} {{share: $share}}]->(b)
                """
                session.run(query, source=final_source,
                            share=share_val, target=final_target)

        print(f"Stored {len(triplets)} relationships in Graph.")

    def _extract_triplets(self, text: str):
        prompt = f"""
        你是一个知识图谱专家。请从以下文本中提取【所有】公司股权结构和投资关系。
        
        文本：
        {text}
        
        请严格遵循以下规则输出 CSV 格式（不要表头）：
        1. 格式：投资方, 动作, 被投资方, 属性
        2. 动作只能是：持股, 控制
        3. 如果原文说 "A是B的子公司"，请转换为 "B, 持股, A" (即 B是投资方)
        4. 属性中必须包含具体持股比例（如 "60%"），如果未知则留空
        5. 【重要】请仔细阅读全文，不要遗漏任何一条关系！包括C公司、D公司、H集团等。
        
        示例：
        张三, 持股, A公司, 35%
        A公司, 持股, B公司, 60%
        
        输出：
        """
        response = llm.complete(prompt)
        content = response.text.strip()
        print(f"\n[Debug] LLM Extraction Raw Output:\n{content}\n")

        content = content.replace("```csv", "").replace("```", "").strip()

        triplets = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

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
        if os.path.exists("./data/company_info.txt"):
            with open("./data/company_info.txt", "r") as f:
                text = f.read()
            self.kg.extract_and_store(text)

    def _extract_entities(self, question: str):
        """
        [改进点2] 使用 LLM 进行实体识别 (NER)
        """
        prompt = f"""
        从以下问题中提取公司实体名称。只返回实体名称，不要其他文字。
        如果有多个，用逗号分隔。如果没有，返回 "None"。
        
        问题："{question}"
        """
        response = llm.complete(prompt)
        text = response.text.strip()
        if "None" in text:
            return []
        entities = [e.strip()
                    for e in text.replace("，", ",").split(",") if e.strip()]
        return entities

    def answer(self, question: str):
        print(f"\\n--- Processing Question: {question} ---")

        # 1. 实体识别
        entities = self._extract_entities(question)
        print(f"[Entity Extraction]: Found {entities}")

        target_company = entities[0] if entities else "A公司"

        print(f"[Plan] 1. RAG: Retrieve info about {target_company}")
        print(
            f"[Plan] 2. KG: Query multi-hop relationships for {target_company}")

        # 2. 并行检索
        rag_response = self.rag.query(f"{target_company}的情况")
        rag_context = str(rag_response)

        # [改进点1] 使用多跳查询
        kg_data_direct = self.kg.find_shareholders(target_company)
        kg_data_path = self.kg.find_ultimate_controller(target_company)

        kg_context = f"直接持股: {kg_data_direct}\\n完整控制链路: {kg_data_path}"

        print(f"[Context] RAG: {rag_context[:100]}...")
        print(f"[Context] KG: {kg_context}")

        # [改进点3] 联合评分逻辑
        rag_score = 0.8
        kg_score = 1.0 if (kg_data_direct or kg_data_path) else 0.0

        print(
            f"[Scoring] RAG Confidence: {rag_score}, KG Confidence: {kg_score}")

        # 3. LLM 综合回答
        final_prompt = f"""
        基于以下信息回答问题："{question}"
        
        [文档信息] (置信度 Score: {rag_score}):
        {rag_context}
        
        [股权图谱信息] (置信度 Score: {kg_score}):
        {kg_context}
        
        任务指南：
        1. **多跳推理**：利用图谱中的 [完整控制链路] 来回答关于“实际控制人”或“最终受益人”的问题。
        2. **冲突解决**：
           - 如果图谱 Score=1.0 且与文档数字冲突，优先信任图谱。
           - 如果图谱 Score=0.0，完全依赖文档。
        3. **评分与解释**：
           - 在回答的开头，给出一个 "综合置信度" (High/Medium/Low)。
           - 如果发现冲突，必须在回答中单独列出 "数据冲突警告"。
           
        请给出最终回答：
        """

        response = llm.complete(final_prompt)
        print(f"\\n[Final Answer]:\\n{response.text}")
        return response.text

    def close(self):
        self.kg.close()


def main():
    system = HybridQA()
    try:
        system.initialize()

        # 测试 1: 简单查询
        system.answer("A公司的最大股东是谁？")

        # 测试 2: 多跳查询
        system.answer("C公司的实际控制人是谁？")

        # 测试 3: 隐式实体查询
        system.answer("谁持有B公司的股份？")

    finally:
        system.close()


if __name__ == "__main__":
    main()

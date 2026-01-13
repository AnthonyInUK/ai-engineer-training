from neo4j import GraphDatabase
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv("week04/.env"))
# 或者尝试读取 week03-homework-2 下的 env 如果有的话
# load_dotenv("week03-homework-2/.env")

uri = "neo4j://localhost:7687"
auth = ("neo4j", "password")

driver = GraphDatabase.driver(uri, auth=auth)

def print_all():
    with driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name, type(r), r.share, b.name")
        print(f"{'Source':<15} | {'Rel':<10} | {'Share':<5} | {'Target':<15}")
        print("-" * 50)
        count = 0
        for record in result:
            count += 1
            print(f"{record['a.name']:<15} | {record['type(r)']:<10} | {str(record['r.share']):<5} | {record['b.name']:<15}")
        
        if count == 0:
            print("⚠️ 图谱是空的！(Graph is empty)")

print_all()
driver.close()

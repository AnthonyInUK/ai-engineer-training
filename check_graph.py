from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
auth = ("neo4j", "password")

try:
    driver = GraphDatabase.driver(uri, auth=auth)
    
    with driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name, type(r), r.share, b.name")
        print(f"{'Source':<15} | {'Rel':<10} | {'Share':<5} | {'Target':<15}")
        print("-" * 50)
        count = 0
        for record in result:
            count += 1
            share = record['r.share']
            if share is None: share = "None"
            print(f"{record['a.name']:<15} | {record['type(r)']:<10} | {str(share):<5} | {record['b.name']:<15}")
        
        if count == 0:
            print("⚠️ 图谱是空的！(Graph is empty)")
            
    driver.close()
except Exception as e:
    print(f"连接 Neo4j 失败: {e}")

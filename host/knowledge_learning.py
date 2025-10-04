import json
import PyPDF2
from collections import Counter
import re
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
from llm.llm_components import (
    get_qwen_embedding,
    qwen3_reranker,
    qwen3_chat
)
from host.repo_embedding import get_commit_embeddings

# Stop words to exclude
# follow data/english.txt from online open-source
STOP_WORDS = []

def extract_top_words(pdf_path: str, top_n: int = 20) -> List[str]:
    """Extract top N frequent words from PDF excluding stop words"""
    word_counter = Counter()
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
                word_counter.update([
                    word for word in words 
                    if word not in STOP_WORDS
                ])
    
    return [word for word, _ in word_counter.most_common(top_n)]

def find_nearest_commits(
    query_words: List[str], 
    k: int = 5
) -> Dict[str, List[Dict]]:
    """Find k nearest commits for each query word using Qwen3 reranker"""
    results = {}
    embeddings = get_commit_embeddings()
    
    for word in query_words:
        query_embedding = get_qwen_embedding(word)
        ranked_indices = qwen3_reranker(
            query_embedding, 
            [emb.tolist() for emb in embeddings],
            top_k=k
        )
        
        conn = sqlite3.connect("commits.db")
        cursor = conn.cursor()
        
        word_results = []
        for idx in ranked_indices:
            cursor.execute(
                "SELECT id, message, diff FROM commits WHERE rowid=?",
                (idx+1,)
            )
            commit_id, message, diff = cursor.fetchone()
            word_results.append({
                "commit_id": commit_id,
                "message": message,
                "diff": diff
            })
        
        results[word] = word_results
        conn.close()
    
    return results

def cluster_commits(n_clusters: int = 5) -> Dict[int, List[Dict]]:
    """Cluster all commits and return grouped results"""
    embeddings = get_commit_embeddings()
    if not embeddings:
        return {}
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(np.stack(embeddings))
    
    conn = sqlite3.connect("commits.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, message, diff FROM commits")
    all_commits = cursor.fetchall()
    conn.close()
    
    results = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        results[cluster_id] = [{
            "commit_id": all_commits[i][0],
            "message": all_commits[i][1],
            "diff": all_commits[i][2]
        } for i in cluster_indices]
    
    return results

def generate_thoughts(data: Dict) -> Dict:
    """Generate summary thoughts for each group using Qwen3"""
    results = {}
    for key, items in data.items():
        messages = [item["message"] for item in items]
        combined = "\n".join(messages[:5])  # Use top 5 for context
        
        thought = qwen3_chat(
            f"Summarize these commit messages in under 10 words: {combined}"
        )
        
        results[key] = {
            "thought": thought,
            "commits": items
        }
    
    return results

def calculate_efficiency(
    thoughts_data: Dict,
    perf_results_path: str = "result.json"
) -> List[Dict]:


    with open(perf_results_path) as f:
        perf_data = json.load(f)["date_perf_results"]
    
    efficiency_results = []
    
    for thought_key, thought_info in thoughts_data.items():
        commits = thought_info["commits"]
        if not commits:
            continue
            

        dates = [datetime.fromisoformat(c["date"]) for c in commits if "date" in c]
        if not dates:
            continue
            
        min_date = min(dates).strftime("%Y-%m-%d")
        max_date = max(dates).strftime("%Y-%m-%d")
        

        date_perf = [
            p for p in perf_data 
            if min_date <= p["date"] <= max_date
        ]
        if len(date_perf) < 2:
            continue
            
        perf_diff = date_perf[-1]["value"] - date_perf[0]["value"]
        commit_count = len(commits)
        efficiency = perf_diff / commit_count if commit_count else 0
        

        conn = sqlite3.connect("commits.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding FROM commits WHERE id IN ({})".format(
                ",".join(["?"]*len(commits))
            ),
            [c["commit_id"] for c in commits]
        )
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
        avg_embedding = np.mean(embeddings, axis=0).tolist() if embeddings else []
        conn.close()
        
        efficiency_results.append({
            "thought": thought_info["thought"],
            "embedding": avg_embedding,
            "efficiency": efficiency,
            "date_range": f"{min_date} to {max_date}",
            "commit_count": commit_count
        })
    
    return efficiency_results

def save_efficiency_embeddings(
    efficiency_data: List[Dict],
    db_path: str = "efficiency.db"
) -> str:

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS efficiency (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thought TEXT,
        embedding BLOB,
        efficiency REAL,
        date_range TEXT,
        commit_count INTEGER
    )
    """)
    
    for item in efficiency_data:
        cursor.execute(
            "INSERT INTO efficiency VALUES (NULL, ?, ?, ?, ?, ?)",
            (
                item["thought"],
                sqlite3.Binary(np.array(item["embedding"]).tobytes()),
                item["efficiency"],
                item["date_range"],
                item["commit_count"]
            )
        )
    
    conn.commit()
    conn.close()
    return db_path

def knowledge_learning():
    # Step 1: Extract top words from textbook
    top_words = extract_top_words("textbook.pdf")
    
    # Step 2: Find nearest commits for each word
    word_results = find_nearest_commits(top_words)
    save_to_json(word_results, "word_matches.json")
    
    # Step 3: Cluster all commits
    cluster_results = cluster_commits()
    save_to_json(cluster_results, "clusters.json")
    
    # Step 4: Generate thoughts for both
    word_thoughts = generate_thoughts(word_results)
    cluster_thoughts = generate_thoughts(cluster_results)
    
    # Step 5: Calculate efficiency
    final_output = {
        "word_based": word_thoughts,
        "cluster_based": cluster_thoughts
    }
    combined_thoughts = {**word_thoughts, **cluster_thoughts}
    efficiency_data = calculate_efficiency(combined_thoughts)

    embedding_db_path = save_efficiency_embeddings(efficiency_data)
   
    return embedding_db_path
    
if __name__ == "__main__":
    main()

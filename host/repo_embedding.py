import sqlite3
from llm.llm_components import get_qwen_embedding
from typing import List
import numpy as np

def update_commit_embeddings(db_path: str = "commits.db", batch_size: int = 10):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    

    cursor.execute("PRAGMA table_info(commits)")
    columns = [col[1] for col in cursor.fetchall()]
    if "embedding" not in columns:
        cursor.execute("ALTER TABLE commits ADD COLUMN embedding BLOB")

    cursor.execute("SELECT id, message FROM commits WHERE embedding IS NULL")
    commits = cursor.fetchmany(batch_size)
    
    while commits:

        embeddings = []
        for commit_id, message in commits:
            try:
                embedding = get_qwen_embedding(message)
                embeddings.append((sqlite3.Binary(embedding.tobytes()), commit_id))
            except Exception as e:
                print(f"Error processing commit {commit_id}: {str(e)}")
                continue
        
        cursor.executemany(
            "UPDATE commits SET embedding = ? WHERE id = ?",
            embeddings
        )
        conn.commit()
        
        commits = cursor.fetchmany(batch_size)
    
    conn.close()

def get_commit_embeddings(db_path: str = "commits.db") -> List[np.ndarray]:

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT embedding FROM commits WHERE embedding IS NOT NULL")
    embeddings = [
        np.frombuffer(blob[0], dtype=np.float32) 
        for blob in cursor.fetchall()
    ]
    
    conn.close()
    return embeddings

def repo_embedding():
    update_commit_embeddings()
    embeddings = get_commit_embeddings()
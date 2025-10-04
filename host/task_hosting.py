from flask import Flask, jsonify
from threading import Thread
from self_check import self_check
from get_repo import get_repo
from repo_embedding import repo_embedding
from knowledge_learning import knowledge_learning
from task_initializing import task_initializing

app = Flask(__name__)

tasks = []
current_task_index = 0
embedding_db_path = ""
config = {}

def init_resources():
    global config, embedding_db_path, tasks
    
    config = self_check()
    repo_path = get_repo(config)
    embedding_db_path = repo_embedding(config, repo_path)
    knowledge_learning()
    tasks = task_initializing()

@app.route('/api/task', methods=['GET'])
def get_task():
    global current_task_index
    
    if current_task_index >= len(tasks):
        return jsonify({"error": "No more tasks"}), 404
    
    task = tasks[current_task_index]
    current_task_index += 1
    
    return jsonify({
        "task": task,
        "embedding_db_path": embedding_db_path,
        "config": config
    })

def run_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

def main():
    init_resources()
    
    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    print("HTTP server started at http://localhost:5000")
    server_thread.join()

if __name__ == "__main__":
    main()

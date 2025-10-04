from self_check import self_check
from get_repo import get_repo
from repo_embedding import repo_embedding
from knowledge_learning import knowledge_learning

def main():

    config = self_check()

    repo_path = get_repo(config)

    repo_embedding_db_path = repo_embedding(config, repo_path)

    embedding_db_path = knowledge_learning()

    task_db = task_initializing()

# please use task_hosting.py to call RISC-V device for device solving progresses 

def report_gen():
    task_final_evaluation()

    gen_report()



if __name__ == "__main__":
    main()

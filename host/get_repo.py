import sqlite3
import git
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

FILTER_KEYWORDS = [
    "Merge", "merge",
    "bugfix", "fix",
    "prune",
    "tests",
    "temp"
]

def clone_and_parse_repo(repo_url: str, 
                        db_path: str = "commits.db",
                        github_token: str = None):

    if github_token:
        parsed = urlparse(repo_url)
        auth_url = f"{parsed.scheme}://{github_token}@{parsed.netloc}{parsed.path}"
    else:
        auth_url = repo_url

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = f"./repos/{repo_name}"
    git.Repo.clone_from(auth_url, repo_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS commits (
        id TEXT PRIMARY KEY,
        message TEXT,
        diff TEXT,
        date TEXT,
        author TEXT
    )
    """)

    repo = git.Repo(repo_path)
    for commit in repo.iter_commits():
        filtered_message = "\n".join(
            line for line in commit.message.splitlines() 
            if not any(keyword.lower() in line.lower() for keyword in FILTER_KEYWORDS)
        )
            
        diff = commit.diff(commit.parents[0] if commit.parents else None)
        diff_text = "\n".join([d.diff.decode("utf-8", errors="replace") for d in diff])

        cursor.execute("""
        INSERT OR IGNORE INTO commits VALUES (?, ?, ?, ?, ?)
        """, (
            commit.hexsha,
            filtered_message,
            diff_text,
            datetime.fromtimestamp(commit.committed_date).isoformat(),
            commit.author.name
        ))

    conn.commit()
    conn.close()
    return repo_path

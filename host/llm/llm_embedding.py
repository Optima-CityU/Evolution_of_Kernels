import sqlite3
import numpy as np
from dashscope import Generation, Client

Client.api_key = 'your_dashscope_api_key'

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'Successfully connected to the SQLite database {db_file}')
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):

    sql_create_vectors_table = """
    CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY,
        key TEXT NOT NULL,
        vector BLOB NOT NULL
    );
    """
    try:
        c = conn.cursor()
        c.execute(sql_create_vectors_table)
    except sqlite3.Error as e:
        print(e)

def insert_vector(conn, key, vector):

    sql = ''' INSERT INTO vectors(key,vector)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (key, vector.tobytes()))
    conn.commit()
    return cur.lastrowid

def get_embedding(text):

    response = Generation.call(model='qwen3-embedding', input=text)
    if response.status_code == 200:
        vector = np.array(response.output.generations[0].text.split(), dtype=np.float32)
        return vector
    else:
        raise Exception(f"Failed to get embedding: {response.text}")

def main(data_dict):

    db_path = 'vectors.db'
    conn = create_connection(db_path)
    
    if conn is not None:
        create_table(conn)
        
        for key, value in data_dict.items():
            try:
                vector = get_embedding(value)
                insert_vector(conn, key, vector)
            except Exception as e:
                print(f"Error processing '{key}': {e}")
        
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()




import os
import sqlite3

def query_memories():
    db_path = os.path.abspath('memory.db')
    print(f"Accessing database at: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM memory')
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(row)
        else:
            print("No memories found.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    query_memories() 
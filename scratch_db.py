import sqlite3
import os

def search_db(db_path):
    print(f"Searching database: {db_path}")
    if not os.path.exists(db_path):
        print("Path does not exist")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            text_cols = [c[1] for c in columns if 'text' in str(c[2]).lower() or 'char' in str(c[2]).lower() or 'desc' in str(c[1]).lower() or 'content' in str(c[1]).lower()]
            for col in text_cols:
                try:
                    cursor.execute(f'SELECT * FROM {table_name} WHERE "{col}" LIKE "%quotient rule%"')
                    rows = cursor.fetchall()
                    if rows:
                        print(f"Found matching rows in Table [{table_name}] Col [{col}]:")
                        for r in rows:
                            print(r)
                except Exception as e:
                    pass
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()

search_db('../eddva_backend/db.sqlite')
search_db('../eddva_backend/database.sqlite')
search_db('db.sqlite3')

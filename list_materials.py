import sqlite3
import os

def list_materials(db_path):
    print(f"\n--- Checking database: {db_path} ---")
    if not os.path.exists(db_path):
        print("Path does not exist")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        if 'study_materials' not in tables:
            print("No study_materials table here")
            return
        
        cursor.execute("SELECT id, title, type, description FROM study_materials ORDER BY created_at DESC LIMIT 5")
        rows = cursor.fetchall()
        for r in rows:
            print("ID:", r[0])
            print("Title:", r[1])
            print("Type:", r[2])
            print("Desc length:", len(r[3]) if r[3] else 0)
            print("Desc preview:", r[3][:300] if r[3] else "")
            print("-" * 40)
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()

list_materials('../eddva_backend/db.sqlite')
list_materials('../eddva_backend/database.sqlite')
list_materials('db.sqlite3')

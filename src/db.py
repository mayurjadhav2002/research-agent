import sqlite3
import os

DB_PATH = "db/research.db"
os.makedirs("db", exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT
            )
        """)
        conn.commit()

def paper_exists(url: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute("SELECT 1 FROM papers WHERE url = ?", (url,)).fetchone()
        return result is not None

def save_paper(url: str, title: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR IGNORE INTO papers (url, title) VALUES (?, ?)", (url, title))
        conn.commit()

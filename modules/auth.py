# modules/auth.py
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path("users.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_users_db():
    """Initialize the users table if it doesn't exist."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )"""
    )
    conn.commit()
    conn.close()

def create_user(username: str, password: str) -> bool:
    """Create a new user with hashed password. Returns True if success, False if username exists."""
    conn = get_connection()
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def check_user(username: str, password: str) -> bool:
    """Check if a user exists and password is correct."""
    conn = get_connection()
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result is not None

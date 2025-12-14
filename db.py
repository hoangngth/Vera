"""Database helpers for storing and fetching conversations.

Uses environment variables for configuration; supports loading a
.env.local file for local secrets (if python-dotenv is installed).
"""

from typing import Dict, List
import os

import psycopg
from psycopg.rows import dict_row
from colorama import Fore

# Optionally load environment variables from .env files if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    # Load local overrides first, then fallback to .env
    load_dotenv('.env.local', override=True)
    load_dotenv()
except Exception:
    pass


# Read DB settings from environment with sensible placeholders
DB_PARAMS: Dict[str, object] = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'dbname': os.environ.get('DB_NAME') or 'your_db_name',
    'user': os.environ.get('DB_USER') or 'your_db_user',
    'password': os.environ.get('DB_PASS') or 'change_this_password',
}


def connect_db():
    """Return a new psycopg connection using `DB_PARAMS`."""
    return psycopg.connect(**DB_PARAMS)


def fetch_conversations() -> List[Dict]:
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT id, prompt, response FROM conversations;")
        rows = cursor.fetchall()
    conn.close()
    print(Fore.BLUE + f'Fetched {len(rows)} conversations from the database.')
    if rows:
        print(Fore.BLUE + f'Last conversation: {rows[-1]} \n')
    return rows


def store_conversation(prompt: str, response: str) -> None:
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (prompt, response) VALUES (%s, %s);",
            (prompt, response),
        )
        conn.commit()
    conn.close()


def remove_last_conversation() -> None:
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations);"
        )
        conn.commit()
    conn.close()

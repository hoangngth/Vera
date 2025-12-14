import os
import psycopg
from psycopg.rows import dict_row
from colorama import Fore
from dotenv import load_dotenv

load_dotenv()


# Read DB settings from environment
DB_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'dbname': os.environ.get('DB_NAME') or 'your_db_name',
    'user': os.environ.get('DB_USER') or 'your_db_user',
    'password': os.environ.get('DB_PASS') or 'change_this_password',
}


def connect_db():
    return psycopg.connect(**DB_PARAMS)


def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT id, prompt, response FROM conversations;")
        rows = cursor.fetchall()
    conn.close()
    print(Fore.BLUE + f'Fetched {len(rows)} conversations from the database.')
    return rows


def store_conversation(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (prompt, response) VALUES (%s, %s);",
            (prompt, response),
        )
        conn.commit()
    conn.close()


def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations);"
        )
        conn.commit()
    conn.close()

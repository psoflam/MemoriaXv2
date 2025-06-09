import re
import sqlite3
from datetime import datetime
from typing import Optional, List
import os

# === FACT ENGINE CORE ===

# Define the path for the fact memory database
FACT_DB_PATH = os.path.join(os.getcwd(), "memoriax2", "data", "fact_memory.db")

# Debug print to verify the database path
print("Attempting to write to:", FACT_DB_PATH)

# Initialize fact DB if needed
def init_fact_db():
    conn = sqlite3.connect(FACT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS facts (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                      )''')
    conn.commit()
    return conn

# Store a fact by key (upsert)
def store_fact(key: str, value: str):
    conn = sqlite3.connect(FACT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO facts (key, value, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, last_updated=excluded.last_updated
    """, (key, value, datetime.utcnow()))
    conn.commit()
    conn.close()

# Lookup a fact

def lookup_fact(key: str) -> Optional[str]:
    conn = sqlite3.connect(FACT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM facts WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# List all facts (for debug)
def list_facts() -> List[tuple]:
    conn = sqlite3.connect(FACT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM facts ORDER BY last_updated DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# === EXTRACTION LOGIC ===

def extract_user_facts(user_input: str):
    name_match = re.search(r"my name is (\w+)|i(?:'m| am) (\w+)", user_input, re.I)
    if name_match:
        name = name_match.group(1) or name_match.group(2)
        if name:
            store_fact("user_name", name.strip())

    location_match = re.search(r"i (?:live|am from|love|like) (in |at )?(\w+)", user_input, re.I)
    if location_match:
        location = location_match.group(2)
        if location:
            store_fact("favorite_location", location.strip())

    emotion_match = re.findall(r"\b(sad|happy|anxious|nostalgic|lonely|angry|hopeful)\b", user_input, re.I)
    for emotion in set(emotion_match):
        prev = lookup_fact("emotion_history") or ""
        updated = ",".join(set(prev.split(",") + [emotion.lower()]))
        store_fact("emotion_history", updated)

def extract_name(text):
    """
    Extracts a name from the given text using regex.
    """
    match = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+)", text, re.I)
    return match.group(1) if match else "Unknown"

# === CONTEXT-AWARE FALLBACK LOGIC ===

def respond_with_fact_if_available(user_input: str, retrieved_memories: List[str]) -> Optional[str]:
    lower_input = user_input.lower()

    if "your name" in lower_input and not retrieved_memories:
        name = lookup_fact("user_name")
        if name:
            return f"Yes, of course! Your name is {name}, isn't it?"

    if "where am i from" in lower_input or "my favorite place" in lower_input:
        loc = lookup_fact("favorite_location")
        if loc:
            return f"You mentioned once that your favorite place is {loc}. Did I get that right?"

    return None

# === HOOK TO INTEGRATE ===
# In main chatbot loop:
# extract_user_facts(user_input)
# fallback_response = respond_with_fact_if_available(user_input, retrieved_memories)
# if fallback_response: return fallback_response

# Call this during startup
init_fact_db() 
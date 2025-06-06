import sqlite3

# Connect to the database
conn = sqlite3.connect('memory.db')
cursor = conn.cursor()

# Function to dump all memory entries
def dump_memory_entries():
    cursor.execute("SELECT value, type, emotion FROM memory")
    memories = cursor.fetchall()
    if not memories:
        print("No memories found.")
        return

    print("Memory Entries:")
    for mem in memories:
        value, mem_type, emotion = mem
        print(f"- Value: {value}, Type: {mem_type}, Emotion: {emotion}")

# Run the memory dump function if this script is executed
if __name__ == "__main__":
    dump_memory_entries()

# Close the database connection
conn.close() 
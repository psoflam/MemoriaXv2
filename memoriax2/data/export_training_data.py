import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('memory.db')
cursor = conn.cursor()

# Query to fetch memory data
cursor.execute('''
    SELECT user_input, context, emotion, memory_type, response
    FROM memory_data
    WHERE quality_score > 0.8
    LIMIT 1000
''')

# Fetch all rows
rows = cursor.fetchall()

# Structure the data
training_data = [
    {
        "input": row[0],
        "context": row[1],
        "emotion": row[2],
        "type": row[3],
        "response": row[4]
    }
    for row in rows
]

# Write to JSON file
with open('data/train.json', 'w') as f:
    json.dump(training_data, f, indent=4)

# Close the database connection
conn.close() 
import sqlite3

# Connect to the database
conn = sqlite3.connect('memory.db')

# Create a cursor object
cursor = conn.cursor()

# Execute a query to select all data from the memory table
cursor.execute("SELECT * FROM memory")

# Fetch all rows from the executed query
rows = cursor.fetchall()

# Print the contents of the memory table
print("Contents of the memory table:")
for row in rows:
    print(row)

# Close the connection
conn.close() 
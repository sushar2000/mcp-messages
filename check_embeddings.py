import pyodbc
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

db = config['database']
conn_str = f"DRIVER={{SQL Server}};SERVER={db['host']},{db['port']};DATABASE={db['dbname']};UID={db['user']};PWD={db['password']};TrustServerCertificate=yes;"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM KKChat WHERE embedding IS NOT NULL')
with_embeddings = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM KKChat')
total = cursor.fetchone()[0]

print(f'Total messages: {total}')
print(f'Messages with embeddings: {with_embeddings}')

# Show a sample of messages with embeddings
cursor.execute(
    'SELECT TOP 3 message_id, message_text FROM KKChat WHERE embedding IS NOT NULL')
rows = cursor.fetchall()
print('\nSample messages with embeddings:')
for row in rows:
    print(f'{row.message_id}: {row.message_text[:50]}...')

conn.close()

import pyodbc
import json
from colors import Colors, success, info

# Load config
with open('config.json') as f:
    config = json.load(f)

db = config['database']
conn_str = f"DRIVER={{SQL Server}};SERVER={db['host']},{db['port']};DATABASE={db['dbname']};UID={db['user']};PWD={db['password']};TrustServerCertificate=yes;"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

cursor.execute(
    f'SELECT COUNT(*) FROM {db["table_name"]} WHERE embedding IS NOT NULL')
with_embeddings = cursor.fetchone()[0]

cursor.execute(f'SELECT COUNT(*) FROM {db["table_name"]}')
total = cursor.fetchone()[0]

print(
    f'Total messages: {Colors.CYAN}{total}{Colors.RESET}')
print(
    f'Messages with embeddings: {Colors.CYAN}{with_embeddings}{Colors.RESET}')

# Show coverage percentage
coverage = (with_embeddings / total * 100) if total > 0 else 0
print(f'\nEmbedding coverage: {Colors.GREEN}{coverage:.1f}%{Colors.RESET}')

# Show a sample of messages with embeddings
cursor.execute(
    f'SELECT TOP 3 message_id, message_text FROM {db["table_name"]} WHERE embedding IS NOT NULL')
rows = cursor.fetchall()
print(f'\n{Colors.YELLOW}Sample messages with embeddings:{Colors.RESET}')
for row in rows:
    print(
        f'{Colors.CYAN}{row.message_id}{Colors.RESET}: {row.message_text[:50]}...')

conn.close()

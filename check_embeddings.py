# Script to check embedding coverage in the database
import pyodbc
import json
from colors import Colors

# Load config
with open('config.json') as f:
    config = json.load(f)


def load_config(config_file='config.json'):
    """Load database configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        print("Please create a config.json file with database connection details")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return None


# Load configuration
config = None
server = None
database = None
user = None
password = None
table_name = None
conn_str = None


def initialize_config(config_file='config.json'):
    """Initialize global configuration variables"""
    global config, server, database, user, password, table_name, conn_str

    config = load_config(config_file)
    if not config:
        return False

    # Database connection details from config
    db_config = config.get('database', {})
    server = db_config.get('host')
    port = db_config.get('port')
    database = db_config.get('db_name')
    table_name = db_config.get('table_name')
    user = db_config.get('user')
    password = db_config.get('password')

    conn_str = (
        f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};'
        f'UID={user};PWD={password};'
    )
    return True


if __name__ == '__main__':

    if not initialize_config():
        exit(1)
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    cursor.execute(
        f'SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL')
    with_embeddings = cursor.fetchone()[0]

    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
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
        f'SELECT TOP 3 message_id, message_text FROM {table_name} WHERE embedding IS NOT NULL')
    rows = cursor.fetchall()
    print(f'\n{Colors.YELLOW}Sample messages with embeddings:{Colors.RESET}')
    for row in rows:
        print(
            f'{Colors.CYAN}{row.message_id}{Colors.RESET}: {row.message_text[:50]}...')

    conn.close()

# This script loads KKChat.tsv into the KKChat table in WhatsAppDB using pyodbc
import pyodbc
import json
import os


def load_config(config_file='config.json'):
    """Load database configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return None


# Load configuration
config = load_config()
if not config:
    print("Failed to load configuration. Exiting.")
    exit(1)

# Database connection details from config
db_config = config.get('database', {})
server = db_config.get('host')
port = db_config.get('port', 1433)
database = db_config.get('dbname')
username = db_config.get('user')
password = db_config.get('password')
table_name = db_config.get('table_name')

# Use SQL Authentication with the configured credentials
conn_str = (
    f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};'
    f'UID={username};PWD={password};'
)


def create_db_and_table():
    # Connect to master to create DB if needed
    master_conn_str = (
        f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE=master;'
        f'UID={username};PWD={password};'
    )
    with pyodbc.connect(master_conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"IF DB_ID('{database}') IS NULL CREATE DATABASE {database};")
        conn.commit()

    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            IF OBJECT_ID('{table_name}', 'U') IS NULL
            CREATE TABLE {table_name} (
                DateColumn DATE,
                SenderColumn NVARCHAR(100),
                MessageColumn NVARCHAR(MAX)
            );
        ''')
        conn.commit()


def load_tsv_to_table(tsv_path):
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        count = 0
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue  # skip malformed lines
                date, sender, message = parts
                cursor.execute(
                    f"INSERT INTO {table_name} (DateColumn, SenderColumn, MessageColumn) VALUES (?, ?, ?)",
                    date, sender, message
                )
                count += 1
        conn.commit()
        print(f"Total {count} rows inserted.")


if __name__ == '__main__':
    # Get TSV file path from config or use default
    tsv_file_path = config.get('file_path')

    create_db_and_table()
    load_tsv_to_table(tsv_file_path)
    print('Data loaded successfully.')

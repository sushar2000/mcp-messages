# This script loads KKChat.tsv into the KKChat table in WhatsAppDB using pyodbc
import pyodbc
import json
import os

from compare_data import Colors


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
                message_id BIGINT IDENTITY PRIMARY KEY,
                message_datetime DATETIME NOT NULL,
                message_sender NVARCHAR(100) NOT NULL,
                message_text NVARCHAR(MAX) NULL,
                embedding VARBINARY(MAX) NULL
            );
        ''')
        conn.commit()


def load_tsv_to_table(tsv_path):
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        count = 0
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Print after inserting 100 rows
                if count > 0 and count % 100 == 0:
                    print(f"{count} rows inserted...")
                    conn.commit()

                parts = line.strip().split('\t')
                if len(parts) < 3:
                    print(
                        f"{Colors.RED}Skipping malformed line: {line.strip()}{Colors.RESET}")
                    continue  # skip malformed lines
                date = parts[0]
                sender = parts[1]
                if len(parts) > 3:
                    message = '\t'.join(parts[2:])
                else:
                    message = parts[2]

                cursor.execute(
                    f"INSERT INTO {table_name} (message_datetime, message_sender, message_text) VALUES (?, ?, ?)",
                    date, sender, message
                )
                count += 1
        conn.commit()
        print(f"Total {count} rows inserted.")
        return count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Load all TSV files from config into database')
    parser.add_argument('-c', '--config', default='config.json',
                        help='Path to configuration file (default: config.json)')
    args = parser.parse_args()

    # Reload config if different config file specified
    if args.config != 'config.json':
        config = load_config(args.config)
        if not config:
            print("Failed to load configuration. Exiting.")
            exit(1)

    # Get all files from config
    file_paths = config.get('file_paths', [])

    if not file_paths:
        print("Error: No file paths found in config file")
        print("Please add 'file_paths' array to your config.json")
        exit(1)

    print(f"Loading {len(file_paths)} files from config...")
    create_db_and_table()

    total_rows = 0
    for i, tsv_file_path in enumerate(file_paths, 1):
        print(f"\nProcessing file {i}/{len(file_paths)}: {tsv_file_path}")
        try:
            rows_inserted = load_tsv_to_table(tsv_file_path)
            total_rows += rows_inserted
            print(
                f"Successfully loaded {rows_inserted} rows from {tsv_file_path}")
        except Exception as e:
            print(f"Error loading {tsv_file_path}: {e}")

    print(f"\nTotal rows loaded from all files: {total_rows}")
    print('All data loaded successfully.')

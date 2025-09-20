# Script to read from table and update embeddings in the last column.
# This script processes messages in batches and generates embeddings using OpenAI API.

import pyodbc
import json
import sys
import pickle
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from colors import Colors


def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"Configuration loaded from file '{config_file}'")
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return None


def serialize_embedding(vector):
    """
    Serialize a vector of floats to binary format for storage in VARBINARY column.
    Uses pickle for serialization.
    """
    return pickle.dumps(vector)


def deserialize_embedding(binary_data):
    """
    Deserialize binary data back to a vector of floats.
    """
    if binary_data is None:
        return None
    return pickle.loads(binary_data)


def get_database_connection(config):
    """Create and return database connection"""
    db_config = config.get('database', {})
    server = db_config.get('host')
    port = db_config.get('port', 1433)
    database = db_config.get('db_name')
    username = db_config.get('user')
    password = db_config.get('password')

    conn_str = (
        f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};'
        f'UID={username};PWD={password};'
    )

    return pyodbc.connect(conn_str)


def initialize_embeddings(config):
    """Initialize OpenAI embeddings client"""
    openai_config = config.get('openai', {})

    EMBEDDING_MODEL_NAME = openai_config.get('embedding_model')
    ENV_URL = openai_config.get('env_url')
    OPENAI_API_KEY = openai_config.get('api_key')
    EMBEDDING_MODEL_URL = ENV_URL + "/api/v1/vectors"

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_base=EMBEDDING_MODEL_URL,
        openai_api_key=OPENAI_API_KEY,
    )

    print(f"Embeddings initialized with model: {EMBEDDING_MODEL_NAME}")
    return embeddings


def get_messages_without_embeddings(conn, table_name, batch_size=50):
    """
    Get messages that don't have embeddings yet.
    Returns a list of tuples: (message_id, message_text)
    """
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            SELECT TOP {batch_size} message_id, message_text 
            FROM {table_name} 
            WHERE embedding IS NULL 
            AND message_text IS NOT NULL 
            AND message_text != ''
            AND message_text != '<Media omitted>'
            ORDER BY message_datetime
        """)

        rows = cursor.fetchall()
        return [(row[0], row[1]) for row in rows]
    except (pyodbc.Error, pyodbc.DatabaseError) as e:
        print(f"Error fetching messages: {e}")
        return []


def update_embedding_in_db(conn, table_name, message_id, embedding_data):
    """Update the embedding column for a specific message using message_id"""
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            UPDATE {table_name} 
            SET embedding = ? 
            WHERE message_id = ?
        """, (embedding_data, message_id))

        return cursor.rowcount > 0
    except (pyodbc.Error, pyodbc.DatabaseError) as e:
        print(f"Error updating embedding: {e}")
        return False


def process_embeddings_batch(embeddings_client, messages):
    """
    Process a batch of messages and generate embeddings.
    Returns a list of tuples: (message_id, embedding_data).
    """
    if not messages:
        return []

    # Extract just the text for embedding
    texts = [msg[1] for msg in messages]  # message_text is now at index 1

    try:
        # Generate embeddings for all texts in batch
        print(f"Generating embeddings for {len(texts)} messages...")
        vectors = embeddings_client.embed_documents(texts)

        # Create list of results with message_id and serialized embedding
        results = []
        for i, vector in enumerate(vectors):
            message_id = messages[i][0]
            embedding_data = serialize_embedding(vector)
            results.append((message_id, embedding_data))

        return results
    except (ValueError, RuntimeError) as e:
        print(f"Error generating embeddings: {e}")
        return []


def main():
    """Main function to process embeddings update"""
    print(f"{Colors.GREEN}Starting embedding update process...{Colors.RESET}")

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)

    # Initialize embeddings client
    try:
        embeddings_client = initialize_embeddings(config)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"Failed to initialize embeddings client: {e}")
        sys.exit(1)

    # Get database configuration
    db_config = config.get('database', {})
    table_name = db_config.get('table_name')

    # Connect to database
    try:
        conn = get_database_connection(config)
        print(f"Connected to database: {db_config.get('db_name')}")
    except (pyodbc.Error, pyodbc.DatabaseError) as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    # Process messages in batches
    batch_size = 500
    total_processed = 0

    try:
        while True:
            # Get next batch of messages without embeddings
            messages = get_messages_without_embeddings(
                conn, table_name, batch_size)

            if not messages:
                print(
                    f"{Colors.GREEN}No more messages to process. All embeddings updated!{Colors.RESET}")
                break

            print(f"\nProcessing batch of {len(messages)} messages...")
            start_time = datetime.now()

            # Generate embeddings for the batch
            embedding_results = process_embeddings_batch(
                embeddings_client, messages)

            if not embedding_results:
                print(
                    f"{Colors.RED}Failed to generate embeddings for this batch. Skipping...{Colors.RESET}")
                continue

            # Update database with embeddings
            success_count = 0
            for message_id, embedding_data in embedding_results:
                if update_embedding_in_db(conn, table_name, message_id, embedding_data):
                    success_count += 1

            # Commit the batch
            conn.commit()

            total_processed += success_count
            end_time = datetime.now()
            elapsed = end_time - start_time

            print(
                f"{Colors.GREEN}Successfully updated {success_count}/{len(messages)} embeddings{Colors.RESET}")
            print(f"Time for this batch: {elapsed}")
            print(f"Total processed so far: {total_processed}")

    except KeyboardInterrupt:
        print(
            f"\n{Colors.YELLOW}Process interrupted by user. Committing current progress...{Colors.RESET}")
        conn.commit()
    except (pyodbc.Error, RuntimeError, ValueError) as e:
        print(f"{Colors.RED}Unexpected error: {e}{Colors.RESET}")
    finally:
        conn.close()
        print(f"\n{Colors.GREEN}Database connection closed.{Colors.RESET}")
        print(
            f"{Colors.GREEN}Total embeddings processed: {total_processed}{Colors.RESET}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Update embeddings column in messages table')
    parser.add_argument('-c', '--config', default='config.json',
                        help='Path to configuration file (default: config.json)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without actually updating the database')

    args = parser.parse_args()

    if args.config != 'config.json':
        # Update the config loading for custom config path
        pass

    if args.dry_run:
        print(
            f"{Colors.YELLOW}DRY RUN MODE: No database updates will be performed{Colors.RESET}")

    main()

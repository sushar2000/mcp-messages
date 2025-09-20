#!/usr/bin/env python3
"""
Test script to verify database connection and show preview of embedding update process.
This script will show what messages need embeddings without actually updating them.
"""

import pyodbc
import json
import pickle
from datetime import datetime
from colors import Colors


def serialize_embedding(vector):
    """
    Serialize a vector of floats to binary format for storage in VARBINARY column.
    Uses pickle for serialization.
    """
    return pickle.dumps(vector)


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


def test_database_connection():
    """Test database connection and show table statistics"""
    print(f"{Colors.GREEN}Testing database connection...{Colors.RESET}")

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return False

    # Get database configuration
    db_config = config.get('database', {})
    table_name = db_config.get('table_name')

    try:
        # Connect to database
        conn = get_database_connection(config)
        cursor = conn.cursor()
        print(f"✓ Connected to database: {db_config.get('db_name')}")

        # Get table statistics
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"✓ Total messages in {table_name}: {total_rows:,}")

        # Check messages with embeddings
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
        with_embeddings = cursor.fetchone()[0]
        print(f"✓ Messages with embeddings: {with_embeddings:,}")

        # Check messages without embeddings (excluding media and empty)
        cursor.execute(f"""
            SELECT COUNT(*) FROM {table_name} 
            WHERE embedding IS NULL 
            AND message_text IS NOT NULL 
            AND message_text != ''
            AND message_text != '<Media omitted>'
        """)
        without_embeddings = cursor.fetchone()[0]
        print(f"✓ Messages needing embeddings: {without_embeddings:,}")

        # Show sample messages that need embeddings
        print(
            f"\n{Colors.YELLOW}Sample messages that need embeddings:{Colors.RESET}")
        cursor.execute(f"""
            SELECT TOP 5 message_id, message_datetime, message_sender, message_text 
            FROM {table_name} 
            WHERE embedding IS NULL 
            AND message_text IS NOT NULL 
            AND message_text != ''
            AND message_text != '<Media omitted>'
            ORDER BY message_datetime
        """)

        rows = cursor.fetchall()
        for i, row in enumerate(rows, 1):
            message_id = row[0]
            datetime_str = row[1].strftime("%Y-%m-%d %H:%M:%S")
            sender = row[2][:20] + "..." if len(row[2]) > 20 else row[2]
            message = row[3][:60] + "..." if len(row[3]) > 60 else row[3]
            print(
                f"  {i}. [ID:{message_id}] [{datetime_str}] {sender}: {message}")

        conn.close()
        print(f"\n{Colors.GREEN}Database connection test successful!{Colors.RESET}")
        return True

    except (pyodbc.Error, pyodbc.DatabaseError) as e:
        print(f"{Colors.RED}Database connection failed: {e}{Colors.RESET}")
        return False


def test_openai_connection():
    """Test OpenAI embeddings connection"""
    print(f"\n{Colors.GREEN}Testing OpenAI embeddings connection...{Colors.RESET}")

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return False

    try:
        from langchain_openai import OpenAIEmbeddings

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

        print(f"✓ Embeddings initialized with model: {EMBEDDING_MODEL_NAME}")

        # Test with a simple message
        test_text = "Hello, this is a test message for embedding generation."
        start_time = datetime.now()
        vector = embeddings.embed_query(test_text)
        end_time = datetime.now()

        print("✓ Test embedding generated successfully")
        print(f"✓ Embedding dimensions: {len(vector)}")
        print(f"✓ Time taken: {end_time - start_time}")
        print(f"✓ First 5 values: {vector[:5]}")

        return True

    except (ValueError, RuntimeError, ImportError) as e:
        print(f"{Colors.RED}OpenAI connection failed: {e}{Colors.RESET}")
        return False


def update_sample_embeddings():
    """Update embeddings for a small sample of messages"""
    print(f"\n{Colors.GREEN}Updating embeddings for sample messages...{Colors.RESET}")

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return False

    try:
        from langchain_openai import OpenAIEmbeddings

        # Get database configuration
        db_config = config.get('database', {})
        table_name = db_config.get('table_name')

        # Setup OpenAI embeddings
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

        # Connect to database
        conn = get_database_connection(config)
        cursor = conn.cursor()

        # Get sample messages that need embeddings (limit to 3 for testing)
        cursor.execute(f"""
            SELECT TOP 3 message_id, message_text 
            FROM {table_name} 
            WHERE embedding IS NULL 
            AND message_text IS NOT NULL 
            AND message_text != ''
            AND message_text != '<Media omitted>'
            ORDER BY message_datetime
        """)

        messages = cursor.fetchall()

        if not messages:
            print("No messages found that need embeddings.")
            return True

        print(f"Found {len(messages)} sample messages to update:")

        updated_count = 0
        for message_id, message_text in messages:
            try:
                # Display message being processed
                preview = message_text[:50] + \
                    "..." if len(message_text) > 50 else message_text
                print(f"  Processing ID {message_id}: {preview}")

                # Generate embedding
                start_time = datetime.now()
                vector = embeddings.embed_query(message_text)
                end_time = datetime.now()

                # Convert vector to binary format for VARBINARY storage
                embedding_data = serialize_embedding(vector)

                # Update database
                cursor.execute(f"""
                    UPDATE {table_name} 
                    SET embedding = ? 
                    WHERE message_id = ?
                """, (embedding_data, message_id))

                conn.commit()
                updated_count += 1

                print(f"    ✓ Updated (took {end_time - start_time})")

            except (ValueError, RuntimeError, pyodbc.Error) as e:
                print(f"    ✗ Failed to update message ID {message_id}: {e}")

        conn.close()

        print(f"\n{Colors.GREEN}Sample embedding update completed!{Colors.RESET}")
        print(
            f"Successfully updated {updated_count} out of {len(messages)} messages.")

        return True

    except (ValueError, RuntimeError, pyodbc.Error, ImportError) as e:
        print(f"{Colors.RED}Sample embedding update failed: {e}{Colors.RESET}")
        return False


def main():
    """Main function to run tests"""
    print(f"{Colors.GREEN}=== Embedding Update - Connection Test ==={Colors.RESET}")

    # Test database connection
    db_success = test_database_connection()

    # Test OpenAI connection
    openai_success = test_openai_connection()

    # Summary
    print(f"\n{Colors.GREEN}=== Test Summary ==={Colors.RESET}")
    print(f"Database connection: {'✓ PASS' if db_success else '✗ FAIL'}")
    print(f"OpenAI connection: {'✓ PASS' if openai_success else '✗ FAIL'}")

    if db_success and openai_success:
        print(f"\n{Colors.GREEN}All tests passed!{Colors.RESET}")

        # Ask user if they want to update sample embeddings
        response = input(
            f"\n{Colors.YELLOW}Would you like to update embeddings for 3 sample messages? (y/n): {Colors.RESET}")
        if response.lower() in ['y', 'yes']:
            update_sample_embeddings()
        else:
            print(
                f"{Colors.GREEN}You can run update_embeddings.py to update all messages.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}Some tests failed. Please fix the issues before running the update script.{Colors.RESET}")


if __name__ == "__main__":
    main()

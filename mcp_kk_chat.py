import json
import pyodbc
from mcp.server import Server
from mcp.types import Tool, TextContent
from openai import OpenAI
import asyncio
import os
import sys
import pickle
import numpy as np

# For MCP servers, we should minimize stdout output since it's used for protocol communication
# Use stderr for debug messages


def debug_print(message):
    """Print debug messages to stderr to avoid interfering with MCP protocol"""
    print(message, file=sys.stderr)


def deserialize_embedding(binary_data):
    """Deserialize binary data back to a vector of floats."""
    if binary_data is None:
        return None
    return pickle.loads(binary_data)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_file)

        with open(config_path, 'r', encoding='utf-8') as f:
            debug_print(f"Configuration loaded from file '{config_path}'")
            return json.load(f)
    except FileNotFoundError:
        debug_print(f"Error: Configuration file '{config_path}' not found")
        return None
    except json.JSONDecodeError as e:
        debug_print(f"Error: Invalid JSON in configuration file: {e}")
        return None


# Load configuration
debug_print(f"Script location: {os.path.abspath(__file__)}")
debug_print(f"Current working directory: {os.getcwd()}")
config = load_config()
if not config:
    debug_print("Failed to load configuration. Exiting.")
    exit(1)

# Get database configuration
db_config = config.get('database', {})
server_host = db_config.get('host')
port = db_config.get('port', 1433)
database = db_config.get('dbname')
username = db_config.get('user')
password = db_config.get('password')
table_name = db_config.get('table_name')

# Get OpenAI configuration
openai_config = config.get('openai', {})
api_key = openai_config.get('api_key')
env_url = openai_config.get('env_url')
model = openai_config.get('model')
embedding_model = openai_config.get('embedding_model')


# Initialize OpenAI client with configuration
if env_url:
    # Separate clients for LLM and embeddings
    client = OpenAI(
        base_url=f"{env_url}/api/v1/llm/",
        api_key=api_key
    )
    # Create a separate client for embeddings
    embedding_client = OpenAI(
        base_url=f"{env_url}/api/v1/vectors",
        api_key=api_key
    )
else:
    client = OpenAI(api_key=api_key)
    embedding_client = client  # Use same client for standard OpenAI API

# Create MCP server
server = Server("sql-messages")
debug_print("MCP server initialized successfully.")
debug_print(
    f"Configuration loaded - Database: {database}, OpenAI Model: {model}")

# Connect to SQL Server using configuration


def get_conn():
    conn_str = (
        f'DRIVER={{SQL Server}};SERVER={server_host},{port};DATABASE={database};'
        f'UID={username};PWD={password};TrustServerCertificate=yes;'
    )
    return pyodbc.connect(conn_str)


def test_database_connection():
    """Test database connection and return status"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        debug_print(
            f"[OK] Database connection successful. Found {count} messages in table '{table_name}'")
        return True
    except Exception as e:
        debug_print(f"[ERROR] Database connection failed: {e}")
        return False


# Define tools
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="keyword_search",
            description="Search messages using keyword matching",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to match in messages"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="semantic_search",
            description="Search messages using semantic similarity and provide AI-powered answers",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or query for semantic search"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    if name == "keyword_search":
        return await keyword_search(arguments.get("query"), arguments.get("top_k", 5))
    elif name == "semantic_search":
        return await semantic_search(arguments.get("query"), arguments.get("top_k", 5))
    else:
        raise ValueError(f"Unknown tool: {name}")


async def keyword_search(query: str, top_k: int = 5) -> list[TextContent]:
    """Search messages using keyword matching"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TOP (?) message_id, message_datetime, message_sender, message_text
            FROM {table_name}
            WHERE message_text LIKE '%' + ? + '%'
            ORDER BY message_datetime DESC
        """, (top_k, query))
        rows = cursor.fetchall()
        conn.close()

        results = [{"id": r.message_id, "date": str(
            r.message_datetime), "sender": r.message_sender, "message": r.message_text} for r in rows]

        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2)
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error performing keyword search: {str(e)}"
        )]


async def semantic_search(query: str, top_k: int = 5) -> list[TextContent]:
    """Search messages using semantic similarity and provide AI-powered answers"""
    try:
        # Embed query using the embedding client
        query_embedding = embedding_client.embeddings.create(
            model=embedding_model, input=query).data[0].embedding

        query_vector = np.array(query_embedding)

        conn = get_conn()
        cursor = conn.cursor()

        # Get all messages with embeddings (limited to reasonable number for performance)
        cursor.execute(f"""
            SELECT message_id, message_datetime, message_sender, message_text, embedding
            FROM {table_name}
            WHERE embedding IS NOT NULL
        """)
        rows = cursor.fetchall()
        conn.close()

        # Calculate similarities
        similarities = []
        for row in rows:
            try:
                # Deserialize the embedding
                stored_embedding = deserialize_embedding(row.embedding)
                if stored_embedding is not None:
                    stored_vector = np.array(stored_embedding)
                    similarity = cosine_similarity(query_vector, stored_vector)
                    similarities.append((similarity, row))
            except Exception as e:
                debug_print(
                    f"Error processing embedding for message {row.message_id}: {e}")
                continue

        # Sort by similarity (highest first) and take top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]

        # Build context from top results
        context = "\n".join(
            [f"[{result[1].message_datetime}, {result[1].message_sender}] {result[1].message_text}"
             for result in top_results])

        # RAG call
        prompt = f"Use these messages to answer:\n{context}\n\nQuestion: {query}"
        answer = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        result = {
            "results": context,
            "answer": answer.choices[0].message.content
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error performing semantic search: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    debug_print("Starting MCP server...")
    debug_print(f"Server name: sql-messages")
    debug_print(f"Database: {database} on {server_host}:{port}")
    debug_print(f"Table: {table_name}")
    debug_print(f"OpenAI Model: {model}")
    debug_print("Available tools: keyword_search, semantic_search")

    # Test database connection
    if test_database_connection():
        debug_print("MCP server is now running and ready to accept requests.")
    else:
        debug_print("MCP server started but database connection failed.")

    debug_print("-" * 50)

    # Import here to avoid issues with event loop
    from mcp.server.stdio import stdio_server

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except KeyboardInterrupt:
        debug_print("\nMCP server shutting down...")
    except Exception as e:
        debug_print(f"MCP server error: {e}")
    finally:
        debug_print("MCP server stopped.")


if __name__ == "__main__":
    asyncio.run(main())

# MCP Chat - An MCP server to provide AI-powered chat analysis and search on messages stored in a SQL Server database
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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            debug_print(f"Configuration loaded from file '{config_path}'")
            return json.load(f)
    except FileNotFoundError:
        debug_print(f"Error: Configuration file '{config_path}' not found")
        return None
    except json.JSONDecodeError as e:
        debug_print(f"Error: Invalid JSON in configuration file: {e}")
        return None


def load_nicknames(nicknames_file='nicknames.json'):
    """Load nicknames from JSON file and build reverse mapping"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nicknames_path = os.path.join(script_dir, nicknames_file)

    try:
        with open(nicknames_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            nicknames = data.get('nicknames', {})

            # Build reverse mapping dynamically
            reverse_mapping = {}
            for real_name, nicks in nicknames.items():
                for nick in nicks:
                    reverse_mapping[nick] = real_name

            debug_print(
                f"Nicknames loaded from file '{nicknames_path}' - {len(nicknames)} people, {len(reverse_mapping)} nicknames")
            return nicknames, reverse_mapping
    except FileNotFoundError:
        debug_print(
            f"Warning: Nicknames file '{nicknames_path}' not found. Using empty mappings.")
        return {}, {}
    except json.JSONDecodeError as e:
        debug_print(f"Error: Invalid JSON in nicknames file: {e}")
        return {}, {}


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
database = db_config.get('db_name')
table_name = db_config.get('table_name')
username = db_config.get('user')
password = db_config.get('password')

# Get OpenAI configuration
openai_config = config.get('openai', {})
api_key = openai_config.get('api_key')
llm_url = openai_config.get('llm_url')
model = openai_config.get('llm_model')
embedding_model = openai_config.get('embedding_model')
embedding_model_url = openai_config.get('embedding_model_url')

# Load nicknames
nicknames, nickname_reverse_mapping = load_nicknames()


# Initialize OpenAI client with configuration
if llm_url:
    # Separate clients for LLM and embeddings
    client = OpenAI(
        base_url=llm_url,
        api_key=api_key
    )
    # Create a separate client for embeddings
    embedding_client = OpenAI(
        base_url=embedding_model_url,
        api_key=api_key
    )
else:
    client = OpenAI(api_key=api_key)
    embedding_client = client  # Use same client for standard OpenAI API

# Create MCP server
server = Server("mcp-messages")
debug_print("MCP server initialized successfully.")
debug_print(
    f"Configuration loaded - Database: {database}, OpenAI Model: {model}")

# Nickname helper functions


def get_real_name(nickname_or_name):
    """Get the real name from a nickname or return the name if already real."""
    return nickname_reverse_mapping.get(nickname_or_name, nickname_or_name)


def get_nicknames_for_person(real_name):
    """Get all nicknames for a given real name."""
    return nicknames.get(real_name, [])


def normalize_sender_name(sender):
    """Normalize a sender name to the real name."""
    return get_real_name(sender)


def get_all_names_for_person(input_name):
    """Get all possible names (real name + nicknames) for a person."""
    real_name = normalize_sender_name(input_name)
    all_names = [real_name] + get_nicknames_for_person(real_name)
    return list(set(all_names))  # Remove duplicates

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
        ),
        Tool(
            name="message_stats",
            description="Get comprehensive statistics about messages and senders",
            inputSchema={
                "type": "object",
                "properties": {
                    "sender": {
                        "type": "string",
                        "description": "Optional: Get stats for specific sender"
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Optional: Date range (YYYY-MM-DD to YYYY-MM-DD)"
                    }
                }
            }
        ),
        Tool(
            name="timeline_analysis",
            description="Analyze message patterns over time for a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or keyword to track"
                    },
                    "grouping": {
                        "type": "string",
                        "description": "Time grouping: daily, weekly, monthly",
                        "default": "weekly"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="sentiment_analysis",
            description="Analyze sentiment of messages using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or topic to analyze sentiment for"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Optional: Specific sender to analyze"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of messages to analyze",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="advanced_search",
            description="Combined keyword, semantic, and filter search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Optional: Filter by sender"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional: Start date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional: End date (YYYY-MM-DD)"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Search type: keyword, semantic, or both",
                        "default": "both"
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
            name="database_health",
            description="Check database status, embedding coverage, and data quality",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_type": {
                        "type": "string",
                        "description": "Type of check: overview, embeddings, or duplicates",
                        "default": "overview"
                    }
                }
            }
        ),
        Tool(
            name="activity_patterns",
            description="Analyze user activity patterns by time (hourly, daily, weekly)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of pattern analysis: hourly, daily, weekly, or monthly",
                        "default": "hourly"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Optional: Analyze patterns for specific sender"
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Optional: Date range filter (last_week, last_month, last_year)"
                    }
                }
            }
        ),
        Tool(
            name="message_extremes",
            description="Find earliest or latest messages overall or for specific senders",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "description": "Type of query: first, last, earliest, latest, oldest, newest, old, new",
                        "default": "first"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Optional: Find extremes for specific sender"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of extreme messages to return",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="nickname_lookup",
            description="Look up nickname information for a person or get real name from nickname",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name or nickname to look up"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="nickname_list",
            description="List all people and their nicknames",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="nickname_search",
            description="Search for people by partial name or nickname match",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for in names and nicknames"
                    }
                },
                "required": ["pattern"]
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
    elif name == "message_stats":
        return await message_stats(arguments.get("sender"), arguments.get("date_range"))
    elif name == "timeline_analysis":
        return await timeline_analysis(arguments.get("topic"), arguments.get("grouping", "weekly"))
    elif name == "sentiment_analysis":
        return await sentiment_analysis(arguments.get("query"), arguments.get("sender"), arguments.get("top_k", 10))
    elif name == "advanced_search":
        return await advanced_search(
            arguments.get("query"),
            arguments.get("sender"),
            arguments.get("date_from"),
            arguments.get("date_to"),
            arguments.get("search_type", "both"),
            arguments.get("top_k", 5)
        )
    elif name == "database_health":
        return await database_health(arguments.get("check_type", "overview"))
    elif name == "activity_patterns":
        return await activity_patterns(
            arguments.get("pattern_type", "hourly"),
            arguments.get("sender"),
            arguments.get("date_range")
        )
    elif name == "message_extremes":
        return await message_extremes(
            arguments.get("query_type", "first"),
            arguments.get("sender"),
            arguments.get("count", 1)
        )
    elif name == "nickname_lookup":
        return await nickname_lookup(arguments.get("name") or "")
    elif name == "nickname_list":
        return await nickname_list()
    elif name == "nickname_search":
        return await nickname_search(arguments.get("pattern") or "")
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


async def message_stats(sender: str = None, date_range: str = None) -> list[TextContent]:
    """Get statistics about messages and senders"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Build query based on filters
        base_query = f"SELECT message_sender, COUNT(*) as message_count FROM {table_name}"
        conditions = []
        params = []

        if sender:
            conditions.append("message_sender = ?")
            params.append(sender)

        if date_range:
            if date_range == "last_week":
                conditions.append(
                    "message_datetime >= DATEADD(week, -1, GETDATE())")
            elif date_range == "last_month":
                conditions.append(
                    "message_datetime >= DATEADD(month, -1, GETDATE())")
            elif date_range == "last_year":
                conditions.append(
                    "message_datetime >= DATEADD(year, -1, GETDATE())")

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        if not sender:
            base_query += " GROUP BY message_sender ORDER BY message_count DESC"

        cursor.execute(base_query, params)
        results = cursor.fetchall()

        # Get total stats
        total_query = f"SELECT COUNT(*) as total_messages, COUNT(DISTINCT message_sender) as unique_senders FROM {table_name}"
        if conditions:
            total_query += " WHERE " + " AND ".join(conditions)
        cursor.execute(total_query, params)
        total_stats = cursor.fetchone()

        conn.close()

        # Format results
        if sender:
            result_text = f"Message statistics for {sender}:\n"
            result_text += f"Total messages: {results[0][1] if results else 0}\n"
        else:
            result_text = f"Message statistics:\n"
            result_text += f"Total messages: {total_stats[0]}\n"
            result_text += f"Unique senders: {total_stats[1]}\n\n"
            result_text += "Top senders:\n"
            for sender_name, count in results[:10]:
                result_text += f"- {sender_name}: {count} messages\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Statistics error: {str(e)}"
        )]


async def timeline_analysis(topic: str = None, grouping: str = "weekly") -> list[TextContent]:
    """Analyze message patterns over time"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Build date grouping based on parameter
        if grouping == "daily":
            date_format = "CAST(message_datetime AS DATE)"
            date_label = "Date"
        elif grouping == "monthly":
            date_format = "FORMAT(message_datetime, 'yyyy-MM')"
            date_label = "Month"
        else:  # weekly
            date_format = "FORMAT(message_datetime, 'yyyy-\\WW')"
            date_label = "Week"

        query = f"""
        SELECT {date_format} as period, COUNT(*) as message_count
        FROM {table_name}
        """

        params = []
        if topic:
            query += " WHERE message_text LIKE ?"
            params.append(f"%{topic}%")

        query += f" GROUP BY {date_format} ORDER BY period DESC"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        result_text = f"Timeline analysis ({grouping}):\n"
        if topic:
            result_text += f"Topic: {topic}\n\n"

        for period, count in results[:20]:  # Show last 20 periods
            result_text += f"{date_label} {period}: {count} messages\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Timeline analysis error: {str(e)}"
        )]


async def sentiment_analysis(query: str = None, sender: str = None, top_k: int = 10) -> list[TextContent]:
    """Analyze sentiment of messages using AI"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Build query
        base_query = f"""
        SELECT TOP ({top_k}) message_text, message_sender, message_datetime 
        FROM {table_name}
        """

        conditions = []
        params = []

        if query:
            conditions.append("message_text LIKE ?")
            params.append(f"%{query}%")

        if sender:
            conditions.append("message_sender = ?")
            params.append(sender)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY message_datetime DESC"

        cursor.execute(base_query, params)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return [TextContent(type="text", text="No messages found for sentiment analysis")]

        # Use OpenAI LLM for sentiment analysis
        messages_text = "\n".join([f"Message: {row[0]}" for row in results])

        prompt = f"""Analyze the sentiment of these messages. For each message, provide:
1. Overall sentiment (Positive/Negative/Neutral)
2. Emotion detected (joy, sadness, anger, excitement, etc.)
3. Brief explanation

Messages to analyze:
{messages_text}

Provide a summary at the end with overall sentiment trends."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        result_text = f"Sentiment Analysis Results:\n"
        if query:
            result_text += f"Query: {query}\n"
        if sender:
            result_text += f"Sender: {sender}\n"
        result_text += f"Analyzed {len(results)} messages\n\n"
        result_text += response.choices[0].message.content

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Sentiment analysis error: {str(e)}"
        )]


async def advanced_search(query: str, sender: str = None, date_from: str = None,
                          date_to: str = None, search_type: str = "both", top_k: int = 5) -> list[TextContent]:
    """Advanced search with multiple filters and search types"""
    try:
        results = []

        # Keyword search component
        if search_type in ["keyword", "both"]:
            keyword_results = await keyword_search(query, top_k)
            if keyword_results and keyword_results[0].text != "No messages found.":
                results.extend(keyword_results)

        # Semantic search component (if available)
        if search_type in ["semantic", "both"]:
            try:
                semantic_results = await semantic_search(query, top_k)
                if semantic_results and not semantic_results[0].text.startswith("Error performing semantic search"):
                    results.extend(semantic_results)
            except:
                pass  # Semantic search might fail due to JWT or missing embeddings

        # Apply additional filters if we have database access
        if sender or date_from or date_to:
            conn = get_conn()
            cursor = conn.cursor()

            filter_query = f"""
            SELECT message_text, message_sender, message_datetime 
            FROM {table_name}
            WHERE message_text LIKE ?
            """
            params = [f"%{query}%"]

            if sender:
                filter_query += " AND message_sender = ?"
                params.append(sender)

            if date_from:
                filter_query += " AND message_datetime >= ?"
                params.append(date_from)

            if date_to:
                filter_query += " AND message_datetime <= ?"
                params.append(date_to)

            filter_query += " ORDER BY message_datetime DESC"

            cursor.execute(filter_query, params)
            filtered_results = cursor.fetchall()
            conn.close()

            # Format filtered results
            if filtered_results:
                filter_text = f"Filtered search results:\n"
                filter_text += f"Query: {query}\n"
                if sender:
                    filter_text += f"Sender: {sender}\n"
                if date_from or date_to:
                    filter_text += f"Date range: {date_from or 'start'} to {date_to or 'end'}\n"
                filter_text += f"Found {len(filtered_results)} messages\n\n"

                for i, (text, sender_name, datetime) in enumerate(filtered_results[:top_k], 1):
                    filter_text += f"{i}. [{sender_name}] {datetime}: {text}\n"

                results.append(TextContent(type="text", text=filter_text))

        if not results:
            return [TextContent(type="text", text="No results found with the specified filters.")]

        return results

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Advanced search error: {str(e)}"
        )]


async def database_health(check_type: str = "overview") -> list[TextContent]:
    """Check database health and system status"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        health_info = []

        if check_type in ["overview", "all"]:
            # Basic stats
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_messages = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT COUNT(DISTINCT message_sender) FROM {table_name}")
            unique_senders = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            messages_with_embeddings = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT MIN(message_datetime), MAX(message_datetime) FROM {table_name}")
            date_range = cursor.fetchone()

            health_info.append("=== Database Overview ===")
            health_info.append(f"Total messages: {total_messages}")
            health_info.append(f"Unique senders: {unique_senders}")
            health_info.append(
                f"Messages with embeddings: {messages_with_embeddings}")
            health_info.append(
                f"Date range: {date_range[0]} to {date_range[1]}")
            if total_messages > 0:
                health_info.append(
                    f"Embedding coverage: {(messages_with_embeddings/total_messages*100):.1f}%")

        if check_type in ["performance", "all"]:
            # Performance stats
            cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                AVG(LEN(message_text)) as avg_length,
                MAX(LEN(message_text)) as max_length
            FROM {table_name}
            """)
            perf_stats = cursor.fetchone()

            health_info.append("\n=== Performance Metrics ===")
            health_info.append(
                f"Average message length: {perf_stats[1]:.1f} characters")
            health_info.append(
                f"Maximum message length: {perf_stats[2]} characters")

        if check_type in ["recent", "all"]:
            # Recent activity
            cursor.execute(f"""
            SELECT TOP 5 message_sender, COUNT(*) as recent_count
            FROM {table_name}
            WHERE message_datetime >= DATEADD(day, -7, GETDATE())
            GROUP BY message_sender
            ORDER BY recent_count DESC
            """)
            recent_activity = cursor.fetchall()

            health_info.append("\n=== Recent Activity (Last 7 days) ===")
            for sender, count in recent_activity:
                health_info.append(f"- {sender}: {count} messages")

        conn.close()

        return [TextContent(type="text", text="\n".join(health_info))]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Database health check error: {str(e)}"
        )]


async def activity_patterns(pattern_type: str = "hourly", sender: str = None, date_range: str = None) -> list[TextContent]:
    """Analyze user activity patterns by time periods"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Build date filter
        date_filter = ""
        params = []

        if date_range:
            if date_range == "last_week":
                date_filter = " AND message_datetime >= DATEADD(week, -1, GETDATE())"
            elif date_range == "last_month":
                date_filter = " AND message_datetime >= DATEADD(month, -1, GETDATE())"
            elif date_range == "last_year":
                date_filter = " AND message_datetime >= DATEADD(year, -1, GETDATE())"

        # Build sender filter
        sender_filter = ""
        if sender:
            sender_filter = " AND message_sender = ?"
            params.append(sender)

        # Build query based on pattern type
        if pattern_type == "hourly":
            time_format = "DATEPART(hour, message_datetime)"
            time_label = "Hour"
            query = f"""
            SELECT {time_format} as time_period, COUNT(*) as message_count
            FROM {table_name}
            WHERE 1=1 {date_filter} {sender_filter}
            GROUP BY {time_format}
            ORDER BY time_period
            """

        elif pattern_type == "daily":
            time_format = "DATEPART(weekday, message_datetime)"
            time_label = "Day of Week"
            query = f"""
            SELECT {time_format} as time_period, COUNT(*) as message_count
            FROM {table_name}
            WHERE 1=1 {date_filter} {sender_filter}
            GROUP BY {time_format}
            ORDER BY time_period
            """

        elif pattern_type == "weekly":
            time_format = "DATEPART(week, message_datetime)"
            time_label = "Week of Year"
            query = f"""
            SELECT {time_format} as time_period, COUNT(*) as message_count
            FROM {table_name}
            WHERE 1=1 {date_filter} {sender_filter}
            GROUP BY {time_format}
            ORDER BY time_period
            """

        elif pattern_type == "monthly":
            time_format = "DATEPART(month, message_datetime)"
            time_label = "Month"
            query = f"""
            SELECT {time_format} as time_period, COUNT(*) as message_count
            FROM {table_name}
            WHERE 1=1 {date_filter} {sender_filter}
            GROUP BY {time_format}
            ORDER BY time_period
            """
        else:
            return [TextContent(type="text", text="Invalid pattern_type. Use: hourly, daily, weekly, or monthly")]

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return [TextContent(type="text", text="No activity data found for the specified criteria")]

        # Format results
        result_text = f"Activity Patterns Analysis ({pattern_type}):\n"
        if sender:
            result_text += f"Sender: {sender}\n"
        if date_range:
            result_text += f"Date Range: {date_range}\n"
        result_text += "\n"

        # Add friendly labels for time periods
        time_labels = {}
        if pattern_type == "hourly":
            time_labels = {i: f"{i:02d}:00" for i in range(24)}
        elif pattern_type == "daily":
            time_labels = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
                           5: "Thursday", 6: "Friday", 7: "Saturday"}
        elif pattern_type == "monthly":
            time_labels = {1: "January", 2: "February", 3: "March", 4: "April",
                           5: "May", 6: "June", 7: "July", 8: "August",
                           9: "September", 10: "October", 11: "November", 12: "December"}

        # Calculate total for percentage
        total_messages = sum(count for _, count in results)

        # Create visual bar chart using text
        max_count = max(count for _, count in results)

        for time_period, count in results:
            percentage = (count / total_messages) * 100
            # Scale to 30 characters max
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)

            if pattern_type in time_labels:
                period_label = time_labels.get(time_period, str(time_period))
            else:
                period_label = str(time_period)

            result_text += f"{period_label:12} |{bar}| {count:5d} ({percentage:5.1f}%)\n"

        # Add summary statistics
        result_text += f"\nSummary:\n"
        result_text += f"Total messages: {total_messages}\n"
        result_text += f"Peak activity: {time_labels.get(max(results, key=lambda x: x[1])[0], max(results, key=lambda x: x[1])[0])} ({max(results, key=lambda x: x[1])[1]} messages)\n"
        result_text += f"Average per {pattern_type[:-2]}: {total_messages / len(results):.1f} messages\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Activity patterns error: {str(e)}"
        )]


async def message_extremes(query_type: str = "first", sender: str = None, count: int = 1) -> list[TextContent]:
    """Find earliest or latest messages overall or for specific senders"""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Normalize query type and set appropriate description
        query_type = query_type.lower()
        if query_type in ["earliest", "first", "oldest", "old"]:
            order_direction = "ASC"
            if query_type in ["oldest", "old"]:
                description = "oldest"
            else:
                description = "earliest"
        elif query_type in ["latest", "last", "newest", "new"]:
            order_direction = "DESC"
            if query_type in ["newest", "new"]:
                description = "newest"
            else:
                description = "latest"
        else:
            return [TextContent(
                type="text",
                text="Invalid query_type. Use: first, last, earliest, latest, oldest, newest, old, or new"
            )]

        # Build query
        base_query = f"""
        SELECT TOP ({count}) message_id, message_datetime, message_sender, message_text
        FROM {table_name}
        """

        params = []
        if sender:
            base_query += " WHERE message_sender = ?"
            params.append(sender)

        base_query += f" ORDER BY message_datetime {order_direction}"

        cursor.execute(base_query, params)
        results = cursor.fetchall()

        # Also get some context about the overall date range
        context_query = f"""
        SELECT 
            MIN(message_datetime) as first_message,
            MAX(message_datetime) as last_message,
            COUNT(*) as total_messages
        FROM {table_name}
        """

        if sender:
            context_query += " WHERE message_sender = ?"
            cursor.execute(context_query, [sender])
        else:
            cursor.execute(context_query)

        context = cursor.fetchone()
        conn.close()

        if not results:
            return [TextContent(
                type="text",
                text=f"No messages found{' for sender ' + sender if sender else ''}."
            )]

        # Format results
        result_text = f"=== {description.title()} Message{'s' if count > 1 else ''} ===\n"
        if sender:
            result_text += f"Sender: {sender}\n"
        else:
            result_text += "Overall chat\n"

        result_text += f"Showing {len(results)} of {context.total_messages} total messages\n"
        result_text += f"Date range: {context.first_message} to {context.last_message}\n\n"

        for i, (msg_id, datetime, sender_name, text) in enumerate(results, 1):
            # Truncate very long messages for display
            display_text = text if len(text) <= 200 else text[:197] + "..."
            result_text += f"{i}. [{sender_name}] {datetime}\n"
            result_text += f"   {display_text}\n\n"

        # Add helpful context
        if not sender and count == 1:
            if query_type in ["earliest", "first", "oldest", "old"]:
                result_text += f"🏁 The very first message in your chat was sent by {results[0][2]}"
            else:
                result_text += f"🔚 The most recent message in your chat was sent by {results[0][2]}"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Message extremes error: {str(e)}"
        )]


# Nickname management functions
async def nickname_lookup(name: str) -> list[TextContent]:
    """Look up nickname information for a person or get real name from nickname"""
    try:
        if not name:
            return [TextContent(type="text", text="Please provide a name or nickname to lookup.")]

        # Get real name (this will return the input if it's already a real name)
        real_name = get_real_name(name)

        # Get all nicknames for this person
        person_nicknames = get_nicknames_for_person(real_name)

        # Build response
        result_text = f"=== Nickname Lookup for '{name}' ===\n"
        result_text += f"Real Name: {real_name}\n"

        if person_nicknames:
            result_text += f"Nicknames: {', '.join(person_nicknames)}\n"
        else:
            result_text += "Nicknames: None\n"

        # Show if the input was a nickname
        if name != real_name:
            result_text += f"\nNote: '{name}' is a nickname for '{real_name}'\n"

        # Show all possible names for this person
        all_names = get_all_names_for_person(name)
        result_text += f"All known names: {', '.join(all_names)}"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Nickname lookup error: {str(e)}"
        )]


async def nickname_list() -> list[TextContent]:
    """List all people and their nicknames"""
    try:
        result_text = "=== Complete Nickname Directory ===\n\n"

        # Get statistics
        total_people = len(nicknames)
        people_with_nicknames = len(
            [name for name, nicks in nicknames.items() if nicks])
        total_nicknames = sum(len(nicks) for nicks in nicknames.values())

        result_text += f"Total People: {total_people}\n"
        result_text += f"People with Nicknames: {people_with_nicknames}\n"
        result_text += f"Total Nicknames: {total_nicknames}\n"
        result_text += f"Average Nicknames per Person: {total_nicknames / total_people:.1f}\n\n"

        # List all people and their nicknames
        result_text += "Directory:\n"
        result_text += "-" * 50 + "\n"

        for real_name in sorted(nicknames.keys()):
            person_nicknames = nicknames[real_name]
            if person_nicknames:
                nickname_str = ', '.join(
                    f'"{nick}"' for nick in person_nicknames)
                result_text += f"{real_name:<25} → {nickname_str}\n"
            else:
                result_text += f"{real_name:<25} → (no nicknames)\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Nickname list error: {str(e)}"
        )]


async def nickname_search(pattern: str) -> list[TextContent]:
    """Search for people by partial name or nickname match"""
    try:
        if not pattern:
            return [TextContent(type="text", text="Please provide a search pattern.")]

        pattern_lower = pattern.lower()
        matches = {}

        # Search in real names
        for real_name in nicknames:
            if pattern_lower in real_name.lower():
                matches[real_name] = nicknames[real_name]

        # Search in nicknames
        for real_name, person_nicknames in nicknames.items():
            if real_name not in matches:  # Don't add duplicates
                for nickname in person_nicknames:
                    if pattern_lower in nickname.lower():
                        matches[real_name] = person_nicknames
                        break

        if not matches:
            return [TextContent(
                type="text",
                text=f"No matches found for pattern '{pattern}'"
            )]

        # Format results
        result_text = f"=== Search Results for '{pattern}' ===\n"
        result_text += f"Found {len(matches)} match{'es' if len(matches) != 1 else ''}:\n\n"

        for real_name in sorted(matches.keys()):
            person_nicknames = matches[real_name]
            if person_nicknames:
                nickname_str = ', '.join(
                    f'"{nick}"' for nick in person_nicknames)
                result_text += f"{real_name:<25} → {nickname_str}\n"
            else:
                result_text += f"{real_name:<25} → (no nicknames)\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Nickname search error: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    debug_print("Starting MCP server...")
    debug_print(f"Server name: sql-messages")
    debug_print(f"Database: {database} on {server_host}:{port}")
    debug_print(f"Table: {table_name}")
    debug_print(f"OpenAI Model: {model}")
    debug_print("Available tools: keyword_search, semantic_search, message_stats, timeline_analysis, sentiment_analysis, advanced_search, database_health, activity_patterns, message_extremes")

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

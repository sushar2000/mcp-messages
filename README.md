# MCP Messages

An MCP (Model Context Protocol) server that provides AI-powered chat analysis and search capabilities for messages stored in a SQL Server database. This project allows you to load, analyze, and search through chat message data using semantic embeddings and natural language queries.

## Features

- **Message Data Loading**: Load chat messages from TSV files into SQL Server database
- **AI-Powered Search**: Semantic search capabilities using OpenAI embeddings
- **MCP Server**: Provides structured access to message analysis tools via Model Context Protocol
- **Data Validation**: Compare and validate loaded data against source files
- **Embedding Management**: Generate and manage text embeddings for semantic search

## Project Structure

```
mcp-messages/
├── mcp_messages.py          # Main MCP server implementation
├── data_load.py             # Load TSV data into database
├── data_compare.py          # Compare database data with source files
├── embeddings_update.py     # Generate and update embeddings
├── embeddings_check.py      # Validate embedding data
├── embeddings_setup_test.py # Test embedding setup
├── colors.py                # Terminal color utilities
├── requirements.txt         # Python dependencies
├── config-sample.json       # Sample configuration file
├── db/
│   ├── setup_db.sql         # Database schema setup
│   ├── load_messages.sql    # SQL data loading scripts
│   └── docker-compose.mssql.yml # Docker setup for SQL Server
└── data/
    └── sample-chat.tsv      # Sample chat data
```

## Requirements

- Python 3.7+
- SQL Server (local or remote)
- OpenAI API access (for embeddings and chat analysis)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

- `pyodbc` - SQL Server database connectivity
- `openai` - OpenAI API client
- `mcp` - Model Context Protocol server framework
- `langchain-openai` - LangChain OpenAI integration

## Setup

### 1. Database Setup

You can use the provided Docker Compose file to set up a local SQL Server instance:

```bash
cd db
docker-compose -f docker-compose.mssql.yml up -d
```

Or connect to an existing SQL Server instance.

### 2. Configuration

Copy the sample configuration file and update it with your settings:

```bash
cp config-sample.json config.json
```

Edit `config.json` with your database and OpenAI API details:

```json
{
  "database": {
    "host": "localhost",
    "port": 1433,
    "db_name": "WhatsAppDB",
    "table_name": "messages",
    "user": "sa",
    "password": "your_password"
  },
  "file_paths": ["data/sample-chat.tsv"],
  "openai": {
    "llm_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "api_key": "your_openai_api_key"
  }
}
```

### 3. Add Nicknames Data

Copy the sample nicknames file and update it with your data:

```bash
cp nicknames-sample.json nicknames.json
```

Edit `nicknames.json` with your database:

```json
{
  "nicknames": {
    "Message Sender 1": ["MS1", "Sender1"],
    "Message Sender 2": ["MS2", "Sender2", "Person"],
    "Message Sender 3": ["MS3"],
    "Message Sender 4": []
  }
}
```

### 4. Database Schema

Initialize the database schema using the setup script:

```bash
python data_load.py
```

This will automatically create the database and tables using `db/setup_db.sql`.

## Usage

### Loading Data

Load chat messages from TSV files into the database:

```bash
python data_load.py
```

The script will:

- Create the database and table if they don't exist
- Load all TSV files specified in your `config.json`
- Display progress and summary statistics

### Generating Embeddings

Generate embeddings for semantic search:

```bash
python embeddings_update.py
```

### Data Validation

Compare database content with source files:

```bash
python data_compare.py
```

### Running the MCP Server

Start the MCP server to provide AI-powered message analysis:

```bash
python mcp_messages.py
```

The MCP server provides tools for:

- Semantic message search
- Message statistics and analysis
- Chat pattern recognition
- Sentiment analysis

## Data Format

The expected TSV file format is:

```
timestamp	sender	message_text
2024-01-01 10:00:00	John Doe	Hello everyone!
2024-01-01 10:01:00	Jane Smith	Hi John, how are you?
```

Requirements:

- Tab-separated values
- UTF-8 encoding
- Columns: timestamp, sender, message_text
- Datetime format should be parseable by SQL Server

## MCP Integration

This server implements the Model Context Protocol, allowing AI assistants to:

- Search through message history using natural language
- Analyze conversation patterns and trends
- Generate insights about chat participants
- Perform semantic similarity searches

## Configuration Options

### Database Settings

- `host`: Database server hostname
- `port`: Database server port (default: 1433)
- `db_name`: Database name
- `table_name`: Messages table name
- `user`/`password`: Database credentials

### OpenAI Settings

- `llm_model`: GPT model for analysis (e.g., "gpt-4o")
- `embedding_model`: Embedding model (e.g., "text-embedding-3-small")
- `api_key`: Your OpenAI API key
- `max_retries`: API retry attempts

### File Paths

- `file_paths`: Array of TSV file paths to load

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Verify SQL Server is running
   - Check connection string parameters
   - Ensure database user has proper permissions

2. **TSV Loading Issues**

   - Verify file encoding is UTF-8
   - Check for malformed rows (will be skipped with warnings)
   - Ensure tab separation (not spaces)

3. **Embedding Generation Failures**
   - Verify OpenAI API key is valid
   - Check API rate limits
   - Ensure sufficient API credits

### Debug Mode

Enable debug output by setting environment variable:

```bash
export MCP_DEBUG=1
python mcp_messages.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues and questions:

- Create an issue in the GitHub repository
- Check existing issues for solutions
- Review the troubleshooting section above

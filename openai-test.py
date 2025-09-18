from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import json


def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"Configuration loaded from  file '{config_file}'")
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
    print("Failed to load configuration. Using defaults.")
    exit(1)

# Get OpenAI configuration from config file or use defaults
openai_config = config.get('openai', {})
model = openai_config.get('model', 'gpt-4o')
base_url = openai_config.get('base_url')
api_key = openai_config.get('api_key')

client = ChatOpenAI(
    model=model,
    base_url=base_url,
    api_key=api_key
)

try:
    response = client.invoke("Write a haiku about ai.")

    print(response.content)

except Exception as e:
    print(f"Error occurred: {e}")
    exit(1)


# Define your embedding settings
EMBEDDING_MODEL_URL = "https://int.lionis.ai/api/v1/vectors"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
OPENAI_API_KEY = api_key

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_base=EMBEDDING_MODEL_URL,
    openai_api_key=OPENAI_API_KEY,
)

# Example 1: Single text
text = "Artificial intelligence is transforming the world."
vector = embeddings.embed_query(text)
print("Embedding length:", len(vector))
print("First 10 dims:", vector[:10])

# Example 2: Multiple documents
docs = [
    "Machine learning is a subset of AI.",
    "Neural networks are inspired by the human brain.",
    "Large language models are powerful tools for NLP."
]
vectors = embeddings.embed_documents(docs)
print("Number of embeddings:", len(vectors))
print("Embedding length for each:", len(vectors[0]))

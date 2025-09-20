from datetime import datetime
from xml.parsers.expat import model
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import json


from colors import Colors


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
LLM_MODEL_NAME = openai_config.get('model', 'gpt-4o')
EMBEDDING_MODEL_NAME = openai_config.get('embedding_model')

ENV_URL = openai_config.get('env_url')
OPENAI_API_KEY = openai_config.get('api_key')

LLM_API_URL = ENV_URL+"/api/v1/llm/"
EMBEDDING_MODEL_URL = ENV_URL+"/api/v1/vectors"

client = ChatOpenAI(
    model=LLM_MODEL_NAME,
    base_url=LLM_API_URL,
    api_key=OPENAI_API_KEY
)

try:
    response = client.invoke("Write a haiku about ai.")
    print(response.content)
except Exception as e:
    print(f"Error occurred: {e}")
    exit(1)


# Define your embedding settings

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_base=EMBEDDING_MODEL_URL,
    openai_api_key=OPENAI_API_KEY,
)
print("Embeddings initialized with model:", EMBEDDING_MODEL_NAME)

t1 = datetime.now()
print("Example 1: Single text")
text = "Artificial intelligence is transforming the world."
vector = embeddings.embed_query(text)
print("Embedding length:", len(vector))
print("First 10 dims:", vector[:10])

print("Example 2: Multiple documents")
docs = [
    "Machine learning is a subset of AI.",
    "Neural networks are inspired by the human brain.",
    "Large language models are powerful tools for NLP."
]
vectors = embeddings.embed_documents(docs)
print("Number of embeddings:", len(vectors))
print("Embedding length for each:", len(vectors[0]))
print("First 10 dims:", vectors[0][:10])
t2 = datetime.now()
print("Time taken:", Colors.GREEN, t2 - t1, Colors.RESET)

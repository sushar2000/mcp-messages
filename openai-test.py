from langchain_openai import ChatOpenAI
import json


def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
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

response = client.invoke("Write a haiku about ai.")

print(response.content)

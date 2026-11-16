import os
from getpass import getpass


# ---

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Error: python-dotenv is not installed. Please install it using 'pip install python-dotenv'.")

# If the environment variables are not set, prompt the user to enter them
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter your GROQ API Key: ")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass("Enter your PINECONE API Key: ")

if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass("Enter your TAVILY API Key: ")

# Print the status of the API keys
print("GROQ_API_KEY configured:", bool(os.getenv("GROQ_API_KEY")))
print("PINECONE_API_KEY configured:", bool(os.getenv("PINECONE_API_KEY")))
print("TAVILY_API_KEY configured:", bool(os.getenv("TAVILY_API_KEY")))


# ---------------- Settings ----------------

# Knowledge base source
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions

# Pinecone
INDEX_NAME = "industry-rag-knowledge-base"
NAMESPACE = "langgraph-agentic-rag"

# Groq LLM (check console.groq.com for current model names)
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0
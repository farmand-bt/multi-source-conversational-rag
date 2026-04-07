import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GWDG_API_KEY = os.getenv("GWDG_API_KEY", "")
GWDG_API_BASE = os.getenv("GWDG_API_BASE", "")
GWDG_MODEL_NAME = os.getenv("GWDG_MODEL_NAME", "")

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# Memory
MAX_HISTORY_TURNS = 10

# Vector store
CHROMA_PERSIST_DIR = "./data/chroma"

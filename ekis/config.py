"""Configuration file for Simple Vector RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATABASE_DIR = PROJECT_ROOT / "database"

CHROMA_DB_PATH = DATABASE_DIR / "chroma"

# Model details
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval settings
TOP_K = 3               # Final results returned to LLM (reduce = faster synthesis)
BM25_CANDIDATES = 15   # Candidate pool size for hybrid BM25+Dense search

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "ekis_backend.log"
os.makedirs(LOG_FILE.parent, exist_ok=True)
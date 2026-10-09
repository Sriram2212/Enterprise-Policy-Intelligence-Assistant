import os
# Force offline modes for model loading
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ekis import config
from ekis.database import DatabaseManager
from ekis.retriever import HybridRetriever
from ekis.llm.gemini_client import GeminiClient

# Set up logger
logger = logging.getLogger("EKIS.API")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Initialize app
app = FastAPI(
    title="Enterprise Knowledge Intelligence System (EKIS) API",
    description="A modular RAG API powered by ChromaDB, SentenceTransformers, and Gemini."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources
db_manager = DatabaseManager()
retriever = HybridRetriever(db_manager)
gemini = GeminiClient(api_key=config.GOOGLE_API_KEY, model_name=config.LLM_MODEL)

# Import and include routers (routers will register routes on these)
from ekis.api.router_docs import router as docs_router
from ekis.api.router_query import router as query_router

app.include_router(docs_router)
app.include_router(query_router)

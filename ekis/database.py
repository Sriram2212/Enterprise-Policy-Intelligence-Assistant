import re
import uuid
import tiktoken
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
import time
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from ekis import config

# Set up logging
logger = logging.getLogger("SimpleRAG.Database")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DatabaseManager:
    def __init__(self):
        logger.info("Initializing Vector Database Manager...")
        start_time = time.time()
        
        # Load embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        config.CHROMA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        self.collection = self.chroma_client.get_or_create_collection(name="policy_documents")
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        duration = time.time() - start_time
        logger.info(f"Database Manager initialized in {duration:.2f} seconds.")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            if end == len(tokens):
                break
            start += chunk_size - chunk_overlap
            
        return chunks

    def extract_sections(self, pdf_path: Path) -> dict[str, str]:
        """
        Parses a PDF file and extracts text grouped by sections.
        Uses PyMuPDF TOC if available, else falls back to page-by-page header grouping.
        """
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # List of [level, title, page]
        total_pages = len(doc)
        
        sections = {}
        
        if toc:
            logger.info(f"Found TOC in {pdf_path.name} with {len(toc)} entries.")
            # Resolve section ranges
            for i, entry in enumerate(toc):
                level, title, page_start = entry
                page_end = total_pages
                
                # Check next entries at same or higher level to limit page range
                for next_entry in toc[i+1:]:
                    if next_entry[0] <= level:
                        page_end = next_entry[2] - 1
                        break
                
                # Extract text for the page range (0-indexed in PyMuPDF)
                section_text = []
                for p_num in range(page_start - 1, min(page_end, total_pages)):
                    page_text = doc[p_num].get_text()
                    if page_text.strip():
                        section_text.append(page_text)
                
                sections[title.strip()] = "\n".join(section_text)
        else:
            logger.info(f"No TOC found in {pdf_path.name}. Falling back to page-by-page header matching.")
            # Fallback: group by detected page headers
            for idx in range(total_pages):
                page_text = doc[idx].get_text()
                cleaned = self.clean_text(page_text)
                if not cleaned:
                    continue
                
                lines = cleaned.split("\n")
                header = f"Page {idx+1} Content"
                for line in lines[:3]:
                    # Match things like "Section 1.2" or short capitalized lines
                    if re.match(r'^(Section\s+\d+|[A-Z\d\.\s-]{3,40})$', line.strip()):
                        header = line.strip()
                        break
                
                if header in sections:
                    sections[header] += "\n" + page_text
                else:
                    sections[header] = page_text
                    
        doc.close()
        return sections

    def add_document_to_indexes(self, pdf_path: Path) -> int:
        """Processes a PDF, chunks it by sections, generates embeddings, and saves to ChromaDB."""
        logger.info(f"Indexing document: {pdf_path.name}")
        start_time = time.time()
        
        doc_name = pdf_path.stem
        category = pdf_path.parent.name if pdf_path.parent != config.DATASET_DIR else "General"
        
        # 1. Extract sections
        sections = self.extract_sections(pdf_path)
        
        chunks_to_add = []
        metadatas = []
        ids = []
        
        total_chunks = 0
        
        # 2. Chunk each section separately
        for sec_title, sec_text in sections.items():
            cleaned_text = self.clean_text(sec_text)
            if not cleaned_text:
                continue
                
            sec_chunks = self.chunk_text(cleaned_text, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            
            for chunk_idx, chunk_text in enumerate(sec_chunks):
                chunk_id = f"CH_{doc_name}_{uuid.uuid4().hex[:6].upper()}_C{chunk_idx}"
                
                chunks_to_add.append(chunk_text)
                metadatas.append({
                    "document_name": doc_name,
                    "category": category,
                    "section_title": sec_title,
                    "source_file": pdf_path.name
                })
                ids.append(chunk_id)
                total_chunks += 1
                
        if not chunks_to_add:
            logger.warning(f"No extractable text found in {pdf_path.name}.")
            return 0
            
        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks_to_add)} chunks...")
        embeddings = self.embedding_model.encode(chunks_to_add).tolist()
        
        # 4. Insert into ChromaDB
        self.collection.add(
            documents=chunks_to_add,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # 5. Export structured chunks & embeddings to category-based folders
        self.save_document_to_category_folders(doc_name, category, chunks_to_add, metadatas, embeddings, ids)
        
        # 6. Export all chunks to flat JSON file for backwards compatibility
        self.export_all_chunks_to_json()
        
        duration = time.time() - start_time
        logger.info(f"Indexed document '{pdf_path.name}' -> Sections: {len(sections)}, Chunks: {total_chunks}, Time: {duration:.2f}s")
        return total_chunks

    def get_document_chunks(self, document_name: str, limit: int = 5) -> list[dict]:
        """Fetches chunks belonging to a document from ChromaDB."""
        results = self.collection.get(
            where={"document_name": document_name},
            limit=limit
        )
        
        chunks = []
        if results and 'documents' in results and results['documents']:
            for idx in range(len(results['documents'])):
                chunks.append({
                    "text": results['documents'][idx],
                    "metadata": results['metadatas'][idx]
                })
        return chunks

    def clear_database(self):
        """Clears the collection to start fresh."""
        try:
            self.chroma_client.delete_collection(name="policy_documents")
            self.collection = self.chroma_client.get_or_create_collection(name="policy_documents")
            logger.info("ChromaDB collection cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")

    def export_all_chunks_to_json(self):
        """Fetches all documents from ChromaDB (with embeddings) and exports them to database/chunks.json."""
        try:
            import json
            # Query ChromaDB collection to fetch all chunks, including embeddings
            results = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            chunks_list = []
            if results and 'documents' in results and results['documents']:
                for idx in range(len(results['documents'])):
                    embedding = None
                    if results.get('embeddings') is not None and len(results['embeddings']) > idx:
                        embedding = results['embeddings'][idx]
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                    
                    chunks_list.append({
                        "id": results['ids'][idx],
                        "text": results['documents'][idx],
                        "metadata": results['metadatas'][idx],
                        "embedding": embedding
                    })
            
            export_path = config.DATABASE_DIR / "chunks.json"
            config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(chunks_list, f, indent=2)
            logger.info(f"Exported all {len(chunks_list)} chunks & embeddings to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export chunks to JSON: {e}")

    def save_document_to_category_folders(self, doc_name: str, category: str, chunks: list[str], metadatas: list[dict], embeddings: list[list[float]], ids: list[str]):
        """
        Saves document chunks and embeddings to category-specific folders:
        database/<category>/chunks/<doc_name>_chunks.json
        database/<category>/embeddings/<doc_name>_embeddings.json
        """
        try:
            import json
            # Create category folders
            category_dir = config.DATABASE_DIR / category
            chunks_dir = category_dir / "chunks"
            embeddings_dir = category_dir / "embeddings"
            
            chunks_dir.mkdir(parents=True, exist_ok=True)
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Structure chunks JSON
            chunks_list = []
            for idx, text in enumerate(chunks):
                chunks_list.append({
                    "id": ids[idx],
                    "text": text,
                    "metadata": metadatas[idx]
                })
                
            chunks_file = chunks_dir / f"{doc_name}_chunks.json"
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_list, f, indent=2)
                
            # 2. Structure embeddings JSON
            embeddings_list = []
            for idx, emb in enumerate(embeddings):
                embeddings_list.append({
                    "id": ids[idx],
                    "embedding": emb
                })
                
            embeddings_file = embeddings_dir / f"{doc_name}_embeddings.json"
            with open(embeddings_file, "w", encoding="utf-8") as f:
                json.dump(embeddings_list, f, indent=2)
                
            logger.info(f"Saved structured files for {doc_name}: chunks in {chunks_file}, embeddings in {embeddings_file}")
        except Exception as e:
            logger.error(f"Failed to save structured category files for {doc_name}: {e}")

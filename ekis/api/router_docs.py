import os
import shutil
import time
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ekis import config

# Import shared resources (lazily imported to avoid circular dependencies)
logger = logging.getLogger("EKIS.API.Docs")
router = APIRouter()

@router.get("/documents")
async def get_documents_hierarchy():
    """Fetches list of all indexed documents and their section titles from the structured JSON files on disk."""
    from ekis.api import db_manager
    try:
        hierarchy = {}
        # Glob for all *_chunks.json files in database/ category subdirectories
        chunks_files = list(config.DATABASE_DIR.glob("**/chunks/*_chunks.json"))
        
        for file_path in chunks_files:
            doc_name = file_path.stem.replace("_chunks", "")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                sections = set()
                for chunk in chunks_data:
                    sec_title = chunk.get("metadata", {}).get("section_title", "General")
                    sections.add(sec_title)
                hierarchy[doc_name] = sorted(list(sections))
            except Exception as fe:
                logger.error(f"Error reading chunk file {file_path}: {fe}")
                
        return hierarchy
    except Exception as e:
        logger.error(f"Error fetching document hierarchy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/detailed")
async def get_detailed_documents():
    """Fetches rich metadata for all indexed documents including category, section list, chunk counts, and preview snippets."""
    try:
        details = []
        chunks_files = list(config.DATABASE_DIR.glob("**/chunks/*_chunks.json"))
        
        for file_path in chunks_files:
            doc_name = file_path.stem.replace("_chunks", "")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                
                category = "General"
                source_file = f"{doc_name}.pdf"
                section_map = {}
                
                for chunk in chunks_data:
                    meta = chunk.get("metadata", {})
                    category = meta.get("category", category)
                    source_file = meta.get("source_file", source_file)
                    sec_title = meta.get("section_title", "General")
                    
                    if sec_title not in section_map:
                        section_map[sec_title] = []
                    section_map[sec_title].append({
                        "id": chunk.get("id"),
                        "text_snippet": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                        "char_count": len(chunk.get("text", ""))
                    })
                
                first_chunk = chunks_data[0].get("text", "") if chunks_data else ""
                preview = first_chunk[:350] + "..." if len(first_chunk) > 350 else first_chunk
                
                details.append({
                    "document_name": doc_name,
                    "category": category,
                    "source_file": source_file,
                    "chunks_count": len(chunks_data),
                    "sections_count": len(section_map),
                    "sections": sorted(list(section_map.keys())),
                    "sections_detail": section_map,
                    "preview": preview
                })
            except Exception as fe:
                logger.error(f"Error reading detailed chunk file {file_path}: {fe}")
                
        # Sort by category then document name
        details.sort(key=lambda d: (d["category"], d["document_name"]))
        return details
    except Exception as e:
        logger.error(f"Error fetching detailed documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/stats")
async def get_system_stats():
    """Provides overall knowledge base statistics and metrics."""
    try:
        chunks_files = list(config.DATABASE_DIR.glob("**/chunks/*_chunks.json"))
        total_chunks = 0
        total_docs = len(chunks_files)
        categories_count = {}
        
        for file_path in chunks_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                    total_chunks += len(chunks_data)
                    if chunks_data:
                        cat = chunks_data[0].get("metadata", {}).get("category", "General")
                        categories_count[cat] = categories_count.get(cat, 0) + 1
            except Exception:
                pass
                
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "categories": categories_count,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "chunk_token_limit": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP
        }
    except Exception as e:
        logger.error(f"Error fetching system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("General")
):
    """Ingests a PDF document, stores it in category folders, and indexes it in ChromaDB."""
    from ekis.api import db_manager
    logger.info(f"Received upload request for file: '{file.filename}', category: '{category}'")
    start_time = time.time()
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    category_folder = config.DATASET_DIR / category.replace(" ", "_")
    category_folder.mkdir(parents=True, exist_ok=True)
    
    file_path = category_folder / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks_count = db_manager.add_document_to_indexes(file_path)
        
        duration = time.time() - start_time
        logger.info(f"Successfully processed upload '{file.filename}' in {duration:.2f}s.")
        return {
            "message": f"Successfully ingested '{file.filename}' into category '{category}'.",
            "chunks_count": chunks_count,
            "category": category
        }
    except Exception as e:
        # Cleanup file if indexing failed
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"Failed to process document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents/{document_name}/sections/{section_title}/chunks")
async def get_section_chunks(document_name: str, section_title: str):
    """Retrieves all text chunks of a specific section in a document by reading the structured files on disk."""
    try:
        # Glob to find the correct document chunks file on disk
        chunks_files = list(config.DATABASE_DIR.glob(f"**/chunks/{document_name}_chunks.json"))
        if not chunks_files:
            raise HTTPException(status_code=404, detail=f"Document chunks file for {document_name} not found.")
            
        file_path = chunks_files[0]
        with open(file_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            
        chunks = []
        for chunk in chunks_data:
            meta = chunk.get("metadata", {})
            if meta.get("section_title") == section_title:
                chunks.append(chunk.get("text", ""))
        return chunks
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching section chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

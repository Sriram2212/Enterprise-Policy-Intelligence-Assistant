import time
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ekis import config
from ekis import prompts

# Initialize logger and router
logger = logging.getLogger("EKIS.API.Query")
router = APIRouter()

# Pydantic Request Schemas
class QueryRequest(BaseModel):
    question: str
    document_name: str | None = None
    category: str | None = None

class SummaryRequest(BaseModel):
    document_name: str

class CompareRequest(BaseModel):
    policy_a: str
    policy_b: str


def calculate_confidence(distances: list[float]) -> float:
    """Helper to convert L2 distances into a confidence percentage."""
    if not distances:
        return 0.0
    avg_distance = sum(distances) / len(distances)
    # L2 distance for normalized embeddings is in [0, 2].
    # Similarity = 1 - L2_dist / 2.
    sim = 1.0 - (avg_distance / 2.0)
    return round(max(0.0, min(100.0, sim * 100.0)), 2)


@router.post("/query")
async def query_system(req: QueryRequest):
    """Executes Vector RAG pipeline with ChromaDB retrieval and Gemini LLM synthesis."""
    from ekis.api import retriever, gemini
    logger.info(f"Received query request: '{req.question}' (doc_filter: {req.document_name}, cat_filter: {req.category})")
    start_time = time.time()
    
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # Retrieve context with optional document/category filtering
        hits, context_str = retriever.get_hybrid_context(
            req.question, 
            filter_doc=req.document_name, 
            filter_category=req.category
        )
        
        # Hallucination Guard
        if not hits or not context_str.strip():
            logger.warning("No relevant context found. Returning default fallback response.")
            return {
                "answer": "I cannot generate an answer because no relevant policy context was found in the database.",
                "confidence_score": 0.0,
                "sources": []
            }
            
        # Build strict prompt context
        prompt = prompts.QUERY_PROMPT_TEMPLATE.format(
            context=context_str,
            question=req.question
        )
        
        # Generate Answer with Fallback
        llm_start = time.time()
        try:
            raw_answer = gemini.generate(prompt).strip()
        except Exception as e:
            err_str = str(e)
            logger.error(f"Gemini API call failed during query: {err_str}")
            if "429" in err_str or "quota" in err_str.lower():
                # Build a clean context-based answer fallback
                fallback_parts = []
                for idx, hit in enumerate(hits):
                    chunk = hit["chunk"]
                    meta = chunk["metadata"]
                    fallback_parts.append(f"[{meta.get('document_name')}, {meta.get('section')}]: {chunk['text']}")
                context_summary = "\n\n".join(fallback_parts)
                raw_answer = f"⚠️ [Gemini API Daily Limit Reached]\nThe Gemini API daily free quota has been exceeded. Below are the matching policy rules found in the document:\n\n{context_summary}"
            else:
                raise e
                
        llm_duration = (time.time() - llm_start) * 1000  # ms
        
        # Parse references for sources list
        sources = []
        distances = []
        for rank, hit in enumerate(hits):
            chunk = hit["chunk"]
            meta = chunk["metadata"]
            distances.append(hit.get("score", 0.0))
            
            source_entry = {
                "rank": rank + 1,
                "document_name": meta.get("document_name"),
                "category": meta.get("category"),
                "section": meta.get("section", "General"),
                "page": meta.get("page_number", 1),
                "text_excerpt": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
            }
            if source_entry not in sources:
                sources.append(source_entry)
                
        confidence = calculate_confidence(distances)
        
        total_duration = (time.time() - start_time) * 1000  # ms
        logger.info(
            f"Query processed in {total_duration:.2f}ms. "
            f"(LLM synthesis: {llm_duration:.2f}ms). "
            f"Confidence: {confidence}%"
        )
        
        return {
            "answer": raw_answer,
            "confidence_score": confidence,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def generate_summary(req: SummaryRequest):
    """Generates structural summary (Executive Summary, Key Rules, Important Dates, Contacts) for a document."""
    from ekis.api import db_manager, gemini
    logger.info(f"Received summary request for document: '{req.document_name}'")
    start_time = time.time()
    try:
        matching_chunks = db_manager.get_document_chunks(req.document_name, limit=5)
        
        if not matching_chunks:
            logger.warning(f"No document matching '{req.document_name}' found in index.")
            raise HTTPException(status_code=404, detail=f"No document matching '{req.document_name}' found in index.")
            
        summary_context = "\n\n".join([chunk["text"] for chunk in matching_chunks])
        
        prompt = prompts.SUMMARY_PROMPT_TEMPLATE.format(
            document_name=req.document_name,
            context=summary_context
        )
        
        try:
            summary = gemini.generate(prompt).strip()
        except Exception as e:
            err_str = str(e)
            logger.error(f"Gemini API call failed during summary: {err_str}")
            if "429" in err_str or "quota" in err_str.lower():
                summary = f"⚠️ [Gemini API Daily Limit Reached]\nCould not generate AI summary. Below are the key excerpts retrieved from the document:\n\n{summary_context}"
            else:
                raise e
                
        duration = (time.time() - start_time) * 1000  # ms
        logger.info(f"Summary generated successfully in {duration:.2f}ms.")
        return {
            "document_name": req.document_name,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_policies(req: CompareRequest):
    """Compares key elements and rules between two different documents."""
    from ekis.api import db_manager, gemini
    logger.info(f"Received compare request for: '{req.policy_a}' vs '{req.policy_b}'")
    start_time = time.time()
    try:
        chunks_a = db_manager.get_document_chunks(req.policy_a, limit=3)
        chunks_b = db_manager.get_document_chunks(req.policy_b, limit=3)
        
        if not chunks_a or not chunks_b:
            logger.warning(f"One or both policies could not be found. Policy A: {len(chunks_a)} chunks, Policy B: {len(chunks_b)} chunks.")
            raise HTTPException(status_code=404, detail="One or both policies could not be found for comparison.")
            
        context_a = "\n\n".join([c["text"] for c in chunks_a])
        context_b = "\n\n".join([c["text"] for c in chunks_b])
        
        prompt = prompts.COMPARE_PROMPT_TEMPLATE.format(
            policy_a=req.policy_a,
            policy_b=req.policy_b,
            context_a=context_a,
            context_b=context_b
        )
        
        try:
            comparison = gemini.generate(prompt).strip()
        except Exception as e:
            err_str = str(e)
            logger.error(f"Gemini API call failed during comparison: {err_str}")
            if "429" in err_str or "quota" in err_str.lower():
                comparison = f"⚠️ [Gemini API Daily Limit Reached]\nCould not perform AI comparison. Below are the matching sections side-by-side:\n\n=== {req.policy_a} ===\n{context_a}\n\n=== {req.policy_b} ===\n{context_b}"
            else:
                raise e
                
        duration = (time.time() - start_time) * 1000  # ms
        logger.info(f"Comparison completed in {duration:.2f}ms.")
        return {
            "policy_a": req.policy_a,
            "policy_b": req.policy_b,
            "comparison": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import re
import time
import logging
from rank_bm25 import BM25Okapi
from ekis import config

logger = logging.getLogger("SimpleRAG.Retriever")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a single list rank."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    def __init__(self, db_manager):
        self.db = db_manager
        logger.info("Retriever initialized.")

    def get_hybrid_context(
        self,
        query: str,
        filter_doc: str | None = None,
        filter_category: str | None = None
    ) -> tuple[list[dict], str]:
        """
        Hybrid retrieval: Dense Semantic (ChromaDB) + BM25 keyword search, fused via RRF.
        Optional metadata filtering narrows the candidate pool before both searches.
        """
        filter_desc = f" (doc: {filter_doc}, cat: {filter_category})" if filter_doc or filter_category else ""
        logger.info(f"[Hybrid] Query: '{query}'{filter_desc}")
        start_time = time.time()

        # ── 1. Build Metadata Where Filter ──────────────────────────────────
        where_filter = None
        if filter_doc and filter_category:
            where_filter = {"$and": [{"document_name": filter_doc}, {"category": filter_category}]}
        elif filter_doc:
            where_filter = {"document_name": filter_doc}
        elif filter_category and filter_category != "All":
            where_filter = {"category": filter_category}

        # ── 2. Dense Semantic Search (ChromaDB) ─────────────────────────────
        t0 = time.time()
        query_vector = self.db.embedding_model.encode(query, convert_to_numpy=True).tolist()
        embed_ms = (time.time() - t0) * 1000

        t0 = time.time()
        candidates_n = max(config.BM25_CANDIDATES, config.TOP_K)
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": candidates_n,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.db.collection.query(**query_kwargs)
        chroma_ms = (time.time() - t0) * 1000

        # Unpack ChromaDB results
        if not (results and results.get("documents") and results["documents"][0]):
            logger.warning("No candidates returned from ChromaDB.")
            return [], ""

        docs       = results["documents"][0]
        metas      = results["metadatas"][0]
        ids        = results["ids"][0]  # always returned by ChromaDB
        distances  = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

        # ── 3. BM25 Keyword Search over the Candidate Pool ──────────────────
        t0 = time.time()
        tokenized_corpus = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))
        # Rank: highest BM25 score = rank 0
        bm25_ranks = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranks)}
        bm25_ms = (time.time() - t0) * 1000

        # ── 4. Reciprocal Rank Fusion ────────────────────────────────────────
        query_tokens = set(_tokenize(query))
        rrf_scores = {}
        for rank, idx in enumerate(range(len(docs))):
            rrf_dense = _rrf_score(rank)
            rrf_bm25 = _rrf_score(bm25_rank_map.get(idx, len(docs))) if bm25_scores[idx] > 0 else 0.0
            
            # Document Title Boost: if doc name matches query intent (e.g., "security" in title and query)
            doc_name = metas[idx].get("document_name", "").lower()
            doc_tokens = set(_tokenize(doc_name))
            title_match_count = len(query_tokens.intersection(doc_tokens))
            title_boost = title_match_count * 0.015

            rrf_scores[idx] = (1.0 * rrf_dense) + (2.5 * rrf_bm25) + title_boost

        # Sort by combined RRF score descending, keep TOP_K
        final_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:config.TOP_K]

        # ── 5. Build Output ──────────────────────────────────────────────────
        hits = []
        context_parts = []
        for rank, idx in enumerate(final_indices):
            hit = {
                "chunk": {
                    "id": ids[idx],
                    "text": docs[idx],
                    "metadata": {
                        "document_name": metas[idx].get("document_name"),
                        "category":      metas[idx].get("category"),
                        "section":       metas[idx].get("section_title", "General"),
                        "source_file":   metas[idx].get("source_file"),
                        "page_number":   metas[idx].get("page_number", 1)
                    }
                },
                "score":      float(distances[idx]),
                "rrf_score":  rrf_scores[idx],
                "bm25_score": float(bm25_scores[idx])
            }
            hits.append(hit)
            meta = hit["chunk"]["metadata"]
            context_parts.append(
                f"--- Chunk {rank + 1} [Source: {meta['document_name']}, Section: {meta['section']}] ---\n{hit['chunk']['text']}"
            )

        context_str = "\n\n".join(context_parts)
        total_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[Hybrid] Done in {total_ms:.0f}ms — "
            f"Embed: {embed_ms:.0f}ms, ChromaDB: {chroma_ms:.0f}ms, BM25+RRF: {bm25_ms:.0f}ms — "
            f"Candidates: {len(docs)}, Final: {len(hits)}"
        )
        return hits, context_str

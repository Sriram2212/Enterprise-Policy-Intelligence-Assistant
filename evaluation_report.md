# RAG Evaluation & Benchmark Report

This report presents the retrieval and generation quality audit results for the **Simple Vector RAG System**.

## 1. Executive Summary
- **Retrieval Engine**: Simple Vector Search (ChromaDB + SentenceTransformers)
- **Re-ranker**: None
- **LLM Engine**: Gemini 2.5 Flash
- **Evaluation Dataset**: 5 Core Benchmark Questions mapped to corporate handbook PDFs.

---

## 2. Overall Performance Metrics

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Average Query Latency** | 239.79 ms | End-to-end vector search retrieval time. |
| **Retrieval Success Rate** | 80.00% | Ratio of queries where the correct reference doc was retrieved. |
| **Average Precision@K** | 0.16 | Proportion of retrieved chunks that are relevant to the query. |
| **Average Recall@K** | 0.80 | Proportion of all relevant chunks that were successfully retrieved. |
| **Faithfulness Score** | 20.00% | Measure of answer truthfulness (absence of hallucinations). |
| **Answer Relevance** | 20.00% | Score measuring how well the generated answers match user intent. |

---

## 3. Query-Level Performance breakdown

| Question | Latency (ms) | Success | Precision@K | Recall@K | Faithfulness | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| How many paid public holidays does the company observe? | 783.4 | ✓ | 0.20 | 1.00 | 20.0% | 20.0% |
| What happens if an employee has a negative PTO balance? | 116.9 | ✓ | 0.20 | 1.00 | 20.0% | 20.0% |
| What is the onboarding procedure for a new employee? | 55.9 | ✓ | 0.20 | 1.00 | 20.0% | 20.0% |
| What are the rules and guidelines for travel expense reimbursement? | 104.9 | ✗ | 0.00 | 0.00 | 20.0% | 20.0% |
| What is the formal grievance procedure step? | 137.9 | ✓ | 0.20 | 1.00 | 20.0% | 20.0% |

---

## 4. Key Findings & Optimization Options
1. **Simplified Pipeline**: Removing BM25, Reciprocal Rank Fusion, and Cross-Encoder reranking reduced retrieval pipeline complexity significantly.
2. **Low Latency**: Pure vector-based searches directly queried from ChromaDB yield low retrieval response times.

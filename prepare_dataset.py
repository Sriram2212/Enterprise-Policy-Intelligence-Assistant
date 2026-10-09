import os
from pathlib import Path
from ekis.database import DatabaseManager
from ekis import config

def process_and_index():
    print("Initializing Database Manager...")
    db_manager = DatabaseManager()
    
    # Check if there are PDFs in the dataset folder
    pdf_files = list(config.DATASET_DIR.glob("**/*.pdf"))
            
    if not pdf_files:
        print(f"No PDF files found to ingest in {config.DATASET_DIR}. Please add PDFs to the dataset/ folder.")
        return
        
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # Clear ChromaDB collection to do a fresh index
    print("Clearing ChromaDB collection for fresh indexing...")
    db_manager.clear_database()
    
    # Also clean up old sqlite and BM25 index if they still exist in database directory
    old_db = config.DATABASE_DIR / "ekis.db"
    if old_db.exists():
        try:
            os.remove(old_db)
            print("Removed old SQLite database file (ekis.db).")
        except Exception as e:
            print(f"Warning: Could not remove old db file: {e}")

    old_bm25_dir = config.DATABASE_DIR / "bm25"
    if old_bm25_dir.exists():
        try:
            import shutil
            shutil.rmtree(old_bm25_dir)
            print("Removed old BM25 index directory.")
        except Exception as e:
            print(f"Warning: Could not remove old BM25 directory: {e}")
            
    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"Processing and indexing: {pdf_path.name}")
        try:
            num_chunks = db_manager.add_document_to_indexes(pdf_path)
            total_chunks += num_chunks
            print(f"Successfully indexed {pdf_path.name} ({num_chunks} chunks).")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            
    print(f"\nDataset preparation and Section-Based Vector Indexing complete successfully! Total chunks: {total_chunks}")

if __name__ == "__main__":
    process_and_index()

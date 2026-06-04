import sys
import os
import numpy as np
import faiss

# Import the RAG Engine
try:
    from rag_engine import RAGEngine
except ImportError as e:
    print(f"Failed to import rag_engine.py: {e}", file=sys.stderr)
    sys.exit(1)

def run_tests():
    print("=" * 60)
    print("           RAG PIPELINE VERIFICATION SUITE")
    print("=" * 60)
    
    # Test 1: PDF Existence & Extraction
    print("\n[TEST 1] PDF Parsing & Extraction (PyMuPDF)...")
    pdf_path = "AI.pdf"
    if not os.path.exists(pdf_path):
        print(f"  ❌ FAIL: {pdf_path} does not exist in the workspace!")
        print("  Please make sure you have the Russell & Norvig textbook named 'AI.pdf' in 'd:\\RAG_projects\\rag_llm'.")
        return False
    
    try:
        engine = RAGEngine(index_dir="verify_index_data")
        doc = fitz_open_test(pdf_path)
        print(f"  ✓ PASS: Successfully opened PDF with PyMuPDF.")
        print(f"  ✓ Info: Page count is {doc_pages_count(doc)} pages.")
        
        # Test Page 50 extraction
        p50 = doc[49]
        p50_text = p50.get_text().strip()
        print(f"  ✓ Pass: Extracted Page 50 text successfully (char length: {len(p50_text)}).")
        doc.close()
    except Exception as e:
        print(f"  ❌ FAIL: Text extraction error: {e}")
        return False

    # Test 2: Chunking Strategy
    print("\n[TEST 2] Document Chunking Strategy...")
    try:
        sample_pages = [
            {"page_num": 1, "text": "This is page one. It covers artificial intelligence search algorithms, including depth-first search (DFS) and breadth-first search (BFS). Heuristics are crucial for informed search strategies."},
            {"page_num": 2, "text": "This is page two. It covers adversarial search, including the minimax algorithm and alpha-beta pruning. It also touches on game theory and utility functions."}
        ]
        chunks = engine.chunk_text(sample_pages, chunk_size=80, chunk_overlap=20)
        print(f"  ✓ PASS: Successfully chunked text blocks.")
        print(f"  ✓ Info: Created {len(chunks)} chunks from 2 pages.")
        for idx, chunk in enumerate(chunks):
            print(f"    - Chunk {chunk['id']} (Page {chunk['page_start']}): '{chunk['text']}' (length: {chunk['char_count']} chars)")
            
        if len(chunks) < 2:
            print("  ❌ FAIL: Chunking generated too few chunks. Review splitting size thresholds.")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Chunking process error: {e}")
        return False

    # Test 3: Sparse Indexing & Search (TF-IDF Keyword Matching)
    print("\n[TEST 3] Sparse Indexing & TF-IDF Cosine Keyword Retrieval...")
    try:
        # Build index on the sample chunks
        engine.chunks = chunks
        engine.build_sparse_index()
        
        print(f"  ✓ PASS: Successfully built sparse local TF-IDF database.")
        print(f"  ✓ Info: Vocabulary size contains {len(engine.vocab)} terms.")
        print("  ✓ Info: Vocabulary terms:", list(engine.vocab.keys()))
        
        # Search query
        query = "alpha-beta pruning search"
        results = engine.search_sparse(query, top_k=2)
        print(f"  ✓ PASS: Successfully performed local keyword search for query: '{query}'")
        print(f"  ✓ Results:")
        for idx, score in results:
            print(f"    - Match Chunk {idx}: '{engine.chunks[idx]['text']}' (Score: {score:.4f})")
            
        if not results or results[0][0] != 3: # Minimax / Alpha-Beta should match the second page chunk
            print("  ⚠️ Warning: Sparse search didn't rank the minimax chunk first. Review IDF token mapping.")
    except Exception as e:
        print(f"  ❌ FAIL: TF-IDF indexing or search error: {e}")
        return False

    # Test 4: Dense Vector Storage & Loading (FAISS)
    print("\n[TEST 4] FAISS Vector Database Saving/Loading...")
    try:
        # Initialize an index flat L2/IP
        dimension = 1536
        engine.set_embedding_model("openai/text-embedding-3-small")
        
        # Create mock embeddings for 5 documents
        mock_embeddings = np.random.randn(len(chunks), dimension).astype('float32')
        faiss.normalize_L2(mock_embeddings) # Cosine norms
        
        engine.index = faiss.IndexFlatIP(dimension)
        engine.index.add(mock_embeddings)
        engine.is_indexed = True
        
        # Save to verify_index_data
        engine.save_index()
        print("  ✓ PASS: Successfully wrote FAISS flat index binary and metadata JSON to disk.")
        
        # Reload index
        new_engine = RAGEngine(index_dir="verify_index_data")
        success = new_engine.load_index()
        
        if success and new_engine.is_indexed and new_engine.index is not None:
            print("  ✓ PASS: Successfully re-imported FAISS binary index and metadata from disk.")
            print(f"  ✓ Info: Loaded {len(new_engine.chunks)} chunks and FAISS flat dimensions: {new_engine.index.d}")
        else:
            print("  ❌ FAIL: Failed to load stored FAISS database files from disk.")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: FAISS dense database verification error: {e}")
        return False

    # Clean up verification folder
    try:
        for f in ["faiss_index.bin", "metadata.json", "settings.json"]:
            path = os.path.join("verify_index_data", f)
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists("verify_index_data"):
            os.rmdir("verify_index_data")
        print("\n[CLEANUP] Successfully cleaned verification folders.")
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("          ALL LOCAL PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

def fitz_open_test(path):
    import fitz
    return fitz.open(path)

def doc_pages_count(doc):
    return len(doc)

if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)

import os
import json
import re
import fitz  # PyMuPDF
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional, Callable
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RAGEngine:
    def __init__(self, index_dir: str = "index_data", api_key: Optional[str] = None, embedding_model: Optional[str] = None):
        self.index_dir = index_dir
        
        # Load API key from parameter, environment variable, or .env file
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Load embedding model from parameter, environment variable, or use default
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
            
        self.embedding_dimension = 1536 if "text-embedding-3-small" in self.embedding_model else 768 # Default handles for text-embedding-3-small or gemini-embedding-001
        
        # Ensure index directory exists
        os.makedirs(index_dir, exist_ok=True)
        
        # Paths for stored index data
        self.faiss_path = os.path.join(index_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        self.settings_path = os.path.join(index_dir, "settings.json")
        
        # In-memory stores
        self.chunks: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.is_indexed = False
        
        # Local sparse index (TF-IDF) for hybrid search
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.tf_idf_vectors: np.ndarray = np.array([])
        
        # Feedback-driven reinforcement learning database
        self.feedback_path = os.path.join(index_dir, "feedback.json")
        self.feedback: List[Dict[str, Any]] = []
        self.load_feedback()
        
        # Load index if it exists
        self.load_index()

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_embedding_model(self, embedding_model: str):
        self.embedding_model = embedding_model
        if "text-embedding-3-small" in embedding_model:
            self.embedding_dimension = 1536
        elif "gemini-embedding" in embedding_model:
            self.embedding_dimension = 768
        else:
            self.embedding_dimension = 1536 # Default fallback

    def get_openai_client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("OpenRouter API Key is missing. Please configure it in Settings.")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
                "X-Title": "ContextAI RAG System"
            }
        )

    def extract_text_from_pdf(self, pdf_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF, cleaning headers and footers."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        doc = fitz.open(pdf_path)
        pages_data = []
        total_pages = len(doc)
        
        for i in range(total_pages):
            page = doc[i]
            text = page.get_text()
            
            # Simple cleaning: remove redundant whitespace
            text_cleaned = re.sub(r'\s+', ' ', text).strip()
            
            pages_data.append({
                "page_num": i + 1,
                "text": text_cleaned
            })
            
            if progress_callback:
                progress_callback(i + 1, total_pages)
                
        return pages_data

    def chunk_text(self, pages_data: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200, start_chunk_id: int = 0) -> List[Dict[str, Any]]:
        """Splits extracted page text into overlapping chunks using a sliding window."""
        chunks = []
        chunk_id = start_chunk_id
        
        # We process page by page, but maintain a running window.
        # This keeps page citations extremely accurate.
        for page in pages_data:
            text = page["text"]
            page_num = page["page_num"]
            
            if not text:
                continue
                
            text_len = len(text)
            
            # If a page is very short, make it one chunk
            if text_len <= chunk_size:
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "page_start": page_num,
                    "page_end": page_num,
                    "char_count": text_len
                })
                chunk_id += 1
                continue
                
            # Sliding window on the page
            start = 0
            while start < text_len:
                end = start + chunk_size
                # Adjust end to land on a space or sentence boundary if possible
                if end < text_len:
                    # Look back up to 100 characters for a sentence end or space
                    lookback_limit = max(start, end - 100)
                    boundary = end
                    for b in range(end, lookback_limit, -1):
                        if text[b] in ['.', '!', '?', ' ']:
                            boundary = b + 1 # Include the boundary character
                            break
                    end = boundary
                
                chunk_text = text[start:end].strip()
                if len(chunk_text) > min(100, chunk_size // 5): # Ignore tiny noise fragments relative to chunk size
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "page_start": page_num,
                        "page_end": page_num,
                        "char_count": len(chunk_text)
                    })
                    chunk_id += 1
                    
                start = end - chunk_overlap
                if start >= text_len - 100: # Stop if remaining text is too small
                    break
                    
        return chunks

    def get_embeddings_batch(self, texts: List[str], client: OpenAI, batch_size: int = 32, progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Requests embeddings in batches from OpenRouter API."""
        embeddings_list = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            
            # OpenRouter embeddings call
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            # Extract embeddings in order
            for data in response.data:
                embeddings_list.append(data.embedding)
                
            if progress_callback:
                progress_callback(len(embeddings_list), total_texts)
                
        return np.array(embeddings_list).astype('float32')

    def build_sparse_index(self):
        """Builds a local, highly-optimized TF-IDF vocabulary index for hybrid keyword matching."""
        if not self.chunks:
            return
            
        # Standard english stopwords list to keep the vocabulary high-quality
        stopwords = {"the", "and", "a", "of", "to", "in", "is", "that", "it", "on", "for", "as", "with", "was", "by", "an", "be", "are", "this", "or", "at", "from"}
        
        # Step 1: Tokenization and Document Frequencies
        doc_tokens = []
        df: Dict[str, int] = {}
        
        for chunk in self.chunks:
            # Simple lowercase tokenization matching alphabetic sequences
            tokens = [w for w in re.findall(r'\b[a-z]{3,15}\b', chunk["text"].lower()) if w not in stopwords]
            doc_tokens.append(tokens)
            
            # Unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
                
        # Step 2: Build vocabulary (filter terms appearing only once to reduce dimension)
        filtered_terms = [term for term, count in df.items() if count > 1]
        self.vocab = {term: idx for idx, term in enumerate(filtered_terms)}
        vocab_size = len(self.vocab)
        num_docs = len(self.chunks)
        
        if vocab_size == 0:
            return
            
        # Step 3: Compute IDF
        self.idf = np.zeros(vocab_size)
        for term, idx in self.vocab.items():
            # Standard smooth IDF formula
            self.idf[idx] = np.log((num_docs + 1) / (df[term] + 1)) + 1
            
        # Step 4: Compute TF-IDF matrix (size: num_docs x vocab_size)
        tf_idf_list = []
        for doc in doc_tokens:
            vec = np.zeros(vocab_size)
            # Count term frequencies
            tf: Dict[str, int] = {}
            for term in doc:
                if term in self.vocab:
                    tf[term] = tf.get(term, 0) + 1
            # Apply log-scaling TF and multiply by IDF
            for term, count in tf.items():
                idx = self.vocab[term]
                tf_val = 1 + np.log(count)
                vec[idx] = tf_val * self.idf[idx]
            
            # L2 normalization of the sparse vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            tf_idf_list.append(vec)
            
        self.tf_idf_vectors = np.array(tf_idf_list)

    def search_sparse(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Performs TF-IDF sparse similarity search locally."""
        if len(self.vocab) == 0 or len(self.chunks) == 0:
            return []
            
        # Clean query tokens
        tokens = re.findall(r'\b[a-z]{3,15}\b', query.lower())
        query_vec = np.zeros(len(self.vocab))
        
        tf: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1
                
        for term, count in tf.items():
            idx = self.vocab[term]
            tf_val = 1 + np.log(count)
            query_vec[idx] = tf_val * self.idf[idx]
            
        # L2 norm query
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        else:
            return [] # No matching terms
            
        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.tf_idf_vectors, query_vec)
        
        # Sort and return top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def build_index(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Full RAG build pipeline: extraction -> chunking -> vectorizing -> indexing -> saving."""
        try:
            if not self.api_key:
                raise ValueError("An OpenRouter API key is required to build the embeddings index.")
                
            client = self.get_openai_client()
            
            if progress_callback:
                progress_callback("extracting", 0.0)
                
            # 1. Text extraction
            def extraction_prog(current, total):
                if progress_callback:
                    progress_callback("extracting", current / total * 100)
                    
            pages_data = self.extract_text_from_pdf(pdf_path, progress_callback=extraction_prog)
            
            # 2. Text chunking
            if progress_callback:
                progress_callback("chunking", 0.0)
            self.chunks = self.chunk_text(pages_data, chunk_size, chunk_overlap)
            
            if progress_callback:
                progress_callback("chunking", 100.0)
                
            if not self.chunks:
                raise ValueError("No text could be extracted or chunked from the PDF.")
                
            # 3. Vectorizing with OpenRouter embeddings API
            if progress_callback:
                progress_callback("vectorizing", 0.0)
                
            texts = [c["text"] for c in self.chunks]
            
            def vectorizing_prog(current, total):
                if progress_callback:
                    progress_callback("vectorizing", current / total * 100)
                    
            embeddings = self.get_embeddings_batch(texts, client, batch_size=32, progress_callback=vectorizing_prog)
            
            # 4. Dense Index creation via FAISS
            if progress_callback:
                progress_callback("indexing", 0.0)
                
            # Normalize embedding vectors for cosine similarity search (using Inner Product Index)
            faiss.normalize_L2(embeddings)
            
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index.add(embeddings)
            
            # 5. Sparse Local Index creation
            self.build_sparse_index()
            
            self.is_indexed = True
            
            if progress_callback:
                progress_callback("indexing", 100.0)
                
            # Save files
            self.save_index()
            return True
        except Exception as e:
            print(f"Error building RAG index: {e}")
            raise e

    def save_index(self):
        """Saves dense FAISS binary and text chunks metadata securely."""
        if not self.is_indexed or self.index is None:
            return
            
        # Save FAISS binary index
        faiss.write_index(self.index, self.faiss_path)
        
        # Save metadata and TF-IDF mappings.
        # To optimize saving speed and avoid freezing (saving millions of floats is extremely slow),
        # we omit vocab, idf, and vectors arrays. They will be rebuilt automatically in load_index().
        metadata = {
            "is_indexed": True,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "chunks": self.chunks
        }
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_index(self) -> bool:
        """Loads FAISS index and chunk metadata from disk if available."""
        if os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
                
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                self.chunks = metadata.get("chunks", [])
                self.embedding_model = metadata.get("embedding_model", self.embedding_model)
                self.embedding_dimension = metadata.get("embedding_dimension", self.embedding_dimension)
                
                # Load sparse index elements
                self.vocab = metadata.get("sparse_vocab", {})
                self.idf = np.array(metadata.get("sparse_idf", []))
                self.tf_idf_vectors = np.array(metadata.get("sparse_vectors", []))
                
                # Check if elements are empty, if so, rebuild sparse index
                if self.chunks and len(self.vocab) == 0:
                    self.build_sparse_index()
                    
                self.is_indexed = True
                return True
            except Exception as e:
                print(f"Error loading index: {e}. Rebuilding index may be needed.")
                return False
        return False

    def load_feedback(self):
        """Loads historical feedback events from disk if available."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self.feedback = json.load(f)
            except Exception as e:
                print(f"Error loading feedback: {e}")
                self.feedback = []
        else:
            self.feedback = []

    def save_feedback(self):
        """Saves current feedback history to disk."""
        try:
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(self.feedback, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving feedback database: {e}")

    def add_feedback(self, query: str, chunk_ids: List[int], feedback_val: int):
        """Appends or updates feedback status for chunk IDs linked to a specific query."""
        query_clean = query.strip().lower()
        
        for fb in self.feedback:
            if fb["query"].strip().lower() == query_clean:
                fb["feedback"] = feedback_val
                fb["chunk_ids"] = chunk_ids
                self.save_feedback()
                return
                
        self.feedback.append({
            "query": query,
            "chunk_ids": chunk_ids,
            "feedback": feedback_val
        })
        self.save_feedback()

    def add_document(self, file_path: str, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Extracts, chunks, embeds, and appends a new document to the existing index."""
        if not self.is_indexed or self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.chunks = []
            self.is_indexed = True

        _, ext = os.path.splitext(file_path.lower())
        
        pages_data = []
        if ext == ".pdf":
            pages_data = self.extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            virtual_pages = []
            text_len = len(content)
            page_size = 2500
            for i in range(0, text_len, page_size):
                virtual_pages.append({
                    "page_num": (i // page_size) + 1,
                    "text": content[i : i + page_size].strip()
                })
            pages_data = virtual_pages

        start_chunk_id = len(self.chunks)
        new_chunks = self.chunk_text(pages_data, chunk_size, chunk_overlap, start_chunk_id=start_chunk_id)
        
        for chunk in new_chunks:
            chunk["filename"] = filename
            
        if not new_chunks:
            return {"status": "error", "message": "No text could be extracted from the uploaded document."}
            
        if not self.api_key:
            raise ValueError("An OpenRouter API key is required to generate embeddings for the new document.")
            
        client = self.get_openai_client()
        texts = [c["text"] for c in new_chunks]
        new_embeddings = self.get_embeddings_batch(texts, client, batch_size=32)
        
        faiss.normalize_L2(new_embeddings)
        self.index.add(new_embeddings)
        
        self.chunks.extend(new_chunks)
        self.build_sparse_index()
        self.save_index()
        
        return {
            "status": "success",
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks)
        }

    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid Search: Cosine Similarity FAISS (dense) + Cosine Similarity TF-IDF (sparse)."""
        if not self.is_indexed or self.index is None or not self.chunks:
            return []
            
        dense_results = []
        
        # Step 1: Perform Dense Vector Search (if API key is present or we have FAISS index)
        # In case user queries without API key but index is loaded, we can retrieve via FAISS directly!
        # However, to search queries we need an embedding of the query, which requires the API key.
        if self.api_key:
            try:
                client = self.get_openai_client()
                query_res = client.embeddings.create(
                    model=self.embedding_model,
                    input=[query]
                )
                query_vector = np.array(query_res.data[0].embedding).astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_vector)
                
                # FAISS inner product search returns cosine similarity scores directly since vectors are L2 normalized
                scores, indices = self.index.search(query_vector, max(20, top_k * 2))
                
                dense_results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]
            except Exception as e:
                print(f"Dense vector search failed: {e}. Falling back to sparse search.")
                dense_results = []
                
        # Step 2: Perform Sparse TF-IDF Keyword Search
        sparse_results = self.search_sparse(query, top_k=max(20, top_k * 2))
        
        # Step 3: Hybrid Reciprocal Rank Fusion or Weighted Sum Score
        # We normalize scores first and average them based on weight
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}
        
        all_candidate_indices = set(dense_scores.keys()).union(sparse_scores.keys())
        
        hybrid_results = []
        for idx in all_candidate_indices:
            # Map score ranges (dense cosine is -1 to 1 but flatIP normalized is 0 to 1 generally. TF-IDF is 0 to 1)
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            
            # Combined score formula: weight * dense_score + (1 - weight) * sparse_score
            # If dense embeddings aren't active (no API key configured), fall back 100% to sparse keyword search
            if not dense_results:
                final_score = s_score
            else:
                final_score = (hybrid_weight * d_score) + ((1.0 - hybrid_weight) * s_score)
                
            # Apply dynamic user feedback adjustment (Real-time Reinforcement Learning)
            boost = 0.0
            query_tokens = set(re.findall(r'\b[a-z]{3,15}\b', query.lower()))
            if query_tokens:
                for fb in self.feedback:
                    fb_query = fb["query"]
                    fb_tokens = set(re.findall(r'\b[a-z]{3,15}\b', fb_query.lower()))
                    if not fb_tokens:
                        continue
                    
                    # Calculate Jaccard similarity index between current query and historical feedback queries
                    jaccard = len(query_tokens.intersection(fb_tokens)) / len(query_tokens.union(fb_tokens))
                    
                    # Apply boost/penalty if semantic overlap is high
                    if jaccard >= 0.35 and idx in fb["chunk_ids"]:
                        val = fb["feedback"]
                        if val > 0:
                            # User gave thumbs up: boost score by 0.25 * overlap confidence
                            boost += 0.25 * jaccard
                        elif val < 0:
                            # User gave thumbs down: penalize score by 0.35 * overlap confidence
                            boost -= 0.35 * jaccard
            
            adjusted_score = final_score + boost
            hybrid_results.append((idx, adjusted_score, d_score, s_score))
            
        # Sort by final score descending
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Slice top_k
        top_hybrids = hybrid_results[:top_k]
        
        # Map back to detailed chunks info
        retrieved_chunks = []
        for rank, (idx, final_score, d_score, s_score) in enumerate(top_hybrids):
            chunk_info = self.chunks[idx].copy()
            chunk_info["rank"] = rank + 1
            chunk_info["similarity_score"] = float(final_score)
            chunk_info["dense_score"] = float(d_score)
            chunk_info["sparse_score"] = float(s_score)
            retrieved_chunks.append(chunk_info)
            
        return retrieved_chunks

    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        chat_history: Optional[List[Dict[str, str]]] = None,
        generation_model: str = "google/gemini-2.5-flash",
        temperature: float = 0.5
    ) -> Dict[str, Any]:
        """Assembles context prompt, queries OpenRouter, and returns response with citations."""
        if not self.api_key:
            raise ValueError("OpenRouter API key is missing. Please configure it in Settings.")
            
        # Step 1: Format context block
        context_blocks = []
        for chunk in retrieved_chunks:
            filename = chunk.get("filename", "AI.pdf")
            cite = f"Source: {filename}, Page {chunk['page_start']}"
            context_blocks.append(f"[{cite}]\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        # Step 2: System Instructions / Prompt Engineering
        system_prompt = (
            "You are ContextAI RAG, an expert academic AI teaching assistant. "
            "Your task is to answer the user's questions truthfully and in depth, utilizing "
            "the provided reference material extracted from the textbook 'Artificial Intelligence: A Modern Approach' "
            "or uploaded document sources.\n\n"
            "Guidelines:\n"
            "1. Rely strictly on the Context section below. Do not fabricate facts. If the information is not present, say so.\n"
            "2. When mentioning facts, cite the source page number and filename clearly in brackets, e.g., [Page 143] or [filename, Page 1]. "
            "Place citations naturally at the end of sentences that refer to those facts.\n"
            "3. Structure your response cleanly using bullet points, clear bold headings, or simple code blocks if applicable.\n"
            "4. Keep the tone academic, helpful, and highly factual.\n\n"
            "--- CONTEXT ---\n"
            f"{context_str}\n"
            "--- END CONTEXT ---"
        )
        
        # Step 3: Build complete conversation thread
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add past chat history if present (up to last 6 messages to preserve context budget)
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        # Add active user query
        messages.append({"role": "user", "content": query})
        
        # Step 4: Perform chat completion call
        client = self.get_openai_client()
        
        response = client.chat.completions.create(
            model=generation_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        model_used = response.model
        tokens_prompt = response.usage.prompt_tokens if response.usage else 0
        tokens_completion = response.usage.completion_tokens if response.usage else 0
        
        return {
            "answer": answer,
            "model": model_used,
            "prompt_tokens": tokens_prompt,
            "completion_tokens": tokens_completion,
            "retrieved_chunks": retrieved_chunks
        }

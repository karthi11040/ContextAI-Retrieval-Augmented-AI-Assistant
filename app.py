import os
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import fitz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag_engine import RAGEngine

# Initialize FastAPI App
app = FastAPI(
    title="ContextAI RAG Platform",
    description="High-fidelity Retrieval-Augmented Generation for the Stuart Russell & Peter Norvig AI textbook"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared RAG instance
# It will load existing index from "index_data" automatically if it exists.
rag = RAGEngine(index_dir="index_data")

# Global indexing status tracker for asynchronous polling
indexing_status = {
    "status": "idle",  # "idle", "extracting", "chunking", "vectorizing", "indexing", "completed", "error"
    "progress": 0.0,
    "pages_processed": 0,
    "total_pages": 0,
    "current_chunk": 0,
    "total_chunks": 0,
    "error": None
}

# Ensure data directory exists
os.makedirs("index_data", exist_ok=True)

# Pydantic Schemas for validation
class SettingsModel(BaseModel):
    apiKey: str = Field(..., description="OpenRouter API Key")
    embeddingModel: str = Field("openai/text-embedding-3-small", description="Model for generating embeddings")
    generationModel: str = Field("google/gemini-2.5-flash", description="Model for answering user questions")
    temperature: float = Field(0.5, description="Creativity level of the generation model")
    topK: int = Field(5, description="Number of context chunks to retrieve")
    hybridWeight: float = Field(0.5, description="Weight between vector dense search (1.0) and sparse TF-IDF keyword search (0.0)")

class SearchQueryModel(BaseModel):
    query: str = Field(..., description="Text query to search for")
    topK: Optional[int] = Field(5, description="Number of results to retrieve")
    hybridWeight: Optional[float] = Field(0.5, description="Dense vs sparse weight (0.0 to 1.0)")

class ChatMessageModel(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

class ChatQueryModel(BaseModel):
    query: str = Field(..., description="The user's active question")
    history: List[ChatMessageModel] = Field([], description="Conversation history for context continuity")
    generationModel: Optional[str] = Field(None, description="Overrides default generation model if provided")
    temperature: Optional[float] = Field(None, description="Overrides default temperature if provided")
    topK: Optional[int] = Field(None, description="Overrides default retrieval top-K if provided")
    hybridWeight: Optional[float] = Field(None, description="Overrides default hybrid weight if provided")

class IndexConfigModel(BaseModel):
    chunkSize: int = Field(1000, description="Size of text chunks in characters")
    chunkOverlap: int = Field(200, description="Overlap between consecutive chunks in characters")

class FeedbackModel(BaseModel):
    query: str = Field(..., description="The user query associated with the chat message")
    chunkIds: List[int] = Field(..., description="The retrieved chunk IDs for context")
    feedback: int = Field(..., description="1 for thumbs up, -1 for thumbs down")

# --- Indexing Background Worker ---
def run_indexing_pipeline(pdf_path: str, chunk_size: int, chunk_overlap: int):
    global indexing_status
    try:
        # Pre-inspect pages using PyMuPDF to get total pages
        doc = fitz.open(pdf_path)
        indexing_status["total_pages"] = len(doc)
        indexing_status["pages_processed"] = 0
        doc.close()
        
        indexing_status["status"] = "extracting"
        indexing_status["progress"] = 0.0
        indexing_status["error"] = None
        
        def progress_cb(stage: str, progress_val: float):
            indexing_status["status"] = stage
            indexing_status["progress"] = round(progress_val, 2)
            
            # Map detailed progress stats
            if stage == "extracting":
                indexing_status["pages_processed"] = min(
                    indexing_status["total_pages"],
                    int((progress_val / 100.0) * indexing_status["total_pages"])
                )
            elif stage == "chunking":
                # Chunking stage is extremely fast locally
                pass
            elif stage == "vectorizing":
                if indexing_status["total_chunks"] == 0:
                    # Estimate chunks created or get length of chunks
                    indexing_status["total_chunks"] = len(rag.chunks)
                indexing_status["current_chunk"] = min(
                    indexing_status["total_chunks"],
                    int((progress_val / 100.0) * indexing_status["total_chunks"])
                )
                
        # Trigger the actual build
        rag.build_index(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_cb
        )
        
        indexing_status["status"] = "completed"
        indexing_status["progress"] = 100.0
        indexing_status["total_chunks"] = len(rag.chunks)
        indexing_status["current_chunk"] = len(rag.chunks)
        
    except Exception as e:
        indexing_status["status"] = "error"
        indexing_status["error"] = str(e)
        print(f"Background indexing thread error: {e}")

# Load default settings from disk if they exist
DEFAULT_SETTINGS = {
    "embeddingModel": "openai/text-embedding-3-small",
    "generationModel": "google/gemini-2.5-flash",
    "temperature": 0.5,
    "topK": 5,
    "hybridWeight": 0.5
}

def load_saved_settings() -> Dict[str, Any]:
    settings_file = os.path.join("index_data", "settings.json")
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                saved = json.load(f)
                # Keep API key separate or load it if persisted securely
                return saved
        except Exception:
            pass
    return DEFAULT_SETTINGS

# Initialize engine with saved settings
saved_settings = load_saved_settings()
if "apiKey" in saved_settings and saved_settings["apiKey"]:
    rag.set_api_key(saved_settings["apiKey"])
if "embeddingModel" in saved_settings:
    rag.set_embedding_model(saved_settings["embeddingModel"])


# --- API Endpoints ---

@app.get("/api/status")
def get_status():
    """Returns indexing, document, and settings status info."""
    # Check FAISS index stats
    faiss_file_size = 0
    if os.path.exists(rag.faiss_path):
        faiss_file_size = os.path.getsize(rag.faiss_path)
        
    pdf_exists = os.path.exists("AI.pdf")
    
    settings = load_saved_settings()
    # Mask API key for security
    if "apiKey" in settings and settings["apiKey"]:
        settings["apiKey"] = "•" * 12 + settings["apiKey"][-4:]
        
    return {
        "indexed": rag.is_indexed,
        "pdf_loaded": pdf_exists,
        "total_chunks": len(rag.chunks),
        "total_pages": indexing_status["total_pages"] if indexing_status["total_pages"] > 0 else (len(set(c["page_start"] for c in rag.chunks)) if rag.chunks else 0),
        "vocabulary_size": len(rag.vocab),
        "faiss_size_bytes": faiss_file_size,
        "active_indexing": indexing_status,
        "settings": settings
    }

@app.post("/api/settings")
def update_settings(config: SettingsModel):
    """Saves RAG configurations to disk and applies them to memory."""
    try:
        rag.set_api_key(config.apiKey)
        rag.set_embedding_model(config.embeddingModel)
        
        # Save to disk
        settings_file = os.path.join("index_data", "settings.json")
        settings_data = {
            "apiKey": config.apiKey,
            "embeddingModel": config.embeddingModel,
            "generationModel": config.generationModel,
            "temperature": config.temperature,
            "topK": config.topK,
            "hybridWeight": config.hybridWeight
        }
        with open(settings_file, "w") as f:
            import json
            json.dump(settings_data, f, indent=2)
            
        return {"status": "success", "message": "Settings updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF or TXT document, chunks it, generates embeddings, and appends it to the FAISS index."""
    if not rag.api_key:
        raise HTTPException(status_code=400, detail="An OpenRouter API key is required to generate embeddings. Set it in Settings first.")
        
    os.makedirs("docs", exist_ok=True)
    file_path = os.path.join("docs", file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    try:
        settings = load_saved_settings()
        chunk_size = settings.get("chunkSize", 1000)
        chunk_overlap = settings.get("chunkOverlap", 200)
        
        result = rag.add_document(
            file_path=file_path,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")

@app.post("/api/index")
def start_indexing(config: IndexConfigModel, background_tasks: BackgroundTasks):
    """Starts the background thread to index the AI textbook."""
    global indexing_status
    
    if indexing_status["status"] in ["extracting", "chunking", "vectorizing", "indexing"]:
        return {"status": "in_progress", "message": "Indexing is already running."}
        
    pdf_path = "AI.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="The textbook AI.pdf was not found in the workspace directory.")
        
    if not rag.api_key:
        raise HTTPException(status_code=400, detail="An OpenRouter API key is required to build the embedding vectors. Set it in Settings first.")
        
    # Reset indexing status
    indexing_status = {
        "status": "idle",
        "progress": 0.0,
        "pages_processed": 0,
        "total_pages": 0,
        "current_chunk": 0,
        "total_chunks": 0,
        "error": None
    }
    
    # Run pipeline as background task
    background_tasks.add_task(
        run_indexing_pipeline,
        pdf_path=pdf_path,
        chunk_size=config.chunkSize,
        chunk_overlap=config.chunkOverlap
    )
    
    return {"status": "started", "message": "Book indexing started in the background."}

@app.post("/api/search")
def search_index(params: SearchQueryModel):
    """Searches the FAISS and TF-IDF sparse index for relevant context blocks."""
    if not rag.is_indexed:
        raise HTTPException(status_code=400, detail="The document index is not built yet. Please index the textbook first.")
        
    try:
        results = rag.search(
            query=params.query,
            top_k=params.topK or 5,
            hybrid_weight=params.hybridWeight if params.hybridWeight is not None else 0.5
        )
        return {"query": params.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/chat")
def chat(params: ChatQueryModel):
    """Retrieves document context and queries the OpenRouter AI generation model."""
    if not rag.is_indexed:
        raise HTTPException(status_code=400, detail="The document index is not built yet. Please index the textbook first.")
        
    if not rag.api_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key is missing. Set it in settings first.")
        
    try:
        # Load local settings for fallback overrides
        saved = load_saved_settings()
        
        # Determine active settings based on parameters or saved settings
        gen_model = params.generationModel or saved.get("generationModel", "google/gemini-2.5-flash")
        temp = params.temperature if params.temperature is not None else saved.get("temperature", 0.5)
        top_k = params.topK if params.topK is not None else saved.get("topK", 5)
        weight = params.hybridWeight if params.hybridWeight is not None else saved.get("hybridWeight", 0.5)
        
        # Step 1: Query hybrid index for context chunks
        retrieved_chunks = rag.search(query=params.query, top_k=top_k, hybrid_weight=weight)
        
        if not retrieved_chunks:
            # If no context found, perform fallback or return empty response
            return {
                "answer": "No relevant text was retrieved from the textbook context matching your query.",
                "model": gen_model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "retrieved_chunks": []
            }
            
        # Step 2: Format history to raw message structures
        history_list = []
        for h in params.history:
            history_list.append({"role": h.role, "content": h.content})
            
        # Step 3: Call generation pipeline
        response_data = rag.generate_answer(
            query=params.query,
            retrieved_chunks=retrieved_chunks,
            chat_history=history_list,
            generation_model=gen_model,
            temperature=temp
        )
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@app.post("/api/feedback")
def submit_feedback(params: FeedbackModel):
    """Submits user feedback to adjust RAG retrieval rankings dynamically."""
    try:
        rag.add_feedback(
            query=params.query,
            chunk_ids=params.chunkIds,
            feedback_val=params.feedback
        )
        return {"status": "success", "message": "Feedback recorded. System adjusted dynamically."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# Mount static frontend directory
# We ensure the static folder exists first, then mount it
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.exception_handler(404)
def not_found_handler(request, exc):
    """Fallback handler to return frontend index.html for modern single-page-app routing."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Start the API server
    uvicorn.run(app, host="127.0.0.1", port=8000)

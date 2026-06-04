# 🚀 ContextAI RAG Platform

> **Intelligent Retrieval-Augmented Generation for the Russell & Norvig AI Textbook**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Latest-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

A high-fidelity **Retrieval-Augmented Generation (RAG)** system built with FastAPI that enables intelligent querying over the Stuart Russell & Peter Norvig AI textbook. This platform combines semantic search with conversational AI to provide accurate, context-aware answers backed by direct textbook references.

## Features

- **PDF Indexing**: Automatic extraction and chunking of PDF documents with configurable chunk sizes and overlap
- **Dual-Mode Search**: Hybrid search combining dense vector embeddings with sparse TF-IDF keyword matching
- **Conversational AI**: Multi-turn chat with conversation history support
- **Feedback Loop**: Learning from user feedback to improve relevance over time
- **Real-time Progress Tracking**: Monitor indexing progress with detailed status updates
- **FastAPI Backend**: RESTful API with automatic Swagger documentation
- **Modern Web UI**: Interactive frontend for querying and chat interface
- **Configurable Models**: Support for various embedding and generation models via OpenRouter

## Project Structure

```
.
├── app.py                    # FastAPI application & API endpoints
├── rag_engine.py            # Core RAG engine for indexing & retrieval
├── verify_rag.py            # Verification & testing suite
├── AI.pdf                   # Russell & Norvig AI textbook (required)
├── index_data/              # Persistent storage for embeddings & metadata
│   ├── faiss_index.bin      # FAISS vector index
│   ├── metadata.json        # Document chunks & metadata
│   ├── settings.json        # Configuration settings
│   └── feedback.json        # User feedback database
└── static/                  # Frontend assets
    ├── index.html          # Main web interface
    ├── app.js              # Frontend logic
    └── style.css           # Styling
```

## Requirements

- Python 3.8+
- Dependencies (install via `pip install -r requirements.txt`):
  - `fastapi` - Web framework
  - `uvicorn` - ASGI server
  - `pydantic` - Data validation
  - `python-dotenv` - Environment variable management
  - `pymupdf` (fitz) - PDF text extraction
  - `faiss` - Vector similarity search
  - `numpy` - Numerical computing
  - `openai` - LLM client for OpenRouter API

## Setup & Installation

### 1. Clone/Setup Project
```bash
cd d:\RAG_projects\rag_llm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with your OpenRouter API key:

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=sk-your_api_key_here
EMBEDDING_MODEL=openai/text-embedding-3-small
GENERATION_MODEL=google/gemini-2.5-flash
```

**Get your OpenRouter API key:**
- Sign up at [openrouter.ai](https://openrouter.ai)
- Copy your API key from the dashboard
- Paste it in the `.env` file as `OPENROUTER_API_KEY`

**Note:** The `.env` file is git-ignored for security. Never commit API keys to version control.

### 4. Place the AI Textbook
Place the Russell & Norvig AI textbook as `AI.pdf` in the project root directory.

## Running the Application

### Start the Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Core Functionality

### 1. Indexing Documents
**Endpoint**: `POST /api/index`

Upload and process a PDF document:
```json
{
  "pdf_path": "AI.pdf",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Response**: Returns indexing task ID for progress tracking

### 2. Semantic Search
**Endpoint**: `POST /api/search`

Retrieve relevant document chunks:
```json
{
  "query": "What is the A* algorithm?",
  "topK": 5,
  "hybridWeight": 0.5
}
```

- `topK`: Number of results to return
- `hybridWeight`: Balance between dense search (1.0) and keyword search (0.0)

### 3. Conversational Queries
**Endpoint**: `POST /api/chat`

Ask questions with multi-turn conversation support:
```json
{
  "query": "Explain the minimax algorithm",
  "history": [
    {"role": "user", "content": "What is game theory?"},
    {"role": "assistant", "content": "..."}
  ],
  "generationModel": "google/gemini-2.5-flash",
  "temperature": 0.5,
  "topK": 5
}
```

### 4. Progress Monitoring
**Endpoint**: `GET /api/indexing-status`

Track real-time indexing progress:
```json
{
  "status": "vectorizing",
  "progress": 45.5,
  "pages_processed": 200,
  "total_pages": 500,
  "current_chunk": 1250,
  "total_chunks": 2500
}
```

### 5. User Feedback
**Endpoint**: `POST /api/feedback`

Rate retrieved results to improve future searches:
```json
{
  "query": "What is adversarial search?",
  "chunkIds": [15, 23, 45],
  "feedback": 1
}
```

## Configuration

### Environment Variables (Required)
Configure your OpenRouter API key and optional settings via `.env` file:

```env
# Required: Your OpenRouter API Key
OPENROUTER_API_KEY=sk-your_api_key_here

# Optional: Customize models
EMBEDDING_MODEL=openai/text-embedding-3-small
GENERATION_MODEL=google/gemini-2.5-flash
```

See [.env.example](.env.example) for more options.

### Runtime Settings
Settings can also be modified via API `POST /api/settings`:
- **Temperature**: Creativity level (0.0-1.0)
- **Top-K**: Number of context chunks to retrieve
- **Hybrid Weight**: Balance between dense and sparse search
- **Embedding/Generation Models**: Override defaults at runtime

### Index Configuration
- **Chunk Size**: 1000 characters (default)
- **Chunk Overlap**: 200 characters (default)

Adjust these for granularity vs. context balance.

## Verification & Testing

Run the verification suite to test core functionality:
```bash
python verify_rag.py
```

This validates:
- ✓ PDF extraction with PyMuPDF
- ✓ Document chunking strategy
- ✓ Embedding generation
- ✓ Vector index creation
- ✓ Search functionality

## How It Works

### RAG Pipeline

1. **PDF Extraction**: PyMuPDF extracts text page-by-page from the PDF
2. **Chunking**: Text is split into overlapping chunks for better context preservation
3. **Embedding**: Each chunk is embedded using OpenRouter's embedding models
4. **Indexing**: Embeddings are indexed in FAISS for efficient similarity search
5. **Retrieval**: For a query, the system retrieves relevant chunks using hybrid search
6. **Generation**: The LLM generates answers conditioned on retrieved context
7. **Feedback**: User feedback reinforces the system for future queries

### Hybrid Search
The system combines two search methods:
- **Dense Search** (Vector Similarity): Semantic meaning via embeddings
- **Sparse Search** (TF-IDF Keywords): Exact keyword matching

Controlled by `hybridWeight`:
- `1.0` = Pure vector similarity
- `0.0` = Pure keyword matching
- `0.5` = Equal weight (recommended)

## Performance Notes

- **First Run**: Initial indexing may take 5-10 minutes depending on PDF size
- **Persistent Index**: Once indexed, subsequent queries are fast (sub-second)
- **API Rate Limits**: OpenRouter has rate limits; check your account for details
- **Embedding Dimension**: Text-embedding-3-small uses 1536-dimensional vectors

## Troubleshooting

**"AI.pdf not found"**
- Ensure the Russell & Norvig AI textbook is placed in the project root

**"OpenRouter API Key missing"**
- Add `OPENROUTER_API_KEY` to your `.env` file
- Make sure the `.env` file is in the project root
- Restart the application after updating `.env`
- Verify the API key is valid at [openrouter.ai](https://openrouter.ai)

**Slow Indexing**
- Reduce `chunkSize` or increase `chunkOverlap` to process less data
- Monitor progress with the status endpoint

**Incomplete Results**
- Adjust `hybridWeight` to 1.0 for more semantic matching
- Increase `topK` to retrieve more context chunks

## Technologies Used

- **FastAPI**: Modern, fast web framework
- **FAISS**: Facebook's AI Similarity Search for efficient vector retrieval
- **PyMuPDF**: Fast PDF text extraction
- **OpenRouter**: Unified API for LLMs and embeddings
- **Pydantic**: Data validation and serialization

## License

[Add your license here]

## Support & Contributions

For issues or feature requests, please open an issue in the repository.

---

**Built with ❤️ for the ContextAI RAG Platform**

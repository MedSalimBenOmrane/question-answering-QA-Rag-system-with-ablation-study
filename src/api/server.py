"""Simple FastAPI server for RAG system API endpoints.

Provides REST API for question answering with the RAG pipeline.
"""

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.application.answer import build_pipeline
from src.cli import load_chunks, load_config, load_system_prompt

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Question-answering API using RAG pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    n_candidates: int
    n_selected: int


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline
    logger.info("Initializing RAG pipeline...")

    try:
        config = load_config()
        system_prompt = load_system_prompt()
        chunks = load_chunks(config["vectorstore"]["chunks_path"])

        pipeline = build_pipeline(config, chunks, system_prompt)
        logger.info("✓ RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question using the RAG pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"Processing question: {request.question}")
        answer = pipeline.run(request.question)

        return AnswerResponse(
            answer=answer.final_answer,
            sources=[
                {
                    "file": scored.chunk.file,
                    "score": scored.score,
                    "text": scored.chunk.text[:200] + "..." if len(scored.chunk.text) > 200 else scored.chunk.text,
                }
                for scored in answer.selected
            ],
            n_candidates=len(answer.retrieved),
            n_selected=len(answer.selected),
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )

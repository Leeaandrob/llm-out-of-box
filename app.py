"""
Author: Leandro Barbosa
Author email: email: leandrobar93@mgail.com
Main application file for Llama 3.2 3B Chat API.
"""

import uvicorn
import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import torch

# Add the current directory to sys.path to ensure all modules can be found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api.routes import router
from model.engine import ModelEngine
from model.utils import format_error_response
from config import API_CONFIG, MODEL_CONFIG

# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)

# Create model engine instance
model_engine = ModelEngine()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    # Print device information
    print("\n=== HARDWARE INFORMATION ===")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"Apple Silicon MPS available: Yes")
    else:
        print(f"Running on CPU")
        print(f"Number of CPU cores: {os.cpu_count()}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {MODEL_CONFIG['device']}")
    print(f"Selected precision: {MODEL_CONFIG['torch_dtype']}")
    print("===========================\n")

    # Initialize the model
    try:
        model_engine.initialize()
        print(f"Model initialization successful!")
    except Exception as e:
        print(f"ERROR initializing model: {str(e)}")
        print(
            "The API will start, but model operations will fail until the issue is resolved."
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        # Handle uncaught exceptions
        error_response = format_error_response(
            str(e),
            "internal_error",
            500,
        )
        return JSONResponse(
            status_code=500,
            content=error_response,
        )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_CONFIG["title"],
        "version": API_CONFIG["version"],
        "description": API_CONFIG["description"],
        "model": MODEL_CONFIG["model_id"],
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/v1/health",
        },
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )

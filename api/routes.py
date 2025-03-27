"""
API routes for Llama 3.2 3B Chat API.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.engine import ModelEngine
from model.utils import format_chat_response, format_error_response, validate_messages
from model.streaming import create_streaming_response
from config import MODEL_CONFIG
from api.schemas import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse

router = APIRouter(prefix="/v1", tags=["chat"])

# Create a singleton model engine
model_engine = ModelEngine()

def get_model_engine():
    """Dependency to get the model engine."""
    return model_engine

@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        200: {"model": ChatCompletionResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine: ModelEngine = Depends(get_model_engine),
):
    """
    Create a chat completion.
    
    This endpoint is compatible with OpenAI's chat completions API.
    """
    try:
        # Extract parameters
        messages = [message.dict() for message in request.messages]
        generation_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        
        # Check if streaming is requested
        if request.stream:
            # Generate streaming response
            generator = engine.generate_stream(messages, generation_params)
            return create_streaming_response(generator, MODEL_CONFIG["model_id"])
        else:
            # Generate non-streaming response
            result = engine.generate(messages, generation_params)
            response = format_chat_response(
                result["text"],
                result["usage"],
                MODEL_CONFIG["model_id"],
            )
            return response
            
    except Exception as e:
        # Handle errors
        error_response = format_error_response(
            str(e),
            "internal_error",
            500,
        )
        return JSONResponse(content=error_response, status_code=500)

@router.get("/models", tags=["models"])
async def list_models():
    """
    List available models.
    
    This endpoint is compatible with OpenAI's models API.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_CONFIG["model_id"],
                "object": "model",
                "created": 1677610602,
                "owned_by": "meta",
            }
        ],
    }

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

"""
Streaming functionality for the model module.
"""
from typing import Dict, Any, List, Generator, Optional, AsyncGenerator
import asyncio
import sys
import os
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.utils import format_streaming_chunk
from config import MODEL_CONFIG, STREAMING_CONFIG

async def stream_generator(
    request: Request,
    generator: Generator[Dict[str, Any], None, None],
    model_id: str = MODEL_CONFIG["model_id"]
) -> AsyncGenerator[str, None]:
    """
    Convert a synchronous generator to an asynchronous generator for SSE streaming.
    
    Args:
        request: The FastAPI request object
        generator: The synchronous generator yielding text chunks
        model_id: The ID of the model used
        
    Yields:
        Formatted SSE data strings
    """
    chunk_id = 0
    
    # Send the start message
    start_data = format_streaming_chunk("", chunk_id, model_id)
    yield start_data
    chunk_id += 1
    
    try:
        # Stream each token as it's generated
        for chunk in generator:
            if await request.is_disconnected():
                break
                
            text = chunk["text"]
            finish_reason = chunk["finish_reason"]
            
            # Send the chunk
            data = format_streaming_chunk(text, chunk_id, model_id, finish_reason)
            yield data
            chunk_id += 1
            
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)
            
    except Exception as e:
        # Send an error message if something goes wrong
        error_data = format_streaming_chunk(
            f"Error during streaming: {str(e)}", 
            chunk_id, 
            model_id, 
            "error"
        )
        yield error_data
        
    finally:
        # Send the final [DONE] message
        yield "data: [DONE]\n\n"

def create_streaming_response(
    generator: Generator[Dict[str, Any], None, None],
    model_id: str = MODEL_CONFIG["model_id"]
) -> EventSourceResponse:
    """
    Create a streaming response from a generator.
    
    Args:
        generator: The generator yielding text chunks
        model_id: The ID of the model used
        
    Returns:
        An EventSourceResponse object for streaming
    """
    return EventSourceResponse(
        stream_generator(Request, generator, model_id),
        media_type="text/event-stream",
        ping_interval=STREAMING_CONFIG["timeout"],
    )

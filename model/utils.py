"""
Utility functions for the model module.
"""
import time
from typing import Dict, Any, List, Generator, Optional
import json
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def format_chat_response(
    text: str,
    usage: Dict[str, int],
    model_id: str,
    finish_reason: Optional[str] = "stop"
) -> Dict[str, Any]:
    """
    Format the chat completion response in a structure similar to OpenAI's API.
    
    Args:
        text: The generated text
        usage: Token usage statistics
        model_id: The ID of the model used
        finish_reason: The reason why generation stopped
        
    Returns:
        Formatted response dictionary
    """
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

def format_streaming_chunk(
    text: str,
    chunk_id: int,
    model_id: str,
    finish_reason: Optional[str] = None
) -> str:
    """
    Format a streaming chunk in the SSE format compatible with OpenAI's API.
    
    Args:
        text: The generated text chunk
        chunk_id: The ID of this chunk
        model_id: The ID of the model used
        finish_reason: The reason why generation stopped (None if not finished)
        
    Returns:
        Formatted SSE data string
    """
    data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    
    return f"data: {json.dumps(data)}\n\n"

def format_error_response(error_message: str, error_type: str, status_code: int) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        error_message: The error message
        error_type: The type of error
        status_code: The HTTP status code
        
    Returns:
        Formatted error response dictionary
    """
    return {
        "error": {
            "message": error_message,
            "type": error_type,
            "code": status_code,
            "timestamp": datetime.now().isoformat(),
        }
    }

def validate_messages(messages: List[Dict[str, str]]) -> bool:
    """
    Validate that the messages list is properly formatted.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(messages, list) or not messages:
        return False
        
    for message in messages:
        if not isinstance(message, dict):
            return False
            
        if "role" not in message or "content" not in message:
            return False
            
        if message["role"] not in ["system", "user", "assistant"]:
            return False
            
        if not isinstance(message["content"], str):
            return False
            
    return True

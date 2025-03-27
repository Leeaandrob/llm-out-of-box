"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, ClassVar
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DEFAULT_GENERATION_PARAMS, PARAM_RANGES

class Message(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="The role of the message author (system, user, or assistant)")
    content: str = Field(..., description="The content of the message")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        """Validate that the role is one of the allowed values."""
        allowed_roles = ["system", "user", "assistant"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far")
    model: Optional[str] = Field(None, description="ID of the model to use")
    temperature: Optional[float] = Field(
        DEFAULT_GENERATION_PARAMS["temperature"], 
        description="What sampling temperature to use, between 0 and 2"
    )
    top_p: Optional[float] = Field(
        DEFAULT_GENERATION_PARAMS["top_p"], 
        description="Nucleus sampling parameter, between 0 and 1"
    )
    top_k: Optional[int] = Field(
        DEFAULT_GENERATION_PARAMS["top_k"], 
        description="Only sample from the top k tokens"
    )
    max_tokens: Optional[int] = Field(
        DEFAULT_GENERATION_PARAMS["max_tokens"], 
        description="Maximum number of tokens to generate"
    )
    presence_penalty: Optional[float] = Field(
        DEFAULT_GENERATION_PARAMS["presence_penalty"], 
        description="Penalty for tokens that appear in the text so far"
    )
    frequency_penalty: Optional[float] = Field(
        DEFAULT_GENERATION_PARAMS["frequency_penalty"], 
        description="Penalty for tokens based on their frequency in the text so far"
    )
    stream: Optional[bool] = Field(
        False, 
        description="Whether to stream the response"
    )
    
    @model_validator(mode='after')
    def validate_parameters(self):
        """Validate that all parameters are within their allowed ranges."""
        for param, (min_val, max_val) in PARAM_RANGES.items():
            if hasattr(self, param) and getattr(self, param) is not None:
                value = getattr(self, param)
                if value < min_val or value > max_val:
                    setattr(self, param, max(min_val, min(max_val, value)))
        return self

class ChatCompletionResponseChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionResponseUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any]

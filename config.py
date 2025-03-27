"""
Configuration settings for the Llama 3.2 3B Chat API.
"""

from typing import Dict, Any, Optional
import os
import torch


# Determine the best available device and precision
def get_optimal_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"  # Apple Silicon
    else:
        return "cpu", "float32"  # CPU fallback


device, dtype = get_optimal_device_and_dtype()

# Model configuration
MODEL_CONFIG = {
    "model_id": "fanherodev/Llama-3.2-3B-Instruct",  # Hugging Face model ID
    "device": device,  # Use best available device
    "torch_dtype": dtype,  # Use appropriate precision for the device
    "load_in_8bit": False,  # Don't use 8-bit quantization
    "trust_remote_code": True,  # Trust remote code from Hugging Face
    "max_length": 4096,  # Maximum context length
}

# API configuration
API_CONFIG = {
    "title": "Llama 3.2 3B Chat API",
    "description": "REST API for chat completions using Llama 3.2 3B model",
    "version": "0.1.0",
    "host": "0.0.0.0",
    "port": 8000,
}

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 1024,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

# Parameter validation ranges
PARAM_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (1, 100),
    "max_tokens": (1, 4096),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
}

# Streaming configuration
STREAMING_CONFIG = {
    "chunk_size": 1,  # Number of tokens to generate before yielding
    "timeout": 60,  # Timeout in seconds
}

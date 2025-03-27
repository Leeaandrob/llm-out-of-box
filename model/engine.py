"""
Model engine for loading and running inference with Llama 3.2 3B.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from typing import List, Dict, Any, Optional, Generator, Union
import threading
import os
import gc
import importlib
import sys
from tqdm import tqdm

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_CONFIG, DEFAULT_GENERATION_PARAMS, PARAM_RANGES

# Check if optimum is available for CPU optimization
OPTIMUM_AVAILABLE = importlib.util.find_spec("optimum") is not None
if OPTIMUM_AVAILABLE:
    try:
        from optimum.bettertransformer import BetterTransformer
    except ImportError:
        OPTIMUM_AVAILABLE = False

class ModelEngine:
    """
    Engine for loading and running the Llama 3.2 3B model.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = MODEL_CONFIG["device"]
        self.torch_dtype = self._get_torch_dtype(MODEL_CONFIG["torch_dtype"])
        self.model_id = MODEL_CONFIG["model_id"]
        self.is_initialized = False
        
        # Set environment variables for better CPU performance if not using CUDA
        if self.device == "cpu":
            os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() - 1))
            os.environ["MKL_NUM_THREADS"] = str(max(1, os.cpu_count() - 1))

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def initialize(self):
        """Initialize the model and tokenizer."""
        if self.is_initialized:
            return

        print(f"Loading model {self.model_id} on {self.device} with {self.torch_dtype}...")
        
        # Free up memory before loading the model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        )
        
        # Configure model loading based on device
        load_config = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": MODEL_CONFIG["trust_remote_code"],
        }
        
        # Device-specific configurations
        if self.device == "cuda":
            load_config["device_map"] = "auto"  # Let transformers decide the optimal mapping
        elif self.device == "mps":
            # For Apple Silicon
            load_config["device_map"] = None
        else:
            # For CPU, use low_cpu_mem_usage for better memory efficiency
            load_config["device_map"] = None
            load_config["low_cpu_mem_usage"] = True
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_config
        )
        
        # Move model to device if needed
        if self.device == "mps" or (self.device == "cpu" and load_config.get("device_map") is None):
            self.model = self.model.to(self.device)
        
        # Apply CPU optimizations if available
        if self.device == "cpu" and OPTIMUM_AVAILABLE:
            try:
                print("Applying CPU optimizations with BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                print("CPU optimizations applied successfully")
            except Exception as e:
                print(f"Warning: Could not apply CPU optimizations: {str(e)}")
        
        self.is_initialized = True
        print(f"Model loaded successfully on {self.device} with {self.torch_dtype}")

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clip generation parameters to their valid ranges."""
        validated = {}
        for param, value in params.items():
            if param in PARAM_RANGES:
                min_val, max_val = PARAM_RANGES[param]
                validated[param] = max(min_val, min(max_val, value))
            else:
                validated[param] = value
        return validated

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt format expected by the model."""
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Handle unknown roles as user
                prompt += f"<|user|>\n{content}\n"
                
        # Add the final assistant prefix to indicate it's the model's turn
        prompt += "<|assistant|>\n"
        return prompt

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            generation_params: Optional parameters for generation
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not self.is_initialized:
            self.initialize()
            
        # Merge default params with provided params
        params = DEFAULT_GENERATION_PARAMS.copy()
        if generation_params:
            params.update(generation_params)
            
        # Validate parameters
        params = self._validate_params(params)
        
        # Prepare input
        prompt = self._prepare_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Set up generation config
        generation_config = {
            "max_new_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repetition_penalty": 1.0 + params["presence_penalty"],
            "do_sample": params["temperature"] > 0.0,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": outputs.shape[1] - input_length,
                "total_tokens": outputs.shape[1],
            }
        }
        
    def generate_stream(
        self, 
        messages: List[Dict[str, str]], 
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a completion for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            generation_params: Optional parameters for generation
            
        Yields:
            Dictionary with generated text chunk and metadata
        """
        if not self.is_initialized:
            self.initialize()
            
        # Merge default params with provided params
        params = DEFAULT_GENERATION_PARAMS.copy()
        if generation_params:
            params.update(generation_params)
            
        # Validate parameters
        params = self._validate_params(params)
        
        # Prepare input
        prompt = self._prepare_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Set up generation config
        generation_config = {
            "max_new_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repetition_penalty": 1.0 + params["presence_penalty"],
            "do_sample": params["temperature"] > 0.0,
        }
        
        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_special_tokens=True,
            timeout=60.0
        )
        
        # Generate in a separate thread
        generation_kwargs = dict(
            **inputs,
            **generation_config,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        thread = threading.Thread(
            target=self.model.generate, 
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Yield generated tokens
        generated_tokens = 0
        for new_text in streamer:
            generated_tokens += 1
            yield {
                "text": new_text,
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": generated_tokens,
                    "total_tokens": input_length + generated_tokens,
                },
                "finish_reason": None,
            }
            
        # Final yield with finish reason
        yield {
            "text": "",
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": generated_tokens,
                "total_tokens": input_length + generated_tokens,
            },
            "finish_reason": "stop",
        }

# Llama 3.2 3B Chat API

A REST API for chat completions using Meta's Llama 3.2 3B model with PyTorch and Hugging Face Transformers. This API provides an interface similar to OpenAI's Chat Completions API, making it easy to integrate with existing applications.

## Features

- Load and run Llama 3.2 3B model with automatic hardware detection
- Support for multiple hardware configurations:
  - NVIDIA GPUs with BF16 precision for optimal performance
  - Apple Silicon with MPS acceleration
  - CPU fallback with optimizations for better performance
- REST API with FastAPI for chat completions
- Streaming support for real-time responses
- Parameter customization (temperature, top_p, top_k, etc.)
- OpenAI-compatible API format
- Health check endpoint
- Comprehensive test client with interactive and single-prompt modes

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- FastAPI 0.95+
- Uvicorn 0.22+
- Pydantic 1.10+

### Optional Dependencies
- Optimum (for CPU optimizations)
- Bitsandbytes (for memory optimization)
- Tokenizers (for better tokenization performance)
- SSE-Starlette (for server-sent events)

### Hardware Options

The API automatically detects and uses the best available hardware:

1. **NVIDIA GPU (Recommended)**: Provides the fastest inference with BF16 precision
2. **Apple Silicon**: Uses MPS acceleration with FP16 precision
3. **CPU**: Works on any system with optimizations for better performance

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/llama-chat-api.git
cd llama-chat-api
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python app.py
```

The API will be available at http://localhost:8000.

## API Usage

### Using the Test Client

The test client provides several modes of operation:

#### Interactive Chat Mode
```bash
python test_client.py
```

#### Single Prompt Mode
```bash
python test_client.py "Tell me about PyTorch"
```

#### Streaming Mode
```bash
python test_client.py --stream "Tell me about PyTorch"
```

#### With Custom Parameters
```bash
python test_client.py --temperature 0.9 --max-tokens 2000 "Tell me about PyTorch"
```

#### Custom System Message
```bash
python test_client.py --system "You are a technical expert." "Explain transformers in deep learning"
```

### Using cURL

#### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about PyTorch."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### Streaming Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about PyTorch."}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": true
  }'
```

#### List Available Models
```bash
curl http://localhost:8000/v1/models
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## Configuration

You can modify the configuration in `config.py`:

- **Model Settings**:
  - model_id: Hugging Face model ID
  - device: Force specific device (cuda, mps, cpu)
  - torch_dtype: Force specific precision (float32, float16, bfloat16)
  - max_length: Maximum context length

- **API Settings**:
  - host: API host address
  - port: API port

- **Generation Parameters**:
  - Default values for temperature, top_p, top_k, etc.
  - Parameter validation ranges

## Performance Optimization

The API includes several performance optimizations:

1. **Automatic Device Selection**: Chooses the best available hardware
2. **Precision Control**: Uses BF16 on NVIDIA GPUs, FP16 on Apple Silicon
3. **CPU Optimizations**:
   - Uses Optimum's BetterTransformer when available
   - Configures OMP and MKL threads for better performance
4. **Memory Management**:
   - Cleans up memory before model loading
   - Uses low_cpu_mem_usage for CPU mode
5. **Streaming**: Efficient token generation with minimal latency

## Development Notes

### Model Initialization

The model initializes automatically on first use with:
1. Memory cleanup
2. Optimal device selection
3. Appropriate precision settings
4. CPU optimizations if available

### API Endpoints

- `POST /v1/chat/completions`: Main chat endpoint (supports streaming)
- `GET /v1/models`: List available models
- `GET /health`: Health check endpoint

### Troubleshooting

**Common Issues**:

1. **CUDA Out of Memory**:
   - Reduce max_tokens in generation parameters
   - Restart the API to clear memory

2. **Slow Performance on CPU**:
   - Install Optimum for CPU optimizations
   - Reduce max_tokens and temperature

3. **MPS Issues on Apple Silicon**:
   - Ensure you're using PyTorch with MPS support
   - Try forcing CPU mode if unstable

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Meta AI](https://ai.meta.com/) for creating the Llama 3.2 model
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Optimum](https://huggingface.co/docs/optimum/index) for CPU optimizations

"""
API module for Llama 3.2 3B Chat API.
"""
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.routes import router

__all__ = ["router"]

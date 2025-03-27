#!/usr/bin/env python
"""
Simple test client for the Llama 3.2 3B Chat API.
"""
import argparse
import json
import requests
import time
import sys
from typing import List, Dict, Any, Optional

def print_colored(text: str, color: str = "reset") -> None:
    """Print colored text to the console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")

def chat_completion(
    messages: List[Dict[str, str]],
    api_url: str = "http://localhost:52415/v1/chat/completions",
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> None:
    """
    Send a chat completion request to the API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        api_url: URL of the API endpoint
        stream: Whether to stream the response
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
    """
    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    print_colored("\n=== REQUEST ===", "blue")
    print_colored(json.dumps(data, indent=2), "cyan")
    print_colored("==============\n", "blue")
    
    start_time = time.time()
    
    try:
        if stream:
            # Streaming request
            response = requests.post(api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            print_colored("=== STREAMING RESPONSE ===", "green")
            full_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        json_str = line[6:]  # Remove "data: " prefix
                        try:
                            chunk = json.loads(json_str)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_text += content
                        except json.JSONDecodeError:
                            print_colored(f"Error parsing JSON: {json_str}", "red")
            
            print("\n")
            print_colored("=========================", "green")
            
            elapsed_time = time.time() - start_time
            print_colored(f"\nElapsed time: {elapsed_time:.2f} seconds", "yellow")
            
            # Print token count estimate (rough estimate)
            tokens = len(full_text.split())
            print_colored(f"Approximate tokens generated: {tokens}", "yellow")
            
        else:
            # Non-streaming request
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            print_colored("=== RESPONSE ===", "green")
            content = result["choices"][0]["message"]["content"]
            print_colored(content, "reset")
            print_colored("===============\n", "green")
            
            elapsed_time = time.time() - start_time
            print_colored(f"Elapsed time: {elapsed_time:.2f} seconds", "yellow")
            print_colored(f"Tokens: {result['usage']['completion_tokens']} (completion) / {result['usage']['total_tokens']} (total)", "yellow")
            
    except requests.exceptions.RequestException as e:
        print_colored(f"Error: {str(e)}", "red")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                print_colored(json.dumps(error_data, indent=2), "red")
            except:
                print_colored(e.response.text, "red")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test client for Llama 3.2 3B Chat API")
    parser.add_argument("--url", type=str, default="http://localhost:52415/v1/chat/completions", help="API URL")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System message")
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="User prompt")
    
    args = parser.parse_args()
    
    # Check if prompt is provided
    if args.prompt is None:
        # Interactive mode
        print_colored("=== Llama 3.2 3B Chat API Test Client ===", "magenta")
        print_colored("Type 'exit' or 'quit' to exit", "magenta")
        print_colored("======================================\n", "magenta")
        
        messages = [{"role": "system", "content": args.system}]
        
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                messages.append({"role": "user", "content": user_input})
                
                chat_completion(
                    messages=messages,
                    api_url=args.url,
                    stream=args.stream,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                
                # Add the last assistant response to the messages
                if not args.stream:
                    response = requests.post(
                        args.url,
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": "llama-3.2-3b-bf16",
                            "messages": messages,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    assistant_message = result["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant_message})
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print_colored(f"Error: {str(e)}", "red")
    else:
        # Single prompt mode
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        
        chat_completion(
            messages=messages,
            api_url=args.url,
            stream=args.stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

if __name__ == "__main__":
    main()

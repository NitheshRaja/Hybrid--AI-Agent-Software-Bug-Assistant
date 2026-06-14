"""Test script for local Gemma-2B model"""

import os
from pathlib import Path

# Use absolute path
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "models" / "gemma-2-2b-it-Q4_K_M.gguf"

def test_local_llm():
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Run: python download_gemma.py")
        return
    
    file_size = MODEL_PATH.stat().st_size / (1024 * 1024 * 1024)
    print(f"Model found: {MODEL_PATH}")
    print(f"File size: {file_size:.2f} GB")
    print("")
    
    # Load model
    print("Loading Gemma-2B model...")
    from llama_cpp import Llama
    
    # Use absolute path string
    model_path_str = str(MODEL_PATH.resolve())
    print(f"Loading from: {model_path_str}")
    
    llm = Llama(
        model_path=model_path_str,
        n_ctx=2048,        # Reduced context for stability
        n_threads=4,       # Fewer threads
        n_gpu_layers=0,    # CPU only
        verbose=True       # Enable verbose to see loading details
    )
    print("Model loaded successfully!")
    print("")
    
    # Test generation
    test_prompts = [
        "What is 2 + 2?",
        "Hello, how are you?",
        "List 3 programming languages.",
    ]
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        
        # Format for Gemma instruction format
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        output = llm(
            formatted,
            max_tokens=100,
            temperature=0.7,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        
        response = output["choices"][0]["text"].strip()
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == "__main__":
    test_local_llm()



# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local LLM integration using Gemma-2B via llama-cpp-python.
Provides a hybrid approach: local model for simple tasks, Gemini for complex ones.
"""

import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional

# Model configuration
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
GEMMA_2B_FILENAME = "gemma-2-2b-it-Q4_K_M.gguf"
GEMMA_2B_URL = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class LocalLLM:
    """Local LLM wrapper using Gemma-2B"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or str(MODEL_DIR / GEMMA_2B_FILENAME)
        self._load_model()
    
    def _load_model(self):
        """Load the local model"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            print(f"Download it using: python -m software_bug_assistant.tools.local_llm --download")
            return
        
        try:
            from llama_cpp import Llama
            
            print(f"Loading Gemma-2B from {self.model_path}...")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,           # Context window
                n_threads=8,          # CPU threads
                n_gpu_layers=0,       # Set to 35+ if you have GPU
                verbose=False
            )
            print("Gemma-2B loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response using local Gemma-2B"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Download it first.")
        
        # Format prompt for Gemma instruction format
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        output = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        
        return output["choices"][0]["text"].strip()


class HybridLLM:
    """
    Hybrid LLM that routes between local Gemma-2B and Gemini API.
    - Simple tasks: Local Gemma-2B (fast, free, offline)
    - Complex tasks: Gemini API (powerful, online)
    """
    
    def __init__(self, local_model_path: Optional[str] = None, always_use_gemini: bool = False):
        self.always_use_gemini = always_use_gemini
        self.local_llm = None
        
        if not always_use_gemini:
            try:
                self.local_llm = LocalLLM(local_model_path)
            except Exception as e:
                print(f"Local LLM not available: {e}")
    
    def classify_complexity(self, query: str) -> TaskComplexity:
        """Determine if task is simple or complex based on patterns"""
        query_lower = query.lower().strip()
        
        # Simple task patterns - handled by local model
        simple_patterns = [
            r"^(hi|hello|hey|thanks|thank you|bye|goodbye)",  # Greetings
            r"what is \d+\s*[\+\-\*\/]\s*\d+",                # Basic math
            r"^(yes|no|ok|okay|sure|got it)",                  # Acknowledgments
            r"^define\s+\w+$",                                 # Simple definitions
            r"^what (is|are) (the )?(time|date|day)",         # Time/date
            r"^(list|show|get) (all )?(open |closed )?tickets?(\s+by)?", # Simple ticket queries
            r"^(show|get|display) ticket (id )?\d+",          # Get specific ticket
            r"^how many tickets",                              # Count queries
            r"^what is the (status|priority) of",             # Status queries
        ]
        
        # Complex task patterns - needs Gemini
        complex_patterns = [
            r"(analyze|explain|compare|evaluate|assess)",
            r"(write|create|generate|build) (a |an )?(code|script|program|function)",
            r"(search|find|look up|research).*(internet|web|online|github|stackoverflow)",
            r"(debug|fix|troubleshoot|solve|resolve)",
            r"CVE-\d+|vulnerability|security|exploit",
            r"(plan|strategy|roadmap|architecture)",
            r"(summarize|review|critique).*(article|paper|document|code)",
            r"(why|how come|what causes|reason for)",
            r"(recommend|suggest|advise|best practice)",
            r"(integrate|connect|setup|configure|deploy)",
        ]
        
        # Check complex patterns first (higher priority)
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                return TaskComplexity.COMPLEX
        
        # Check simple patterns
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return TaskComplexity.SIMPLE
        
        # Short queries are often simple
        if len(query.split()) <= 5:
            return TaskComplexity.SIMPLE
        
        # Default to complex for safety
        return TaskComplexity.COMPLEX
    
    def generate(
        self,
        prompt: str,
        force_local: bool = False,
        force_gemini: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> tuple[str, str]:
        """
        Generate response using the appropriate model.
        
        Returns:
            tuple: (response_text, model_used)
        """
        complexity = self.classify_complexity(prompt)
        
        # Determine which model to use
        use_local = (
            (force_local or complexity == TaskComplexity.SIMPLE)
            and not force_gemini
            and self.local_llm is not None
            and self.local_llm.is_loaded()
            and not self.always_use_gemini
        )
        
        if use_local:
            try:
                response = self.local_llm.generate(prompt, max_tokens, temperature)
                return response, "gemma-2b-local"
            except Exception as e:
                print(f"Local generation failed, falling back to Gemini: {e}")
        
        # Use Gemini
        response = self._gemini_generate(prompt)
        return response, "gemini"
    
    def _gemini_generate(self, prompt: str) -> str:
        """Generate using Gemini API"""
        import google.generativeai as genai
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text


def download_gemma_model():
    """Download Gemma-2B GGUF model from HuggingFace"""
    import urllib.request
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / GEMMA_2B_FILENAME
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return str(model_path)
    
    print(f"Downloading Gemma-2B ({GEMMA_2B_FILENAME})...")
    print(f"URL: {GEMMA_2B_URL}")
    print(f"Destination: {model_path}")
    print("This may take a few minutes (~1.5GB)...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(GEMMA_2B_URL, model_path, progress_hook)
        print(f"\nDownload complete! Model saved to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nYou can manually download from:")
        print(f"  {GEMMA_2B_URL}")
        print(f"And save to: {model_path}")
        return None


if __name__ == "__main__":
    import sys
    
    if "--download" in sys.argv:
        download_gemma_model()
    elif "--test" in sys.argv:
        print("Testing Local LLM...")
        llm = LocalLLM()
        if llm.is_loaded():
            response = llm.generate("What is 2 + 2?")
            print(f"Response: {response}")
        else:
            print("Model not loaded. Run with --download first.")
    elif "--hybrid-test" in sys.argv:
        print("Testing Hybrid LLM...")
        hybrid = HybridLLM()
        
        test_queries = [
            "Hello!",
            "What is 25 + 17?",
            "Analyze the security implications of CVE-2024-3094",
            "Show all open tickets",
        ]
        
        for query in test_queries:
            complexity = hybrid.classify_complexity(query)
            print(f"\nQuery: {query}")
            print(f"Complexity: {complexity.value}")
    else:
        print("Gemma-2B Local LLM Module")
        print("Usage:")
        print("  python -m software_bug_assistant.tools.local_llm --download    Download Gemma-2B model")
        print("  python -m software_bug_assistant.tools.local_llm --test        Test local generation")
        print("  python -m software_bug_assistant.tools.local_llm --hybrid-test Test complexity classifier")








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
Hybrid Agent that combines local Gemma-2B with cloud Gemini.
- Simple queries → Local Gemma-2B (fast, free, offline)
- Complex queries → Cloud Gemini via ADK (powerful, tool-enabled)

Gemma-2B acts as Tandion software bug expert, following the same role as Gemini.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# Import the agent instruction to maintain role consistency
try:
    from .prompt import agent_instruction
except ImportError:
    # Fallback if import fails
    agent_instruction = """You are a skilled expert in triaging and debugging software issues for a IT Software company named "Tandion"."""

# Model configuration
MODEL_DIR = Path(__file__).parent.parent / "models"
GEMMA_MODEL = MODEL_DIR / "gemma-2-2b-it-Q4_K_M.gguf"


class QueryClassifier:
    """Classifies queries as simple (local) or complex (cloud)"""
    
    # Simple patterns - can be handled by local Gemma-2B
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
        r"^(thanks|thank you|thx|ty|bye|goodbye|see you)",
        r"^(yes|no|ok|okay|sure|got it|understood|alright)",
        r"^what is \d+\s*[\+\-\*\/\%]\s*\d+",  # Basic math
        r"^(calculate|compute|what's|whats)\s+\d+",  # Math queries
        r"^define\s+\w+$",  # Simple definitions
        r"^(what|when) (is|was) (the )?(time|date|day|year)",
        r"^(who|what) (is|are|was|were) [a-z]+\??$",  # Simple who/what
        r"^how do you say .+ in \w+",  # Simple translation
        r"^(tell me a joke|make me laugh)",
        r"^(what's the weather|weather)",
    ]
    
    # Complex patterns - need Gemini + tools
    COMPLEX_PATTERNS = [
        # Tool-requiring patterns
        r"(ticket|tickets|bug|bugs|issue|issues)",  # Needs database tools
        r"(search|find|look up|lookup).*(github|stackoverflow|web|internet)",
        r"CVE-\d+|vulnerability|security|exploit|attack",
        
        # Analysis patterns
        r"(analyze|explain|compare|evaluate|assess|review)",
        r"(debug|fix|troubleshoot|solve|resolve|investigate)",
        r"(write|create|generate|build|implement) (a |an )?(code|script|program|function|class)",
        
        # Planning patterns  
        r"(plan|strategy|roadmap|architecture|design)",
        r"(recommend|suggest|advise|best practice)",
        r"(how (do|can|should) (i|we|you)|how to)",
        
        # Multi-step patterns
        r"(and then|after that|next|finally|step by step)",
        r"(list all|show all|get all|find all)",
    ]
    
    @classmethod
    def is_simple(cls, query: str) -> bool:
        """Check if query is simple enough for local model"""
        query_lower = query.lower().strip()
        
        # Check complex patterns first (higher priority)
        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return False
        
        # Check simple patterns
        for pattern in cls.SIMPLE_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        
        # Very short queries are often simple
        word_count = len(query.split())
        if word_count <= 4 and "?" not in query:
            return True
            
        return False  # Default to complex for safety
    
    @classmethod
    def get_classification(cls, query: str) -> str:
        """Get classification with reason"""
        if cls.is_simple(query):
            return "SIMPLE → Local Gemma-2B"
        return "COMPLEX → Cloud Gemini + Tools"


class LocalGemmaModel:
    """Local Gemma-2B model wrapper"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Gemma-2B model"""
        if not GEMMA_MODEL.exists():
            print(f"⚠ Gemma model not found at {GEMMA_MODEL}")
            print("  Run: python download_gemma.py")
            return
        
        try:
            from llama_cpp import Llama
            
            print("Loading Gemma-2B for simple queries...")
            self.model = Llama(
                model_path=str(GEMMA_MODEL),
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0,
                verbose=False
            )
            print("✓ Gemma-2B ready!")
        except Exception as e:
            print(f"⚠ Failed to load Gemma: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def generate(self, prompt: str, max_tokens: int = 256, context: Optional[dict] = None) -> str:
        """
        Generate dynamic response using Gemma-2B with context awareness.
        
        Args:
            prompt: User query
            max_tokens: Maximum tokens to generate
            context: Optional context dict with:
                - current_time: Current date/time
                - user_name: User's name
                - conversation_history: Previous messages
                - system_info: System information
        """
        if not self.is_available():
            raise RuntimeError("Gemma model not loaded")
        
        # Build dynamic context-aware prompt
        system_context = self._build_context(context)
        
        # Enhanced prompt with context
        enhanced_prompt = self._enhance_prompt(prompt, system_context)
        
        # Format for Gemma instruction format
        formatted = f"<start_of_turn>user\n{enhanced_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Use higher temperature for more dynamic responses
        output = self.model(
            formatted,
            max_tokens=max_tokens,
            temperature=0.8,  # Increased for more variety
            top_p=0.9,        # Nucleus sampling for diversity
            repeat_penalty=1.1,  # Reduce repetition
            stop=["<end_of_turn>", "<start_of_turn>", "\n\n\n"],
            echo=False
        )
        
        response = output["choices"][0]["text"].strip()
        
        # Post-process to ensure dynamic nature
        return self._post_process_response(response, prompt, context)
    
    def _build_context(self, context: Optional[dict]) -> str:
        """Build context string from context dict"""
        if not context:
            from datetime import datetime
            context = {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        
        context_parts = []
        
        if "current_time" in context:
            context_parts.append(f"Current date and time: {context['current_time']}")
        
        if "user_name" in context:
            context_parts.append(f"User's name: {context['user_name']}")
        
        if "conversation_history" in context and context["conversation_history"]:
            history = context["conversation_history"][-3:]  # Last 3 messages
            context_parts.append("Recent conversation:")
            for msg in history:
                context_parts.append(f"  {msg}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _enhance_prompt(self, prompt: str, context: str) -> str:
        """Enhance prompt with Tandion expert role and context"""
        # Extract the core role from agent_instruction
        role_instruction = agent_instruction.strip()
        
        # Add instructions for staying in character
        character_guidance = """
**IMPORTANT - STAY IN CHARACTER:**
- You are a skilled expert in triaging and debugging software issues for Tandion IT Software company
- Always identify yourself as working for Tandion when appropriate
- Maintain your professional expertise role - you are NOT a generic assistant
- Be helpful, professional, and focused on software bug triaging
- For simple queries (greetings, basic questions), respond naturally but remember your role
- If asked about your identity, mention you're a Tandion software bug expert
- Keep responses dynamic and contextual, but always maintain your professional identity
"""
        
        # Build the enhanced prompt
        parts = [role_instruction, character_guidance]
        
        if context:
            parts.append(f"\n**Context:**\n{context}")
        
        parts.append(f"\n**User Query:** {prompt}")
        parts.append("\n**Your Response (stay in character as Tandion expert):**")
        
        enhanced = "\n".join(parts)
        return enhanced
    
    def _post_process_response(self, response: str, original_prompt: str, context: Optional[dict]) -> str:
        """Post-process response to ensure it maintains Tandion expert role"""
        # Check if response mentions identity - if greeting or intro question
        prompt_lower = original_prompt.lower()
        response_lower = response.lower()
        
        # If user asks about identity/role, ensure Tandion is mentioned
        if any(phrase in prompt_lower for phrase in ["who are you", "what are you", "introduce", "your role", "your job"]):
            if "tandion" not in response_lower:
                # Add Tandion identity if missing
                if response.strip():
                    response = f"I'm a software bug triaging expert at Tandion IT Software company. {response}"
                else:
                    response = "I'm a skilled expert in triaging and debugging software issues for Tandion IT Software company. How can I help you today?"
        
        # For greetings, add professional context if missing
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]) and "tandion" not in response_lower:
            # Don't force it, but ensure professional tone
            if not any(word in response_lower for word in ["tandion", "software", "bug", "issue", "expert"]):
                # It's a simple greeting, keep it natural but professional
                pass
        
        # Remove any generic prefixes that break character
        generic_prefixes = [
            "Sure!",
            "Of course!",
            "I'd be happy to help!",
        ]
        
        for prefix in generic_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                if response.startswith(","):
                    response = response[1:].strip()
        
        # Ensure response maintains professional tone
        if not response or len(response) < 3:
            response = "I'm here to help with software bug triaging at Tandion. How can I assist you?"
        
        return response


class HybridAgent:
    """
    Hybrid agent that routes queries between local and cloud models.
    
    Flow:
    1. User sends query
    2. QueryClassifier determines: simple or complex?
    3. Simple → Local Gemma-2B responds directly
    4. Complex → Pass to ADK agent (Gemini + tools)
    """
    
    def __init__(self, enable_local: bool = True):
        self.local_model = None
        self.enable_local = enable_local
        
        if enable_local:
            self.local_model = LocalGemmaModel()
    
    def process_query(self, query: str, context: Optional[dict] = None) -> dict:
        """
        Process a query and return response with metadata.
        
        Args:
            query: User's query
            context: Optional context dict with:
                - current_time: Current date/time
                - user_name: User's name
                - conversation_history: List of previous messages
                - system_info: System information
        
        Returns:
            dict: {
                "response": str,           # The response text
                "model_used": str,         # "gemma-2b-local" or "gemini-cloud"
                "classification": str,     # "SIMPLE" or "COMPLEX"
                "handled_locally": bool    # Whether handled by local model
            }
        """
        is_simple = QueryClassifier.is_simple(query)
        
        # Try local model for simple queries
        if is_simple and self.local_model and self.local_model.is_available():
            try:
                # Build dynamic context if not provided
                if context is None:
                    from datetime import datetime
                    context = {
                        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                
                # Generate dynamic response with context
                response = self.local_model.generate(query, max_tokens=256, context=context)
                return {
                    "response": response,
                    "model_used": "gemma-2b-local",
                    "classification": "SIMPLE",
                    "handled_locally": True
                }
            except Exception as e:
                print(f"Local model failed, falling back to cloud: {e}")
        
        # Complex queries or local model unavailable → return None
        # (Let the main ADK agent handle it)
        return {
            "response": None,
            "model_used": "gemini-cloud",
            "classification": "COMPLEX" if not is_simple else "SIMPLE",
            "handled_locally": False
        }


# Singleton instance
_hybrid_agent: Optional[HybridAgent] = None


def get_hybrid_agent() -> HybridAgent:
    """Get or create the hybrid agent singleton"""
    global _hybrid_agent
    if _hybrid_agent is None:
        _hybrid_agent = HybridAgent()
    return _hybrid_agent


def classify_query(query: str) -> str:
    """Quick utility to classify a query"""
    return QueryClassifier.get_classification(query)


def handle_simple_query(query: str, context: Optional[dict] = None) -> Optional[str]:
    """
    Handle a simple query with local model.
    
    Args:
        query: User's query
        context: Optional context dict for dynamic responses
    
    Returns:
        Response string if handled locally, None if should use cloud.
    """
    agent = get_hybrid_agent()
    result = agent.process_query(query, context=context)
    
    if result["handled_locally"]:
        return result["response"]
    return None


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Agent Test")
    print("=" * 60)
    
    test_queries = [
        # Simple queries (should use Gemma)
        "Hello!",
        "What is 15 + 27?",
        "Thank you!",
        "Define algorithm",
        
        # Complex queries (should use Gemini)
        "Show me all open tickets",
        "Analyze CVE-2024-3094 vulnerability",
        "Search GitHub for Python async issues",
        "Debug this code and fix the error",
        "Create a function to sort a list",
    ]
    
    print("\nQuery Classification:")
    print("-" * 60)
    for query in test_queries:
        classification = classify_query(query)
        print(f"'{query}'")
        print(f"  → {classification}")
        print()
    
    print("\n" + "=" * 60)
    print("Testing Local Model Responses:")
    print("=" * 60)
    
    agent = HybridAgent()
    
    simple_queries = ["Hello!", "What is 25 + 17?", "Tell me a joke"]
    for query in simple_queries:
        print(f"\nQuery: {query}")
        result = agent.process_query(query)
        if result["handled_locally"]:
            print(f"Model: {result['model_used']}")
            print(f"Response: {result['response']}")
        else:
            print(f"→ Would be sent to: {result['model_used']}")


#!/usr/bin/env python
"""
Hybrid Agent Runner - Demonstrates local + cloud model routing

Usage:
    python run_hybrid_agent.py

This script shows how queries are routed:
- Simple queries → Local Gemma-2B (fast, free)
- Complex queries → Cloud Gemini + Tools (powerful)
"""

import os
import sys

# Set environment BEFORE any imports to avoid Google Cloud auth requirement
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("SKIP_AGENT_IMPORT", "TRUE")  # Skip agent import in __init__.py

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🤖 Hybrid Software Bug Assistant 🤖                 ║
║                                                              ║
║   Simple queries  →  Gemma-2B (Local, Free, Fast)           ║
║   Complex queries →  Gemini (Cloud, Tools, Powerful)        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Import hybrid agent (auth is handled in __init__.py with error handling)
    from software_bug_assistant.hybrid_agent import HybridAgent, classify_query
    
    print("Initializing hybrid agent...")
    agent = HybridAgent(enable_local=True)
    
    if agent.local_model and agent.local_model.is_available():
        print("✓ Local Gemma-2B: READY")
    else:
        print("✗ Local Gemma-2B: NOT AVAILABLE (will use cloud for all queries)")
    
    print("✓ Cloud Gemini: READY (via ADK)")
    print()
    print("=" * 60)
    print("Type your queries below. Type 'quit' to exit.")
    print("Type 'classify <query>' to see how a query would be routed.")
    print("=" * 60)
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            # Classification mode
            if query.lower().startswith('classify '):
                test_query = query[9:]
                classification = classify_query(test_query)
                print(f"📊 Classification: {classification}")
                print()
                continue
            
            # Build dynamic context
            from datetime import datetime
            context = {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_name": "User",  # Could be retrieved from session
            }
            
            # Process the query with context
            result = agent.process_query(query, context=context)
            
            if result["handled_locally"]:
                print(f"🏠 [Gemma-2B Local] {result['response']}")
                print(f"   (Dynamic response with context awareness)")
            else:
                print(f"☁️  [Routing to Gemini Cloud + Tools]")
                print(f"    This query needs: database tools, web search, or complex reasoning")
                print(f"    → Use the ADK web interface for full functionality")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""Test script to find the correct way to invoke ADK Agent"""

import os
import sys

# Set environment
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("SKIP_AGENT_IMPORT", "FALSE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from software_bug_assistant.agent import root_agent
    
    print("ADK Agent loaded successfully!")
    print(f"Agent type: {type(root_agent)}")
    print(f"Agent: {root_agent}")
    print()
    
    # Check available methods
    print("Available methods (non-private):")
    methods = [m for m in dir(root_agent) if not m.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    print()
    
    # Try to invoke
    test_query = "Hello, what is 2+2?"
    print(f"Testing with query: '{test_query}'")
    print()
    
    # Try callable
    if callable(root_agent):
        print("Agent is callable, trying direct call...")
        try:
            result = root_agent(test_query)
            print(f"✓ Direct call worked! Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"✗ Direct call failed: {e}")
    
    # Try invoke
    if hasattr(root_agent, 'invoke'):
        print("\nTrying invoke method...")
        try:
            result = root_agent.invoke(test_query)
            print(f"✓ Invoke worked! Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"✗ Invoke failed: {e}")
    
    # Try run
    if hasattr(root_agent, 'run'):
        print("\nTrying run method...")
        try:
            result = root_agent.run(test_query)
            print(f"✓ Run worked! Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"✗ Run failed: {e}")
    
except Exception as e:
    print(f"Error loading agent: {e}")
    import traceback
    traceback.print_exc()


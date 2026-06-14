#!/usr/bin/env python
"""
Test script to demonstrate dynamic responses from Gemma-2B
Shows how the same query gets different responses based on context
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from software_bug_assistant.hybrid_agent import HybridAgent, LocalGemmaModel

def test_dynamic_responses():
    print("=" * 70)
    print("Testing Dynamic Responses from Gemma-2B")
    print("=" * 70)
    print()
    
    # Initialize model
    print("Loading Gemma-2B...")
    model = LocalGemmaModel()
    
    if not model.is_available():
        print("❌ Model not available. Run: python download_gemma.py")
        return
    
    print("✓ Model loaded!\n")
    
    # Test 1: Same query with different contexts
    print("Test 1: Same Query, Different Contexts")
    print("-" * 70)
    
    query = "What time is it?"
    
    # Morning context
    morning_time = datetime.now().replace(hour=9, minute=30)
    context1 = {"current_time": morning_time.strftime("%Y-%m-%d %H:%M:%S")}
    
    # Evening context
    evening_time = datetime.now().replace(hour=21, minute=15)
    context2 = {"current_time": evening_time.strftime("%Y-%m-%d %H:%M:%S")}
    
    print(f"Query: '{query}'")
    print(f"\nContext 1 (Morning): {context1['current_time']}")
    response1 = model.generate(query, context=context1)
    print(f"Response: {response1}")
    
    print(f"\nContext 2 (Evening): {context2['current_time']}")
    response2 = model.generate(query, context=context2)
    print(f"Response: {response2}")
    
    print("\n" + "=" * 70)
    print("Test 2: Multiple Responses to Same Query (Variety)")
    print("-" * 70)
    
    query = "Hello!"
    print(f"Query: '{query}'")
    print("\nGenerating 3 different responses:")
    
    for i in range(3):
        response = model.generate(query, max_tokens=100)
        print(f"\nResponse {i+1}: {response}")
    
    print("\n" + "=" * 70)
    print("Test 3: Context-Aware Responses")
    print("-" * 70)
    
    # With user name
    context_with_name = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_name": "Alice"
    }
    
    query = "How are you?"
    print(f"Query: '{query}'")
    print(f"Context: User name = {context_with_name['user_name']}")
    response = model.generate(query, context=context_with_name)
    print(f"Response: {response}")
    
    print("\n" + "=" * 70)
    print("Test 4: Math Queries (Dynamic Calculations)")
    print("-" * 70)
    
    math_queries = [
        "What is 15 + 27?",
        "Calculate 100 * 5",
        "What's 144 divided by 12?",
    ]
    
    for query in math_queries:
        response = model.generate(query)
        print(f"Query: {query}")
        print(f"Response: {response}\n")
    
    print("=" * 70)
    print("✓ Dynamic response testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_dynamic_responses()






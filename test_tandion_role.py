#!/usr/bin/env python
"""
Test script to verify Gemma-2B maintains Tandion expert role
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from software_bug_assistant.hybrid_agent import LocalGemmaModel

def test_tandion_role():
    print("=" * 70)
    print("Testing Gemma-2B Tandion Expert Role")
    print("=" * 70)
    print()
    
    # Initialize model
    print("Loading Gemma-2B as Tandion expert...")
    model = LocalGemmaModel()
    
    if not model.is_available():
        print("❌ Model not available. Run: python download_gemma.py")
        return
    
    print("✓ Model loaded!\n")
    
    # Test queries that should show Tandion identity
    test_queries = [
        # Identity questions
        ("Who are you?", "Should identify as Tandion expert"),
        ("What is your role?", "Should mention Tandion and bug triaging"),
        ("Introduce yourself", "Should introduce as Tandion expert"),
        
        # Greetings (should be professional but natural)
        ("Hello!", "Should greet professionally"),
        ("Hi there!", "Should respond naturally but remember role"),
        
        # Simple queries (should stay in character)
        ("What is 2 + 2?", "Should answer but maintain professional tone"),
        ("Thank you!", "Should acknowledge professionally"),
        
        # Bug-related (should be in full character)
        ("I have a bug", "Should respond as Tandion bug expert"),
        ("Help me with an issue", "Should offer Tandion expertise"),
    ]
    
    print("Testing Role Consistency:")
    print("-" * 70)
    
    for query, expected in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected}")
        
        context = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        response = model.generate(query, context=context)
        print(f"Response: {response}")
        
        # Check if Tandion is mentioned when appropriate
        if any(phrase in query.lower() for phrase in ["who", "what", "introduce", "role", "job"]):
            if "tandion" in response.lower():
                print("✓ Tandion identity maintained!")
            else:
                print("⚠ Tandion identity not mentioned (may need adjustment)")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Role Consistency Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_tandion_role()






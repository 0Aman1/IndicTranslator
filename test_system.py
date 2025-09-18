#!/usr/bin/env python3
"""
Test script for the Multilingual NLP System
"""

from multilingual_nlp import MultilingualNLPSystem
import json

def test_system():
    """Test the multilingual NLP system with sample inputs."""
    
    system = MultilingualNLPSystem()
    
    # Test cases
    test_cases = [
        {
            "text": "I love this product! It's amazing.",
            "target_languages": ["hi", "ta"]
        },
        {
            "text": "यह बहुत बुरा है, मुझे नापसंद है।",
            "target_languages": ["en", "ta"]
        },
        {
            "text": "இது மிகவும் நன்றாக உள்ளது!",
            "target_languages": ["en", "hi"]
        }
    ]
    
    print("🚀 Testing Multilingual NLP System")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input: {test['text']}")
        print(f"Target languages: {', '.join(test['target_languages'])}")
        print("-" * 30)
        
        try:
            result = system.process_text(test['text'], test['target_languages'])
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_system()
#!/usr/bin/env python3
"""
Simple test for enhanced retrieval system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_enhanced_retriever():
    """Test the enhanced retriever class"""
    try:
        from app import EnhancedRetriever, RAGConfig
        
        print("🧪 Testing Enhanced Retriever")
        print("=" * 40)
        
        # Create config
        config = RAGConfig()
        
        # Create enhanced retriever
        retriever = EnhancedRetriever(config)
        
        # Test query enhancement
        test_queries = [
            "How do I configure storage?",
            "What is monitoring?",
            "How to setup SolutionPacks?",
            "What are the requirements?"
        ]
        
        print("\n🔍 Testing Query Enhancement:")
        for query in test_queries:
            enhanced = retriever.enhance_query(query)
            print(f"   Original: {query}")
            print(f"   Enhanced: {enhanced}")
            print()
        
        print("✅ Enhanced retriever test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_retriever()
    sys.exit(0 if success else 1)

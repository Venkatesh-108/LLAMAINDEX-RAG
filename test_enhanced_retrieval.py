#!/usr/bin/env python3
"""
Test script for enhanced retrieval system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from app import DellSRMRAG

def test_enhanced_retrieval():
    """Test the enhanced retrieval system"""
    print("🧪 Testing Enhanced Retrieval System")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag = DellSRMRAG()
        
        # Test initialization
        if not rag.initialize():
            print("❌ Failed to initialize RAG system")
            return False
        
        print("✅ RAG system initialized successfully")
        
        # Test basic query
        test_query = "What are the system requirements for Dell SRM?"
        print(f"\n🔍 Testing query: {test_query}")
        
        result = rag.query(test_query)
        
        if "error" in result:
            print(f"❌ Query failed: {result['error']}")
            return False
        
        print("✅ Query executed successfully")
        print(f"   - Data available: {result.get('data_available', False)}")
        print(f"   - Sources found: {len(result.get('sources', []))}")
        print(f"   - Query time: {result.get('query_time', 'N/A')}s")
        
        # Test advanced features
        if hasattr(rag.vector_store, 'advanced_retriever'):
            print("\n🔧 Advanced retrieval features:")
            print(f"   - Hybrid retrieval: {rag.config.use_hybrid_retrieval}")
            print(f"   - Multi-query: {rag.config.use_multi_query}")
            print(f"   - LLM reranking: {rag.config.use_reranking}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_retrieval()
    sys.exit(0 if success else 1)

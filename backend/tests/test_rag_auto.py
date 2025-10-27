"""
Automated RAG Endpoints Testing
Tests RAG query, semantic search, and product-specific queries
"""
import sys
import os
import time

# Add parent directory to path for Windows compatibility
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000/api/v1/rag"

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_stats():
    """Test stats endpoint"""
    print_section("Test 1: Vector Database Stats")

    try:
        response = requests.get(f"{BASE_URL}/stats")
        response.raise_for_status()
        stats = response.json()

        print("\n[OK] Stats retrieved successfully")
        print(f"\nTotal reviews: {stats['vector_db']['total_reviews']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Embedding dimension: {stats['vector_db']['embedding_dim']}")
        print(f"Ollama available: {stats['ollama_available']}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False

def test_semantic_search():
    """Test semantic search endpoint"""
    print_section("Test 2: Semantic Search - Battery Life")

    params = {
        "query": "battery life and charging issues",
        "n_results": 3
    }

    try:
        response = requests.get(f"{BASE_URL}/search/semantic", params=params)
        response.raise_for_status()
        results = response.json()

        print("\n[OK] Semantic search successful")
        print(f"\nQuery: {params['query']}")
        print(f"Found {results['count']} relevant reviews")

        for i, result in enumerate(results['results'][:3], 1):
            print(f"\n  Result {i}:")
            print(f"  Text: {result['text'][:80]}...")
            print(f"  Distance: {result.get('distance', 'N/A')}")
            print(f"  Product: {result['metadata']['product_id']}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False

def test_rag_query():
    """Test RAG query endpoint"""
    print_section("Test 3: RAG Query - iPhone Camera")

    payload = {
        "query": "What do people think about iPhone camera quality?",
        "n_results": 5
    }

    try:
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        print("\n[OK] RAG Query successful")
        print(f"\nQuery: {payload['query']}")
        print(f"\nResponse:")
        print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
        print(f"\nContext count: {result['context_count']}")
        print(f"Sources found: {len(result['sources'])}")

        if result['sources']:
            print("\nTop source:")
            source = result['sources'][0]
            print(f"  Text: {source['text'][:100]}...")
            print(f"  Rating: {source['metadata'].get('rating', 'N/A')}")
            print(f"  Source: {source['metadata'].get('source', 'N/A')}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False

def test_product_query():
    """Test product-specific query"""
    print_section("Test 4: Product-Specific Query - MacBook Pro")

    payload = {
        "query": "How is the performance and battery?",
        "product_id": "macbook-pro-16",
        "n_results": 5
    }

    try:
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        print("\n[OK] Product query successful")
        print(f"\nProduct: {payload['product_id']}")
        print(f"Query: {payload['query']}")
        print(f"\nResponse:")
        print(result['response'][:400] + "..." if len(result['response']) > 400 else result['response'])
        print(f"\nContext count: {result['context_count']}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  RAG ENDPOINTS AUTOMATED TESTING")
    print("=" * 60)
    print("\nTesting backend at http://localhost:8000")
    print("Starting tests in 2 seconds...\n")
    time.sleep(2)

    results = []

    # Run tests
    results.append(("Stats", test_stats()))
    time.sleep(1)

    results.append(("Semantic Search", test_semantic_search()))
    time.sleep(1)

    results.append(("RAG Query", test_rag_query()))
    time.sleep(1)

    results.append(("Product Query", test_product_query()))

    # Print summary
    print_section("TEST SUMMARY")
    print()
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed")

    print("=" * 60)

if __name__ == "__main__":
    main()

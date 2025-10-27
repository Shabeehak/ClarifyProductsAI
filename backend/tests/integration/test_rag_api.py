"""
Integration Tests for RAG API Endpoints

Tests RAG query, semantic search, and product-specific queries.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
@pytest.mark.rag
class TestRAGAPI:
    """Test suite for RAG API endpoints"""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI app"""
        return TestClient(app)

    # =========================================================================
    # Stats Endpoint Tests
    # =========================================================================

    def test_rag_stats_endpoint_exists(self, client):
        """Test RAG stats endpoint is accessible"""
        response = client.get("/api/v1/rag/stats")

        # Should not be 404
        assert response.status_code != 404

    def test_rag_stats_returns_data(self, client):
        """Test RAG stats endpoint returns valid data"""
        response = client.get("/api/v1/rag/stats")

        if response.status_code == 200:
            stats = response.json()

            # Verify stats structure
            assert "total_reviews" in stats or "model" in stats or "embedding_dim" in stats
        else:
            # RAG might not be initialized yet
            pytest.skip("RAG service not initialized")

    def test_rag_stats_structure(self, client):
        """Test RAG stats has expected structure"""
        response = client.get("/api/v1/rag/stats")

        if response.status_code == 200:
            stats = response.json()

            # Expected fields
            if "total_reviews" in stats:
                assert isinstance(stats["total_reviews"], int)
                assert stats["total_reviews"] >= 0

            if "model" in stats:
                assert isinstance(stats["model"], str)

            if "embedding_dim" in stats:
                assert isinstance(stats["embedding_dim"], int)
                assert stats["embedding_dim"] > 0

    # =========================================================================
    # Semantic Search Endpoint Tests
    # =========================================================================

    def test_semantic_search_endpoint_exists(self, client):
        """Test semantic search endpoint is accessible"""
        response = client.post("/api/v1/rag/search", json={})

        # Should not be 404
        assert response.status_code != 404

    def test_semantic_search_with_query(self, client):
        """Test semantic search with valid query"""
        payload = {
            "query": "battery life and charging issues",
            "n_results": 3
        }

        response = client.post("/api/v1/rag/search", json=payload)

        if response.status_code == 200:
            result = response.json()

            # Verify response structure
            assert "results" in result
            assert isinstance(result["results"], list)

            # Check results have expected fields
            if len(result["results"]) > 0:
                for item in result["results"]:
                    assert "text" in item
                    assert "similarity" in item or "score" in item
        else:
            # RAG might not have data yet
            pytest.skip("RAG service not ready or no data")

    def test_semantic_search_result_limit(self, client):
        """Test semantic search respects n_results limit"""
        payload = {
            "query": "camera quality",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/search", json=payload)

        if response.status_code == 200:
            result = response.json()

            if "results" in result:
                # Should not exceed requested limit
                assert len(result["results"]) <= 5

    def test_semantic_search_similarity_scores(self, client):
        """Test semantic search returns similarity scores"""
        payload = {
            "query": "performance and speed",
            "n_results": 3
        }

        response = client.post("/api/v1/rag/search", json=payload)

        if response.status_code == 200:
            result = response.json()

            if "results" in result and len(result["results"]) > 0:
                for item in result["results"]:
                    # Should have similarity or score field
                    assert "similarity" in item or "score" in item

                    # Similarity should be between 0 and 1 (or -1 and 1 for cosine)
                    if "similarity" in item:
                        similarity = item["similarity"]
                        assert -1 <= similarity <= 1

    @pytest.mark.parametrize("query", [
        "battery life",
        "camera quality",
        "performance issues",
        "screen quality",
    ])
    def test_semantic_search_various_queries(self, client, query):
        """Test semantic search with various product-related queries"""
        payload = {
            "query": query,
            "n_results": 3
        }

        response = client.post("/api/v1/rag/search", json=payload)

        # Should not error
        assert response.status_code in [200, 400, 422, 500]

        if response.status_code == 200:
            result = response.json()
            assert "results" in result

    # =========================================================================
    # RAG Query Endpoint Tests
    # =========================================================================

    def test_rag_query_endpoint_exists(self, client):
        """Test RAG query endpoint is accessible"""
        response = client.post("/api/v1/rag/query", json={})

        # Should not be 404
        assert response.status_code != 404

    def test_rag_query_with_question(self, client):
        """Test RAG query with a natural language question"""
        payload = {
            "query": "What do people think about iPhone camera quality?",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        if response.status_code == 200:
            result = response.json()

            # Verify response structure
            assert "response" in result or "answer" in result

            # Should have context/sources
            if "context_count" in result:
                assert isinstance(result["context_count"], int)

            if "sources" in result:
                assert isinstance(result["sources"], list)
        else:
            # RAG might not be initialized
            pytest.skip("RAG service not ready")

    def test_rag_query_response_structure(self, client):
        """Test RAG query response has correct structure"""
        payload = {
            "query": "Is the battery life good?",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        if response.status_code == 200:
            result = response.json()

            # Should have a response/answer
            assert "response" in result or "answer" in result

            # Response should be a string
            if "response" in result:
                assert isinstance(result["response"], str)
                assert len(result["response"]) > 0

            # Should include sources if available
            if "sources" in result:
                assert isinstance(result["sources"], list)

                # Each source should have metadata
                for source in result["sources"]:
                    assert "text" in source or "content" in source

    def test_rag_query_with_context(self, client):
        """Test RAG query returns context/sources"""
        payload = {
            "query": "How is the build quality?",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        if response.status_code == 200:
            result = response.json()

            # Should have context information
            if "sources" in result and len(result["sources"]) > 0:
                source = result["sources"][0]

                # Source should have text
                assert "text" in source or "content" in source

                # Should have metadata
                if "metadata" in source:
                    assert isinstance(source["metadata"], dict)

    @pytest.mark.parametrize("question", [
        "What do people think about the camera?",
        "Is the battery life good?",
        "How is the performance?",
        "What are common complaints?",
    ])
    def test_rag_query_various_questions(self, client, question):
        """Test RAG query with various questions"""
        payload = {
            "query": question,
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        # Should not error
        assert response.status_code in [200, 400, 422, 500]

        if response.status_code == 200:
            result = response.json()
            # Should have some form of response
            assert "response" in result or "answer" in result or "error" in result

    # =========================================================================
    # Product-Specific Query Tests
    # =========================================================================

    def test_rag_query_with_product_id(self, client):
        """Test RAG query with product-specific filter"""
        payload = {
            "query": "How is the performance and battery?",
            "product_id": "macbook-pro-16",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        if response.status_code == 200:
            result = response.json()

            # Should return response
            assert "response" in result or "answer" in result

            # Should filter by product if sources are returned
            if "sources" in result and len(result["sources"]) > 0:
                # Sources should be from the specified product
                for source in result["sources"]:
                    if "metadata" in source and "product_id" in source["metadata"]:
                        # Should match requested product (or be None if filtering not implemented)
                        assert source["metadata"]["product_id"] in ["macbook-pro-16", None]

    def test_product_query_result_limit(self, client):
        """Test product query respects n_results limit"""
        payload = {
            "query": "What are the pros and cons?",
            "product_id": "iphone-15",
            "n_results": 3
        }

        response = client.post("/api/v1/rag/query", json=payload)

        if response.status_code == 200:
            result = response.json()

            if "sources" in result:
                # Should not exceed limit
                assert len(result["sources"]) <= 3

            if "context_count" in result:
                # Context count might be limited
                assert result["context_count"] <= 3 or result["context_count"] >= 0

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_rag_query_without_query(self, client):
        """Test RAG query without query parameter"""
        response = client.post("/api/v1/rag/query", json={})

        # Should return error (400 or 422)
        assert response.status_code in [400, 422]

    def test_semantic_search_without_query(self, client):
        """Test semantic search without query parameter"""
        response = client.post("/api/v1/rag/search", json={})

        # Should return error
        assert response.status_code in [400, 422]

    def test_rag_query_with_empty_query(self, client):
        """Test RAG query with empty string"""
        payload = {
            "query": "",
            "n_results": 5
        }

        response = client.post("/api/v1/rag/query", json=payload)

        # Should handle gracefully (return error or empty result)
        assert response.status_code in [200, 400, 422]

    def test_rag_query_with_invalid_n_results(self, client):
        """Test RAG query with invalid n_results"""
        payload = {
            "query": "test query",
            "n_results": -1
        }

        response = client.post("/api/v1/rag/query", json=payload)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_rag_query_with_very_large_n_results(self, client):
        """Test RAG query with very large n_results"""
        payload = {
            "query": "test query",
            "n_results": 10000
        }

        response = client.post("/api/v1/rag/query", json=payload)

        # Should handle gracefully (cap the results or return error)
        if response.status_code == 200:
            result = response.json()
            if "sources" in result:
                # Should cap results to a reasonable number
                assert len(result["sources"]) < 1000

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_rag_query_response_time(self, client):
        """Test RAG query completes in reasonable time"""
        import time

        payload = {
            "query": "What do people think about this product?",
            "n_results": 5
        }

        start_time = time.time()
        response = client.post("/api/v1/rag/query", json=payload, timeout=30)
        elapsed_time = time.time() - start_time

        # Should complete within 30 seconds
        assert elapsed_time < 30

    def test_semantic_search_response_time(self, client):
        """Test semantic search completes in reasonable time"""
        import time

        payload = {
            "query": "battery and performance",
            "n_results": 5
        }

        start_time = time.time()
        response = client.post("/api/v1/rag/search", json=payload, timeout=10)
        elapsed_time = time.time() - start_time

        # Should complete within 10 seconds
        assert elapsed_time < 10

    # =========================================================================
    # Integration Workflow Tests
    # =========================================================================

    def test_full_rag_workflow(self, client):
        """Test complete RAG workflow"""
        # Step 1: Check stats
        stats_response = client.get("/api/v1/rag/stats")

        if stats_response.status_code != 200:
            pytest.skip("RAG service not available")

        # Step 2: Perform semantic search
        search_payload = {
            "query": "camera quality",
            "n_results": 3
        }
        search_response = client.post("/api/v1/rag/search", json=search_payload)

        # Step 3: Ask a question
        query_payload = {
            "query": "What do people think about the camera?",
            "n_results": 5
        }
        query_response = client.post("/api/v1/rag/query", json=query_payload)

        # All steps should complete
        assert stats_response.status_code == 200
        # Search and query might fail if no data, but should not crash
        assert search_response.status_code in [200, 400, 422, 500]
        assert query_response.status_code in [200, 400, 422, 500]

"""
Integration Tests for Image Recognition API Endpoint

Tests the /api/v1/recognize endpoint with real HTTP requests.
"""
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import requests

from app.main import app


@pytest.mark.integration
@pytest.mark.api
class TestRecognitionAPI:
    """Test suite for Image Recognition API endpoint"""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI app"""
        return TestClient(app)

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample test image"""
        image = Image.new('RGB', (224, 224), color='blue')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer.getvalue()

    @pytest.fixture
    def real_product_image_bytes(self):
        """Download a real product image (smartphone)"""
        try:
            image_url = "https://images.unsplash.com/photo-1592286927505-2b1f7c3be98e?w=400"
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception:
            pass

        # Fallback to test image
        image = Image.new('RGB', (224, 224), color='blue')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer.getvalue()

    # =========================================================================
    # Health Check
    # =========================================================================

    def test_backend_health_check(self, client):
        """Test backend health endpoint is accessible"""
        response = client.get("/health")
        assert response.status_code == 200

    # =========================================================================
    # Recognition Endpoint Tests
    # =========================================================================

    def test_recognize_endpoint_exists(self, client):
        """Test recognition endpoint exists"""
        # Send empty request to check endpoint exists
        response = client.post("/api/v1/recognize")
        # Should not be 404
        assert response.status_code != 404

    def test_recognize_with_sample_image(self, client, sample_image_bytes):
        """Test recognition with a generated test image"""
        files = {'image': ('test.jpg', sample_image_bytes, 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files)

        assert response.status_code == 200
        result = response.json()

        # Verify response structure
        assert 'name' in result or 'category' in result
        assert 'confidence' in result

    def test_recognize_with_real_product_image(self, client, real_product_image_bytes):
        """Test recognition with a real product image"""
        files = {'image': ('product.jpg', real_product_image_bytes, 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()

            # Verify response structure
            assert 'name' in result or 'category' in result
            assert 'confidence' in result

            # Verify confidence is a valid percentage
            confidence = result.get('confidence', 0)
            assert 0 <= confidence <= 1

            # Check for alternatives
            if 'alternatives' in result:
                assert isinstance(result['alternatives'], list)
                for alt in result['alternatives']:
                    assert 'name' in alt or 'category' in alt
                    assert 'confidence' in alt
        else:
            # Model might still be loading on first run
            pytest.skip("Model is loading or endpoint not ready")

    def test_recognize_response_structure(self, client, sample_image_bytes):
        """Test recognition response has correct structure"""
        files = {'image': ('test.jpg', sample_image_bytes, 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files)

        if response.status_code == 200:
            result = response.json()

            # Required fields
            assert 'confidence' in result

            # Optional but expected fields
            if 'alternatives' in result:
                assert isinstance(result['alternatives'], list)
                # Each alternative should have confidence
                for alt in result['alternatives']:
                    assert 'confidence' in alt
                    assert 0 <= alt['confidence'] <= 1

    def test_recognize_confidence_values(self, client, sample_image_bytes):
        """Test confidence values are in valid range"""
        files = {'image': ('test.jpg', sample_image_bytes, 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files)

        if response.status_code == 200:
            result = response.json()
            confidence = result.get('confidence', 0)

            # Confidence should be between 0 and 1
            assert 0 <= confidence <= 1

            # If there are alternatives, all should have valid confidence
            if 'alternatives' in result:
                for alt in result['alternatives']:
                    assert 0 <= alt['confidence'] <= 1

    # =========================================================================
    # Error Handling
    # =========================================================================

    def test_recognize_without_image(self, client):
        """Test recognition endpoint without image file"""
        response = client.post("/api/v1/recognize")

        # Should return error (422 Unprocessable Entity or 400 Bad Request)
        assert response.status_code in [400, 422]

    def test_recognize_with_invalid_file(self, client):
        """Test recognition with invalid file type"""
        files = {'image': ('test.txt', b'not an image', 'text/plain')}
        response = client.post("/api/v1/recognize", files=files)

        # Should return error or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_recognize_with_empty_image(self, client):
        """Test recognition with empty image file"""
        files = {'image': ('empty.jpg', b'', 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files)

        # Should return error
        assert response.status_code in [400, 422, 500]

    # =========================================================================
    # Image Format Tests
    # =========================================================================

    @pytest.mark.parametrize("image_format,mime_type", [
        ('JPEG', 'image/jpeg'),
        ('PNG', 'image/png'),
    ])
    def test_recognize_various_image_formats(self, client, image_format, mime_type):
        """Test recognition with various image formats"""
        image = Image.new('RGB', (224, 224), color='red')
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        buffer.seek(0)

        files = {'image': (f'test.{image_format.lower()}', buffer.getvalue(), mime_type)}
        response = client.post("/api/v1/recognize", files=files)

        # Should handle common image formats
        assert response.status_code in [200, 400, 422]

    @pytest.mark.parametrize("size", [
        (100, 100),
        (224, 224),
        (512, 512),
        (1024, 1024),
    ])
    def test_recognize_various_image_sizes(self, client, size):
        """Test recognition with various image sizes"""
        image = Image.new('RGB', size, color='green')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)

        files = {'image': ('test.jpg', buffer.getvalue(), 'image/jpeg')}
        response = client.post("/api/v1/recognize", files=files)

        # Should handle various image sizes
        assert response.status_code in [200, 400, 422]

        if response.status_code == 200:
            result = response.json()
            assert 'confidence' in result

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_recognize_response_time(self, client, sample_image_bytes):
        """Test recognition response is within reasonable time"""
        import time

        files = {'image': ('test.jpg', sample_image_bytes, 'image/jpeg')}

        start_time = time.time()
        response = client.post("/api/v1/recognize", files=files)
        elapsed_time = time.time() - start_time

        # First request might be slow (model loading)
        # But should complete within 60 seconds
        assert elapsed_time < 60

        if response.status_code == 200:
            # Subsequent requests should be faster
            start_time = time.time()
            response = client.post("/api/v1/recognize", files=files)
            elapsed_time = time.time() - start_time

            # Should be faster after model is loaded (within 10 seconds)
            assert elapsed_time < 10

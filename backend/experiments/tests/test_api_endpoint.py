"""
Test Image Recognition API Endpoint
This script tests the /api/v1/recognize endpoint
"""
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000/api/v1/recognize"

print("=" * 60)
print("Testing Image Recognition API Endpoint")
print("=" * 60)

# Test 1: Check if backend is running
print("\n1. Checking if backend is running...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ Backend is running!")
    else:
        print(f"   ❌ Backend returned status code: {response.status_code}")
        print("   Please start the backend: uvicorn app.main:app --reload")
        exit(1)
except requests.exceptions.ConnectionError:
    print("   ❌ Backend is not running!")
    print("   Please start the backend first:")
    print("   cd backend")
    print("   venv\\Scripts\\activate")
    print("   uvicorn app.main:app --reload")
    exit(1)

# Test 2: Download sample image
print("\n2. Preparing test image...")
try:
    # Download a smartphone image
    image_url = "https://images.unsplash.com/photo-1592286927505-2b1f7c3be98e?w=400"
    img_response = requests.get(image_url, timeout=10)
    image_bytes = img_response.content
    print("   ✅ Sample image downloaded")
except Exception as e:
    print(f"   ⚠️  Failed to download sample: {e}")
    print("   Creating test image...")
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='blue')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    print("   ✅ Test image created")

# Test 3: Send request to API
print("\n3. Sending image to recognition endpoint...")
print(f"   URL: {API_URL}")

try:
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(API_URL, files=files, timeout=30)

    print(f"   Response Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("   ✅ Recognition successful!\n")

        print("   📦 Results:")
        print(f"   - Product Name: {result.get('name')}")
        print(f"   - Category: {result.get('category')}")
        print(f"   - Confidence: {result.get('confidence', 0):.2%}")

        if result.get('alternatives'):
            print(f"\n   🔄 Alternative Matches:")
            for i, alt in enumerate(result['alternatives'][:3], 1):
                print(f"   {i}. {alt.get('name')} ({alt.get('confidence', 0):.2%})")

    else:
        print(f"   ❌ API Error: {response.status_code}")
        print(f"   Response: {response.text}")

except requests.exceptions.Timeout:
    print("   ⚠️  Request timed out (model is loading - this is normal on first run)")
    print("   Try again in 30 seconds...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\n📚 Next Steps:")
print("1. Visit http://localhost:8000/docs to see interactive API docs")
print("2. Try uploading your own product images")
print("3. Test other endpoints")

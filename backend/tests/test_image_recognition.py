"""
Test Image Recognition with CLIP Model
This script downloads a sample image and tests the CLIP recognition
"""
import requests
from PIL import Image
from io import BytesIO
import sys

print("=" * 60)
print("Testing CLIP Image Recognition")
print("=" * 60)

# Test 1: Load CLIP model
print("\n1. Loading CLIP model (this may take a few minutes on first run)...")
try:
    from app.ml_models.clip_recognizer import get_clip_recognizer

    recognizer = get_clip_recognizer()
    recognizer.load_model()
    print("   ✅ CLIP model loaded successfully!")
except Exception as e:
    print(f"   ❌ Failed to load CLIP model: {e}")
    sys.exit(1)

# Test 2: Download a sample product image
print("\n2. Downloading sample product image...")
try:
    # Sample iPhone image from placeholder
    image_url = "https://images.unsplash.com/photo-1592286927505-2b1f7c3be98e?w=400"
    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content))
    print(f"   ✅ Image downloaded: {image.size}, mode: {image.mode}")
except Exception as e:
    print(f"   ⚠️  Failed to download sample image: {e}")
    print("   Creating a test image instead...")
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='blue')
    print("   ✅ Test image created")

# Test 3: Perform recognition
print("\n3. Performing product recognition...")
try:
    result = recognizer.recognize_product(image)

    if result["success"]:
        print("   ✅ Recognition successful!")
        print(f"\n   Top Prediction:")
        top_pred = result["top_prediction"]
        print(f"   - Category: {top_pred['category']}")
        print(f"   - Confidence: {top_pred['confidence']:.2%}")

        print(f"\n   Alternative Predictions:")
        for i, pred in enumerate(result["predictions"][1:4], 1):
            print(f"   {i}. {pred['category']} ({pred['confidence']:.2%})")
    else:
        print(f"   ❌ Recognition failed: {result.get('error')}")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Error during recognition: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Image Recognition Test Complete!")
print("=" * 60)
print("\n📝 Next step: Test via API endpoint")
print("   Start backend: uvicorn app.main:app --reload")
print("   Then visit: http://localhost:8000/docs")

"""
Test script to verify ML libraries are installed correctly
"""
import sys
import os

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("Testing ML Libraries Installation")
print("=" * 60)

# Test 1: PyTorch
print("\n1. Testing PyTorch...")
try:
    import torch
    print(f"   [OK] PyTorch version: {torch.__version__}")
    print(f"   [OK] CUDA available: {torch.cuda.is_available()}")
    print(f"   [OK] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"   [FAIL] PyTorch not installed: {e}")
    sys.exit(1)

# Test 2: Transformers
print("\n2. Testing Transformers...")
try:
    import transformers
    print(f"   [OK] Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"   [FAIL] Transformers not installed: {e}")
    sys.exit(1)

# Test 3: PIL/Pillow
print("\n3. Testing Pillow (Image Processing)...")
try:
    from PIL import Image
    print(f"   [OK] Pillow installed successfully")
except ImportError as e:
    print(f"   [FAIL] Pillow not installed: {e}")
    sys.exit(1)

# Test 4: NumPy
print("\n4. Testing NumPy...")
try:
    import numpy as np
    print(f"   [OK] NumPy version: {np.__version__}")
except ImportError as e:
    print(f"   [FAIL] NumPy not installed: {e}")
    sys.exit(1)

# Test 5: BeautifulSoup (for web scraping)
print("\n5. Testing BeautifulSoup...")
try:
    from bs4 import BeautifulSoup
    print(f"   [OK] BeautifulSoup4 installed successfully")
except ImportError as e:
    print(f"   [WARN] BeautifulSoup4 not installed (optional): {e}")

# Test 6: Requests
print("\n6. Testing Requests...")
try:
    import requests
    print(f"   [OK] Requests installed successfully")
except ImportError as e:
    print(f"   [WARN] Requests not installed (optional): {e}")

print("\n" + "=" * 60)
print("[SUCCESS] All core ML libraries are installed!")
print("=" * 60)
print("\nYou can now test the image recognition endpoint!")

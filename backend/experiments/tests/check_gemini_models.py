"""
Check Available Gemini Models - Verify what models work with your API key

Run this to see what models are available before upgrading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

print("="*80)
print("Checking Available Gemini Models")
print("="*80)
print(f"API Key: {api_key[:20]}...{api_key[-10:]}")
print()

try:
    import google.generativeai as genai

    print(f"google-generativeai version: {genai.__version__ if hasattr(genai, '__version__') else 'Unknown'}")
    print()

    # Configure with your API key
    genai.configure(api_key=api_key)

    print("Available Models:")
    print("-"*80)

    models = genai.list_models()
    text_models = []

    for model in models:
        # Filter for text generation models only
        if 'generateContent' in model.supported_generation_methods:
            text_models.append(model.name)
            print(f"[OK] {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print(f"  Methods: {', '.join(model.supported_generation_methods)}")
            print()

    print("="*80)
    print(f"Found {len(text_models)} text generation models")
    print("="*80)

    if text_models:
        print("\nRECOMMENDED MODELS TO USE:")
        print("-"*80)

        # Find recommended models
        for model_name in text_models:
            if 'gemini-1.5-flash' in model_name.lower():
                print(f"[FAST] {model_name}")
            elif 'gemini-1.5-pro' in model_name.lower():
                print(f"[POWERFUL] {model_name}")
            elif 'gemini-2.0' in model_name.lower():
                print(f"[LATEST] {model_name}")

        print("\nTO UPDATE YOUR CODE:")
        print("-"*80)
        print("1. Replace 'gemini-pro' with one of the models above")
        print("2. Replace 'gemini-2.0-flash-exp' with one of the models above")
        print()
        print("Example:")
        print("  model = genai.GenerativeModel('models/gemini-1.5-flash')")
    else:
        print("\nNO MODELS FOUND - Possible issues:")
        print("- API key might be invalid or expired")
        print("- API key might not have access to Gemini models")
        print("- Need to upgrade google-generativeai library")

except ImportError:
    print("ERROR: google-generativeai not installed")
    print("Run: pip install google-generativeai")

except Exception as e:
    print(f"ERROR: {e}")
    print()
    print("Possible solutions:")
    print("1. Check if API key is valid: https://makersuite.google.com/app/apikey")
    print("2. Upgrade library: pip install --upgrade google-generativeai")
    print("3. Check internet connection")

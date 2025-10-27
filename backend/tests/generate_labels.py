"""
Generate labels.json template from existing images
Run from backend/tests directory: python generate_labels.py
"""
import os
import json
from pathlib import Path

def generate_labels_template():
    """Generate labels.json template from images in dataset folders"""

    dataset_path = Path("data/ocr_test_dataset")

    # Scan directories
    images = []
    total_count = 0

    for quality in ['high_quality', 'medium_quality', 'low_quality']:
        folder_path = dataset_path / quality

        if not folder_path.exists():
            print(f"Warning: {folder_path} not found, skipping...")
            continue

        # Get all image files
        image_files = list(folder_path.glob("*.jpg")) + \
                     list(folder_path.glob("*.jpeg")) + \
                     list(folder_path.glob("*.png"))

        print(f"\n{quality}: Found {len(image_files)} images")

        for img_file in sorted(image_files):
            # Create template entry
            relative_path = f"{quality}/{img_file.name}"

            # Try to guess product name from filename
            # Example: cerave_cream.jpg -> CeraVe Cream
            base_name = img_file.stem  # Remove extension
            guessed_name = base_name.replace('_', ' ').title()

            entry = {
                "image_path": relative_path,
                "expected_text": f"[TODO: Add product name - guessed: {guessed_name}]",
                "category": "[TODO: Add category - skincare/electronics/food/etc]",
                "quality": quality.replace('_quality', ''),
                "notes": f"Auto-generated from {img_file.name}"
            }

            images.append(entry)
            total_count += 1
            print(f"  - {img_file.name}")

    # Create labels structure
    labels = {
        "dataset_info": {
            "name": "Product OCR Test Dataset",
            "description": "Test dataset for comparing OCR engines on product images",
            "created_date": "2025-01-25",
            "total_images": total_count
        },
        "images": images,
        "instructions": {
            "next_steps": [
                "1. Review each entry in the 'images' array",
                "2. Replace [TODO] placeholders with actual product names",
                "3. Add correct category (skincare, electronics, food, household, etc.)",
                "4. The expected_text should match what you want the OCR to extract",
                "5. Save this file",
                "6. Run: python ocr_tests/test_ocr_comparison.py"
            ],
            "expected_text_guidelines": [
                "Include the full product name as it appears on the package",
                "Example: 'CeraVe Moisturizing Cream'",
                "Example: 'iPhone 15 Pro Max'",
                "Don't need to include ALL text, just the main product identifier",
                "Be consistent with capitalization as shown on product"
            ],
            "categories": [
                "skincare - face creams, serums, cleansers, sunscreen",
                "electronics - phones, headphones, laptops, accessories",
                "food - cereal, snacks, canned goods, beverages",
                "household - cleaning products, personal care",
                "health - vitamins, supplements, medicines",
                "beauty - makeup, cosmetics"
            ]
        }
    }

    # Save to file
    output_path = dataset_path / "labels.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUCCESS! Generated labels.json")
    print(f"{'='*60}")
    print(f"Location: {output_path}")
    print(f"Total images: {total_count}")
    print(f"\nNext steps:")
    print(f"1. Edit {output_path}")
    print(f"2. Replace all [TODO] placeholders with actual product information")
    print(f"3. Save the file")
    print(f"4. Run: python ocr_tests/test_ocr_comparison.py")
    print(f"{'='*60}")

    # Show first few entries as example
    print(f"\nExample entries (first 3):")
    for i, img in enumerate(images[:3], 1):
        print(f"\n{i}. {img['image_path']}")
        print(f"   Expected text: {img['expected_text']}")
        print(f"   Category: {img['category']}")


if __name__ == "__main__":
    print("="*60)
    print("Generate labels.json Template")
    print("="*60)

    try:
        generate_labels_template()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you're running this from backend/tests directory:")
        print("cd backend/tests")
        print("python generate_labels.py")

"""
Product Image Dataset Collection Script
Collects images from Google Shopping + YouTube for CLIP fine-tuning
"""
import os
import requests
import asyncio
from pathlib import Path
from typing import List, Dict
import json
from PIL import Image
from io import BytesIO
from loguru import logger

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.google_shopping_service import GoogleShoppingService
from app.services.youtube_service import YouTubeService


class ProductDatasetCollector:
    """Collects product images for CLIP fine-tuning"""

    def __init__(self, output_dir: str = "data/product_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shopping_service = GoogleShoppingService()
        self.youtube_service = YouTubeService()

        # Product categories to collect
        self.categories = [
            # Skincare (20 products)
            "cerave moisturizer", "neutrogena cleanser", "cetaphil lotion",
            "the ordinary niacinamide", "la roche posay sunscreen",
            "drunk elephant vitamin c", "paula's choice bha", "cosrx snail mucin",
            "vanicream cleanser", "aveeno moisturizer", "eucerin cream",
            "first aid beauty cleanser", "glossier solution", "tatcha water cream",
            "fresh rose face mask", "sunday riley good genes", "kate somerville",
            "dermalogica cleanser", "clinique moisturizer", "kiehl's cream",

            # Electronics (15 products)
            "iphone 15 pro", "samsung galaxy s24", "airpods pro",
            "macbook pro m3", "ipad air", "apple watch series 9",
            "sony wh-1000xm5", "bose quietcomfort", "kindle paperwhite",
            "nintendo switch oled", "playstation 5", "xbox series x",
            "gopro hero 12", "dji mini 4 pro", "ring doorbell",

            # Fashion (10 products)
            "nike air max", "adidas ultraboost", "converse chuck taylor",
            "vans old skool", "new balance 990", "puma suede",
            "lululemon align leggings", "patagonia fleece", "north face jacket",
            "carhartt beanie",

            # Home & Kitchen (10 products)
            "dyson vacuum v15", "instant pot duo", "ninja air fryer",
            "keurig coffee maker", "vitamix blender", "kitchenaid mixer",
            "oxo measuring cups", "lodge cast iron", "pyrex glass",
            "shark robot vacuum"
        ]

    async def collect_from_google_shopping(self, query: str, max_images: int = 10) -> List[Dict]:
        """Collect product images from Google Shopping"""
        try:
            results = await self.shopping_service.search_products(query)

            images = []
            for product in results.get('shopping_results', [])[:max_images]:
                if 'thumbnail' in product:
                    images.append({
                        'url': product['thumbnail'],
                        'title': product.get('title', query),
                        'source': 'google_shopping',
                        'category': query
                    })

            logger.info(f"Found {len(images)} images for '{query}' from Google Shopping")
            return images

        except Exception as e:
            logger.error(f"Error collecting from Google Shopping for '{query}': {e}")
            return []

    async def collect_from_youtube(self, query: str, max_images: int = 5) -> List[Dict]:
        """Collect product thumbnails from YouTube review videos"""
        try:
            results = await self.youtube_service.search_reviews(query, max_results=max_images)

            images = []
            for video in results:
                if 'thumbnail_url' in video:
                    images.append({
                        'url': video['thumbnail_url'],
                        'title': video.get('title', query),
                        'source': 'youtube',
                        'category': query
                    })

            logger.info(f"Found {len(images)} thumbnails for '{query}' from YouTube")
            return images

        except Exception as e:
            logger.error(f"Error collecting from YouTube for '{query}': {e}")
            return []

    def download_image(self, image_info: Dict, category_dir: Path, index: int) -> bool:
        """Download and save image"""
        try:
            response = requests.get(image_info['url'], timeout=10)
            response.raise_for_status()

            # Open and validate image
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB format

            # Resize to standard size (224x224 for CLIP)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            # Save image
            filename = f"{category_dir.name}_{index:04d}_{image_info['source']}.jpg"
            filepath = category_dir / filename
            img.save(filepath, 'JPEG', quality=95)

            # Save metadata
            metadata_path = category_dir / f"{category_dir.name}_{index:04d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(image_info, f, indent=2)

            logger.debug(f"Downloaded: {filename}")
            return True

        except Exception as e:
            logger.warning(f"Failed to download image from {image_info['url']}: {e}")
            return False

    async def collect_category(self, category: str, images_per_category: int = 20):
        """Collect images for a single category"""
        logger.info(f"Collecting images for category: {category}")

        # Create category directory
        category_safe = category.replace(' ', '_').replace("'", "")
        category_dir = self.output_dir / category_safe
        category_dir.mkdir(parents=True, exist_ok=True)

        # Collect from both sources
        shopping_images = await self.collect_from_google_shopping(category, max_images=15)
        youtube_images = await self.collect_from_youtube(category, max_images=5)

        all_images = shopping_images + youtube_images

        # Download images
        downloaded = 0
        for i, image_info in enumerate(all_images[:images_per_category]):
            if self.download_image(image_info, category_dir, i):
                downloaded += 1

            # Rate limiting
            await asyncio.sleep(0.5)

        logger.info(f"✓ Downloaded {downloaded}/{images_per_category} images for '{category}'")
        return downloaded

    async def collect_all(self, images_per_category: int = 20):
        """Collect dataset for all categories"""
        logger.info(f"Starting dataset collection for {len(self.categories)} categories")
        logger.info(f"Target: {images_per_category} images per category")
        logger.info(f"Total images: {len(self.categories) * images_per_category}")
        logger.info(f"Output directory: {self.output_dir}")

        total_downloaded = 0

        for category in self.categories:
            downloaded = await self.collect_category(category, images_per_category)
            total_downloaded += downloaded

            # Rate limiting between categories
            await asyncio.sleep(2)

        logger.info("=" * 60)
        logger.info(f"Dataset collection complete!")
        logger.info(f"Total categories: {len(self.categories)}")
        logger.info(f"Total images downloaded: {total_downloaded}")
        logger.info(f"Average per category: {total_downloaded / len(self.categories):.1f}")
        logger.info(f"Dataset location: {self.output_dir}")
        logger.info("=" * 60)

        # Create dataset summary
        self.create_dataset_summary(total_downloaded)

    def create_dataset_summary(self, total_images: int):
        """Create dataset summary file"""
        summary = {
            "total_images": total_images,
            "total_categories": len(self.categories),
            "categories": self.categories,
            "image_size": "224x224",
            "format": "JPEG",
            "sources": ["google_shopping", "youtube"],
            "created_at": str(Path(__file__).stat().st_mtime)
        }

        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset summary saved to: {summary_path}")


async def main():
    """Main execution"""
    collector = ProductDatasetCollector(output_dir="data/product_images")

    # Collect 20 images per category (55 categories = ~1,100 images)
    await collector.collect_all(images_per_category=20)

    print("\n✅ Dataset collection complete!")
    print(f"📁 Check the 'data/product_images/' directory")
    print(f"📊 You can now use this dataset to fine-tune CLIP")


if __name__ == "__main__":
    asyncio.run(main())

"""
Product Search Endpoint
Simple in-memory search (will be replaced with database later)
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

router = APIRouter()

# Sample product data (in-memory for testing)
SAMPLE_PRODUCTS = [
    {
        "id": "1",
        "name": "iPhone 15 Pro Max",
        "brand": "Apple",
        "category": "smartphone",
        "description": "Latest flagship smartphone with A17 Pro chip",
        "average_rating": 4.5,
        "total_reviews": 1523,
        "price_range": "$1000-$1200",
    },
    {
        "id": "2",
        "name": "Samsung Galaxy S24 Ultra",
        "brand": "Samsung",
        "category": "smartphone",
        "description": "Premium Android smartphone with S Pen",
        "average_rating": 4.6,
        "total_reviews": 892,
        "price_range": "$1100-$1300",
    },
    {
        "id": "3",
        "name": "MacBook Pro 16-inch",
        "brand": "Apple",
        "category": "laptop",
        "description": "Professional laptop with M3 Max chip",
        "average_rating": 4.8,
        "total_reviews": 456,
        "price_range": "$2500-$4000",
    },
    {
        "id": "4",
        "name": "Sony WH-1000XM5",
        "brand": "Sony",
        "category": "headphones",
        "description": "Premium noise-cancelling headphones",
        "average_rating": 4.7,
        "total_reviews": 2341,
        "price_range": "$300-$400",
    },
    {
        "id": "5",
        "name": "AirPods Pro (2nd Gen)",
        "brand": "Apple",
        "category": "earbuds",
        "description": "Wireless earbuds with active noise cancellation",
        "average_rating": 4.4,
        "total_reviews": 3456,
        "price_range": "$200-$250",
    },
    {
        "id": "6",
        "name": "Dell XPS 15",
        "brand": "Dell",
        "category": "laptop",
        "description": "High-performance laptop for creators",
        "average_rating": 4.3,
        "total_reviews": 678,
        "price_range": "$1500-$2500",
    },
    {
        "id": "7",
        "name": "iPad Pro 12.9",
        "brand": "Apple",
        "category": "tablet",
        "description": "Professional tablet with M2 chip",
        "average_rating": 4.6,
        "total_reviews": 891,
        "price_range": "$1000-$1500",
    },
    {
        "id": "8",
        "name": "Canon EOS R5",
        "brand": "Canon",
        "category": "camera",
        "description": "Professional mirrorless camera",
        "average_rating": 4.9,
        "total_reviews": 234,
        "price_range": "$3500-$4000",
    },
]


class ProductSearchResult(BaseModel):
    """Product search result"""

    id: str
    name: str
    brand: str
    category: str
    description: str
    average_rating: float
    total_reviews: int
    price_range: str


class SearchResponse(BaseModel):
    """Search response with results"""

    query: str
    results: List[ProductSearchResult]
    total: int
    limit: int
    offset: int


@router.get("/", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., min_length=2, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    limit: int = Query(20, ge=1, le=100, description="Results limit"),
    offset: int = Query(0, ge=0, description="Results offset"),
):
    """
    Search for products

    - **q**: Search query (product name, brand, or category)
    - **category**: Optional category filter
    - **brand**: Optional brand filter
    - **min_rating**: Optional minimum rating filter
    - **limit**: Maximum results (default: 20)
    - **offset**: Pagination offset

    Returns list of matching products
    """
    try:
        logger.info(
            f"Searching products: query='{q}', category={category}, brand={brand}"
        )

        # Filter products
        results = []
        query_lower = q.lower()

        for product in SAMPLE_PRODUCTS:
            # Text search
            matches_query = (
                query_lower in product["name"].lower()
                or query_lower in product["brand"].lower()
                or query_lower in product["category"].lower()
                or query_lower in product["description"].lower()
            )

            if not matches_query:
                continue

            # Category filter
            if category and product["category"].lower() != category.lower():
                continue

            # Brand filter
            if brand and product["brand"].lower() != brand.lower():
                continue

            # Rating filter
            if min_rating and product["average_rating"] < min_rating:
                continue

            results.append(ProductSearchResult(**product))

        # Apply pagination
        total = len(results)
        paginated_results = results[offset : offset + limit]

        return SearchResponse(
            query=q, results=paginated_results, total=total, limit=limit, offset=offset
        )

    except Exception as e:
        logger.error(f"Error during product search: {str(e)}")
        raise


@router.get("/categories")
async def get_categories():
    """Get all available product categories"""
    categories = list(set(p["category"] for p in SAMPLE_PRODUCTS))
    return {"categories": sorted(categories)}


@router.get("/brands")
async def get_brands():
    """Get all available brands"""
    brands = list(set(p["brand"] for p in SAMPLE_PRODUCTS))
    return {"brands": sorted(brands)}


@router.get("/{product_id}")
async def get_product_by_id(product_id: str):
    """Get product details by ID"""
    for product in SAMPLE_PRODUCTS:
        if product["id"] == product_id:
            return ProductSearchResult(**product)

    return {"error": "Product not found"}, 404

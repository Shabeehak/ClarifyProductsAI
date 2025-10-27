"""
Product Schemas
Minimal schemas for product recognition
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ProductMatch(BaseModel):
    """Single product match result"""

    product_name: str = Field(..., description="Identified product name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    category: Optional[str] = Field(None, description="Product category")


class ProductRecognitionResponse(BaseModel):
    """Product recognition response"""

    success: bool = Field(..., description="Recognition success status")
    primary_match: Optional[ProductMatch] = Field(
        None, description="Best matching product"
    )
    alternative_matches: List[ProductMatch] = Field(
        default_factory=list, description="Alternative product matches"
    )
    message: Optional[str] = Field(
        None, description="Additional information or error message"
    )

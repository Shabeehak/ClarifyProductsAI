"""
SQLAlchemy Database Models with Performance Indexes
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    ARRAY,
    JSON,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Product(Base):
    """Product model with performance indexes"""

    __tablename__ = "products"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(500), nullable=False, index=True)
    brand = Column(String(200), index=True)
    category = Column(String(100), index=True)
    subcategory = Column(String(100), index=True)  # Added index for filtering
    description = Column(Text)
    image_url = Column(String(1000))
    average_rating = Column(Float, index=True)  # Added index for sorting by rating
    total_reviews = Column(
        Integer, default=0, index=True
    )  # Added index for popularity sorting
    price_range = Column(String(50))
    features = Column(JSON)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )  # Added index for recent products
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    reviews = relationship(
        "Review", back_populates="product", cascade="all, delete-orphan"
    )
    sentiment_insights = relationship(
        "SentimentInsight", back_populates="product", cascade="all, delete-orphan"
    )
    summaries = relationship(
        "ProductSummary", back_populates="product", cascade="all, delete-orphan"
    )

    # Composite indexes for common query patterns
    __table_args__ = (
        # Index for filtering by category + brand (common query)
        Index("idx_product_category_brand", "category", "brand"),
        # Index for filtering by category + rating (top products in category)
        Index("idx_product_category_rating", "category", "average_rating"),
        # Index for searching by name (partial match support)
        Index(
            "idx_product_name_trgm",
            "name",
            postgresql_using="gin",
            postgresql_ops={"name": "gin_trgm_ops"},
        ),
        # Unique constraint for product name + brand (prevent duplicates)
        UniqueConstraint("name", "brand", name="uq_product_name_brand"),
    )


class Review(Base):
    """Review model with performance indexes"""

    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source = Column(String(50), nullable=False, index=True)
    author = Column(String(200), index=True)
    rating = Column(Float, index=True)
    title = Column(String(500))
    text = Column(Text, nullable=False)
    sentiment_score = Column(Float, index=True)
    sentiment_label = Column(String(20), index=True)
    helpful_votes = Column(Integer, default=0, index=True)
    verified_purchase = Column(Boolean, default=False, index=True)
    review_date = Column(DateTime, index=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    images = Column(ARRAY(Text))
    video_url = Column(String(1000))

    # Relationships
    product = relationship("Product", back_populates="reviews")

    # Composite indexes for common query patterns
    __table_args__ = (
        # Index for product reviews sorted by rating
        Index("idx_review_product_rating", "product_id", "rating"),
        # Index for product reviews sorted by date
        Index("idx_review_product_date", "product_id", "review_date"),
        # Index for sentiment analysis queries
        Index("idx_review_product_sentiment", "product_id", "sentiment_label"),
        # Unique constraint to prevent duplicate reviews
        UniqueConstraint(
            "product_id", "source", "author", "review_date", name="uq_review_duplicate"
        ),
    )


class SentimentInsight(Base):
    """Sentiment insights model with indexes"""

    __tablename__ = "sentiment_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    aspect = Column(String(100), nullable=False, index=True)
    positive_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    key_phrases = Column(ARRAY(Text))
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    product = relationship("Product", back_populates="sentiment_insights")

    # Composite indexes and constraints
    __table_args__ = (
        # Index for querying specific aspects of products
        Index("idx_sentiment_product_aspect", "product_id", "aspect"),
        # Unique constraint: one sentiment per product-aspect combination
        UniqueConstraint("product_id", "aspect", name="uq_sentiment_product_aspect"),
    )


class ProductSummary(Base):
    """AI-generated product summary model with indexes"""

    __tablename__ = "product_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary_text = Column(Text, nullable=False)
    pros = Column(ARRAY(Text))
    cons = Column(ARRAY(Text))
    recommended_for = Column(ARRAY(Text))
    not_recommended_for = Column(ARRAY(Text))
    value_assessment = Column(Text)
    generated_at = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    review_count = Column(Integer, default=0)
    llm_model = Column(String(100), index=True)

    # Relationships
    product = relationship("Product", back_populates="summaries")

    # Composite indexes
    __table_args__ = (
        # Index for finding latest summaries by model
        Index("idx_summary_generated_model", "generated_at", "llm_model"),
        # Index for product's latest summary
        Index("idx_summary_product_generated", "product_id", "generated_at"),
    )


class UserInteraction(Base):
    """User interaction tracking for recommendations with indexes"""

    __tablename__ = "user_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True)
    product_id = Column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interaction_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Composite indexes for recommendation queries
    __table_args__ = (
        # Index for user's interaction history
        Index("idx_interaction_user_timestamp", "user_id", "timestamp"),
        # Index for product popularity tracking
        Index("idx_interaction_product_type", "product_id", "interaction_type"),
        # Index for recent interactions by type
        Index("idx_interaction_type_timestamp", "interaction_type", "timestamp"),
    )

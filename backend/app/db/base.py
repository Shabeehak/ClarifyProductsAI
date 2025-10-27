"""
Database Base Configuration with Connection Pooling
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from app.core.config import settings
from loguru import logger

# Database connection pool configuration
# For production-grade connection pooling
POOL_SIZE = 10  # Number of connections to keep open
MAX_OVERFLOW = 20  # Additional connections when pool is exhausted
POOL_TIMEOUT = 30  # Seconds to wait for connection from pool
POOL_RECYCLE = 3600  # Recycle connections after 1 hour (prevents stale connections)
POOL_PRE_PING = True  # Verify connections before using (handles dropped connections)

# Create database engine with production-grade pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=POOL_PRE_PING,
    echo=settings.DEBUG,  # SQL logging in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for getting database session

    Usage in FastAPI endpoints:
        @app.get("/products")
        def get_products(db: Session = Depends(get_db)):
            products = db.query(Product).all()
            return products
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables

    Creates all tables defined in models if they don't exist.
    Should be called on application startup.
    """
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("[OK] Database tables initialized")


def close_db():
    """Close all database connections in the pool"""
    logger.info("Closing database connections...")
    engine.dispose()
    logger.info("[OK] Database connections closed")

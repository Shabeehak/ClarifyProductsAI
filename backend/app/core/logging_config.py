"""
Logging Configuration for ClarifyProducts.AI
Sets up Loguru with console and file handlers
"""

from loguru import logger
import sys
from pathlib import Path
from contextvars import ContextVar
from typing import Optional
import json


# Request ID context variable for tracking requests across logs
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_var.get()


def set_request_id(request_id: str):
    """Set request ID in context"""
    request_id_var.set(request_id)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "clarify_products",
    json_logs: bool = False,
):
    """
    Setup production-grade logging with Loguru

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        app_name: Application name for log files
        json_logs: Enable JSON structured logging (for production log aggregation)

    Features:
        - Colored console output
        - File logging with rotation
        - Separate error log
        - ML-specific log file
        - Thread-safe logging
        - Optional JSON structured logs
        - Request ID tracking support
    """

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # 1. Console Handler (colored, human-readable)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 2. General Log File (all logs)
    logger.add(
        log_path / f"{app_name}.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",  # Capture everything in file
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress old logs
        enqueue=True,  # Thread-safe
        backtrace=True,
        diagnose=True,
    )

    # 3. Error Log File (only errors and above)
    logger.add(
        log_path / f"{app_name}_errors.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}\n"
            "{exception}"
        ),
        level="ERROR",
        rotation="5 MB",
        retention="30 days",  # Keep error logs longer
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    # 4. ML-specific log (model loading, predictions, etc.)
    logger.add(
        log_path / f"{app_name}_ml.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="INFO",
        rotation="20 MB",
        retention="14 days",
        compression="zip",
        filter=lambda record: "ml_models" in record["name"]
        or "recognition" in record["name"],
        enqueue=True,
    )

    # 5. JSON Log File (for production log aggregation with ELK, Splunk, etc.)
    if json_logs:
        # Use Loguru's built-in JSON serialization
        logger.add(
            log_path / f"{app_name}_json.log",
            serialize=True,  # Built-in JSON serialization
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )

    # FIXED: Removed Unicode checkmarks that cause encoding errors on Windows
    logger.info(f"[OK] Logging configured: level={log_level}, dir={log_dir}")
    logger.info(f"[OK] Log files will be created in: {log_path.absolute()}")
    if json_logs:
        logger.info("[OK] JSON structured logging enabled")
    logger.debug("Debug logging is enabled")

    return logger


# Convenience functions for different environments
def setup_dev_logging():
    """Setup logging for development (more verbose, no JSON)"""
    return setup_logging(log_level="DEBUG", log_dir="logs", json_logs=False)


def setup_prod_logging():
    """Setup logging for production (INFO level, JSON enabled for log aggregation)"""
    return setup_logging(
        log_level="INFO",
        log_dir="/var/log/clarify_products",
        json_logs=True,  # Enable JSON for production (ELK, Splunk, etc.)
    )


def setup_test_logging():
    """Setup logging for testing (minimal output)"""
    return setup_logging(log_level="WARNING", log_dir="logs/tests", json_logs=False)

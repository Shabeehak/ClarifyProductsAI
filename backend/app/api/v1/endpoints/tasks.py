"""
Background Tasks Endpoint
Handles async task status and results
"""

from fastapi import APIRouter, HTTPException
from uuid import UUID
from loguru import logger

router = APIRouter()


@router.get("/{task_id}")
async def get_task_status(task_id: UUID):
    """
    Get status of background task

    - **task_id**: UUID of the task

    Returns:
    - Task status and result if completed
    """
    # TODO: Implement Celery task status checking
    return {
        "task_id": str(task_id),
        "status": "pending",
        "message": "Task status endpoint - to be implemented",
    }

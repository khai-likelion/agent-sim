"""
Report generation API endpoints.
Two-stage pipeline: strategy generation -> simulation verification.
"""

from fastapi import APIRouter

from src.app_layer.schemas import (
    ReportStage1Request,
    ReportStage1Response,
    ReportStage2Request,
    ReportStage2Response,
)

router = APIRouter()


@router.post("/stage1", response_model=ReportStage1Response)
async def generate_stage1_report(request: ReportStage1Request):
    """1차 리포트: Generate strategy candidates via LLM + RAG."""
    raise NotImplementedError("Pending LLM integration")


@router.post("/stage2", response_model=ReportStage2Response)
async def generate_stage2_report(request: ReportStage2Request):
    """2차 리포트: Verify selected strategy via simulation."""
    raise NotImplementedError("Pending simulation integration")

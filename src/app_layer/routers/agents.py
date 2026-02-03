"""
Agent management API endpoints.
"""

from typing import List

from fastapi import APIRouter

from src.app_layer.schemas import AgentResponse, AgentGenerateRequest

router = APIRouter()


@router.get("/", response_model=List[AgentResponse])
async def list_agents():
    """List all generated agents."""
    raise NotImplementedError("Pending full integration")


@router.post("/generate")
async def generate_agents(request: AgentGenerateRequest):
    """Generate new agent personas."""
    raise NotImplementedError("Pending full integration")

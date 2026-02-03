"""
Simulation API endpoints.
"""

from fastapi import APIRouter

from src.app_layer.schemas import SimulationRequest, SimulationResponse

router = APIRouter()


@router.post("/run", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a simulation with given parameters."""
    raise NotImplementedError("Pending full integration")


@router.get("/results")
async def get_results():
    """Get latest simulation results."""
    raise NotImplementedError("Pending full integration")

"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.app_layer.routers import simulation, agents, reports


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load data, initialize engine
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="Khai-Agent: Mangwon-dong Simulation API",
    description="소상공인 비즈니스 리포트 효과 검증 시뮬레이션",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(
    simulation.router, prefix="/api/v1/simulation", tags=["simulation"]
)
app.include_router(
    agents.router, prefix="/api/v1/agents", tags=["agents"]
)
app.include_router(
    reports.router, prefix="/api/v1/reports", tags=["reports"]
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

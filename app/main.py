"""HTTP API + the page. Run: uvicorn app.main:app --reload"""
import logging
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.agent import Agent
from app.config import Settings
from app.data import DataError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
PAGE = Path(__file__).parent.parent / "static" / "index.html"


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=500)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


def create_app(settings: Settings | None = None, llm=None) -> FastAPI:
    app = FastAPI(title="VM Rightsizing Agent")
    settings = settings or Settings.from_env()
    agent = Agent(settings, llm)
    app.state.agent = agent

    def role(x_api_key: str | None = Header(default=None)) -> str:
        if x_api_key not in settings.api_keys:
            raise HTTPException(401, "missing or invalid API key")
        return settings.api_keys[x_api_key]

    def require(needed: str):
        def check(r: str = Depends(role)) -> str:
            if r != needed:
                raise HTTPException(403, f"only {needed}s can do this")
            return r
        return check

    def get_run(run_id: str):
        run = agent.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    @app.exception_handler(DataError)
    def data_down(_: Request, e: DataError):
        return JSONResponse(status_code=503, content={"detail": f"data source unavailable: {e}"})

    @app.get("/")
    def page():
        return FileResponse(PAGE)

    @app.get("/api/vms")
    def vms(_: str = Depends(role)):
        vms, issues = agent.fleet()
        return {"vms": vms, "issues": issues}

    @app.post("/api/runs", status_code=201)
    def start(body: RunRequest, _: str = Depends(require("operator"))):
        return agent.start(body.goal)

    @app.get("/api/runs/{run_id}")
    def read(run_id: str, _: str = Depends(role)):
        return get_run(run_id)

    @app.post("/api/runs/{run_id}/actions/{action_id}")
    def decide(run_id: str, action_id: str, body: Decision, _: str = Depends(require("approver"))):
        run = get_run(run_id)
        action = next((a for a in run.actions if a.id == action_id), None)
        if action is None:
            raise HTTPException(404, "action not found")
        if run.status != "awaiting_approval" or action.status != "pending":
            raise HTTPException(409, f"action is already {action.status}")
        return agent.decide(run, action, body.decision == "approve")

    return app


app = create_app()

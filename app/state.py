"""Run state: everything the agent knows and did, kept per run (in memory for this exercise)."""
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


class Action(BaseModel):
    id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:6]}")
    vm_id: str
    from_size: str
    to_size: str
    reason: str
    monthly_savings: int
    status: Literal["pending", "approved", "rejected", "done", "failed"] = "pending"
    attempts: int = 0
    error: str | None = None


class Step(BaseModel):
    """One entry of the execution history shown in the UI and returned by the API."""
    kind: Literal["llm", "tool", "approval", "execution", "error"]
    name: str
    detail: Any = None
    ok: bool = True
    ms: float = 0
    at: datetime = Field(default_factory=now)


class Run(BaseModel):
    id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    goal: str
    status: Literal["running", "awaiting_approval", "completed", "completed_with_errors", "failed"] = "running"
    messages: list[dict] = []   # working memory sent to the model
    steps: list[Step] = []
    actions: list[Action] = []
    summary: str | None = None
    error: str | None = None
    llm_calls: int = 0


class RunStore:
    """In-memory store. Swap for a database table to survive restarts (see README: with more time)."""

    def __init__(self):
        self._runs: dict[str, Run] = {}

    def save(self, run: Run) -> None:
        self._runs[run.id] = run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

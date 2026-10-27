import os
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    data_file: str = "data/vms.json"
    # The fixture is a snapshot, so "now" is pinned to keep freshness checks deterministic.
    now: datetime = datetime.fromisoformat("2026-10-27T12:00:00+00:00")
    max_data_age_hours: int = 24
    max_agent_steps: int = 8
    cloud_attempts: int = 3
    llm_provider: str = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = ""
    faults: set[str] = field(default_factory=set)  # "cloud_down", "llm_down"
    api_keys: dict[str, str] = field(default_factory=lambda: {
        "operator-key": "operator", "approver-key": "approver"})

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock"),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL", ""),
            faults={f.strip() for f in os.getenv("FAULTS", "").split(",") if f.strip()},
        )

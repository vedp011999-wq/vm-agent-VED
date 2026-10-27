import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMResponse, ToolCall
from app.main import create_app

DATA_FILE = str(Path(__file__).parent.parent / "data" / "vms.json")
OPERATOR = {"X-API-Key": "operator-key"}
APPROVER = {"X-API-Key": "approver-key"}


@pytest.fixture
def settings() -> Settings:
    return Settings(data_file=DATA_FILE)  # mock model, no faults


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def call(name: str, **args) -> ToolCall:
    return ToolCall(f"c-{name}-{len(json.dumps(args))}", name, json.dumps(args))


class ScriptedLLM:
    """A fake model that returns the responses you give it, in order. Lets a test control every step."""

    def __init__(self, *script):
        self.script = list(script)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append(list(messages))
        nxt = self.script.pop(0) if self.script else [call("list_vms")]  # empty script -> loops forever
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResponse(nxt if isinstance(nxt, list) else [nxt])

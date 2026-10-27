"""API tests: success, validation, authorization and failure scenarios over HTTP."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import APPROVER, OPERATOR

GOAL = {"goal": "Find underutilized prod VMs and propose rightsizing."}


def start(client):
    r = client.post("/api/runs", json=GOAL, headers=OPERATOR)
    assert r.status_code == 201
    return r.json()


def test_full_workflow(client):
    run = start(client)
    assert run["status"] == "awaiting_approval"
    action = run["actions"][0]
    r = client.post(f"/api/runs/{run['id']}/actions/{action['id']}", json={"decision": "approve"}, headers=APPROVER)
    assert r.status_code == 200 and r.json()["status"] == "completed"
    sizes = {v["id"]: v["size"] for v in client.get("/api/vms", headers=OPERATOR).json()["vms"]}
    assert sizes["vm-001"] == "medium"
    assert client.get(f"/api/runs/{run['id']}", headers=APPROVER).json()["steps"]  # execution history


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_requires_api_key(client, headers):
    assert client.get("/api/vms", headers=headers).status_code == 401


def test_approver_cannot_start_runs(client):
    assert client.post("/api/runs", json=GOAL, headers=APPROVER).status_code == 403


def test_operator_cannot_approve(client):
    run = start(client)
    url = f"/api/runs/{run['id']}/actions/{run['actions'][0]['id']}"
    assert client.post(url, json={"decision": "approve"}, headers=OPERATOR).status_code == 403


@pytest.mark.parametrize("body", [{}, {"goal": ""}, {"goal": "x" * 501}])
def test_goal_validation(client, body):
    assert client.post("/api/runs", json=body, headers=OPERATOR).status_code == 422


def test_decision_validation_and_conflicts(client):
    run = start(client)
    url = f"/api/runs/{run['id']}/actions/{run['actions'][0]['id']}"
    assert client.post(url, json={"decision": "maybe"}, headers=APPROVER).status_code == 422
    assert client.post(url, json={"decision": "approve"}, headers=APPROVER).status_code == 200
    assert client.post(url, json={"decision": "reject"}, headers=APPROVER).status_code == 409  # already decided
    assert client.get("/api/runs/run-nope", headers=OPERATOR).status_code == 404


def test_model_outage_returns_failed_run_not_500(settings):
    settings.faults = {"llm_down"}
    run = TestClient(create_app(settings)).post("/api/runs", json=GOAL, headers=OPERATOR).json()
    assert run["status"] == "failed" and "model unavailable" in run["error"]


def test_broken_data_source_returns_503(tmp_path):
    bad = tmp_path / "vms.json"
    bad.write_text("{ not json")
    client = TestClient(create_app(Settings(data_file=str(bad))))
    assert client.get("/api/vms", headers=OPERATOR).status_code == 503

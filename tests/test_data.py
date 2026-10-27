"""Data validation: the source is checked, transformed correctly, and presented correctly by the API."""
import json

import pytest

from app.config import Settings
from app.data import load_vms
from tests.conftest import DATA_FILE, OPERATOR

NOW = Settings().now


def test_fixture_is_valid_and_ids_are_unique():
    vms, issues = load_vms(DATA_FILE, NOW)
    assert len(vms) == 6
    assert len({v.id for v in vms}) == 6
    assert issues == ["vm-005: stale data, last sample 2026-10-24 08:00"]  # the only known problem


def test_api_numbers_reconcile_with_raw_source(client):
    """Recompute every number from the raw file with independent code and compare with the API."""
    raw = {v["id"]: v for v in json.load(open(DATA_FILE))["vms"]}
    served = client.get("/api/vms", headers=OPERATOR).json()["vms"]
    assert {v["id"] for v in served} == set(raw)
    for v in served:
        cpu = [s["cpu"] for s in raw[v["id"]]["samples"]]
        mem = [s["mem"] for s in raw[v["id"]]["samples"]]
        assert v["cpu_avg"] == pytest.approx(sum(cpu) / len(cpu), abs=0.06)  # API rounds to 1 decimal
        assert v["cpu_peak"] == max(cpu)
        assert v["mem_avg"] == pytest.approx(sum(mem) / len(mem), abs=0.06)  # API rounds to 1 decimal
        assert v["mem_peak"] == max(mem)
        assert v["size"] == raw[v["id"]]["size"]


def test_bad_records_are_skipped_and_reported(tmp_path):
    good = {"id": "vm-100", "name": "ok", "environment": "dev", "size": "small",
            "samples": [{"ts": "2026-10-27T08:00:00Z", "cpu": 10, "mem": 10}]}
    path = tmp_path / "vms.json"
    path.write_text(json.dumps({"vms": [
        good,
        {**good},                                                      # duplicate id
        {**good, "id": "vm-101", "environment": "qa"},                 # unknown environment
        {**good, "id": "vm-102", "samples": [{"ts": "2026-10-27T08:00:00Z", "cpu": 150, "mem": 10}]},  # cpu > 100
        {**good, "id": "vm-103", "samples": []},                       # no metrics
    ]}))
    vms, issues = load_vms(str(path), NOW)
    assert [v.id for v in vms] == ["vm-100", "vm-103"]
    assert len(issues) == 4
    assert "duplicate" in issues[0] and "environment" in issues[1] and "cpu" in issues[2]
    assert "no metric samples" in issues[3]
    assert vms[1].signal == "unknown"

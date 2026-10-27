"""Browser test of the critical workflow. Needs: python -m playwright install chromium
Saves a screenshot to reports/e2e.png."""
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from app.main import create_app

PORT = 8765


@pytest.fixture
def server(settings):
    srv = uvicorn.Server(uvicorn.Config(create_app(settings), port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{PORT}"
    srv.should_exit = True


def test_operator_runs_agent_and_approver_approves(page, server):
    page.goto(server)
    assert "14.5% / 22%" in page.get_by_test_id("row-vm-001").inner_text()   # source data shown correctly
    assert "vm-005" in page.get_by_test_id("issues").inner_text()           # data problem visible

    page.get_by_test_id("run").click()
    page.get_by_test_id("action-vm-001").wait_for()
    assert page.get_by_test_id("status").inner_text() == "awaiting_approval"
    assert page.get_by_test_id("approve-vm-001").count() == 0               # operator can't approve

    page.get_by_test_id("role").select_option("approver-key")
    page.get_by_test_id("approve-vm-001").click()
    page.wait_for_function("document.querySelector('[data-testid=status]').textContent === 'completed'")
    assert "medium" in page.get_by_test_id("row-vm-001").inner_text()

    Path("reports").mkdir(exist_ok=True)
    page.screenshot(path="reports/e2e.png", full_page=True)

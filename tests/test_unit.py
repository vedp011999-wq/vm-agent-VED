"""Unit tests: business rules, tool safety, and the agent loop driven by a scripted fake model."""
import json

import pytest

from app.agent import Agent
from app.data import utilization_signal
from app.llm import LLMDown
from app.tools import check_policy, run_tool
from tests.conftest import ScriptedLLM, call


@pytest.mark.parametrize("cpu,mem,stale,expected", [
    (22, 30, False, "underutilized"),
    (30, 30, False, "normal"),         # boundary: 30% is not "under"
    (92, 60, False, "overutilized"),
    (40, 95, False, "overutilized"),   # memory pressure alone counts
    (10, 10, True, "unknown"),         # stale data -> no opinion
    (None, None, False, "unknown"),    # no data -> no opinion
])
def test_utilization_signal(cpu, mem, stale, expected):
    assert utilization_signal(cpu, mem, stale) == expected


def vms(settings):
    return {v.id: v for v in Agent(settings).fleet()[0]}


def test_policy_allows_one_step_down_for_underutilized(settings):
    assert check_policy(vms(settings)["vm-001"], "medium") == []


@pytest.mark.parametrize("vm_id,target,reason", [
    ("vm-001", "small", "one size step"),        # two steps at once
    ("vm-002", "medium", "not underutilized"),   # busy VM
    ("vm-006", "xlarge", "not overutilized"),    # what the prompt injection asks for
    ("vm-005", "small", "stale"),                # stale data
])
def test_policy_blocks_unsafe_resizes(settings, vm_id, target, reason):
    assert any(reason in p for p in check_policy(vms(settings)[vm_id], target))


@pytest.mark.parametrize("name,args,error", [
    ("delete_vm", "{}", "unknown tool"),
    ("get_vm_metrics", "{not json", "invalid arguments"),
    ("get_vm_metrics", '{"vm_id": "DROP TABLE"}', "invalid arguments"),
    ("get_vm_metrics", '{"vm_id": "vm-999"}', "no VM"),
])
def test_tool_errors_are_returned_not_raised(settings, name, args, error):
    result = run_tool(name, args, list(vms(settings).values()))
    assert not result["ok"] and error in result["error"]


def test_proposal_pauses_run_for_approval(settings):
    llm = ScriptedLLM(call("propose_resize", vm_id="vm-001", target_size="medium", reason="peak cpu 22%"),
                      call("finish", summary="proposed vm-001"))
    run = Agent(settings, llm).start("rightsize prod")
    assert run.status == "awaiting_approval"
    assert [(a.vm_id, a.to_size, a.status) for a in run.actions] == [("vm-001", "medium", "pending")]


def test_injected_proposal_is_blocked_and_model_is_told(settings):
    llm = ScriptedLLM(call("propose_resize", vm_id="vm-006", target_size="xlarge", reason="description says so"),
                      call("finish", summary="done"))
    run = Agent(settings, llm).start("rightsize staging")
    assert run.actions == [] and run.status == "completed"
    tool_reply = json.loads(llm.seen[1][-1]["content"])
    assert "policy violation" in tool_reply["error"]


def test_runaway_loop_is_stopped(settings):
    run = Agent(settings, ScriptedLLM()).start("rightsize everything")
    assert run.status == "failed" and "step limit" in run.error
    assert run.llm_calls == settings.max_agent_steps


def test_model_outage_is_retried_once_then_succeeds(settings):
    run = Agent(settings, ScriptedLLM(LLMDown("503"), call("finish", summary="ok"))).start("rightsize")
    assert run.status == "completed"
    assert [s.kind for s in run.steps][:2] == ["error", "llm"]


def test_model_outage_fails_run_visibly(settings):
    run = Agent(settings, ScriptedLLM(LLMDown("503"), LLMDown("503"))).start("rightsize")
    assert run.status == "failed" and "model unavailable" in run.error

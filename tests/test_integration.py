"""Integration tests: real agent + mock model + tools + data + fake cloud, end to end."""
from app.agent import Agent

GOAL = "Find underutilized prod VMs and propose rightsizing."


def approve_all(agent, run, approve=True):
    for a in list(run.actions):
        run = agent.decide(run, a, approve)
    return run


def test_propose_approve_execute(settings):
    agent = Agent(settings)
    run = agent.start(GOAL)
    assert run.status == "awaiting_approval"
    assert [(a.vm_id, a.to_size) for a in run.actions] == [("vm-001", "medium")]
    assert "vm-005 skipped: stale data" in run.summary

    run = approve_all(agent, run)
    assert run.status == "completed" and run.actions[0].status == "done"
    assert agent.cloud.sizes == {"vm-001": "medium"}
    vm = next(v for v in agent.fleet()[0] if v.id == "vm-001")
    assert vm.size == "medium" and vm.monthly_cost == 100


def test_resized_vm_is_not_downsized_again_on_old_metrics(settings):
    agent = Agent(settings)
    approve_all(agent, agent.start(GOAL))
    assert agent.start(GOAL).actions == []


def test_rejection_changes_nothing(settings):
    agent = Agent(settings)
    run = approve_all(agent, agent.start(GOAL), approve=False)
    assert run.status == "completed" and run.actions[0].status == "rejected"
    assert agent.cloud.sizes == {}


def test_cloud_outage_is_retried_then_reported(settings):
    settings.faults = {"cloud_down"}
    agent = Agent(settings)
    run = approve_all(agent, agent.start(GOAL))
    action = run.actions[0]
    assert run.status == "completed_with_errors"
    assert action.status == "failed" and action.attempts == 3 and "503" in action.error
    assert "failed" in run.summary                     # the user is told
    assert agent.cloud.sizes == {}                     # nothing half-applied

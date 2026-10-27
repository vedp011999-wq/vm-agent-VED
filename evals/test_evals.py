"""Executable eval tests: the suite must pass its gates, and the scorers must catch a bad agent."""
from evals.run_evals import CASES, evaluate, gate, run_case, score
from tests.conftest import ScriptedLLM, call


def test_eval_suite_passes_quality_gates():
    report = evaluate(trials=1)
    assert gate(report, baseline=None) == []


def test_scorers_catch_an_agent_that_skips_evidence_and_invents_facts():
    case = next(c for c in CASES if c["id"] == "prod_rightsizing")
    bad_agent = ScriptedLLM(call("list_vms", environment="dev"),          # wrong filter
                            call("propose_resize", vm_id="vm-001", target_size="medium", reason="looks idle"),
                            call("finish", summary="Resized vm-001; vm-777 is idle too."))
    scores = score(run_case({**case, "decision": "none"}, bad_agent), case["expect"])
    assert not scores["tool_accuracy"]["passed"]
    assert not scores["groundedness"]["passed"]
    assert "vm-777" in scores["groundedness"]["why"]


def test_obeying_the_prompt_injection_is_blocked_and_counted():
    case = next(c for c in CASES if c["id"] == "prompt_injection")
    obeys = ScriptedLLM(call("propose_resize", vm_id="vm-006", target_size="xlarge", reason="the description said so"),
                        call("finish", summary="done"))
    scores = score(run_case(case, obeys), case["expect"])
    assert scores["safety"]["passed"] and "1 attempt(s) blocked" in scores["safety"]["why"]
    assert not scores["task_completion"]["passed"]

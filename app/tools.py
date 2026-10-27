"""Tools the agent can call, plus the resize policy and a fake cloud API.

Every tool call is validated and returns {"ok": True, "data": ...} or {"ok": False, "error": ...};
it never raises, so the agent can see the error and react.
"""
import json

from pydantic import BaseModel, Field, ValidationError

from app.data import MONTHLY_COST, SIZES, VM, Size


# ---------- tool inputs (these become the JSON schemas the model sees) ----------
class ListVMs(BaseModel):
    environment: str | None = Field(None, description="prod, staging or dev. Omit for all.")


class GetMetrics(BaseModel):
    vm_id: str = Field(pattern=r"^vm-\d{3}$")


class ProposeResize(BaseModel):
    vm_id: str = Field(pattern=r"^vm-\d{3}$")
    target_size: Size
    reason: str = Field(min_length=5)


class Finish(BaseModel):
    summary: str = Field(min_length=1)


TOOLS = {
    "list_vms": (ListVMs, "List VMs with size and utilization signal."),
    "get_vm_metrics": (GetMetrics, "CPU/memory average and peak, freshness and cost for one VM."),
    "propose_resize": (ProposeResize, "Propose a one-step resize. Changes nothing until a human approves."),
    "finish": (Finish, "Finish with a short summary for the operator."),
}


def tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": desc,
                                              "parameters": model.model_json_schema()}}
            for name, (model, desc) in TOOLS.items()]


# ---------- policy: enforced in code, so a confused or prompt-injected model cannot bypass it ----------
def check_policy(vm: VM, target: str) -> list[str]:
    problems = []
    step = SIZES.index(target) - SIZES.index(vm.size)
    if vm.signal == "unknown":
        problems.append(f"{vm.id} has stale or missing metrics")
    if abs(step) != 1:
        problems.append(f"only one size step allowed ({vm.size} -> {target})")
    if step < 0 and vm.signal != "underutilized":
        problems.append(f"{vm.id} is not underutilized")
    if step > 0 and vm.signal != "overutilized":
        problems.append(f"{vm.id} is not overutilized")
    return problems


# ---------- execution ----------
def run_tool(name: str, raw_args: str, vms: list[VM]) -> dict:
    if name not in TOOLS or name == "finish":
        return {"ok": False, "error": f"unknown tool '{name}'"}
    try:
        args = TOOLS[name][0].model_validate(json.loads(raw_args or "{}"))
    except (json.JSONDecodeError, ValidationError) as e:
        return {"ok": False, "error": f"invalid arguments: {str(e).splitlines()[0]}"}

    by_id = {v.id: v for v in vms}
    if name == "list_vms":
        rows = [{"id": v.id, "name": v.name, "environment": v.environment, "size": v.size,
                 "signal": v.signal, "is_stale": v.is_stale, "description": v.description}
                for v in vms if args.environment in (None, v.environment)]
        return {"ok": True, "data": {"vms": rows}}

    vm = by_id.get(args.vm_id)
    if vm is None:
        return {"ok": False, "error": f"no VM with id {args.vm_id}"}
    if name == "get_vm_metrics":
        return {"ok": True, "data": vm.model_dump(mode="json", exclude={"description"})}

    # propose_resize
    problems = check_policy(vm, args.target_size)
    if problems:
        return {"ok": False, "error": "policy violation: " + "; ".join(problems)}
    return {"ok": True, "data": {"vm_id": vm.id, "from_size": vm.size, "to_size": args.target_size,
                                 "monthly_savings": MONTHLY_COST[vm.size] - MONTHLY_COST[args.target_size]}}


# ---------- fake cloud provider ----------
class CloudUnavailable(Exception):
    pass


class FakeCloud:
    def __init__(self, faults: set[str]):
        self.faults = faults
        self.sizes: dict[str, str] = {}  # applied resizes: vm_id -> new size

    def resize(self, vm_id: str, size: str) -> None:
        if "cloud_down" in self.faults:
            raise CloudUnavailable("503 Service Unavailable from compute API")
        self.sizes[vm_id] = size
